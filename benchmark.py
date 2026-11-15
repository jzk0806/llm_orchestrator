from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import statistics
from typing import Any, Sequence

from .local_inference import AssessmentClient
from .experiment import FreshRun
from .models import APIUsage, EventOutcome, Observation
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .workflow import OrchestratorConfig


def _usage(value: APIUsage | None) -> APIUsage:
    usage = APIUsage() if value is None else APIUsage.model_validate(value.model_dump())
    for name, item in usage.model_dump().items():
        values = item if isinstance(item, list) else [item]
        if any(isinstance(x, bool) or not math.isfinite(float(x)) or x < 0 for x in values):
            raise ValueError(f'Usage field {name} must contain finite nonnegative values.')
    return usage


def _accumulate(total: APIUsage, addition: APIUsage | None) -> None:
    checked = _usage(addition)
    for name in type(total).model_fields:
        value = getattr(checked, name)
        if isinstance(value, list):
            getattr(total, name).extend(value)
        else:
            setattr(total, name, getattr(total, name) + value)


def _latency_metrics(usage: APIUsage) -> dict[str, Any]:
    latencies = list(usage.request_latencies_ms)
    complete = len(latencies) == usage.calls and math.isclose(
        sum(latencies), usage.latency_ms, rel_tol=1e-9, abs_tol=1e-7)
    mean = statistics.mean(latencies) / 1000 if complete and latencies else None
    sd = statistics.stdev(latencies) / 1000 if complete and len(latencies) > 1 else None
    return {
        'latency_samples': len(latencies),
        'latency_complete': complete,
        'latency_per_call_seconds': {'mean': mean, 'sample_sd': sd},
        'cumulative_latency_hours': usage.latency_ms / 3_600_000,
        'cumulative_latency_status': 'complete' if complete else 'partial_or_inconsistent',
        'request_latencies_ms': latencies,
    }


def _metrics(usage: APIUsage, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        'intervals': len(records),
        'assessment_intervals': sum(record['assessment_scheduled'] for record in records),
        'natural_event_intervals': sum(record['natural_event_triggered'] for record in records),
        'failed_assessments': sum(record['assessment_failed'] for record in records),
        'calls': usage.calls,
        'schema_repairs': usage.schema_repairs,
        'input_tokens': usage.input_tokens,
        'output_tokens': usage.output_tokens,
        'input_tokens_millions': usage.input_tokens / 1_000_000,
        'output_tokens_millions': usage.output_tokens / 1_000_000,
        'embedding_calls': usage.embedding_calls,
        'embedding_input_tokens': usage.embedding_input_tokens,
        'embedding_latency_ms': usage.embedding_latency_ms,
        **_latency_metrics(usage),
    }


def _reduction(reference: float, triggered: float) -> float | None:
    return None if reference == 0 else 100.0 * (reference - triggered) / reference


def compare_inference_overhead(
    *, root: str | Path, observations: Sequence[Observation], outcomes: Sequence[EventOutcome],
    config: OrchestratorConfig, assessment_client: AssessmentClient,
    replication: int = 1, warmup_assessments: int = 1, run_id: str = 'table_iii_b',
    branch_order: tuple[str, str] = ('every_step', 'event_triggered'),
) -> dict[str, Any]:
    if isinstance(replication, bool) or not isinstance(replication, int) or replication < 1:
        raise ValueError('replication must be a positive integer.')
    if isinstance(warmup_assessments, bool) or not isinstance(warmup_assessments, int) or warmup_assessments < 0:
        raise ValueError('warmup_assessments must be a nonnegative integer.')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,49}', run_id):
        raise ValueError('run_id must be a safe identifier of at most 50 characters.')
    if len(branch_order) != 2 or set(branch_order) != {'every_step', 'event_triggered'}:
        raise ValueError('branch_order must contain every_step and event_triggered once each.')
    observed = tuple(Observation.model_validate(item).model_copy(deep=True) for item in observations)
    completed = tuple(EventOutcome.model_validate(item).model_copy(deep=True) for item in outcomes)
    if not observed or len(observed) != len(completed):
        raise ValueError('Supply one completed outcome per observation and at least one interval.')
    for index, point in enumerate(observed):
        if point.prosumer_id != observed[0].prosumer_id or point.step != index + 1:
            raise ValueError('Observations must cover one prosumer, starting at step 1 without gaps.')
        if index and (point.timestamp - observed[index-1].timestamp).total_seconds() != 900:
            raise ValueError('Observation timestamps must be consecutive 15-minute intervals.')
    settings = OrchestratorConfig.model_validate(config.model_dump()).model_copy(deep=True)
    base = Path(root).expanduser().resolve()
    controller_ids = {mode: f'{run_id}--{mode}' for mode in branch_order}
    for identifier in controller_ids.values():
        target = base / identifier / f'replication-{replication:04d}'
        if target.exists():
            raise FileExistsError('A benchmark controller/replication directory already exists.')
    runs = {
        mode: FreshRun.create(root=base, controller_id=controller_ids[mode], replication=replication,
                              config=settings, assessment_client=assessment_client, assessment_mode=mode)
        for mode in branch_order
    }
    warmup_controller = runs[branch_order[0]].controller
    session = warmup_controller.memory.load_session(observed[0].prosumer_id, settings.initial_agent)
    initial_state = {'observation': observed[0].model_copy(deep=True), 'session': session,
                     'active_before': session.active_agent}
    perceived = warmup_controller._perception_node(initial_state)
    reasons = warmup_controller._event_node({**initial_state, **perceived})['trigger_reasons']
    warmup_prompt = build_user_prompt(semantic_context=perceived['semantic_context'],
                                      current_step=observed[0].step, event_reasons=reasons,
                                      short_term_memory=[], long_term_memory=[])
    warmup_usage = APIUsage()
    warmup_failures = []
    for index in range(warmup_assessments):
        try:
            _, usage, _ = assessment_client.assess(warmup_prompt)
        except Exception as error:
            _accumulate(warmup_usage, getattr(error, 'usage', None))
            warmup_failures.append({'assessment': index + 1, 'error_type': type(error).__name__})
        else:
            _accumulate(warmup_usage, usage)
    arms: dict[str, dict[str, Any]] = {}
    for mode in branch_order:
        run = runs[mode]
        totals = APIUsage()
        records = []
        for point, outcome in zip(observed, completed):
            decision = run.controller.decide(point.model_copy(deep=True))
            _accumulate(totals, decision.api_usage)
            if decision.event_id is not None:
                completion = run.controller.complete_event(event_id=decision.event_id,
                    completed_step=point.step, outcome=outcome.model_copy(deep=True))
                _accumulate(totals, completion.api_usage)
            records.append({
                'step': point.step,
                'natural_event_triggered': decision.natural_event_triggered,
                'assessment_scheduled': decision.event_triggered,
                'assessment_failed': decision.event_triggered and decision.fallback_used,
                'selected_agent': decision.selected_agent.value,
                'trigger_reasons': list(decision.trigger_reasons),
                'calls': decision.api_usage.calls,
            })
        arms[mode] = {'directory': str(run.directory), 'database_path': str(run.database_path),
                      'metrics': _metrics(totals, records), 'usage': totals.model_dump(), 'records': records}
    reference = arms['every_step']['metrics']
    triggered = arms['event_triggered']['metrics']
    reductions = {name: _reduction(reference[name], triggered[name])
                  for name in ['calls', 'input_tokens', 'output_tokens']}
    reductions['cumulative_latency'] = (
        _reduction(reference['cumulative_latency_hours'], triggered['cumulative_latency_hours'])
        if reference['latency_complete'] and triggered['latency_complete'] else None)
    return {
        'scope': 'Local-generation overhead on identical caller-supplied observations and outcomes; no plant simulation or paper-result reproduction.',
        'branch_order': list(branch_order),
        'replication': replication,
        'input_intervals': len(observed),
        'complete_96_interval_days': len(observed) // 96,
        'remaining_intervals': len(observed) % 96,
        'warmup': {'assessments': warmup_assessments, 'usage': warmup_usage.model_dump(),
                   'failures': warmup_failures, 'included_in_measured_arms': False,
                   'prompt_sha256': hashlib.sha256((SYSTEM_PROMPT + '\n' + warmup_prompt).encode('utf-8')).hexdigest(),
                   **_latency_metrics(warmup_usage)},
        'arms': arms,
        'reductions_percent': reductions,
        'latency_definition': 'Individual local-generation durations, including repairs and failed generations; sample SD is unavailable with fewer than two complete samples.',
        'token_definition': 'Local-backend prompt and completion token counts; failed generations retain the usage available from the client.',
    }
