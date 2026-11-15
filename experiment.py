from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .local_inference import AssessmentCallError, AssessmentClient, DeterministicHashEmbeddingClient, EmbeddingClient, LocalAssessmentClient
from .memory import SQLiteMemoryStore
from .models import AGENT_ORDER, APIUsage, AgentName, LLMAssessment, MemoryRecord
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .workflow import OrchestratorConfig, PaperLLMOrchestrator
from .models import margin_reached, score_agent_candidates


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return value


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite.')
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be finite and satisfy its lower bound.')
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def global_step(day: int, interval: int) -> int:
    _integer(day, 'day')
    _integer(interval, 'interval')
    if interval > 96:
        raise ValueError('interval must be in 1..96.')
    return 96 * (day - 1) + interval


def pre_control_net_load(requested_load: float, pv_power: float) -> float:
    return _finite(requested_load, 'requested_load', 0.0) - _finite(pv_power, 'pv_power', 0.0)


def fit_seasonal_ramp_thresholds(
    training_days_by_season: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, float]:
    if not training_days_by_season:
        raise ValueError('At least one training season is required.')
    thresholds: dict[str, float] = {}
    for season, days in training_days_by_season.items():
        if not isinstance(season, str) or not season:
            raise ValueError('Each season needs a nonempty string identifier.')
        differences: list[float] = []
        for day in days:
            values = [_finite(value, 'training pre-control net load') for value in day]
            if len(values) != 96:
                raise ValueError('Each training day must contain exactly 96 pre-control net loads.')
            differences.extend(abs(right - left) for left, right in zip(values, values[1:]))
        if not differences:
            raise ValueError(f'Season {season!r} contains no complete training days.')
        differences.sort()
        position = 0.95 * (len(differences) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        thresholds[season] = differences[lower] + (position - lower) * (differences[upper] - differences[lower])
    return thresholds


@dataclass(frozen=True)
class AlertThresholds:
    pcc_import_rating_kw: float
    seasonal_ramp_thresholds_kw: Mapping[str, float]

    def __post_init__(self) -> None:
        rating = _finite(self.pcc_import_rating_kw, 'pcc_import_rating_kw', 0.0)
        if rating == 0.0:
            raise ValueError('PCC import rating must be positive.')
        if not self.seasonal_ramp_thresholds_kw:
            raise ValueError('Fixed training-season thresholds are required.')
        thresholds = {}
        for season, value in self.seasonal_ramp_thresholds_kw.items():
            if not isinstance(season, str) or not season:
                raise ValueError('Each season needs a nonempty string identifier.')
            thresholds[season] = _finite(value, 'seasonal ramp threshold', 0.0)
        object.__setattr__(self, 'pcc_import_rating_kw', rating)
        object.__setattr__(self, 'seasonal_ramp_thresholds_kw', MappingProxyType(thresholds))


@dataclass(frozen=True)
class AlertResult:
    step: int
    pre_control_net_load_kw: float
    high_import: bool
    two_step_import_alert: bool
    absolute_ramp_kw: float | None
    ramp_alert: bool
    active: bool
    onset: bool
    episodes_started: int


class GridAlertTracker:
    def __init__(self, thresholds: AlertThresholds) -> None:
        self._thresholds = thresholds
        self._previous_step = 0
        self._previous_net_load: float | None = None
        self._previous_high_import = False
        self._previous_active = False
        self._episodes = 0

    @property
    def thresholds(self) -> AlertThresholds:
        return self._thresholds

    def update(self, *, day: int, interval: int, requested_load: float, pv_power: float, season: str) -> AlertResult:
        step = global_step(day, interval)
        if step != self._previous_step + 1:
            raise ValueError('Alert inputs must start at day 1, interval 1 and remain consecutive across days.')
        if season not in self.thresholds.seasonal_ramp_thresholds_kw:
            raise ValueError('No fixed training threshold is available for this season.')
        net_load = pre_control_net_load(requested_load, pv_power)
        high_import = max(net_load, 0.0) > 0.9 * self.thresholds.pcc_import_rating_kw
        import_alert = high_import and self._previous_high_import
        ramp = None if self._previous_net_load is None else abs(net_load - self._previous_net_load)
        ramp_alert = ramp is not None and ramp > self.thresholds.seasonal_ramp_thresholds_kw[season]
        active = import_alert or ramp_alert
        onset = active and not self._previous_active
        episodes = self._episodes + int(onset)
        result = AlertResult(step, net_load, high_import, import_alert, ramp, ramp_alert, active, onset, episodes)
        self._previous_step = step
        self._previous_net_load = net_load
        self._previous_high_import = high_import
        self._previous_active = active
        self._episodes = episodes
        return result


@dataclass(frozen=True)
class FreshRun:
    directory: Path
    database_path: Path
    controller: PaperLLMOrchestrator

    @property
    def decisions_path(self) -> Path:
        return self.directory / 'decisions.jsonl'

    @property
    def snapshots_path(self) -> Path:
        return self.directory / 'snapshots.jsonl'

    @classmethod
    def create(
        cls, *, root: str | Path, controller_id: str, replication: int,
        config: OrchestratorConfig, assessment_client: AssessmentClient,
        embedding_client: EmbeddingClient | None = None,
        assessment_mode: str = 'event_triggered',
    ) -> FreshRun:
        if not isinstance(controller_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', controller_id):
            raise ValueError('controller_id must be a safe, nonempty run identifier.')
        _integer(replication, 'replication')
        if assessment_mode not in {'event_triggered', 'every_step'}:
            raise ValueError('assessment_mode must be event_triggered or every_step.')
        base = Path(root).expanduser().resolve()
        directory = base / controller_id / f'replication-{replication:04d}'
        if not directory.resolve().is_relative_to(base):
            raise ValueError('Run directory must stay within root.')
        directory.mkdir(parents=True, exist_ok=False)
        database = directory / 'memory.sqlite3'
        with database.open('xb'):
            pass
        memory = SQLiteMemoryStore(database)
        controller = PaperLLMOrchestrator(
            config=config.model_copy(deep=True), memory_store=memory,
            assessment_client=assessment_client,
            embedding_client=embedding_client if embedding_client is not None else DeterministicHashEmbeddingClient(),
            audit_log_path=directory / 'decisions.jsonl',
            snapshot_log_path=directory / 'snapshots.jsonl',
            assessment_mode=assessment_mode,
        )
        manifest = {'controller_id': controller_id, 'replication': replication,
                    'database_path': str(database), 'initial_memory': 'empty',
                    'assessment_mode': assessment_mode,
                    'decisions_path': str(directory / 'decisions.jsonl'),
                    'snapshots_path': str(directory / 'snapshots.jsonl'),
                    'step_convention': '96*(day-1)+interval',
                    'config': config.model_dump(mode='json')}
        with (directory / 'run.json').open('x', encoding='utf-8') as handle:
            handle.write(_json(manifest) + '\n')
        return cls(directory, database, controller)


@dataclass(frozen=True)
class FrozenReplayInput:
    input_id: str
    system_prompt: str
    user_prompt: str
    snapshot_json: str
    incumbent: AgentName
    grid_alert: bool
    switching_penalty: float
    handover_margin: float
    reference_agent: AgentName | None
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.input_id, self.system_prompt, self.user_prompt)):
            raise ValueError('Replay identifiers and prompts must be nonempty.')
        if not isinstance(self.grid_alert, bool):
            raise ValueError('grid_alert must be boolean.')
        object.__setattr__(self, 'switching_penalty', _finite(self.switching_penalty, 'switching_penalty', 0.0))
        object.__setattr__(self, 'handover_margin', _finite(self.handover_margin, 'handover_margin', 0.0))
        object.__setattr__(self, 'incumbent', AgentName(self.incumbent))
        if self.reference_agent is not None:
            object.__setattr__(self, 'reference_agent', AgentName(self.reference_agent))
        expected = hashlib.sha256((self.system_prompt + '\n' + self.user_prompt).encode('utf-8')).hexdigest()
        if self.prompt_sha256 != expected:
            raise ValueError('Frozen prompt hash does not match the supplied text.')
        json.loads(self.snapshot_json)


def freeze_replay_input(
    *, input_id: str, semantic_context: Mapping[str, Any], current_step: int,
    event_reasons: Sequence[str], short_term_memory: Sequence[MemoryRecord],
    long_term_memory: Sequence[MemoryRecord], incumbent: AgentName, grid_alert: bool,
    switching_penalty: float = 0.05, handover_margin: float = 0.05,
    reference_agent: AgentName | None = None, system_prompt: str = SYSTEM_PROMPT,
) -> FrozenReplayInput:
    _integer(current_step, 'current_step')
    context = json.loads(_json(dict(semantic_context)))
    if 'grid_alert' in context and context['grid_alert'] is not grid_alert:
        raise ValueError('The frozen context and decision gate must have the same grid-alert state.')
    short = [MemoryRecord.model_validate(item.model_dump()).model_copy(deep=True) for item in short_term_memory]
    long = [MemoryRecord.model_validate(item.model_dump()).model_copy(deep=True) for item in long_term_memory]
    for record in short + long:
        if record.completed_step >= current_step:
            raise ValueError('Replay memories must complete before the current decision.')
    prompt = build_user_prompt(semantic_context=context, current_step=current_step,
                               event_reasons=event_reasons, short_term_memory=short, long_term_memory=long)
    snapshot = _json({'current_step': current_step, 'semantic_context': context,
                      'event_reasons': list(event_reasons),
                      'short_term_memory': [item.model_dump(mode='json') for item in short],
                      'long_term_memory': [item.model_dump(mode='json') for item in long]})
    digest = hashlib.sha256((system_prompt + '\n' + prompt).encode('utf-8')).hexdigest()
    return FrozenReplayInput(input_id, system_prompt, prompt, snapshot, AgentName(incumbent), grid_alert,
                             switching_penalty, handover_margin,
                             None if reference_agent is None else AgentName(reference_agent), digest)


def load_replay_inputs(
    path: str | Path, *, reference_labels: Mapping[str, AgentName | str] | None = None,
) -> tuple[FrozenReplayInput, ...]:
    labels = {} if reference_labels is None else {key: AgentName(value) for key, value in reference_labels.items()}
    items = []
    seen = set()
    with Path(path).expanduser().open('r', encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                snapshot = json.loads(record['snapshot_json'])
                input_id = record['input_id']
                if not isinstance(input_id, str) or not input_id or input_id in seen:
                    raise ValueError('Snapshot input identifiers must be nonempty and unique.')
                expected = freeze_replay_input(
                    input_id=input_id, semantic_context=snapshot['semantic_context'],
                    current_step=snapshot['current_step'], event_reasons=snapshot['event_reasons'],
                    short_term_memory=[MemoryRecord.model_validate(value) for value in snapshot['short_term_memory']],
                    long_term_memory=[MemoryRecord.model_validate(value) for value in snapshot['long_term_memory']],
                    incumbent=record['incumbent'], grid_alert=record['grid_alert'],
                    switching_penalty=record['switching_penalty'], handover_margin=record['handover_margin'],
                    reference_agent=labels.get(input_id), system_prompt=record['system_prompt'],
                )
                if record['user_prompt'] != expected.user_prompt or record['prompt_sha256'] != expected.prompt_sha256:
                    raise ValueError('Frozen prompt, snapshot content and prompt hash must agree.')
                item = FrozenReplayInput(
                    input_id, record['system_prompt'], record['user_prompt'], record['snapshot_json'],
                    expected.incumbent, expected.grid_alert, expected.switching_penalty, expected.handover_margin,
                    labels.get(input_id), record['prompt_sha256'],
                )
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise ValueError(f'Invalid replay snapshot at line {line_number}: {type(error).__name__}.') from None
            seen.add(input_id)
            items.append(item)
    if not items:
        raise ValueError('The snapshot file contains no replay inputs.')
    return tuple(items)


@dataclass(frozen=True)
class ReplayAssessment:
    assessment: LLMAssessment | None
    usage: APIUsage = field(default_factory=APIUsage)
    failure: str | None = None


def _validated_usage(value: APIUsage | Mapping[str, Any]) -> APIUsage:
    payload = value.model_dump() if isinstance(value, APIUsage) else dict(value)
    usage = APIUsage.model_validate(payload)
    for name, amount in usage.model_dump().items():
        if name == 'request_latencies_ms':
            for duration in amount:
                _finite(duration, 'request latency', 0.0)
        else:
            _finite(amount, name, 0.0)
    return usage


def _exception_usage(error: Exception) -> APIUsage | None:
    value = getattr(error, 'usage', None)
    if value is None:
        return None
    try:
        return _validated_usage(value)
    except (ValueError, TypeError):
        return None


def _accumulate_usage(total: APIUsage, usage: APIUsage) -> None:
    for name in type(usage).model_fields:
        if name == 'request_latencies_ms':
            getattr(total, name).extend(getattr(usage, name))
        else:
            setattr(total, name, getattr(total, name) + getattr(usage, name))


def make_replay_assessor(
    client: LocalAssessmentClient,
) -> Callable[[str, str], ReplayAssessment]:
    if not isinstance(client, LocalAssessmentClient):
        raise TypeError('A configured LocalAssessmentClient is required.')

    def assess(system_prompt: str, user_prompt: str) -> ReplayAssessment:
        if system_prompt != SYSTEM_PROMPT:
            return ReplayAssessment(None, APIUsage(), 'system_prompt_mismatch')
        expected_hash = hashlib.sha256((system_prompt + '\n' + user_prompt).encode('utf-8')).hexdigest()
        try:
            assessment, usage, observed_hash = client.assess(user_prompt)
        except AssessmentCallError as error:
            usage = _validated_usage(error.usage)
            if error.prompt_sha256 != expected_hash:
                return ReplayAssessment(None, usage, 'error_prompt_hash_mismatch')
            code = str(error)
            allowed = {'prompt_too_large', 'call_budget_exhausted', 'invalid_assessment',
                       'local_generation_failed', 'local_backend_unavailable', 'local_checkpoint_invalid'}
            if code not in allowed:
                code = 'AssessmentCallError'
            return ReplayAssessment(None, usage, code)
        usage = _validated_usage(usage)
        if observed_hash != expected_hash:
            return ReplayAssessment(None, usage, 'prompt_hash_mismatch')
        return ReplayAssessment(assessment, usage)

    return assess


def _select(item: FrozenReplayInput, assessment: LLMAssessment | None) -> tuple[AgentName | None, AgentName]:
    active = AgentName(item.incumbent)
    if assessment is None:
        return None, AgentName.GRID if item.grid_alert else active
    _, _, proposal, gain = score_agent_candidates(assessment, active, item.switching_penalty)
    selected = AgentName.GRID if item.grid_alert else (
        proposal if margin_reached(gain, item.handover_margin) else active)
    return proposal, selected


def _mode(values: Sequence[str | None]) -> str | None:
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None
    return max((agent.value for agent in AGENT_ORDER), key=lambda value: counts[value])


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    valid = [value for value in values if value is not None]
    return {'per_replay': list(values), 'mean': statistics.mean(valid) if valid else None,
            'sample_sd': statistics.stdev(valid) if len(valid) > 1 else None}


def replay_frozen_inputs(
    inputs: Sequence[FrozenReplayInput],
    assessor: Callable[[str, str], ReplayAssessment], *, repetitions: int = 10,
) -> dict[str, Any]:
    _integer(repetitions, 'repetitions')
    items = tuple(inputs)
    if not items or len({item.input_id for item in items}) != len(items):
        raise ValueError('A nonempty set of uniquely identified frozen inputs is required.')
    runs: list[list[dict[str, Any]]] = []
    totals = APIUsage()
    unknown_usage = 0
    for repeat in range(repetitions):
        records = []
        for item in items:
            failure = None
            usage_known = True
            usage = None
            try:
                result = assessor(item.system_prompt, item.user_prompt)
                if not isinstance(result, ReplayAssessment):
                    raise TypeError('Assessor must return ReplayAssessment.')
                usage = _validated_usage(result.usage)
                if result.failure is not None and result.assessment is not None:
                    raise ValueError('An assessment cannot be both valid and failed.')
                assessment = None if result.assessment is None else LLMAssessment.model_validate(result.assessment.model_dump())
                failure = result.failure or ('invalid_assessment' if assessment is None else None)
            except Exception as error:
                assessment = None
                failure = type(error).__name__
                recovered_usage = _exception_usage(error)
                if recovered_usage is not None:
                    usage = recovered_usage
                if usage is None:
                    usage = APIUsage()
                    usage_known = False
                    unknown_usage += 1
            proposal, final = _select(item, assessment)
            _accumulate_usage(totals, usage)
            records.append({'input_id': item.input_id, 'replay': repeat + 1,
                            'prompt_sha256': item.prompt_sha256,
                            'proposal': None if proposal is None else proposal.value,
                            'final_decision': final.value, 'failure': failure,
                            'usage_known': usage_known, 'usage': usage.model_dump()})
        runs.append(records)
    proposal_modes = [_mode([run[index]['proposal'] for run in runs]) for index in range(len(items))]
    final_modes = [_mode([run[index]['final_decision'] for run in runs]) for index in range(len(items))]
    proposal_rates, final_rates, reference_rates = [], [], []
    labelled = sum(item.reference_agent is not None for item in items)
    for run in runs:
        proposal_rates.append(100.0 * sum(record['proposal'] is not None and record['proposal'] == proposal_modes[index]
                                         for index, record in enumerate(run)) / len(items))
        final_rates.append(100.0 * sum(record['final_decision'] == final_modes[index]
                                      for index, record in enumerate(run)) / len(items))
        reference_rates.append(None if not labelled else 100.0 * sum(
            record['proposal'] is not None and item.reference_agent is not None
            and record['proposal'] == AgentName(item.reference_agent).value
            for item, record in zip(items, run)) / labelled)
    return {'repetitions': repetitions, 'input_count': len(items), 'reference_label_count': labelled,
            'scheduled_assessments': repetitions * len(items),
            'failed_assessments': sum(record['failure'] is not None for run in runs for record in run),
            'modal_proposal_agreement_percent': _summary(proposal_rates),
            'final_decision_consistency_percent': _summary(final_rates),
            'initial_reference_agreement_percent': _summary(reference_rates),
            'proposal_modes': proposal_modes, 'final_decision_modes': final_modes,
            'usage_totals': totals.model_dump(), 'usage_source': 'assessor-reported',
            'unknown_usage_assessments': unknown_usage,
            'usage_totals_are_partial': bool(unknown_usage), 'records': runs}


def replay(
    snapshot_path: str | Path, client: LocalAssessmentClient, *, repetitions: int = 10,
    reference_labels: Mapping[str, AgentName | str] | None = None,
) -> dict[str, Any]:
    inputs = load_replay_inputs(snapshot_path, reference_labels=reference_labels)
    return replay_frozen_inputs(inputs, make_replay_assessor(client), repetitions=repetitions)
