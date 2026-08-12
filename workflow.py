from __future__ import annotations
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import NotRequired, TypedDict
from .api import AssessmentCallError, AssessmentClient, ChatAPISettings, DeterministicHashEmbeddingClient, EmbeddingCallError, EmbeddingAPISettings, EmbeddingClient, OpenAICompatibleAssessmentClient, OpenAICompatibleEmbeddingClient
from .memory import SQLiteMemoryStore
from .models import AGENT_ORDER, FEATURE_ORDER, APIUsage, AgentName, ControllerSession, CriterionScores, DecisionResult, EventCompletionResult, EventOutcome, FeatureStats, LLMAssessment, MemoryRecord, Observation
from .prompts import SYSTEM_PROMPT, build_user_prompt

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    lookback_intervals: int = Field(default=16, ge=4)
    price_z_trigger_threshold: float = Field(default=2.0, gt=0.0)
    state_change_trigger_threshold: float = Field(default=0.75, gt=0.0)
    maximum_event_gap: int = Field(default=16, ge=1)
    short_memory_capacity: int = Field(default=8, ge=1)
    long_memory_top_k: int = Field(default=4, ge=1)
    switching_penalty: float = Field(default=0.05, ge=0.0)
    handover_margin: float = Field(default=0.05, ge=0.0)
    salience_outcome_threshold: float = Field(default=0.2, ge=0.0)
    bin_low_z: float = -0.5
    bin_high_z: float = 0.5
    trend_delta_z: float = Field(default=0.25, gt=0.0)
    minimum_price_history: int = Field(default=4, ge=2)
    initial_agent: AgentName = AgentName.SELF
    feature_stats: Dict[str, FeatureStats]

    @model_validator(mode='after')
    def validate_configuration(self) -> 'OrchestratorConfig':
        missing = set(FEATURE_ORDER) - set(self.feature_stats)
        extra = set(self.feature_stats) - set(FEATURE_ORDER)
        if missing or extra:
            raise ValueError(f'feature_stats mismatch; missing={missing}, extra={extra}')
        if self.bin_low_z >= self.bin_high_z:
            raise ValueError('bin_low_z must be smaller than bin_high_z')
        if self.minimum_price_history > self.lookback_intervals:
            raise ValueError('minimum_price_history cannot exceed lookback_intervals')
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> 'OrchestratorConfig':
        return cls.model_validate_json(Path(path).read_text(encoding='utf-8'))

class WorkflowState(TypedDict):
    observation: Observation
    session: ControllerSession
    active_before: AgentName
    semantic_context: NotRequired[Dict[str, Any]]
    normalized_current: NotRequired[Dict[str, float]]
    price_z_score: NotRequired[float]
    state_change_inf: NotRequired[float]
    event_triggered: NotRequired[bool]
    trigger_reasons: NotRequired[List[str]]
    short_memory: NotRequired[List[MemoryRecord]]
    long_memory: NotRequired[List[MemoryRecord]]
    assessment: NotRequired[Optional[LLMAssessment]]
    api_usage: NotRequired[APIUsage]
    embedding_usage: NotRequired[APIUsage]
    prompt_sha256: NotRequired[Optional[str]]
    retrieval_warning: NotRequired[Optional[str]]
    fallback_used: NotRequired[bool]
    fallback_reason: NotRequired[Optional[str]]
    q_scores: NotRequired[Dict[str, float]]
    priority_weights: NotRequired[Dict[str, float]]
    agent_scores: NotRequired[Dict[str, Dict[str, float]]]
    reasoning_summary: NotRequired[Dict[str, str]]
    confidence: NotRequired[float]
    pre_gate_proposal: NotRequired[AgentName]
    selected_agent: NotRequired[AgentName]
    handover_gain: NotRequired[float]
    event_id: NotRequired[Optional[str]]

class PaperLLMOrchestrator:

    def __init__(self, *, config: OrchestratorConfig, memory_store: SQLiteMemoryStore, assessment_client: AssessmentClient, embedding_client: EmbeddingClient, audit_log_path: Optional[str | Path]=None) -> None:
        self.config = config
        self.memory = memory_store
        self.assessment_client = assessment_client
        self.embedding_client = embedding_client
        self.audit_log_path = Path(audit_log_path).expanduser().resolve() if audit_log_path else None
        if self.audit_log_path:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_guard = threading.Lock()
        self._prosumer_locks: Dict[str, threading.RLock] = {}
        self.graph = self._build_graph()

    def _prosumer_lock(self, prosumer_id: str) -> threading.RLock:
        with self._lock_guard:
            return self._prosumer_locks.setdefault(prosumer_id, threading.RLock())

    @classmethod
    def from_environment(cls, *, config_path: str | Path, database_path: str | Path, env_file: Optional[str | Path]=None, audit_log_path: Optional[str | Path]=None, assessment_client: Optional[AssessmentClient]=None, embedding_client: Optional[EmbeddingClient]=None) -> 'PaperLLMOrchestrator':
        if env_file:
            load_dotenv(Path(env_file), override=False)
        config = OrchestratorConfig.from_json(config_path)
        if assessment_client is None:
            assessment_client = OpenAICompatibleAssessmentClient(ChatAPISettings(base_url=os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8000/v1'), api_key=os.getenv('LLM_API_KEY', 'EMPTY'), model=os.getenv('LLM_MODEL', 'Qwen/Qwen3.5-9B'), max_tokens=int(os.getenv('LLM_MAX_TOKENS', '512')), timeout_seconds=float(os.getenv('LLM_TIMEOUT_SECONDS', '60')), seed=int(os.environ['LLM_SEED']) if os.getenv('LLM_SEED', '').strip() else 42, constrained_json_schema=_env_bool('LLM_CONSTRAINED_JSON_SCHEMA', True), enable_thinking=_env_bool('LLM_ENABLE_THINKING', False), schema_repair_attempts=int(os.getenv('LLM_SCHEMA_REPAIRS', '1')), provider=os.getenv('LLM_PROVIDER', 'vllm').strip().lower()))
        if embedding_client is None:
            backend = os.getenv('EMBEDDING_BACKEND', 'api').strip().lower()
            if backend == 'api':
                embedding_client = OpenAICompatibleEmbeddingClient(EmbeddingAPISettings(base_url=os.getenv('EMBEDDING_BASE_URL', 'http://127.0.0.1:8001/v1'), api_key=os.getenv('EMBEDDING_API_KEY', 'EMPTY'), model=os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'), revision=os.getenv('EMBEDDING_REVISION', 'frozen-server-revision'), timeout_seconds=float(os.getenv('EMBEDDING_TIMEOUT_SECONDS', '30')), dimensions=int(os.environ['EMBEDDING_DIMENSIONS']) if os.getenv('EMBEDDING_DIMENSIONS', '').strip() else None, encoding_format=os.getenv('EMBEDDING_ENCODING_FORMAT', '').strip() or None))
            elif backend == 'hash':
                embedding_client = DeterministicHashEmbeddingClient(dimensions=int(os.getenv('HASH_EMBEDDING_DIMENSIONS', '256')))
            else:
                raise ValueError("EMBEDDING_BACKEND must be 'api' or 'hash'.")
        return cls(config=config, memory_store=SQLiteMemoryStore(database_path), assessment_client=assessment_client, embedding_client=embedding_client, audit_log_path=audit_log_path)

    def _normalize(self, observation: Observation) -> Dict[str, float]:
        vector = observation.numerical_vector()
        return {name: (vector[name] - self.config.feature_stats[name].mean) / self.config.feature_stats[name].std for name in FEATURE_ORDER}

    def _bin(self, normalized_value: float) -> str:
        if normalized_value < self.config.bin_low_z:
            return 'low'
        if normalized_value > self.config.bin_high_z:
            return 'high'
        return 'normal'

    def _trend(self, values: Sequence[float]) -> str:
        if len(values) < 2:
            return 'insufficient_history'
        split = max(1, len(values) // 2)
        earlier = values[:split]
        later = values[split:]
        delta = sum(later) / len(later) - sum(earlier) / len(earlier)
        if delta >= self.config.trend_delta_z:
            return 'increasing'
        if delta <= -self.config.trend_delta_z:
            return 'decreasing'
        return 'stable'

    def _price_z_score(self, previous_history: Sequence[Observation], current_price: float) -> float:
        prices = [item.cleared_price for item in previous_history[-self.config.lookback_intervals:]]
        if len(prices) < self.config.minimum_price_history:
            return 0.0
        mean = sum(prices) / len(prices)
        variance = sum(((price - mean) ** 2 for price in prices)) / len(prices)
        std = math.sqrt(variance)
        if std <= 1e-09:
            return 0.0 if abs(current_price - mean) <= 1e-09 else math.copysign(10.0, current_price - mean)
        return (current_price - mean) / std

    def _semantic_mapping(self, state: WorkflowState) -> Dict[str, Any]:
        observation = state['observation']
        previous_history = state['session'].history
        window = previous_history[-self.config.lookback_intervals:] + [observation]
        normalized_window = [self._normalize(item) for item in window]
        normalized_current = normalized_window[-1]
        binned_window = [{**{name: self._bin(row[name]) for name in FEATURE_ORDER}} for item, row in zip(window, normalized_window)]
        trends = {name: self._trend([row[name] for row in normalized_window]) for name in FEATURE_ORDER}
        price_z = self._price_z_score(previous_history, observation.cleared_price)
        if previous_history:
            previous_normalized = self._normalize(previous_history[-1])
            state_change_inf = max((abs(normalized_current[name] - previous_normalized[name]) for name in FEATURE_ORDER))
        else:
            state_change_inf = 0.0
        context = {'causal_boundary': 'All values were available before the control action at this step; no realized value after the decision step is included.', 'price_regime': {'level': self._bin(price_z), 'definition': 'relative to preceding observations only', 'preceding_observations': min(len(previous_history), self.config.lookback_intervals)}, 'trends': trends, 'asset_state': {'ev_available': observation.ev_available, 'normalized_bins': {name: self._bin(normalized_current[name]) for name in ('pv_power', 'requested_load', 'battery_soc', 'ev_soc', 'predicted_net_power')}}, 'pcc_state': {'exchange_bin': self._bin(normalized_current['exchange_power']), 'exchange_trend': trends['exchange_power']}, 'instruction': observation.instruction, 'grid_alert': observation.grid_alert, 'binned_trailing_window': binned_window}
        return {'semantic_context': context, 'normalized_current': normalized_current, 'price_z_score': price_z, 'state_change_inf': state_change_inf}

    def _perception_node(self, state: WorkflowState) -> Dict[str, Any]:
        return self._semantic_mapping(state)

    def _event_node(self, state: WorkflowState) -> Dict[str, Any]:
        observation = state['observation']
        session = state['session']
        reasons: List[str] = []
        if not session.history:
            reasons.append('initialization')
        if session.history and observation.instruction != session.last_instruction:
            reasons.append('instruction_change')
        if observation.grid_alert:
            reasons.append('grid_alert')
        if abs(state['price_z_score']) >= self.config.price_z_trigger_threshold:
            reasons.append('trailing_price_deviation')
        if state['state_change_inf'] >= self.config.state_change_trigger_threshold:
            reasons.append('material_state_change')
        if session.last_event_step is not None and observation.step - session.last_event_step >= self.config.maximum_event_gap:
            reasons.append('timeout')
        return {'event_triggered': bool(reasons), 'trigger_reasons': reasons}

    @staticmethod
    def _route_event(state: WorkflowState) -> str:
        return 'retrieve' if state['event_triggered'] else 'retain'

    def _retrieve_node(self, state: WorkflowState) -> Dict[str, Any]:
        observation = state['observation']
        context_text = _canonical_json(state['semantic_context'])
        short_memory = self.memory.recent_completed(prosumer_id=observation.prosumer_id, before_step=observation.step, limit=self.config.short_memory_capacity)
        try:
            embedding_result = self.embedding_client.embed(context_text)
            long_memory = self.memory.similar_salient(prosumer_id=observation.prosumer_id, before_step=observation.step, query_embedding=embedding_result.vector, embedding_identifier=embedding_result.identifier, top_k=self.config.long_memory_top_k)
        except Exception as error:
            long_memory = []
            failed_usage = error.usage if isinstance(error, EmbeddingCallError) else APIUsage()
            return {'short_memory': short_memory, 'long_memory': long_memory, 'embedding_usage': failed_usage, 'retrieval_warning': f'LTM retrieval unavailable: {type(error).__name__}: {error}'}
        return {'short_memory': short_memory, 'long_memory': long_memory, 'embedding_usage': embedding_result.usage}

    def _reason_node(self, state: WorkflowState) -> Dict[str, Any]:
        prompt = build_user_prompt(semantic_context=state['semantic_context'], current_step=state['observation'].step, event_reasons=state['trigger_reasons'], short_term_memory=state.get('short_memory', []), long_term_memory=state.get('long_memory', []))
        try:
            assessment, usage, prompt_hash = self.assessment_client.assess(prompt)
            return {'assessment': assessment, 'api_usage': state.get('embedding_usage', APIUsage()).add(usage), 'prompt_sha256': prompt_hash, 'fallback_used': False}
        except Exception as error:
            retrieval_warning = state.get('retrieval_warning')
            parts = [f'LLM assessment unavailable: {type(error).__name__}: {error}']
            if retrieval_warning:
                parts.append(retrieval_warning)
            if isinstance(error, AssessmentCallError):
                chat_usage = error.usage
                prompt_hash = error.prompt_sha256
            else:
                chat_usage = APIUsage(calls=1)
                prompt_hash = hashlib.sha256((SYSTEM_PROMPT + '\n' + prompt).encode('utf-8')).hexdigest()
            return {'assessment': None, 'api_usage': state.get('embedding_usage', APIUsage()).add(chat_usage), 'prompt_sha256': prompt_hash, 'fallback_used': True, 'fallback_reason': ' | '.join(parts)}

    def _score_node(self, state: WorkflowState) -> Dict[str, Any]:
        active = state['active_before']
        assessment = state.get('assessment')
        if assessment is None:
            return {'q_scores': {agent.value: 0.0 if agent == active else -1.0 for agent in AGENT_ORDER}, 'priority_weights': {}, 'agent_scores': {}, 'reasoning_summary': {}, 'confidence': 0.0, 'pre_gate_proposal': active, 'handover_gain': 0.0}
        weights = assessment.priority_weights.normalized().as_dict()
        score_matrix = assessment.agent_scores.as_mapping()
        q_scores: Dict[str, float] = {}
        serializable_scores: Dict[str, Dict[str, float]] = {}
        for agent in AGENT_ORDER:
            scores = score_matrix[agent].as_dict()
            serializable_scores[agent.value] = scores
            weighted_score = sum((weights[key] * scores[key] for key in weights))
            if agent != active:
                weighted_score -= self.config.switching_penalty
            q_scores[agent.value] = weighted_score
        proposal = active
        best_score = q_scores[active.value]
        for agent in AGENT_ORDER:
            candidate = q_scores[agent.value]
            if candidate > best_score + 1e-12:
                proposal = agent
                best_score = candidate
        handover_gain = q_scores[proposal.value] - q_scores[active.value]
        evidence = assessment.evidence
        return {'q_scores': q_scores, 'priority_weights': weights, 'agent_scores': serializable_scores, 'reasoning_summary': {'grid_analysis': evidence.grid_analysis, 'prosumer_analysis': evidence.prosumer_analysis, 'market_analysis': evidence.market_analysis, 'memory_analysis': evidence.memory_analysis, 'synthesis': evidence.synthesis}, 'confidence': assessment.confidence, 'pre_gate_proposal': proposal, 'handover_gain': handover_gain}

    def _safety_handover_node(self, state: WorkflowState) -> Dict[str, Any]:
        active = state['active_before']
        if state['observation'].grid_alert:
            selected = AgentName.GRID
        elif state.get('fallback_used', False):
            selected = active
        elif state['pre_gate_proposal'] != active and state['handover_gain'] >= self.config.handover_margin:
            selected = state['pre_gate_proposal']
        else:
            selected = active
        return {'selected_agent': selected}

    @staticmethod
    def _retain_node(state: WorkflowState) -> Dict[str, Any]:
        active = state['active_before']
        return {'short_memory': [], 'long_memory': [], 'assessment': None, 'api_usage': APIUsage(), 'embedding_usage': APIUsage(), 'prompt_sha256': None, 'fallback_used': False, 'fallback_reason': None, 'q_scores': {}, 'priority_weights': {}, 'agent_scores': {}, 'reasoning_summary': {}, 'confidence': 0.0, 'pre_gate_proposal': active, 'selected_agent': active, 'handover_gain': 0.0}

    def _persist_node(self, state: WorkflowState) -> Dict[str, Any]:
        observation = state['observation']
        session = state['session'].model_copy(deep=True)
        event_id: Optional[str] = None
        if state['event_triggered']:
            event_id = self.memory.stage_event(prosumer_id=observation.prosumer_id, decision_step=observation.step, context=state['semantic_context'], selected_agent=state['selected_agent'], trigger_reasons=state['trigger_reasons'])
            session.last_event_step = observation.step
        session.active_agent = state['selected_agent']
        session.last_instruction = observation.instruction
        session.history = (session.history + [observation])[-self.config.lookback_intervals:]
        self.memory.save_session(session)
        return {'session': session, 'event_id': event_id}

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node('causal_perception', self._perception_node)
        builder.add_node('event_trigger', self._event_node)
        builder.add_node('retrieve_dual_memory', self._retrieve_node)
        builder.add_node('react_cot_assessment', self._reason_node)
        builder.add_node('deterministic_score', self._score_node)
        builder.add_node('safety_and_handover_gate', self._safety_handover_node)
        builder.add_node('retain_active_agent', self._retain_node)
        builder.add_node('persist_state', self._persist_node)
        builder.add_edge(START, 'causal_perception')
        builder.add_edge('causal_perception', 'event_trigger')
        builder.add_conditional_edges('event_trigger', self._route_event, {'retrieve': 'retrieve_dual_memory', 'retain': 'retain_active_agent'})
        builder.add_edge('retrieve_dual_memory', 'react_cot_assessment')
        builder.add_edge('react_cot_assessment', 'deterministic_score')
        builder.add_edge('deterministic_score', 'safety_and_handover_gate')
        builder.add_edge('safety_and_handover_gate', 'persist_state')
        builder.add_edge('retain_active_agent', 'persist_state')
        builder.add_edge('persist_state', END)
        return builder.compile()

    def decide(self, observation: Observation) -> DecisionResult:
        with self._prosumer_lock(observation.prosumer_id):
            return self._decide_locked(observation)

    def _decide_locked(self, observation: Observation) -> DecisionResult:
        session = self.memory.load_session(observation.prosumer_id, self.config.initial_agent)
        if session.history and observation.step <= session.history[-1].step:
            raise ValueError('Observation steps must be strictly increasing for each prosumer.')
        final_state = self.graph.invoke({'observation': observation, 'session': session, 'active_before': session.active_agent})
        result = DecisionResult(prosumer_id=observation.prosumer_id, step=observation.step, event_triggered=final_state['event_triggered'], trigger_reasons=final_state['trigger_reasons'], api_called=final_state['api_usage'].calls > 0, active_agent_before=final_state['active_before'], pre_gate_proposal=final_state['pre_gate_proposal'], selected_agent=final_state['selected_agent'], switched=final_state['selected_agent'] != final_state['active_before'], handover_gain=final_state['handover_gain'], q_scores=final_state['q_scores'], priority_weights=final_state['priority_weights'], agent_scores=final_state['agent_scores'], reasoning_summary=final_state['reasoning_summary'], confidence=final_state['confidence'], event_id=final_state.get('event_id'), retrieval_warning=final_state.get('retrieval_warning'), fallback_used=final_state.get('fallback_used', False), fallback_reason=final_state.get('fallback_reason'), prompt_sha256=final_state.get('prompt_sha256'), api_usage=final_state['api_usage'])
        self._write_audit(result)
        return result

    def complete_event(self, *, event_id: str, completed_step: int, outcome: EventOutcome) -> EventCompletionResult:
        row = self.memory.pending_event(event_id)
        with self._prosumer_lock(row['prosumer_id']):
            embedding_warning: Optional[str] = None
            embedding_usage = APIUsage()
            embedding_vector = None
            embedding_identifier = None
            should_index = self.memory.will_be_salient(row=row, outcome=outcome, salience_outcome_threshold=self.config.salience_outcome_threshold)
            if should_index:
                try:
                    embedding_result = self.embedding_client.embed(row['context_text'])
                    embedding_vector = embedding_result.vector
                    embedding_identifier = embedding_result.identifier
                    embedding_usage = embedding_result.usage
                except Exception as error:
                    if isinstance(error, EmbeddingCallError):
                        embedding_usage = error.usage
                    embedding_warning = f'LTM indexing unavailable: {type(error).__name__}: {error}'
            salient = self.memory.complete_event(event_id=event_id, completed_step=completed_step, outcome=outcome, embedding=embedding_vector, embedding_identifier=embedding_identifier, salience_outcome_threshold=self.config.salience_outcome_threshold)
            return EventCompletionResult(event_id=event_id, salient=salient, long_term_indexed=salient and embedding_vector is not None, embedding_warning=embedding_warning, api_usage=embedding_usage)

    def _write_audit(self, result: DecisionResult) -> None:
        if self.audit_log_path is None:
            return
        with self.audit_log_path.open('a', encoding='utf-8') as stream:
            stream.write(result.model_dump_json() + '\n')

class RLPolicy(Protocol):

    def __call__(self, state: Mapping[str, Any]) -> Any:
        pass

class RLAgentRegistry:

    def __init__(self) -> None:
        self._policies: Dict[AgentName, RLPolicy] = {}

    def register(self, agent: AgentName, policy: RLPolicy) -> None:
        self._policies[agent] = policy

    def validate_complete(self) -> None:
        missing = set(AGENT_ORDER) - set(self._policies)
        if missing:
            raise RuntimeError(f'Missing RL policies: {sorted((item.value for item in missing))}')

    def dispatch(self, decision: DecisionResult, rl_state: Mapping[str, Any]) -> Any:
        self.validate_complete()
        return self._policies[decision.selected_agent](rl_state)
