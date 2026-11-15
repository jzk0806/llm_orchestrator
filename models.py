from __future__ import annotations
import hashlib
import json
from fractions import Fraction
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class AgentName(str, Enum):
    PROFIT = 'Profit'
    COMFORT = 'Comfort'
    SELF = 'Self'
    GRID = 'Grid'
AGENT_ORDER = (AgentName.PROFIT, AgentName.COMFORT, AgentName.SELF, AgentName.GRID)
FEATURE_ORDER = ('cleared_price', 'pv_power', 'requested_load', 'battery_soc', 'ev_soc', 'ev_available', 'pre_control_net_power', 'exchange_power')

class Observation(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    prosumer_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    timestamp: datetime
    cleared_price: float
    pv_power: float = Field(ge=0.0)
    requested_load: float = Field(ge=0.0)
    battery_soc: float = Field(ge=0.0, le=1.0)
    ev_soc: float = Field(ge=0.0, le=1.0)
    ev_available: bool
    exchange_power: float
    instruction: str = Field(default='none', max_length=240)
    grid_alert: bool = False

    @property
    def pre_control_net_power(self) -> float:
        return self.requested_load - self.pv_power

    @field_validator('instruction')
    @classmethod
    def canonical_instruction(cls, value: str) -> str:
        value = ''.join((character for character in value if character in '\t\n\r' or ord(character) >= 32))
        value = ' '.join(value.strip().split())
        return value if value else 'none'

    def numerical_vector(self) -> Dict[str, float]:
        return {'cleared_price': float(self.cleared_price), 'pv_power': float(self.pv_power), 'requested_load': float(self.requested_load), 'battery_soc': float(self.battery_soc), 'ev_soc': float(self.ev_soc), 'ev_available': float(self.ev_available), 'pre_control_net_power': self.pre_control_net_power, 'exchange_power': float(self.exchange_power)}

class FeatureStats(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    mean: float
    std: float = Field(gt=0.0)

class CriterionWeights(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, allow_inf_nan=False, strict=True)
    profit: float = Field(alias='p', ge=0.0, le=1.0)
    comfort: float = Field(alias='c', ge=0.0, le=1.0)
    sufficiency: float = Field(alias='s', ge=0.0, le=1.0)
    grid: float = Field(alias='g', ge=0.0, le=1.0)

    @model_validator(mode='after')
    def require_positive_total(self) -> 'CriterionWeights':
        if sum(self.as_dict().values()) <= 1e-12:
            raise ValueError('Priority weights must sum to more than 1e-12.')
        return self

    def normalized(self) -> 'CriterionWeights':
        values = self.as_dict()
        total = sum(values.values())
        return CriterionWeights(**{key: value / total for key, value in values.items()})

    def as_dict(self) -> Dict[str, float]:
        return {'profit': self.profit, 'comfort': self.comfort, 'sufficiency': self.sufficiency, 'grid': self.grid}

class CriterionScores(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, allow_inf_nan=False, strict=True)
    profit: float = Field(alias='p', ge=0.0, le=1.0)
    comfort: float = Field(alias='c', ge=0.0, le=1.0)
    sufficiency: float = Field(alias='s', ge=0.0, le=1.0)
    grid: float = Field(alias='g', ge=0.0, le=1.0)

    def as_dict(self) -> Dict[str, float]:
        return {'profit': self.profit, 'comfort': self.comfort, 'sufficiency': self.sufficiency, 'grid': self.grid}

class AgentScoreMatrix(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, allow_inf_nan=False)
    profit_agent: CriterionScores = Field(alias='Profit')
    comfort_agent: CriterionScores = Field(alias='Comfort')
    self_agent: CriterionScores = Field(alias='Self')
    grid_agent: CriterionScores = Field(alias='Grid')

    def as_mapping(self) -> Dict[AgentName, CriterionScores]:
        return {AgentName.PROFIT: self.profit_agent, AgentName.COMFORT: self.comfort_agent, AgentName.SELF: self.self_agent, AgentName.GRID: self.grid_agent}

class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, allow_inf_nan=False, strict=True)
    grid_analysis: str = Field(alias='g', min_length=1, max_length=180)
    prosumer_analysis: str = Field(alias='p', min_length=1, max_length=180)
    market_analysis: str = Field(alias='m', min_length=1, max_length=180)
    memory_analysis: str = Field(alias='r', min_length=1, max_length=180)
    synthesis: str = Field(alias='s', min_length=1, max_length=220)

class LLMAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, allow_inf_nan=False, strict=True)
    priority_weights: CriterionWeights = Field(alias='w')
    agent_scores: AgentScoreMatrix = Field(alias='u')
    evidence: EvidenceSummary = Field(alias='e')
    confidence: float = Field(alias='c', ge=0.0, le=1.0)

def score_agent_candidates(assessment: LLMAssessment, incumbent: AgentName, switching_penalty: float) -> tuple[Dict[str, float], Dict[str, float], AgentName, Fraction]:
    active = AgentName(incumbent)
    raw_weights = {name: Fraction(str(value)) for name, value in assessment.priority_weights.as_dict().items()}
    total = sum(raw_weights.values(), Fraction(0))
    penalty = Fraction(str(switching_penalty))
    matrix = assessment.agent_scores.as_mapping()
    exact_scores = {}
    for agent in AGENT_ORDER:
        row = matrix[agent].as_dict()
        score = sum((weight * Fraction(str(row[name])) for name, weight in raw_weights.items()), Fraction(0)) / total
        exact_scores[agent] = score - (penalty if agent != active else Fraction(0))
    maximum = max(exact_scores.values())
    tied = [agent for agent in AGENT_ORDER if maximum - exact_scores[agent] <= Fraction('1e-12')]
    proposal = active if active in tied else tied[0]
    gain = exact_scores[proposal] - exact_scores[active]
    weights = {name: float(weight / total) for name, weight in raw_weights.items()}
    scores = {agent.value: float(exact_scores[agent]) for agent in AGENT_ORDER}
    return weights, scores, proposal, gain

def margin_reached(gain: Fraction | str | float, margin: float) -> bool:
    return Fraction(str(gain)) >= Fraction(str(margin))

class OutcomeDelta(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    profit: float = Field(default=0.0, ge=-1.0, le=1.0)
    comfort: float = Field(default=0.0, ge=-1.0, le=1.0)
    sufficiency: float = Field(default=0.0, ge=-1.0, le=1.0)
    grid: float = Field(default=0.0, ge=-1.0, le=1.0)

    def magnitude(self) -> float:
        return max(abs(self.profit), abs(self.comfort), abs(self.sufficiency), abs(self.grid))

class EventOutcome(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    metric_improvement: OutcomeDelta
    constraint_violation: bool = False
    note: str = Field(default='', max_length=500)

class APIUsage(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    request_latencies_ms: List[float] = Field(default_factory=list)
    schema_repairs: int = 0
    embedding_calls: int = 0
    embedding_input_tokens: int = 0
    embedding_latency_ms: float = 0.0

    def add(self, other: 'APIUsage') -> 'APIUsage':
        return APIUsage(calls=self.calls + other.calls, input_tokens=self.input_tokens + other.input_tokens, output_tokens=self.output_tokens + other.output_tokens, latency_ms=self.latency_ms + other.latency_ms, request_latencies_ms=self.request_latencies_ms + other.request_latencies_ms, schema_repairs=self.schema_repairs + other.schema_repairs, embedding_calls=self.embedding_calls + other.embedding_calls, embedding_input_tokens=self.embedding_input_tokens + other.embedding_input_tokens, embedding_latency_ms=self.embedding_latency_ms + other.embedding_latency_ms)

class EmbeddingResult(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    vector: List[float] = Field(min_length=1)
    identifier: str = Field(min_length=1)
    usage: APIUsage = Field(default_factory=APIUsage)

class EventCompletionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str
    salient: bool
    long_term_indexed: bool
    embedding_warning: Optional[str] = None
    api_usage: APIUsage = Field(default_factory=APIUsage)

class ControllerSession(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prosumer_id: str
    active_agent: AgentName = AgentName.SELF
    last_event_step: Optional[int] = None
    last_instruction: str = 'none'
    last_grid_alert: bool = False
    history: List[Observation] = Field(default_factory=list)

class DecisionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prosumer_id: str
    step: int
    event_triggered: bool
    natural_event_triggered: bool = False
    trigger_reasons: List[str]
    api_called: bool
    active_agent_before: AgentName
    pre_gate_proposal: Optional[AgentName] = Field(default=None, description='Initial proposal before alert and handover rules, or null when no valid assessment exists; legacy serialized field name.')
    selected_agent: AgentName
    switched: bool
    handover_gain: float = 0.0
    q_scores: Dict[str, float] = Field(default_factory=dict)
    priority_weights: Dict[str, float] = Field(default_factory=dict)
    agent_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    reasoning_summary: Dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    event_id: Optional[str] = None
    retrieval_warning: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    prompt_sha256: Optional[str] = None
    api_usage: APIUsage = Field(default_factory=APIUsage)

    @property
    def initial_proposal(self) -> Optional[AgentName]:
        return self.pre_gate_proposal

class DecisionSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, frozen=True)
    input_id: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    snapshot_json: str
    incumbent: AgentName
    grid_alert: bool
    switching_penalty: float = Field(ge=0.0)
    handover_margin: float = Field(ge=0.0)
    prompt_sha256: str
    result_json: str

    @model_validator(mode='after')
    def validate_capture(self) -> 'DecisionSnapshot':
        expected = hashlib.sha256((self.system_prompt + '\n' + self.user_prompt).encode('utf-8')).hexdigest()
        if self.prompt_sha256 != expected:
            raise ValueError('The captured prompt hash does not match its exact text.')
        if not isinstance(json.loads(self.snapshot_json), dict):
            raise ValueError('The decision snapshot must encode an object.')
        result = json.loads(self.result_json)
        if not isinstance(result, dict) or result.get('prompt_sha256') != expected:
            raise ValueError('The recorded decision must use the captured prompt hash.')
        return self

class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str
    decision_step: int
    completed_step: int
    selected_agent: AgentName
    trigger_reasons: List[str]
    context_summary: Mapping[str, Any]
    outcome: EventOutcome
    similarity: Optional[float] = None

