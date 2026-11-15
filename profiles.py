from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .local_inference import CallBudget, LocalAssessmentClient, LocalGeneration, LocalInferenceSettings
from .models import FeatureStats
from .workflow import OrchestratorConfig


def paper_controller_config(
    feature_stats: Mapping[str, FeatureStats | Mapping[str, float]], *,
    matched_switching: bool = False
) -> OrchestratorConfig:
    return OrchestratorConfig(
        lookback_intervals=16,
        price_z_trigger_threshold=2.0,
        state_change_trigger_threshold=0.75,
        maximum_event_gap=16,
        short_memory_capacity=8,
        long_memory_top_k=4,
        switching_penalty=0.0 if matched_switching else 0.05,
        handover_margin=0.0 if matched_switching else 0.05,
        salience_outcome_threshold=0.20,
        bin_low_z=-0.5,
        bin_high_z=0.5,
        trend_delta_z=0.25,
        minimum_price_history=4,
        feature_stats=dict(feature_stats),
    )


def paper_serving(
    checkpoint_dir: str | Path, *, max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.85, tensor_parallel_size: int = 1,
) -> LocalInferenceSettings:
    return LocalInferenceSettings(
        checkpoint_dir=str(checkpoint_dir), max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size,
    )


def make_assessment_client(
    settings: LocalInferenceSettings, *, request_limit: int | None = None,
    backend: Callable[[list[dict[str, str]], dict[str, Any]], LocalGeneration] | None = None,
) -> LocalAssessmentClient:
    return LocalAssessmentClient(
        settings, budget=CallBudget(request_limit) if request_limit is not None else None,
        backend=backend,
    )
