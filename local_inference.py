from __future__ import annotations
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .models import APIUsage, EmbeddingResult, LLMAssessment
from .prompts import SYSTEM_PROMPT, build_repair_prompt

MODEL_ID = 'Qwen/Qwen3.5-9B'


class AssessmentCallError(RuntimeError):
    def __init__(self, code: str, usage: APIUsage | None = None, prompt_sha256: str | None = None):
        super().__init__(code)
        self.usage = usage or APIUsage()
        self.prompt_sha256 = prompt_sha256


class EmbeddingCallError(RuntimeError):
    def __init__(self, code: str, usage: APIUsage | None = None):
        super().__init__(code)
        self.usage = usage or APIUsage()


class AssessmentClient(Protocol):
    def assess(self, prompt: str) -> tuple[LLMAssessment, APIUsage, str]:
        ...


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> EmbeddingResult:
        ...


def _checkpoint_directory(value: str) -> Path:
    if not isinstance(value, str) or not value or '://' in value or value.startswith(('\\\\', '//')):
        raise ValueError('An existing absolute local checkpoint directory is required.')
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('An existing absolute local checkpoint directory is required.')
    path = path.resolve(strict=True)
    if str(path).startswith(('\\\\', '//')):
        raise ValueError('The checkpoint must be on a local filesystem.')
    try:
        config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        text = config.get('text_config', config)
        correct = (config.get('model_type') in {'qwen3_5', 'qwen3_5_text'}
                   and text.get('hidden_size') == 4096 and text.get('num_hidden_layers') == 32
                   and text.get('intermediate_size', 12288) == 12288)
    except (OSError, ValueError, TypeError, AttributeError):
        correct = False
    if not correct:
        raise ValueError('The local configuration must match Qwen3.5-9B.')
    return path


def _offline_runtime() -> None:
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
                 'VLLM_NO_USAGE_STATS', 'DO_NOT_TRACK'):
        os.environ[name] = '1'
    os.environ['VLLM_USE_MODELSCOPE'] = '0'


class LocalInferenceSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, frozen=True)
    checkpoint_dir: str
    max_tokens: int = Field(default=512, ge=1, le=512)
    seed: int = 42
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    top_p: float = Field(default=1.0, ge=1.0, le=1.0)
    enable_thinking: bool = False
    schema_repair_attempts: int = Field(default=1, ge=0, le=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    dtype: str = 'auto'
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)
    max_model_len: int = Field(default=8192, ge=1024)

    @model_validator(mode='after')
    def validate_settings(self):
        if self.enable_thinking:
            raise ValueError('Thinking must remain disabled.')
        if self.dtype not in {'auto', 'float16', 'bfloat16', 'float32'}:
            raise ValueError('Unsupported local inference dtype.')
        object.__setattr__(self, 'checkpoint_dir', str(_checkpoint_directory(self.checkpoint_dir)))
        return self


@dataclass(frozen=True)
class LocalGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = 'stop'

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError('Generation text must be a string.')
        for value in (self.prompt_tokens, self.completion_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Token counts must be nonnegative integers.')
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ValueError('The finish reason must be a string or None.')


class CallBudget:
    def __init__(self, limit: int = 12):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('Call budget must be a positive integer.')
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self):
        with self._lock:
            if self.used >= self.limit:
                raise RuntimeError('call_budget_exhausted')
            self.used += 1


def _load_assessment_json(text: str) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_JSON_key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('nonfinite_JSON_number')

    return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def parse_assessment(text: str) -> LLMAssessment:
    value = _load_assessment_json(text)
    if not isinstance(value, dict) or set(value) != {'w', 'u', 'e', 'c'}:
        raise ValueError('invalid_top_level_fields')
    if not isinstance(value['w'], dict) or set(value['w']) != {'p', 'c', 's', 'g'}:
        raise ValueError('invalid_weight_fields')
    if not isinstance(value['u'], dict) or set(value['u']) != {'Profit', 'Comfort', 'Self', 'Grid'}:
        raise ValueError('invalid_expert_fields')
    if any(not isinstance(row, dict) or set(row) != {'p', 'c', 's', 'g'} for row in value['u'].values()):
        raise ValueError('invalid_score_fields')
    if not isinstance(value['e'], dict) or set(value['e']) != {'g', 'p', 'm', 'r', 's'}:
        raise ValueError('invalid_evidence_fields')
    return LLMAssessment.model_validate(value)


def _repair_data(content: Any, response_status: str) -> tuple[str, str]:
    issues: list[dict[str, Any]] = []
    partial: dict[str, Any] = {}
    if response_status != 'schema_validation':
        issues.append({'path': '$', 'error': response_status})
    if not isinstance(content, str):
        issues.append({'path': '$', 'error': 'content_not_string'})
    elif not content.strip():
        issues.append({'path': '$', 'error': 'empty_content'})
    else:
        try:
            value = _load_assessment_json(content)
        except json.JSONDecodeError:
            issues.append({'path': '$', 'error': 'invalid_JSON'})
        except ValueError as exc:
            code = str(exc)
            issues.append({'path': '$', 'error': code if code in {'duplicate_JSON_key', 'nonfinite_JSON_number'} else 'invalid_JSON'})
        except (TypeError, RecursionError):
            issues.append({'path': '$', 'error': 'invalid_JSON'})
        else:
            def fields(item: Any, expected: tuple[str, ...], path: str) -> bool:
                if not isinstance(item, dict):
                    issues.append({'path': path, 'error': 'expected_object'})
                    return False
                missing = [key for key in expected if key not in item]
                if missing:
                    issues.append({'path': path, 'error': 'missing_keys', 'keys': missing})
                extra_count = len(set(item) - set(expected))
                if extra_count:
                    issues.append({'path': path, 'error': 'extra_keys', 'count': extra_count})
                return True

            def number(item: Any, path: str) -> bool:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    issues.append({'path': path, 'error': 'expected_number'})
                    return False
                if not 0 <= item <= 1:
                    issues.append({'path': path, 'error': 'number_outside_0_1'})
                    return False
                return True

            def numeric_row(item: Any, path: str) -> dict[str, Any]:
                row = {}
                if fields(item, ('p', 'c', 's', 'g'), path):
                    for key in ('p', 'c', 's', 'g'):
                        if key in item and number(item[key], path + '.' + key):
                            row[key] = item[key]
                return row

            if fields(value, ('w', 'u', 'e', 'c'), '$'):
                if 'w' in value:
                    weights = numeric_row(value['w'], '$.w')
                    if len(weights) == 4 and sum(weights.values()) <= 1e-12:
                        issues.append({'path': '$.w', 'error': 'weight_total_must_exceed_1e-12'})
                    elif weights:
                        partial['w'] = weights
                if 'u' in value and fields(value['u'], ('Profit', 'Comfort', 'Self', 'Grid'), '$.u'):
                    scores = {}
                    for agent in ('Profit', 'Comfort', 'Self', 'Grid'):
                        if agent in value['u']:
                            row = numeric_row(value['u'][agent], '$.u.' + agent)
                            if row:
                                scores[agent] = row
                    if scores:
                        partial['u'] = scores
                if 'e' in value and fields(value['e'], ('g', 'p', 'm', 'r', 's'), '$.e'):
                    evidence = {}
                    for key in ('g', 'p', 'm', 'r', 's'):
                        if key not in value['e']:
                            continue
                        item = value['e'][key]
                        error = None
                        if not isinstance(item, str):
                            error = 'expected_string'
                        elif not item:
                            error = 'empty_string'
                        elif len(item) > (220 if key == 's' else 180):
                            error = 'string_too_long'
                        if error:
                            issues.append({'path': '$.e.' + key, 'error': error})
                        else:
                            evidence[key] = item
                    if evidence:
                        partial['e'] = evidence
                if 'c' in value and number(value['c'], '$.c'):
                    partial['c'] = value['c']
    if not issues:
        issues.append({'path': '$', 'error': 'schema_validation_failed'})
    return (json.dumps(partial, ensure_ascii=False, separators=(',', ':'), allow_nan=False),
            json.dumps({'issues': issues}, ensure_ascii=False, separators=(',', ':')))


class LocalAssessmentClient:
    def __init__(self, settings: LocalInferenceSettings, *, budget: CallBudget | None = None,
                 backend: Callable[[list[dict[str, str]], dict[str, Any]], LocalGeneration] | None = None):
        if not isinstance(settings, LocalInferenceSettings):
            raise TypeError('LocalInferenceSettings is required.')
        self.settings = settings
        self.budget = budget
        self._backend = backend
        self._engine = None
        self._sampling_params = None
        self._lock = threading.RLock()
        _offline_runtime()

    def generation_parameters(self) -> dict[str, Any]:
        return {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': self.settings.max_tokens,
                'seed': self.settings.seed, 'n': 1,
                'json_schema': LLMAssessment.model_json_schema(by_alias=True),
                'chat_template_kwargs': {'enable_thinking': False}}

    def _load_engine(self) -> None:
        if self._backend is not None or self._engine is not None:
            return
        _offline_runtime()
        try:
            path = _checkpoint_directory(self.settings.checkpoint_dir)
            if not (path / 'tokenizer_config.json').is_file():
                raise ValueError('missing_tokenizer')
            if not any((path / name).is_file() for name in ('tokenizer.json', 'tokenizer.model')):
                raise ValueError('missing_tokenizer')
            if not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
                raise ValueError('missing_weights')
        except (OSError, ValueError):
            raise RuntimeError('local_checkpoint_invalid') from None
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError:
            raise RuntimeError('local_backend_unavailable') from None
        parameters = self.generation_parameters()
        try:
            sampling = SamplingParams(
                temperature=parameters['temperature'], top_p=parameters['top_p'],
                max_tokens=parameters['max_tokens'], seed=parameters['seed'], n=1,
                structured_outputs=StructuredOutputsParams(json=parameters['json_schema']),
            )
            engine = LLM(
                model=str(path), tokenizer=str(path), trust_remote_code=False,
                dtype=self.settings.dtype, tensor_parallel_size=self.settings.tensor_parallel_size,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
                max_model_len=self.settings.max_model_len, seed=self.settings.seed,
                generation_config='vllm', disable_log_stats=True,
                language_model_only=True,
            )
        except Exception:
            raise RuntimeError('local_backend_unavailable') from None
        self._sampling_params = sampling
        self._engine = engine

    def _generate(self, messages: list[dict[str, str]]) -> LocalGeneration:
        if self._backend is not None:
            result = self._backend(messages, self.generation_parameters())
            if not isinstance(result, LocalGeneration):
                raise TypeError('The local backend must return LocalGeneration.')
            return result
        outputs = self._engine.chat(
            messages, sampling_params=self._sampling_params, use_tqdm=False,
            chat_template_kwargs={'enable_thinking': False},
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError('One local completion is required.')
        output, completion = outputs[0], outputs[0].outputs[0]
        if output.prompt_token_ids is None:
            raise ValueError('Local prompt token counts are unavailable.')
        return LocalGeneration(completion.text, len(output.prompt_token_ids),
                               len(completion.token_ids), completion.finish_reason)

    def assess(self, prompt: str) -> tuple[LLMAssessment, APIUsage, str]:
        with self._lock:
            return self._assess_locked(prompt)

    def _assess_locked(self, prompt: str) -> tuple[LLMAssessment, APIUsage, str]:
        prompt_hash = hashlib.sha256((SYSTEM_PROMPT + '\n' + prompt).encode('utf-8')).hexdigest()
        usage = APIUsage()
        if len(prompt) > 60_000:
            raise AssessmentCallError('prompt_too_large', usage, prompt_hash)
        try:
            self._load_engine()
        except RuntimeError as error:
            code = str(error)
            if code not in {'local_checkpoint_invalid', 'local_backend_unavailable'}:
                code = 'local_backend_unavailable'
            raise AssessmentCallError(code, usage, prompt_hash) from None
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
        for attempt in range(self.settings.schema_repair_attempts + 1):
            try:
                if self.budget is not None:
                    self.budget.take()
            except RuntimeError:
                raise AssessmentCallError('call_budget_exhausted', usage, prompt_hash) from None
            started = time.perf_counter()
            usage.calls += 1
            usage.schema_repairs += int(attempt > 0)
            try:
                result = self._generate(messages)
            except Exception:
                elapsed = 1000 * (time.perf_counter() - started)
                usage.latency_ms += elapsed
                usage.request_latencies_ms.append(elapsed)
                raise AssessmentCallError('local_generation_failed', usage, prompt_hash) from None
            elapsed = 1000 * (time.perf_counter() - started)
            usage.latency_ms += elapsed
            usage.request_latencies_ms.append(elapsed)
            usage.input_tokens += result.prompt_tokens
            usage.output_tokens += result.completion_tokens
            content = result.text
            response_status = 'schema_validation'
            try:
                if result.finish_reason not in {'stop', None}:
                    response_status = 'output_token_limit' if result.finish_reason == 'length' else 'incomplete_response'
                    raise ValueError('incomplete_assessment')
                assessment = parse_assessment(content)
                return assessment, usage, prompt_hash
            except Exception:
                if attempt >= self.settings.schema_repair_attempts:
                    raise AssessmentCallError('invalid_assessment', usage, prompt_hash) from None
                partial, diagnostic = _repair_data(content, response_status)
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt},
                            {'role': 'user', 'content': build_repair_prompt(partial, diagnostic)}]
        raise AssessmentCallError('invalid_assessment', usage, prompt_hash)


class DeterministicHashEmbeddingClient:
    def __init__(self, dimensions: int = 256):
        if not 32 <= dimensions <= 4096:
            raise ValueError('Hash embedding dimensions must be between 32 and 4096.')
        self.dimensions = dimensions

    def embed(self, text: str) -> EmbeddingResult:
        vector = [0.0] * self.dimensions
        for token in re.findall(r'[\w.-]+', text.casefold()):
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            index = int.from_bytes(digest[:8], 'big') % self.dimensions
            vector[index] += 1.0 if digest[8] & 1 else -1.0
        norm = math.sqrt(sum(x*x for x in vector))
        if norm:
            vector = [x/norm for x in vector]
        return EmbeddingResult(vector=vector, identifier=f'signed-token-sha256-v1:{self.dimensions}')


