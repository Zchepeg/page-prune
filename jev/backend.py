from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .questions import AD_TREATMENT, TypedQuestion
from .types import DecisionLevel
from runtime.state_serialization import validate_state_schema


@dataclass(frozen=True, slots=True)
class BackendResult:
    probabilities: dict[str, float]
    level: DecisionLevel
    model_version: str
    diagnostics: dict[str, object]


class DecisionBackend(Protocol):
    def decide(self, state: str, question: TypedQuestion) -> BackendResult: ...


class HeuristicBackend:
    """Deterministic development backend; replace with AnyJev in production."""

    model_version = "heuristic-dev-v1"

    def runtime_status(self) -> dict[str, object]:
        return {"backend": "heuristic", "loaded": True}

    def decide(self, state: str, question: TypedQuestion) -> BackendResult:
        payload = json.loads(state)
        evidence = set(payload.get("detector_evidence", []))
        positive = sum(
            weight
            for name, weight in {
                "known_ad_host": 2.8,
                "ad_semantic_marker": 1.5,
                "ad_like_url": 1.1,
                "common_ad_dimensions": 0.7,
                "large_fixed_overlay": 0.4,
            }.items()
            if name in evidence
        )
        negative = 2.5 if "protected_element_type" in evidence else 0.0
        logit = positive - negative - 1.0
        block = 1.0 / (1.0 + math.exp(-logit))
        uncertainty = max(0.05, 0.42 - abs(block - 0.5))
        allow = max(0.0, 1.0 - block - uncertainty)
        total = block + allow + uncertainty
        return BackendResult(
            probabilities={
                "ad": block / total,
                "non_ad": allow / total,
                "insufficient_evidence": uncertainty / total,
            },
            level=DecisionLevel.L0,
            model_version=self.model_version,
            diagnostics={"backend": "heuristic", "production_ready": False},
        )


class QwenPointerBackend:
    """Project-owned Qwen + LoRA + pointer-head inference backend."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        quantization: str = "",
        gpu_memory_budget_mb: int = 0,
        inference_max_length: int = 0,
        micro_batch_size: int = 2,
    ):
        import os

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch
        from modeling.decision_model import load_decision_runtime
        from modeling.encoding import DecisionExample, encode_example

        self._torch = torch
        self._DecisionExample = DecisionExample
        self._encode_example = encode_example
        self.quantization = quantization.lower().strip()
        self.gpu_memory_budget_mb = max(0, int(gpu_memory_budget_mb))
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.model, self.tokenizer, metadata = load_decision_runtime(
            checkpoint,
            device,
            self.quantization,
            self.gpu_memory_budget_mb,
        )
        self.device = device
        self.metadata = metadata
        self.model_labels = tuple(str(label) for label in metadata.get("labels", ("ad", "non_ad", "insufficient_evidence")))
        checkpoint_max_length = int(self.metadata["max_length"])
        self.max_length = min(checkpoint_max_length, inference_max_length) if inference_max_length > 0 else checkpoint_max_length
        self._inference_lock = threading.Lock()

    def runtime_status(self) -> dict[str, object]:
        status: dict[str, object] = {
            "backend": "qwen_pointer",
            "loaded": True,
            "quantization": self.quantization or "bf16",
            "inference_max_length": self.max_length,
            "micro_batch_size": self.micro_batch_size,
        }
        if self.device.startswith("cuda"):
            status["cuda_allocated_mb"] = round(self._torch.cuda.memory_allocated(self.device) / 1024**2, 1)
            status["cuda_reserved_mb"] = round(self._torch.cuda.memory_reserved(self.device) / 1024**2, 1)
        return status

    def decide(self, state: str, question: TypedQuestion) -> BackendResult:
        return self.decide_many([state], question)[0]

    def decide_many(self, states: list[str], question: TypedQuestion) -> list[BackendResult]:
        if question.name == AD_TREATMENT.name:
            for state in states:
                validate_state_schema(state, self.metadata.get("state_schema"))
        encoded_batch = [
            self._encode_example(
                self.tokenizer,
                self._DecisionExample(state, self.metadata.get("instruction", question.text) if question.name == AD_TREATMENT.name else question.text, self.model_labels),
                self.max_length,
            )
            for state in states
        ]
        values_batch: list[list[float] | None] = [None] * len(encoded_batch)
        order = sorted(range(len(encoded_batch)), key=lambda index: int(encoded_batch[index]["input_ids"].shape[0]))
        with self._inference_lock, self._torch.inference_mode():
            for start in range(0, len(order), self.micro_batch_size):
                indices = order[start : start + self.micro_batch_size]
                chunk = [encoded_batch[index] for index in indices]
                logits_batch = self.model.forward_encoded_batch(chunk)
                for index, logits in zip(indices, logits_batch):
                    values_batch[index] = self._torch.softmax(logits.float(), dim=-1).cpu().tolist()
        if any(values is None for values in values_batch):
            raise RuntimeError("inference batch did not produce every result")
        return [
            self._result(values, encoded["truncated"])
            for values, encoded in zip(values_batch, encoded_batch)
        ]

    def _result(self, values: list[float], truncated: bool) -> BackendResult:
        diagnostics = {
            "backend": "qwen_pointer",
            "base_model": str(self.metadata.get("requested_model") or self.metadata["base_model"]),
            "temperature": self.metadata.get("temperature", 1.0),
            "truncated": truncated,
            "quantization": self.quantization or "bf16",
            "inference_max_length": self.max_length,
            "gpu_memory_budget_mb": self.gpu_memory_budget_mb,
            "micro_batch_size": self.micro_batch_size,
        }
        if self.device.startswith("cuda"):
            diagnostics.update(
                {
                    "cuda_allocated_mb": round(self._torch.cuda.memory_allocated(self.device) / 1024**2, 1),
                    "cuda_reserved_mb": round(self._torch.cuda.memory_reserved(self.device) / 1024**2, 1),
                }
            )
        return BackendResult(
            probabilities=dict(zip(self.model_labels, values)),
            level=DecisionLevel.L2,
            model_version=str(self.metadata.get("architecture", "qwen-pointer-decision-v1")),
            diagnostics=diagnostics,
        )


class LazyQwenPointerBackend:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        quantization: str = "",
        gpu_memory_budget_mb: int = 0,
        inference_max_length: int = 0,
        micro_batch_size: int = 2,
    ):
        self._settings = (
            checkpoint,
            device,
            quantization,
            gpu_memory_budget_mb,
            inference_max_length,
            micro_batch_size,
        )
        self._backend: QwenPointerBackend | None = None
        self._load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._phase = 'idle'
        self._load_ms = None
        self._warmup_ms = None
        self._last_error = None
        self._retry_at = 0.0

    def prepare(self, states, question):
        if not states:
            return True
        with self._state_lock:
            if self._phase == 'ready':
                return True
            if self._phase in {'loading', 'warming'} or time.monotonic() < self._retry_at:
                return False
            self._phase = 'loading'
            warm_states = list(states[:max(1, int(self._settings[-1]))])
            threading.Thread(target=self._prepare_worker, args=(warm_states, question), daemon=True).start()
            return False

    def _prepare_worker(self, states, question):
        try:
            started = time.perf_counter()
            backend = self._get_backend()
            with self._state_lock:
                self._load_ms = round((time.perf_counter()-started)*1000, 3)
                self._phase = 'warming'
            started = time.perf_counter()
            backend.decide_many(states, question)
            if len(states) > 1:
                backend.decide_many(states[:1], question)
            with self._state_lock:
                self._warmup_ms = round((time.perf_counter()-started)*1000, 3)
                self._phase = 'ready'
                self._last_error = None
        except Exception as exc:
            with self._state_lock:
                self._last_error = type(exc).__name__
                self._phase = 'failed'
                self._retry_at = time.monotonic()+30

    def _get_backend(self) -> QwenPointerBackend:
        if self._backend is None:
            with self._load_lock:
                if self._backend is None:
                    self._backend = QwenPointerBackend(*self._settings)
        return self._backend

    def decide(self, state: str, question: TypedQuestion) -> BackendResult:
        result = self._get_backend().decide(state, question)
        self._mark_direct_ready()
        return result

    def decide_many(self, states: list[str], question: TypedQuestion) -> list[BackendResult]:
        results = self._get_backend().decide_many(states, question)
        self._mark_direct_ready()
        return results

    def _mark_direct_ready(self):
        with self._state_lock:
            if self._phase == 'idle':
                self._phase = 'ready'

    def runtime_status(self) -> dict[str, object]:
        with self._state_lock:
            phase = self._phase
            if phase == 'idle' and self._backend is not None:
                phase = 'warming'
            status = {'phase': phase, 'ready': phase == 'ready', 'load_ms': self._load_ms, 'warmup_ms': self._warmup_ms, 'last_error_type': self._last_error}
        if self._backend is None:
            return {"backend": "qwen_pointer", "loaded": False, "lazy_load": True, **status}
        return {**self._backend.runtime_status(), "lazy_load": True, **status}
