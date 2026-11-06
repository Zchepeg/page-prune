from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auto_dataset import dataset_readiness, read_records, write_snapshot


class AutoTrainer:
    """Gate and supervise background training; never promotes a checkpoint automatically."""

    def __init__(
        self,
        project_root: str | Path,
        config: dict[str, Any],
        inference_config: dict[str, Any] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.inference_config = inference_config or {}
        self.enabled = bool(config.get("enabled", False))
        self.state_path = self.project_root / config.get("state_path", "data/training/auto_train_state.json")
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"status": "idle", "last_trained_examples": 0}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "idle", "last_trained_examples": 0}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _records(self) -> tuple[list[dict], list[dict]]:
        sources = [self.project_root / path for path in self.config.get("training_sources", [])]
        calibration = self.project_root / self.config.get("calibration_path", "data/training/calibration.jsonl")
        return read_records(sources), read_records([calibration])

    def _readiness(self):
        training, calibration = self._records()
        label_mode = str(self.config.get("label_mode", "binary_selective"))
        if label_mode == "binary_selective":
            training = [record for record in training if int(record["question"]["label"]) in {0, 1}]
            calibration = [record for record in calibration if int(record["question"]["label"]) in {0, 1}]
        readiness = dataset_readiness(
            training,
            calibration,
            min_examples=int(self.config.get("min_examples", 300)),
            min_per_label=int(self.config.get("min_per_label", 60)),
            min_domains=int(self.config.get("min_domains", 20)),
            min_calibration_per_label=int(self.config.get("min_calibration_per_label", 15)),
            required_labels=(0, 1) if label_mode == "binary_selective" else (0, 1, 2),
            min_human_examples=int(self.config.get("min_human_examples", 0)),
            min_human_domains=int(self.config.get("min_human_domains", 0)),
            min_human_per_label=int(self.config.get("min_human_per_label", 0)),
            min_human_calibration_per_label=int(self.config.get("min_human_calibration_per_label", 0)),
        )
        return training, calibration, readiness

    def maybe_start(self) -> bool:
        with self._lock:
            if not self.enabled or (self._thread and self._thread.is_alive()):
                return False
            training, calibration, readiness = self._readiness()
            self._state["readiness"] = readiness.to_dict()
            if not readiness.ready:
                self._state["status"] = "waiting_for_data"
                self._save_state()
                return False
            last_count = int(self._state.get("last_trained_examples", 0))
            min_new = int(self.config.get("retrain_min_new", 100))
            if last_count and readiness.examples - last_count < min_new:
                self._state["status"] = "waiting_for_new_data"
                self._save_state()
                return False
            run_id = datetime.now(timezone.utc).strftime("auto-%Y%m%dT%H%M%SZ")
            self._state.update({"status": "starting", "run_id": run_id, "error": ""})
            self._save_state()
            self._thread = threading.Thread(
                target=self._run,
                args=(run_id, training, calibration),
                name="qwen-auto-trainer",
                daemon=True,
            )
            self._thread.start()
            return True

    def _run(self, run_id: str, training: list[dict], calibration: list[dict]) -> None:
        snapshot_root = self.project_root / "data" / "training" / "snapshots" / run_id
        output = self.project_root / "runs" / run_id
        log_path = self.project_root / "data" / "training" / "logs" / f"{run_id}.log"
        try:
            label_mode = str(self.config.get("label_mode", "binary_selective"))
            required_labels = (0, 1) if label_mode == "binary_selective" else (0, 1, 2)
            train_path, validation_path, calibration_path = write_snapshot(
                snapshot_root,
                training,
                calibration,
                float(self.config.get("validation_fraction", 0.15)),
                int(self.config.get("seed", 42)),
                required_labels,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            train_command = [
                sys.executable,
                "-m",
                "training.train",
                "--data",
                str(train_path),
                "--validation",
                str(validation_path),
                "--model",
                str(self.config.get("model", "Qwen/Qwen3-4B-Base")),
                "--epochs",
                str(self.config.get("epochs", 2)),
                "--accum",
                str(self.config.get("accum", 8)),
                "--max-length",
                str(self.config.get("max_length", 512)),
                "--label-mode",
                label_mode,
                "--patience",
                str(self.config.get("early_stopping_patience", 2)),
                "--evidence-dropout",
                str(self.config.get("evidence_dropout", 0.5)),
                "--feature-dropout",
                str(self.config.get("feature_dropout", 0.1)),
                "--domain-balance-power",
                str(self.config.get("domain_balance_power", 0.5)),
                "--label-smoothing",
                str(self.config.get("label_smoothing", 0.05)),
                "--local-files-only",
                "--out",
                str(output),
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with log_path.open("a", encoding="utf-8") as log:
                self._set_status("training", checkpoint=str(output), log=str(log_path))
                subprocess.run(
                    train_command,
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    creationflags=creationflags,
                )
                self._set_status("calibrating")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "training.calibrate",
                        "--checkpoint",
                        str(output),
                        "--data",
                        str(calibration_path),
                        "--device",
                        str(self.inference_config.get("device", "cuda")),
                        "--quantization",
                        str(self.inference_config.get("quantization", "none") or "none"),
                        "--gpu-memory-budget-mb",
                        str(self.inference_config.get("gpu_memory_budget_mb", 0)),
                        "--max-length",
                        str(self.inference_config.get("inference_max_length", self.config.get("max_length", 384))),
                    ],
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    creationflags=creationflags,
                )
            with self._lock:
                self._state.update(
                    {
                        "status": "candidate_ready",
                        "checkpoint": str(output),
                        "last_trained_examples": len(training),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self._save_state()
        except Exception as exc:
            self._set_status("failed", error=f"{type(exc).__name__}: {exc}")

    def _set_status(self, status: str, **updates: Any) -> None:
        with self._lock:
            self._state.update({"status": status, **updates})
            self._save_state()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if not self.enabled:
                return {"enabled": False, "status": "disabled", "running": False}
            result = {"enabled": self.enabled, **self._state}
            result["running"] = bool(self._thread and self._thread.is_alive())
            return result
