from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM

from .encoding import option_isolation_mask
from .pointer_head import PointerHead

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_decision_runtime(
    checkpoint: str | Path,
    device: str = "cuda",
    quantization: str = "",
    gpu_memory_budget_mb: int = 0,
):
    from transformers import AutoTokenizer, BitsAndBytesConfig

    checkpoint = Path(checkpoint)
    quantization = quantization.lower().strip()
    if quantization not in {"", "none", "nf4"}:
        raise ValueError(f"unsupported inference quantization: {quantization}")
    if device.startswith("cuda") and gpu_memory_budget_mb:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the configured inference device")
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device_index).total_memory
        fraction = min(1.0, int(gpu_memory_budget_mb) * 1024**2 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    saved = torch.load(checkpoint / "pointer_head.pt", map_location="cpu", weights_only=True)
    metadata = dict(saved["metadata"])
    base_model = str(metadata.get("requested_model") or metadata["base_model"])
    quantization_config = None
    device_map = None
    if quantization == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = {"": device}
    model, metadata = QwenDecisionModel.load_checkpoint(
        base_model,
        checkpoint,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    if quantization == "nf4":
        model.head.to(device)
    else:
        model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    return model, tokenizer, metadata


class QwenDecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, pointer_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.head = PointerHead(hidden_size, pointer_dim)

    @classmethod
    def create_for_training(
        cls,
        model_name: str,
        revision: str | None = None,
        lora_rank: int = 16,
        pointer_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "QwenDecisionModel":
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        ).model
        base.gradient_checkpointing_enable()
        config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.05,
            target_modules=list(LORA_TARGETS),
        )
        return cls(get_peft_model(base, config), pointer_dim=pointer_dim)

    def forward_encoded(self, encoded: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        positions = encoded["position_ids"].to(device)
        owners = encoded["option_owner"].to(device)
        model_dtype = next(self.backbone.parameters()).dtype
        mask = option_isolation_mask(owners, model_dtype)
        hidden = self.backbone(
            input_ids=input_ids[None],
            position_ids=positions[None],
            attention_mask=mask,
            use_cache=False,
        ).last_hidden_state[0].float()
        return self.head(hidden[encoded["decide_index"]], hidden[encoded["option_indices"].to(device)])

    def forward_encoded_batch(self, encoded_batch: list[dict]) -> list[torch.Tensor]:
        if not encoded_batch:
            return []
        device = next(self.parameters()).device
        model_dtype = next(self.backbone.parameters()).dtype
        batch_size = len(encoded_batch)
        max_length = max(int(encoded["input_ids"].shape[0]) for encoded in encoded_batch)
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        position_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        attention_mask = torch.full(
            (batch_size, 1, max_length, max_length),
            torch.finfo(model_dtype).min,
            dtype=model_dtype,
            device=device,
        )
        for index, encoded in enumerate(encoded_batch):
            length = int(encoded["input_ids"].shape[0])
            input_ids[index, :length] = encoded["input_ids"].to(device)
            position_ids[index, :length] = encoded["position_ids"].to(device)
            owners = encoded["option_owner"].to(device)
            attention_mask[index, :, :length, :length] = option_isolation_mask(owners, model_dtype)[0]
        hidden = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state.float()
        return [
            self.head(
                hidden[index, encoded["decide_index"]],
                hidden[index, encoded["option_indices"].to(device)],
            )
            for index, encoded in enumerate(encoded_batch)
        ]

    def save_checkpoint(self, directory: str | Path, metadata: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(directory / "adapter")
        torch.save({"state_dict": self.head.state_dict(), "metadata": metadata}, directory / "pointer_head.pt")

    @classmethod
    def load_checkpoint(
        cls,
        base_model: str,
        checkpoint: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
        quantization_config: Any | None = None,
        device_map: Any | None = None,
    ) -> tuple["QwenDecisionModel", dict]:
        checkpoint = Path(checkpoint)
        options = {
            "dtype": dtype,
            "attn_implementation": "sdpa",
            "local_files_only": local_files_only,
        }
        if quantization_config is not None:
            options["quantization_config"] = quantization_config
        if device_map is not None:
            options["device_map"] = device_map
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            **options,
        ).model
        if quantization_config is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        backbone = PeftModel.from_pretrained(
            base,
            checkpoint / "adapter",
            low_cpu_mem_usage=quantization_config is not None,
            autocast_adapter_dtype=quantization_config is None,
        )
        saved = torch.load(checkpoint / "pointer_head.pt", map_location="cpu", weights_only=True)
        metadata = dict(saved["metadata"])
        model = cls(backbone, pointer_dim=int(metadata["pointer_dim"]))
        model.head.load_state_dict(saved["state_dict"])
        return model, metadata
