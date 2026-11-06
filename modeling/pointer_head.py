from __future__ import annotations

import math

import torch
from torch import nn


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, pointer_dim: int = 256):
        super().__init__()
        self.query = nn.Linear(hidden_size, pointer_dim)
        self.key = nn.Linear(hidden_size, pointer_dim)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def raw_logits(self, decide_state: torch.Tensor, option_states: torch.Tensor) -> torch.Tensor:
        return (self.key(option_states) @ self.query(decide_state)) * self.scale

    def forward(self, decide_state: torch.Tensor, option_states: torch.Tensor) -> torch.Tensor:
        logits = self.raw_logits(decide_state, option_states)
        return logits if self.training else logits / self.temperature.clamp_min(1e-4)
