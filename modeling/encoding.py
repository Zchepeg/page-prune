from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from runtime.state_serialization import serialize_state

# Existing rare Qwen tokens: no embedding resize or giant trainable embedding table.
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
OPT_NONE, OPT_DECIDE = -1, -2
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass(frozen=True, slots=True)
class DecisionExample:
    state: str
    instruction: str
    options: tuple[str, ...]
    label: int | None = None
    example_id: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionExample":
        state = serialize_state(value.get("state", ""))
        question = value.get("question", {})
        options = tuple(str(item) for item in question.get("options", ()))
        label = question.get("label")
        if len(options) < 2:
            raise ValueError("a decision example needs at least two options")
        if label is not None and not 0 <= int(label) < len(options):
            raise ValueError("label is outside the option range")
        return cls(
            state=state,
            instruction=str(question.get("instruction", "")),
            options=options,
            label=None if label is None else int(label),
            example_id=str(value.get("id", "")),
            source=str(value.get("source", "")),
        )


def _user_tokens(tokenizer, text: str) -> list[int]:
    escaped = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return tokenizer(escaped, add_special_tokens=False).input_ids


def encode_example(tokenizer, example: DecisionExample, max_length: int = 512) -> dict[str, Any]:
    state_id, question_id, option_id, option_end_id, decide_id = (
        tokenizer.convert_tokens_to_ids(token) for token in SPECIAL
    )
    if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in (state_id, question_id, option_id, option_end_id, decide_id)):
        raise ValueError("the tokenizer does not contain the required Qwen delimiter tokens")

    state_tokens = _user_tokens(tokenizer, example.state)
    state = [state_id] + state_tokens
    instruction = [question_id] + _user_tokens(tokenizer, example.instruction)
    spans = [[option_id] + _user_tokens(tokenizer, option) + [option_end_id] for option in example.options]
    fixed = len(instruction) + sum(map(len, spans)) + 1
    keep_state = max_length - fixed
    if keep_state < 1:
        raise ValueError("question and options exceed max_length")
    truncated = len(state) > keep_state
    state = state[:keep_state]

    ids = state + instruction
    option_owner = [OPT_NONE] * len(ids)
    option_indices: list[int] = []
    for index, span in enumerate(spans):
        ids.extend(span)
        option_owner.extend([index] * len(span))
        option_indices.append(len(ids) - 1)
    ids.append(decide_id)
    option_owner.append(OPT_DECIDE)
    decide_index = len(ids) - 1

    position_ids = list(range(len(ids)))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "position_ids": torch.tensor(position_ids, dtype=torch.long),
        "option_owner": torch.tensor(option_owner, dtype=torch.long),
        "option_indices": torch.tensor(option_indices, dtype=torch.long),
        "decide_index": decide_index,
        "label": example.label,
        "truncated": truncated,
    }


def option_isolation_mask(option_owner: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Causal mask where an option sees state/instruction and itself; decide sees all."""
    length = option_owner.shape[0]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=option_owner.device))
    key_is_option = option_owner[None, :] >= 0
    query_is_decide = option_owner[:, None] == OPT_DECIDE
    same_option = option_owner[:, None] == option_owner[None, :]
    allow = causal & (~key_is_option | query_is_decide | same_option)
    mask = torch.zeros(length, length, dtype=dtype, device=option_owner.device)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)[None, None]
