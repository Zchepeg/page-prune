import json
import random

import torch
from torch import nn
from types import SimpleNamespace

from modeling.encoding import OPT_DECIDE, OPT_NONE, DecisionExample, encode_example, option_isolation_mask
from modeling.decision_model import QwenDecisionModel
from training.augment import dropout_state_features, permute_options


class TokenizerStub:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {
            "<|fim_prefix|>": 1,
            "<|fim_middle|>": 2,
            "<|box_start|>": 3,
            "<|box_end|>": 4,
            "<|fim_suffix|>": 5,
        }[token]

    def __call__(self, text, add_special_tokens=False):
        return type("Tokens", (), {"input_ids": list(range(10, 10 + len(text)))})()


def test_option_isolation_mask_hides_other_option_spans():
    owners = torch.tensor([OPT_NONE, OPT_NONE, 0, 0, 1, 1, OPT_DECIDE])
    mask = option_isolation_mask(owners, torch.float32)[0, 0]
    assert mask[3, 2] == 0
    assert mask[3, 4] < -1e20
    assert mask[5, 2] < -1e20
    assert mask[6, 2] == 0 and mask[6, 4] == 0


def test_option_permutation_tracks_label():
    example = DecisionExample("state", "question", ("ad", "non_ad", "unknown"), label=0)

    class Reverse:
        @staticmethod
        def shuffle(values):
            values.reverse()

    changed = permute_options(example, Reverse())
    assert changed.options == ("unknown", "non_ad", "ad")
    assert changed.label == 2


def test_training_feature_dropout_removes_shortcuts_without_changing_label():
    state = '{"attributes":{"role":"banner"},"classes":["ad"],"detector_evidence":["ad_semantic_marker"],"element_id":"ad-slot","resource_urls":["https://ads.test/x"],"text":"Sponsored"}'
    example = DecisionExample(state, "question", ("ad", "non_ad"), label=0, source="example.test")

    changed = dropout_state_features(example, random.Random(1), evidence_dropout=1.0, feature_dropout=1.0)
    payload = json.loads(changed.state)

    assert payload["detector_evidence"] == []
    assert payload["classes"] == []
    assert payload["resource_urls"] == []
    assert changed.label == 0
    assert changed.source == "example.test"


def test_encoding_reports_actual_truncation():
    tokenizer = TokenizerStub()
    short = encode_example(tokenizer, DecisionExample("x", "q", ("ad", "non_ad")), max_length=32)
    long = encode_example(tokenizer, DecisionExample("x" * 40, "q", ("ad", "non_ad")), max_length=32)
    assert short["truncated"] is False
    assert long["truncated"] is True


class BackboneStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(256, 8)

    def forward(self, input_ids, position_ids, attention_mask, use_cache):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_batched_forward_matches_single_forward_for_variable_lengths():
    tokenizer = TokenizerStub()
    encoded = [
        encode_example(tokenizer, DecisionExample("x", "q", ("ad", "non_ad")), max_length=64),
        encode_example(tokenizer, DecisionExample("long-state", "q", ("ad", "non_ad")), max_length=64),
    ]
    model = QwenDecisionModel(BackboneStub(), pointer_dim=4).eval()
    singles = [model.forward_encoded(value) for value in encoded]
    batched = model.forward_encoded_batch(encoded)
    assert len(batched) == len(singles)
    assert all(torch.allclose(single, batch, atol=1e-6) for single, batch in zip(singles, batched))
