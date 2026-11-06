from __future__ import annotations

import json
import random

from modeling.encoding import DecisionExample
from runtime.state_serialization import serialize_state


def permute_options(example: DecisionExample, rng: random.Random) -> DecisionExample:
    order = list(range(len(example.options)))
    rng.shuffle(order)
    label = None if example.label is None else order.index(example.label)
    return DecisionExample(
        state=example.state,
        instruction=example.instruction,
        options=tuple(example.options[index] for index in order),
        label=label,
        example_id=example.example_id,
        source=example.source,
    )


def dropout_state_features(
    example: DecisionExample,
    rng: random.Random,
    evidence_dropout: float = 0.0,
    feature_dropout: float = 0.0,
) -> DecisionExample:
    payload = json.loads(example.state)
    if rng.random() < evidence_dropout:
        payload["detector_evidence"] = []
    defaults = {
        "text": "",
        "element_id": "",
        "classes": [],
        "attributes": {},
        "resource_urls": [],
    }
    for name, default in defaults.items():
        if rng.random() < feature_dropout:
            payload[name] = default
    return DecisionExample(
        state=serialize_state(payload),
        instruction=example.instruction,
        options=example.options,
        label=example.label,
        example_id=example.example_id,
        source=example.source,
    )


def augment_example(
    example: DecisionExample,
    rng: random.Random,
    evidence_dropout: float = 0.0,
    feature_dropout: float = 0.0,
) -> DecisionExample:
    return permute_options(
        dropout_state_features(example, rng, evidence_dropout, feature_dropout),
        rng,
    )
