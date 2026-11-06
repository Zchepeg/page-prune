import json
import pytest

from collector.schemas import Candidate, PageContext
from collector.state_builder import STATE_SCHEMA_VERSION, build_state, build_state_payload
from modeling.encoding import DecisionExample
from training.dataset import load_jsonl, normalize_record
from collector.state_builder import OWNERSHIP_SCHEMA_VERSION
from runtime.state_serialization import validate_state_schema


def test_checkpoint_schema_guard_rejects_mixed_input_and_keeps_legacy_optional(tmp_path):
    path = tmp_path/'data.jsonl'
    path.write_text(json.dumps(record()), encoding='utf-8')
    assert len(load_jsonl(path, 'binary_selective', state_schema=STATE_SCHEMA_VERSION)) == 1
    with pytest.raises(ValueError, match='schema does not match'):
        load_jsonl(path, 'binary_selective', state_schema=OWNERSHIP_SCHEMA_VERSION)
    validate_state_schema('{}', None)
    with pytest.raises(ValueError, match='schema does not match'):
        validate_state_schema('{}', STATE_SCHEMA_VERSION)


def record(label=0):
    return {
        "id": "sample",
        "source": "example.test",
        "state": {
            "page_domain": "example.test",
            "tag": "DIV",
            "text": "Sponsored",
            "element_id": "ad-slot",
            "classes": ["advertisement"],
            "attributes": {},
            "resource_urls": [],
            "width": 300,
            "height": 250,
            "fixed": False,
        },
        "question": {
            "instruction": "Classify this candidate web element.",
            "options": ["ad", "non_ad", "insufficient_evidence"],
            "label": label,
        },
    }


def test_checkpoint_instruction_overrides_source_without_changing_label(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(record(1)), encoding="utf-8")
    original = load_jsonl(path, "binary_selective")[0]
    bound = load_jsonl(path, "binary_selective", "Bound task instruction")[0]
    assert original.instruction == "Classify this candidate web element."
    assert bound.instruction == "Bound task instruction"
    assert bound.label == original.label == 1
    assert bound.state == original.state
    assert bound.options == original.options


def test_v3_preserves_known_empty_direct_resources_on_roundtrip():
    candidate = Candidate('container', PageContext('', '', 'example.test'), resource_urls=['https://ads.doubleclick.net/banner'], resource_ownership_known=True)
    state = build_state_payload(candidate, [], OWNERSHIP_SCHEMA_VERSION)
    value = record()
    value['state'] = state
    normalized = normalize_record(value, 'binary_selective')['state']
    assert normalized['schema_version'] == OWNERSHIP_SCHEMA_VERSION
    assert normalized['resource_ownership_known'] is True
    assert normalized['direct_resource_urls'] == []
    assert 'descendant_known_ad_host' in normalized['detector_evidence']
    assert 'known_ad_host' not in normalized['detector_evidence']
    legacy = build_state_payload(candidate, [])
    assert legacy['schema_version'] == STATE_SCHEMA_VERSION
    assert 'direct_resource_urls' not in legacy


def test_v3_direct_resource_is_not_reassigned_to_descendants():
    url = 'https://ads.doubleclick.net/banner'
    candidate = Candidate('ad', PageContext('', '', 'example.test'), resource_urls=[url], direct_resource_urls=[url], resource_ownership_known=True)
    value = record()
    value['state'] = build_state_payload(candidate, [], OWNERSHIP_SCHEMA_VERSION)
    normalized = normalize_record(value, 'binary_selective')['state']
    assert normalized['direct_resource_urls'] == [url]
    assert 'known_ad_host' in normalized['detector_evidence']


def test_legacy_training_state_is_normalized_to_runtime_contract():
    normalized = normalize_record(record())
    candidate = Candidate(
        "",
        PageContext("", "", "example.test"),
        tag="DIV",
        text="Sponsored",
        element_id="ad-slot",
        classes=["advertisement"],
        width=300,
        height=250,
    )
    expected = build_state_payload(
        candidate,
        [
            "ad_semantic_marker",
            "common_ad_dimensions",
            "known_ad_slot_marker",
            "multiple_ad_identity_markers",
        ],
    )
    assert normalized["state"] == expected
    assert normalized["state"]["schema_version"] == STATE_SCHEMA_VERSION
    assert DecisionExample.from_dict(normalized).state == build_state(
        candidate,
        [
            "ad_semantic_marker",
            "common_ad_dimensions",
            "known_ad_slot_marker",
            "multiple_ad_identity_markers",
        ],
    )


def test_binary_loader_excludes_abstention_examples(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps(record(label)) for label in range(3)), encoding="utf-8")
    examples = load_jsonl(path, "binary_selective")
    assert [example.label for example in examples] == [0, 1]
    assert all(example.options == ("ad", "non_ad") for example in examples)
