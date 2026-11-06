from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from collector.schemas import Candidate, PageContext
from collector.state_builder import STATE_SCHEMA_VERSION, OWNERSHIP_SCHEMA_VERSION, build_state_payload
from detector.candidate_ranker import CandidateRanker
from modeling.encoding import DecisionExample
from runtime.state_serialization import validate_state_schema


def _candidate_from_state(state: dict, source: str) -> Candidate:
    if state.get("schema_version") in {STATE_SCHEMA_VERSION, OWNERSHIP_SCHEMA_VERSION}:
        width, height = (list(state.get("geometry", [])) + [0, 0])[:2]
        return Candidate(
            candidate_id="",
            page=PageContext("", "", str(state.get("page_domain", source))),
            tag=str(state.get("tag", "DIV")).upper(),
            text=str(state.get("text", "")),
            element_id=str(state.get("element_id", "")),
            classes=[str(item) for item in state.get("classes", [])],
            attributes={str(key): str(value) for key, value in state.get("attributes", {}).items()},
            resource_urls=[str(item) for item in state.get("resource_urls", [])],
            direct_resource_urls=[str(item) for item in state.get("direct_resource_urls", [])],
            resource_ownership_known=bool(state.get("resource_ownership_known", "direct_resource_urls" in state)),
            width=float(width or 0),
            height=float(height or 0),
            fixed=bool(state.get("fixed", False)),
            visible=bool(state.get("visible", True)),
        )
    nested = state.get("candidate") if isinstance(state.get("candidate"), dict) else state
    page = state.get("page") if isinstance(state.get("page"), dict) else {}
    geometry = list(nested.get("geometry", []))
    width = nested.get("width", geometry[0] if geometry else 0)
    height = nested.get("height", geometry[1] if len(geometry) > 1 else 0)
    return Candidate(
        candidate_id="",
        page=PageContext(
            str(page.get("url", "")),
            str(page.get("title", "")),
            str(page.get("top_level_domain", state.get("page_domain", source))),
        ),
        tag=str(nested.get("tag", "DIV")).upper(),
        text=str(nested.get("text", "")),
        element_id=str(nested.get("element_id", nested.get("id", ""))),
        classes=[str(item) for item in nested.get("classes", [])],
        attributes={str(key): str(value) for key, value in nested.get("attributes", {}).items()},
        resource_urls=[str(item) for item in nested.get("resource_urls", [])],
        direct_resource_urls=[str(item) for item in nested.get("direct_resource_urls", [])],
        resource_ownership_known=bool(nested.get("resource_ownership_known", "direct_resource_urls" in nested)),
        width=float(width or 0),
        height=float(height or 0),
        fixed=bool(nested.get("fixed", False)),
        visible=bool(nested.get("visible", True)),
    )


def normalize_record(value: dict, label_mode: str = "three_class") -> dict | None:
    record = deepcopy(value)
    question = record.get("question", {})
    label = question.get("label")
    if label_mode == "binary_selective":
        if label is not None and int(label) == 2:
            return None
        question["options"] = ["ad", "non_ad"]
    elif label_mode != "three_class":
        raise ValueError(f"unknown label mode: {label_mode}")
    state = record.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    candidate = _candidate_from_state(state, str(record.get("source", "")))
    existing_evidence = state.get("detector_evidence", [])
    evidence = [str(item) for item in existing_evidence] if existing_evidence else CandidateRanker().assess(candidate).evidence
    version = OWNERSHIP_SCHEMA_VERSION if state.get("schema_version") == OWNERSHIP_SCHEMA_VERSION else STATE_SCHEMA_VERSION
    record["state"] = build_state_payload(candidate, evidence, version)
    record["question"] = question
    return record


def load_jsonl(path: str | Path, label_mode: str = "three_class", instruction: str | None = None, state_schema: str | None = None) -> list[DecisionExample]:
    examples: list[DecisionExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = normalize_record(json.loads(line), label_mode)
                if record is not None:
                    if instruction is not None:
                        record["question"]["instruction"] = instruction
                    example = DecisionExample.from_dict(record)
                    validate_state_schema(example.state, state_schema)
                    examples.append(example)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid example at {path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples
