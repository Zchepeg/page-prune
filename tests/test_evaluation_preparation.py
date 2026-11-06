from pathlib import Path

from evaluation.preparation import candidate_fingerprint, prepare_records


def item(identifier: str, domain: str = "one.test") -> dict:
    return {
        "id": identifier,
        "domain": domain,
        "candidate": {
            "candidate_id": identifier,
            "page": {"url": f"https://{domain}/", "top_level_domain": domain},
            "tag": "DIV",
            "text": "Sponsored",
            "element_id": "ad-slot",
            "classes": ["advertisement"],
            "resource_urls": ["https://ads.doubleclick.net/banner?id=1"],
            "width": 300,
            "height": 250,
            "visible": True,
        },
        "review_context": {"candidate_kind": "suspicious", "screenshot": ""},
    }


def test_fingerprint_ignores_candidate_identifier_and_url_query() -> None:
    first = item("first")
    second = item("second")
    second["candidate"]["resource_urls"] = ["https://ads.doubleclick.net/banner?id=2"]
    assert candidate_fingerprint(first) == candidate_fingerprint(second)


def test_preparation_deduplicates_and_stratifies(tmp_path: Path) -> None:
    pool = [item("first"), item("duplicate"), item("other", "two.test")]
    curated, prelabels, report = prepare_records(pool, tmp_path)
    assert len(curated) == 2
    assert {value["domain"] for value in curated} == {"one.test", "two.test"}
    assert report["rejected"] == {"duplicate_candidate": 1}
    assert report["screenshots"] == {"missing": 2}
    assert all(value["suggested_label"] == "ad" for value in prelabels)


def test_review_ui_distinguishes_blind_and_assisted_annotation() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "review_evaluation_pool.py").read_text(encoding="utf-8")
    assert "机器建议" in script
    assert 'if self.server.settings["blind"]' in script
    assert 'parser.add_argument("--blind"' in script
    assert "human_confirmed_machine_suggestion" in script
    assert "human_independent" in script
    assert 'assisted=false' in script
