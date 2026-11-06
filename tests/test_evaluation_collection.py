from scripts.build_evaluation_pool import load_catalog, select_rows


def test_load_catalog_selects_valid_domains_after_offset(tmp_path):
    source = tmp_path / "sites.csv"
    source.write_text(
        "GlobalRank,Domain\n1,first.example\n2,invalid domain\n3,second.example\n4,third.example\n",
        encoding="utf-8",
    )

    assert load_catalog(source, offset=1, limit=2) == ["https://second.example/", "https://third.example/"]


def test_select_rows_can_restrict_collection_to_ad_enriched_kinds():
    rows = [
        {"kind": "control", "index": 1},
        {"kind": "suspicious", "index": 2},
        {"kind": "ambiguous", "index": 3},
        {"kind": "neutral", "index": 4},
    ]

    selected = select_rows(rows, per_domain=10, seed=42, candidate_kinds=("suspicious", "ambiguous"))

    assert {item["kind"] for item in selected} == {"suspicious", "ambiguous"}
