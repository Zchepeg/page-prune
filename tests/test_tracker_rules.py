import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_tracker_dataset", ROOT / "scripts" / "build_tracker_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_runtime_tracker_hosts_are_compact_and_normalized() -> None:
    payload = json.loads(
        (ROOT / "data/runtime/tracker_hosts.json").read_text(encoding="utf-8")
    )
    hosts = payload["hosts"]

    assert payload["version"] == 1
    assert hosts == sorted(set(hosts))
    assert len(hosts) == 190
    assert "pagead2.googlesyndication.com" in hosts
    assert "events.reddit.com" in hosts
    assert "browser.sentry-cdn.com" in hosts


def test_generated_rules_cover_hosts_and_safe_first_party_paths(tmp_path: Path) -> None:
    output = tmp_path / "rules.json"
    count = MODULE.write_dnr_rules(
        output,
        [{"host": "ads.example.com", "category": "ads", "vendor": "example"}],
    )
    rules = json.loads(output.read_text(encoding="utf-8"))
    filters = {rule["condition"]["urlFilter"] for rule in rules}

    assert count == len(rules)
    assert "||ads.example.com^" in filters
    assert "/ads.js" in filters
    assert "/ads/banner/" in filters
    assert "/ads/adframe." in filters
    assert "/banners/pr_advertising_ads_banner." in filters
    assert "/ad" not in filters
    assert all(rule["condition"].get("excludedResourceTypes") == ["main_frame"] for rule in rules)


def test_checked_in_rules_retain_benchmark_coverage() -> None:
    rules = json.loads(
        (ROOT / "browser/extension/rules/tracker_rules.json").read_text(encoding="utf-8")
    )
    filters = {rule["condition"]["urlFilter"] for rule in rules}

    assert len(rules) == 1085
    assert {
        "||pagead2.googlesyndication.com^",
        "||events.reddit.com^",
        "||samsungads.com^",
        "||browser.sentry-cdn.com^",
        "||ib.adnxs.com^",
        "||metrics5.data.hicloud.com^",
    } <= filters


def test_anti_ad_rules_are_complete_chunked_and_subresource_only(tmp_path: Path) -> None:
    checked_in = json.loads(
        (ROOT / "browser/extension/rules/anti_ad_rules.json").read_text(
            encoding="utf-8"
        )
    )
    hosts = [
        host for rule in checked_in for host in rule["condition"]["requestDomains"]
    ]
    output = tmp_path / "anti_ad_rules.json"
    count = MODULE.write_grouped_domain_rules(output, hosts)
    rules = json.loads(output.read_text(encoding="utf-8"))
    encoded_hosts = {
        host for rule in rules for host in rule["condition"]["requestDomains"]
    }

    assert len(hosts) == 108_911
    assert count == len(rules)
    assert encoded_hosts == set(hosts)
    assert all(len(rule["condition"]["requestDomains"]) <= 1000 for rule in rules)
    assert all(rule["condition"]["excludedResourceTypes"] == ["main_frame"] for rule in rules)
    assert all(rule["condition"]["domainType"] == "thirdParty" for rule in rules)


def test_checked_in_anti_ad_rules_have_unique_hosts() -> None:
    rules = json.loads(
        (ROOT / "browser/extension/rules/anti_ad_rules.json").read_text(
            encoding="utf-8"
        )
    )
    hosts = [
        host for rule in rules for host in rule["condition"]["requestDomains"]
    ]

    assert len(hosts) == len(set(hosts)) == 108_911


def test_manifest_enables_static_rulesets() -> None:
    manifest = json.loads(
        (ROOT / "browser/extension/manifest.json").read_text(encoding="utf-8")
    )
    enabled = {
        item["id"]: item["path"]
        for item in manifest["declarative_net_request"]["rule_resources"]
        if item["enabled"]
    }

    assert enabled == {
        "tracker_hosts": "rules/tracker_rules.json",
        "anti_ad_hosts": "rules/anti_ad_rules.json",
        "privacy_transport": "rules/privacy_rules.json",
    }


def test_manifest_loads_cosmetic_filter_at_document_start_in_frames() -> None:
    manifest = json.loads(
        (ROOT / "browser/extension/manifest.json").read_text(encoding="utf-8")
    )
    cosmetic = next(item for item in manifest["content_scripts"] if "css" in item)

    assert cosmetic["css"] == ["cosmetic.css"]
    assert cosmetic["run_at"] == "document_start"
    assert cosmetic["all_frames"] is True


def test_rules_cover_local_probe_paths_and_extreme_ad_hosts() -> None:
    rules = json.loads(
        (ROOT / "browser/extension/rules/tracker_rules.json").read_text(encoding="utf-8")
    )
    filters = {rule["condition"]["urlFilter"] for rule in rules}
    probe_urls = [
        "https://ads-blocker.com/ads/banner/ad-banner.svg",
        "https://ads-blocker.com/ads/advertisement.js",
        "https://ads-blocker.com/ads/adframe.html",
        "https://ads-blocker.com/ads/banner/advertisement.json",
        "https://adblock-tester.com/banners/pr_advertising_ads_banner.png",
    ]
    control_urls = [
        "https://ads-blocker.com/checks/control/banner.svg",
        "https://ads-blocker.com/checks/control/script.js",
        "https://ads-blocker.com/checks/control/frame.html",
        "https://ads-blocker.com/checks/control/background.json",
    ]
    path_filters = {value for value in filters if not value.startswith("||")}

    assert all(any(value in url for value in path_filters) for url in probe_urls)
    assert all(not any(value in url for value in path_filters) for url in control_urls)
    assert {
        "||12ezo5v60.com^",
        "||fvcwqkkqmuv.com^",
        "||ybs2ffs7v.com^",
    } <= filters


def test_cosmetic_rules_cover_adblock_tester_and_extreme_widgets() -> None:
    css = (ROOT / "browser/extension/cosmetic.css").read_text(encoding="utf-8")

    assert ".adsbygoogle" in css
    assert ".banner_ads" in css
    assert ".textads" in css
    assert ".sponsored_ad" in css
    assert "#ad" in css
    assert '[data-bait=".adsbygoogle2"]' in css
    assert "#abt-baits > #ad" in css
    assert ".video-ad-container" in css
    assert ".sticky-ad-container" in css
    assert "[data-ads]" in css
    assert "pr_advertising_ads_banner" in css
    assert 'script[src*="ybs2ffs7v.com"]' in css


def test_popup_is_minimal() -> None:
    html = (ROOT / "browser/extension/popup.html").read_text(encoding="utf-8")

    assert "PagePrune" in html
    assert "点选并标记广告" in html
    assert "浏览器已发送" not in html
    assert "训练集样本" not in html


def test_privacy_guard_runs_in_main_world_at_document_start() -> None:
    manifest = json.loads(
        (ROOT / "browser/extension/manifest.json").read_text(encoding="utf-8")
    )
    guard = next(
        item for item in manifest["content_scripts"] if item.get("js") == ["privacy_guard.js"]
    )

    assert guard["run_at"] == "document_start"
    assert guard["all_frames"] is True
    assert guard["world"] == "MAIN"


def test_privacy_guard_covers_ad_apis_gpc_and_fingerprint_readback() -> None:
    script = (ROOT / "browser/extension/privacy_guard.js").read_text(encoding="utf-8")

    assert "browsingTopics" in script
    assert "joinAdInterestGroup" in script
    assert "runAdAuction" in script
    assert "attributionSrc" in script
    assert "hasPrivateToken" in script
    assert "globalPrivacyControl" in script
    assert "CanvasRenderingContext2D" in script
    assert "AudioBuffer" in script


def test_privacy_transport_rules_cover_analytics_beacons() -> None:
    rules = json.loads(
        (ROOT / "browser/extension/rules/privacy_rules.json").read_text(encoding="utf-8")
    )

    assert rules == [
        {
            "id": 1,
            "priority": 2,
            "action": {"type": "block"},
            "condition": {
                "urlFilter": "||google-analytics.com/g/collect",
                "resourceTypes": ["ping"],
            },
        }
    ]
