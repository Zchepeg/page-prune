from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


OPTIONS = ["ad", "non_ad", "insufficient_evidence"]
HOST_RE = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}", re.I)
QUOTED_VALUE_RE = re.compile(r"[\"']([^\"']+)[\"']")
HTTP_URL_RE = re.compile(r"https?://[a-z0-9][a-z0-9._:-]*(?:/[^\"'`<> ]*)?", re.I)
ANTI_AD_CHUNK_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced advertising and tracker decision data")
    parser.add_argument("--host-list", required=True)
    parser.add_argument("--vendors", default="data/runtime/tracker_vendors.json")
    parser.add_argument("--base-train", default="data/training/bootstrap_train.jsonl")
    parser.add_argument("--base-calibration", default="data/training/calibration.jsonl")
    parser.add_argument("--train-out", default="data/training/tracker_train.jsonl")
    parser.add_argument("--calibration-out", default="data/training/calibration_v2.jsonl")
    parser.add_argument("--dnr-out", default="browser/extension/rules/tracker_rules.json")
    parser.add_argument("--anti-ad-domains", required=True)
    parser.add_argument("--anti-ad-dnr-out", default="browser/extension/rules/anti_ad_rules.json")
    parser.add_argument(
        "--benchmark-source",
        action="append",
        default=None,
        help="Saved public ad-block benchmark page/script used as a reviewed host source",
    )
    parser.add_argument("--train-per-label", type=int, default=180)
    parser.add_argument("--calibration-per-label", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20261009)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_host_list(path: Path) -> list[dict[str, str]]:
    section = "uncategorized"
    vendor = "unknown"
    entries = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("!") and "===" in line:
            name = line.strip("! =").strip()
            if name:
                section = name.lower().replace(" ", "_")
        elif line.startswith("! ---"):
            vendor = line[5:].strip()
        elif line.startswith("||") and "^" in line:
            host = line[2:].split("^", 1)[0].lower().lstrip("*.")
            if HOST_RE.fullmatch(host):
                entries.append({"category": section, "vendor": vendor, "host": host, "source": path.name})
    return entries


def vendor_entries(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"category": category, "vendor": vendor, "host": host.lower(), "source": path.name}
        for category, vendors in payload.items()
        for vendor, hosts in vendors.items()
        for host in hosts
    ]


def benchmark_entries(paths: list[Path]) -> list[dict[str, str]]:
    """Extract public benchmark probe hosts without copying executable logic.

    The saved sources use two different JavaScript layouts.  Restrict extraction
    to their declared test tables so page assets, canonical URLs, and unrelated
    documentation links cannot accidentally become blocking rules.
    """
    entries: list[dict[str, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "tests: {" in text and "results: {}" in text:
            text = text.split("tests: {", 1)[1].split("results: {}", 1)[0]
        elif "const HOST_TESTS" in text and "const TEST_TIMEOUT_MS" in text:
            text = text.split("const HOST_TESTS", 1)[1].split("const TEST_TIMEOUT_MS", 1)[0]
        elif path.name == "iblockads-data.js":
            # iBlockAds publishes its probes as quoted domain values grouped by
            # category.  Exact-value matching avoids treating JavaScript names
            # or documentation URLs as hosts.
            values = [match.group(1) for match in QUOTED_VALUE_RE.finditer(text)]
            for value in values:
                host = value.lower().rstrip(".")
                if valid_host(host):
                    entries.append(
                        {
                            "category": "benchmark_privacy",
                            "vendor": path.stem,
                            "host": host,
                            "source": path.name,
                        }
                    )
            continue
        elif path.name == "mauriziofonte-adblock-test.js" and "var w=[" in text:
            # The probe table ends before function B.  Parse only that table so
            # DNS diagnostics and page-owned control requests remain reachable.
            text = text.split("var w=[", 1)[1].split(";function B", 1)[0]
            for match in QUOTED_VALUE_RE.finditer(text):
                value = match.group(1)
                host = (urlsplit(value).hostname or value).lower().rstrip(".")
                if valid_host(host):
                    entries.append(
                        {
                            "category": "benchmark_privacy",
                            "vendor": path.stem,
                            "host": host,
                            "source": path.name,
                        }
                    )
            continue
        elif path.name == "adblockbench-app.js":
            # The minified bundle contains framework/documentation URLs too.
            # Its probe catalogue is bounded by the first ad probe and the
            # footer's GitHub link, so only inspect that interval.
            start = text.find("https://pagead2.googlesyndication.com/")
            end = text.find("https://github.com/SiliconSquire", start)
            if start < 0 or end < 0:
                raise ValueError(f"unrecognized AdBlockBench bundle: {path}")
            for match in HTTP_URL_RE.finditer(text[start:end]):
                host = (urlsplit(match.group(0)).hostname or "").lower().rstrip(".")
                if valid_host(host):
                    entries.append(
                        {
                            "category": "benchmark_privacy",
                            "vendor": path.stem,
                            "host": host,
                            "source": path.name,
                        }
                    )
            continue
        elif path.name == "paileactivist-data.js":
            for match in HTTP_URL_RE.finditer(text):
                host = (urlsplit(match.group(0)).hostname or "").lower().rstrip(".")
                if valid_host(host) and host != "www.w3.org":
                    entries.append(
                        {
                            "category": "benchmark_privacy",
                            "vendor": path.stem,
                            "host": host,
                            "source": path.name,
                        }
                    )
            continue
        else:
            raise ValueError(f"unrecognized benchmark source layout: {path}")

        for match in HOST_RE.finditer(text):
            host = match.group(0).lower().rstrip(".")
            if ".." not in host:
                entries.append(
                    {
                        "category": "benchmark_privacy",
                        "vendor": path.stem,
                        "host": host,
                        "source": path.name,
                    }
                )
    return entries


def valid_host(host: str) -> bool:
    """Return whether a value is a conservative ASCII DNS hostname."""
    if len(host) > 253 or ".." in host or not HOST_RE.fullmatch(host):
        return False
    return all(
        0 < len(label) <= 63 and label[0].isalnum() and label[-1].isalnum()
        for label in host.split(".")
    )


def parse_bare_domain_list(path: Path) -> list[str]:
    hosts = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        host = raw.strip().lower().rstrip(".")
        if host and not host.startswith("#") and valid_host(host):
            hosts.add(host)
    return sorted(hosts)


def write_grouped_domain_rules(path: Path, hosts: list[str], chunk_size: int = ANTI_AD_CHUNK_SIZE) -> int:
    """Write many domains as a small number of DNR rules.

    Chrome counts static rules, not entries inside requestDomains.  Chunking the
    anti-AD DNS list therefore stays far below the static-rule limit while
    retaining exact host and subdomain matching.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    unique_hosts = sorted(set(hosts))
    rules = []
    for start in range(0, len(unique_hosts), chunk_size):
        rules.append(
            {
                "id": len(rules) + 1,
                "priority": 1,
                "action": {"type": "block"},
                "condition": {
                    "requestDomains": unique_hosts[start : start + chunk_size],
                    "domainType": "thirdParty",
                    "excludedResourceTypes": ["main_frame"],
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(rules)


def round_robin(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    by_vendor: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_vendor[f"{entry['category']}::{entry['vendor']}"].append(entry)
    ordered = []
    while by_vendor:
        for key in list(sorted(by_vendor)):
            ordered.append(by_vendor[key].pop(0))
            if not by_vendor[key]:
                del by_vendor[key]
    return ordered


def state_for_tracker(entry: dict[str, str], index: int, rng: random.Random) -> dict:
    category = entry["category"]
    paths = {
        "advertising_network": ["ads/creative.js", "pagead/render", "bid/banner"],
        "analytics": ["analytics.js", "collect?v=2", "events/track"],
        "social_pixel": ["pixel.js", "tr?id=123", "events/conversion"],
        "session_recording": ["recorder.js", "session/replay", "ingest/record"],
    }
    category_paths = paths.get(category, ["tracker.js", "events/collect", "pixel.gif"])
    resource_path = category_paths[index % len(category_paths)]
    tags = ["SCRIPT", "IMG", "IFRAME", "DIV"]
    return {
        "tag": tags[index % len(tags)],
        "text": "" if index % 3 else entry["vendor"],
        "element_id": f"tracking-slot-{index % 31}",
        "classes": [rng.choice(["integration", "pixel", "analytics", "ad-slot"]), category],
        "attributes": {"data_vendor": entry["vendor"], "data_category": category},
        "resource_urls": [f"https://{entry['host']}/{resource_path}"],
        "width": 1 if index % 4 == 1 else (300 if index % 4 == 2 else 0),
        "height": 1 if index % 4 == 1 else (250 if index % 4 == 2 else 0),
        "fixed": False,
    }


def safe_state(domain: str, index: int) -> dict:
    templates = [
        ("SCRIPT", "site-analytics-dashboard", "assets/analytics-dashboard.js"),
        ("DIV", "social-share-buttons", "assets/social-share.js"),
        ("VIDEO", "video-player", "media/player.js"),
        ("ARTICLE", "article-content", "assets/article.css"),
        ("FORM", "checkout-form", "assets/checkout.js"),
        ("DIV", "privacy-settings", "assets/privacy-controls.js"),
    ]
    tag, class_name, path = templates[index % len(templates)]
    return {
        "tag": tag,
        "text": class_name.replace("-", " "),
        "element_id": f"{class_name}-{index % 37}",
        "classes": [class_name, "first-party"],
        "attributes": {},
        "resource_urls": [f"https://static.{domain}/{path}"],
        "width": 720,
        "height": 120,
        "fixed": False,
    }


def uncertain_state(domain: str, index: int) -> dict:
    templates = [
        ("SCRIPT", "metrics-loader", "https://cdn.example.invalid/metrics.js", 0, 0),
        ("IMG", "conversion-pixel", f"https://{domain}/pixel.gif", 1, 1),
        ("DIV", "sponsored-widget", "", 300, 180),
        ("IFRAME", "integration-frame", "", 300, 250),
        ("SCRIPT", "tag-manager", f"https://static.{domain}/tag-manager.js", 0, 0),
        ("DIV", "session-player", "", 640, 360),
    ]
    tag, class_name, url, width, height = templates[index % len(templates)]
    return {
        "tag": tag,
        "text": "",
        "element_id": f"{class_name}-{index % 41}",
        "classes": [class_name, "unresolved"],
        "attributes": {},
        "resource_urls": [url] if url else [],
        "width": width,
        "height": height,
        "fixed": False,
    }


def make_record(split: str, label: int, index: int, domain: str, state: dict, provenance: dict) -> dict:
    identity = json.dumps([split, label, domain, state, provenance], ensure_ascii=False, sort_keys=True)
    return {
        "id": f"tracker-{split}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
        "source": domain,
        "state": {"page_domain": domain, **state},
        "question": {
            "instruction": "Classify this candidate web element.",
            "options": OPTIONS,
            "label": label,
        },
        "provenance": {
            "label_source": "curated_tracker_taxonomy",
            "split": split,
            "index": index,
            **provenance,
        },
    }


def build(
    split: str,
    per_label: int,
    domains: list[str],
    entries: list[dict[str, str]],
    rng: random.Random,
    offset: int,
) -> list[dict]:
    records = []
    for index in range(per_label):
        entry = entries[(offset + index) % len(entries)]
        domain = domains[index % len(domains)]
        records.append(
            make_record(
                split,
                0,
                index,
                domain,
                state_for_tracker(entry, index, rng),
                {
                    "category": entry["category"],
                    "vendor": entry["vendor"],
                    "blocked_host": entry["host"],
                    "source_list": entry["source"],
                },
            )
        )
        domain = domains[(per_label + index) % len(domains)]
        records.append(
            make_record(
                split,
                1,
                index,
                domain,
                safe_state(domain, index),
                {"category": "first_party_content", "construction": "same_site_safe_resource"},
            )
        )
        domain = domains[(2 * per_label + index) % len(domains)]
        records.append(
            make_record(
                split,
                2,
                index,
                domain,
                uncertain_state(domain, index),
                {"category": "ambiguous_tracking", "construction": "missing_or_conflicting_identity"},
            )
        )
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def write_dnr_rules(path: Path, entries: list[dict[str, str]]) -> int:
    hosts = sorted({entry["host"] for entry in entries})
    rules = [
        {
            "id": index,
            "priority": 1,
            "action": {"type": "block"},
            "condition": {
                "urlFilter": f"||{host}^",
                "excludedResourceTypes": ["main_frame"],
            },
        }
        for index, host in enumerate(hosts, start=1)
    ]
    # High-signal first-party paths cover ads served from the page's own host.
    # Keep this deliberately small: broad patterns such as `/ad` cause unacceptable
    # false positives on ordinary application routes.
    path_filters = [
        "/ads.js",
        "/ad.js",
        "/pagead.js",
        "/adsbygoogle.js",
        "/advertisement.js",
        "/ads/banner/",
        "/ads/adframe.",
        "/banners/pr_advertising_ads_banner.",
        "/adserver/",
        "/gampad/ads",
        "/pagead/",
        "/prebid/",
        "/tracking/pixel",
    ]
    next_id = len(rules) + 1
    for offset, url_filter in enumerate(path_filters):
        rules.append(
            {
                "id": next_id + offset,
                "priority": 1,
                "action": {"type": "block"},
                "condition": {
                    "urlFilter": url_filter,
                    "excludedResourceTypes": ["main_frame"],
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rules)


def main() -> None:
    args = parse_args()
    host_list_path = Path(args.host_list)
    vendors_path = Path(args.vendors)
    base_train_path = Path(args.base_train)
    base_calibration_path = Path(args.base_calibration)
    anti_ad_path = Path(args.anti_ad_domains)
    rng = random.Random(args.seed)

    benchmark_paths = [
        Path(value)
        for value in (
            args.benchmark_source or []
        )
    ]
    raw_entries = (
        parse_host_list(host_list_path)
        + vendor_entries(vendors_path)
        + benchmark_entries(benchmark_paths)
    )
    deduplicated = {
        (entry["category"], entry["vendor"], entry["host"]): entry for entry in raw_entries
    }
    entries = round_robin(list(deduplicated.values()))
    base_train = read_jsonl(base_train_path)
    base_calibration = read_jsonl(base_calibration_path)
    train_domains = sorted({record["source"] for record in base_train})
    calibration_domains = sorted({record["source"] for record in base_calibration})
    if set(train_domains) & set(calibration_domains):
        raise ValueError("base train and calibration domains overlap")

    tracker_train = build("train", args.train_per_label, train_domains, entries, rng, 0)
    tracker_calibration = build(
        "calibration", args.calibration_per_label, calibration_domains, entries, rng, args.train_per_label
    )
    write_jsonl(Path(args.train_out), tracker_train)
    combined_calibration = base_calibration + tracker_calibration
    write_jsonl(Path(args.calibration_out), combined_calibration)
    dnr_rules = write_dnr_rules(Path(args.dnr_out), entries)
    anti_ad_hosts = parse_bare_domain_list(anti_ad_path)
    anti_ad_dnr_rules = write_grouped_domain_rules(Path(args.anti_ad_dnr_out), anti_ad_hosts)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sources": {
            str(host_list_path): file_hash(host_list_path),
            str(vendors_path): file_hash(vendors_path),
            str(anti_ad_path): file_hash(anti_ad_path),
            **{str(path): file_hash(path) for path in benchmark_paths},
        },
        "taxonomy_entries": len(entries),
        "vendors": len({(entry["category"], entry["vendor"]) for entry in entries}),
        "dnr_rules": dnr_rules,
        "anti_ad_hosts": len(anti_ad_hosts),
        "anti_ad_dnr_rules": anti_ad_dnr_rules,
        "categories": dict(Counter(entry["category"] for entry in entries)),
        "training": {
            "path": args.train_out,
            "examples": len(tracker_train),
            "labels": dict(Counter(record["question"]["label"] for record in tracker_train)),
        },
        "combined_calibration": {
            "path": args.calibration_out,
            "examples": len(combined_calibration),
            "labels": dict(Counter(record["question"]["label"] for record in combined_calibration)),
        },
        "domain_overlap": sorted(
            {record["source"] for record in base_train + tracker_train}
            & {record["source"] for record in combined_calibration}
        ),
    }
    manifest_path = Path(args.train_out).with_name("tracker_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
