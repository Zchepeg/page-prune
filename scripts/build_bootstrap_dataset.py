from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HOST_RE = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}", re.I)
OPTIONS = ["ad", "non_ad", "insufficient_evidence"]
AD_SIZES = [(300, 250), (336, 280), (728, 90), (970, 250), (320, 50)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced, traceable ad-decision bootstrap dataset")
    parser.add_argument("--easylist", required=True)
    parser.add_argument("--adguard", required=True)
    parser.add_argument("--tranco", required=True)
    parser.add_argument("--train-out", default="data/training/bootstrap_train.jsonl")
    parser.add_argument("--calibration-out", default="data/training/calibration.jsonl")
    parser.add_argument("--train-per-label", type=int, default=150)
    parser.add_argument("--calibration-per-label", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261008)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_ad_hosts(paths: list[Path]) -> list[tuple[str, str]]:
    hosts: dict[str, str] = {}
    for path in paths:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line.startswith("||") or line.startswith("@@") or "^" not in line:
                    continue
                host = line[2:].split("^", 1)[0].lower().lstrip("*.")
                if HOST_RE.fullmatch(host) and ".." not in host:
                    hosts.setdefault(host, path.name)
    return sorted(hosts.items())


def read_domains(path: Path) -> list[str]:
    domains = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        domain = line.strip().lower()
        if HOST_RE.fullmatch(domain):
            domains.append(domain)
    return list(dict.fromkeys(domains))


def make_record(
    split: str,
    index: int,
    label: int,
    page_domain: str,
    state: dict,
    provenance: dict,
) -> dict:
    identity = json.dumps([split, label, page_domain, state], ensure_ascii=False, sort_keys=True)
    record_id = f"bootstrap-{split}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return {
        "id": record_id,
        "source": page_domain,
        "state": {"page_domain": page_domain, **state},
        "question": {
            "instruction": "Classify this candidate web element.",
            "options": OPTIONS,
            "label": label,
        },
        "provenance": {"label_source": "external_rule_bootstrap", "split": split, "index": index, **provenance},
    }


def ad_state(page_domain: str, ad_host: str, index: int, rng: random.Random) -> dict:
    width, height = AD_SIZES[index % len(AD_SIZES)]
    token = hashlib.sha256(f"{ad_host}:{index}".encode()).hexdigest()[:10]
    tag = rng.choice(["IFRAME", "INS", "DIV", "IMG"])
    marker = rng.choice(["ad-slot", "advertisement", "sponsored-banner", "promoted-ad"])
    return {
        "tag": tag,
        "text": rng.choice(["Advertisement", "Sponsored", "Promoted", ""]),
        "element_id": f"{marker}-{index % 19}",
        "classes": [marker, rng.choice(["sidebar", "leaderboard", "placement", "creative"])],
        "attributes": {"aria_label": rng.choice(["Advertisement", "Sponsored content", ""])},
        "resource_urls": [f"https://{ad_host}/ads/{token}/creative.js"],
        "width": width,
        "height": height,
        "fixed": index % 13 == 0,
    }


def non_ad_state(page_domain: str, index: int, rng: random.Random) -> dict:
    templates = [
        ("ARTICLE", "article-content", "Article body", 760, 900, "article.css"),
        ("NAV", "site-navigation", "Navigation", 1180, 64, "navigation.js"),
        ("DIV", "product-card", "Product details", 320, 420, "product-card.js"),
        ("FORM", "login-form", "Sign in", 360, 280, "login.js"),
        ("DIV", "user-comment", "Reader comment", 680, 140, "comments.js"),
        ("IMG", "hero-image", "", 960, 540, "hero.webp"),
    ]
    tag, class_name, text, width, height, asset = templates[index % len(templates)]
    subdomain = rng.choice(["static", "cdn", "assets", "www"])
    return {
        "tag": tag,
        "text": text,
        "element_id": f"{class_name}-{index % 23}",
        "classes": [class_name, "content-module"],
        "attributes": {"role": "navigation" if tag == "NAV" else ""},
        "resource_urls": [f"https://{subdomain}.{page_domain}/assets/{asset}"],
        "width": width,
        "height": height,
        "fixed": False,
    }


def uncertain_state(page_domain: str, index: int, rng: random.Random) -> dict:
    templates = [
        ("DIV", "ad-container", "", [], 640, 80, False),
        ("IFRAME", "embed-frame", "", [], 300, 250, False),
        ("ASIDE", "partner-widget", "Partner", [], 320, 180, False),
        ("DIV", "banner", "Recommended", [], 728, 90, False),
        ("DIV", "promoted", "", [f"https://{page_domain}/content/promo"], 300, 180, False),
        ("IFRAME", "ad-slot", "", [f"https://{page_domain}/ads/placeholder"], 0, 0, False),
    ]
    tag, class_name, text, urls, width, height, fixed = templates[index % len(templates)]
    return {
        "tag": tag,
        "text": text,
        "element_id": f"{class_name}-{index % 29}",
        "classes": [class_name, rng.choice(["widget", "module", "slot", "placeholder"])],
        "attributes": {},
        "resource_urls": urls,
        "width": width,
        "height": height,
        "fixed": fixed,
    }


def build_split(
    split: str,
    per_label: int,
    page_domains: list[str],
    ad_hosts: list[tuple[str, str]],
    rng: random.Random,
) -> list[dict]:
    records = []
    domain_order = page_domains.copy()
    rng.shuffle(domain_order)
    for label in range(3):
        for index in range(per_label):
            page_domain = domain_order[(label * per_label + index) % len(domain_order)]
            if label == 0:
                ad_host, source_name = ad_hosts[(index + (0 if split == "train" else per_label)) % len(ad_hosts)]
                state = ad_state(page_domain, ad_host, index, rng)
                provenance = {"source_list": source_name, "matched_ad_host": ad_host}
            elif label == 1:
                state = non_ad_state(page_domain, index, rng)
                provenance = {"source_list": "tranco_top_500.txt", "construction": "same-site benign asset"}
            else:
                state = uncertain_state(page_domain, index, rng)
                provenance = {"source_list": "mixed-ambiguity-templates", "construction": "single_or_conflicting_evidence"}
            records.append(make_record(split, index, label, page_domain, state, provenance))
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    easylist = Path(args.easylist)
    adguard = Path(args.adguard)
    tranco = Path(args.tranco)
    ad_hosts = extract_ad_hosts([easylist, adguard])
    domains = [domain for domain in read_domains(tranco) if domain not in {host for host, _ in ad_hosts}]
    required_domains = args.train_per_label * 2 + args.calibration_per_label * 3
    if len(ad_hosts) < args.train_per_label + args.calibration_per_label:
        raise ValueError("not enough ad hosts extracted from source lists")
    if len(domains) < min(required_domains, 400):
        raise ValueError("not enough benign page domains")

    rng = random.Random(args.seed)
    rng.shuffle(domains)
    calibration_domain_count = args.calibration_per_label * 3
    calibration_domains = domains[:calibration_domain_count]
    training_domains = domains[calibration_domain_count:]
    training = build_split("train", args.train_per_label, training_domains, ad_hosts, rng)
    calibration = build_split("calibration", args.calibration_per_label, calibration_domains, ad_hosts, rng)
    write_jsonl(Path(args.train_out), training)
    write_jsonl(Path(args.calibration_out), calibration)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sources": {
            str(easylist): sha256(easylist),
            str(adguard): sha256(adguard),
            str(tranco): sha256(tranco),
        },
        "extracted_ad_hosts": len(ad_hosts),
        "training": {
            "path": args.train_out,
            "examples": len(training),
            "labels": dict(Counter(record["question"]["label"] for record in training)),
            "domains": len({record["source"] for record in training}),
        },
        "calibration": {
            "path": args.calibration_out,
            "examples": len(calibration),
            "labels": dict(Counter(record["question"]["label"] for record in calibration)),
            "domains": len({record["source"] for record in calibration}),
        },
        "domain_overlap": sorted(
            {record["source"] for record in training} & {record["source"] for record in calibration}
        ),
    }
    manifest_path = Path(args.train_out).with_name("bootstrap_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
