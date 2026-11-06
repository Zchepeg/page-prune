from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.annotation import clean_url, load_jsonl
from evaluation.benchmark import training_domains
from service.config import load_config


EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
DOMAIN_PATTERN = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
DEFAULT_SITES = [
    "https://www.wikipedia.org/",
    "https://developer.mozilla.org/en-US/",
    "https://www.python.org/",
    "https://www.rust-lang.org/",
    "https://www.gnu.org/",
    "https://www.nasa.gov/",
    "https://www.bbc.com/",
    "https://www.reuters.com/",
    "https://apnews.com/",
    "https://www.theguardian.com/international",
    "https://www.npr.org/",
    "https://www.cnn.com/",
    "https://www.nytimes.com/",
    "https://www.washingtonpost.com/",
    "https://www.forbes.com/",
    "https://www.theverge.com/",
    "https://arstechnica.com/",
    "https://www.tomshardware.com/",
    "https://www.techradar.com/",
    "https://www.wired.com/",
    "https://www.engadget.com/",
    "https://www.cnet.com/",
    "https://www.imdb.com/",
    "https://www.weather.com/",
    "https://www.accuweather.com/",
    "https://www.espn.com/",
    "https://www.booking.com/",
    "https://www.tripadvisor.com/",
    "https://www.ebay.com/",
    "https://www.etsy.com/",
    "https://www.aliexpress.com/",
    "https://www.aljazeera.com/",
    "https://www.scmp.com/",
    "https://www.channelnewsasia.com/",
    "https://www.japantimes.co.jp/",
    "https://www.lemonde.fr/en/",
    "https://www.dw.com/en/",
    "https://www.euronews.com/",
    "https://www.cbssports.com/",
    "https://www.gamespot.com/",
    "https://www.ign.com/",
    "https://stackoverflow.com/",
    "https://www.geeksforgeeks.org/",
    "https://medium.com/",
    "https://www.quora.com/",
    "https://www.zhihu.com/",
    "https://www.sohu.com/",
    "https://www.163.com/",
    "https://www.ifeng.com/",
    "https://www.hupu.com/",
    "https://www.douban.com/",
    "https://www.autohome.com.cn/",
    "https://www.ctrip.com/",
    "https://www.zol.com.cn/",
    "https://www.gsmarena.com/",
    "https://www.allrecipes.com/",
    "https://www.investopedia.com/",
    "https://www.healthline.com/",
    "https://www.webmd.com/",
    "https://www.dictionary.com/",
    "https://www.openstreetmap.org/",
    "https://archive.org/",
    "https://www.britannica.com/",
    "https://dev.to/",
    "https://gitlab.com/",
    "https://bitbucket.org/",
    "https://www.npmjs.com/",
    "https://pypi.org/",
    "https://huggingface.co/",
    "https://www.kaggle.com/",
    "https://www.coursera.org/",
    "https://www.edx.org/",
    "https://www.udemy.com/",
    "https://www.axios.com/",
    "https://www.politico.com/",
    "https://www.latimes.com/",
    "https://www.usatoday.com/",
    "https://www.nbcnews.com/",
    "https://abcnews.go.com/",
    "https://www.cbsnews.com/",
    "https://time.com/",
    "https://www.newsweek.com/",
    "https://www.rollingstone.com/",
    "https://variety.com/",
    "https://www.hollywoodreporter.com/",
    "https://www.independent.co.uk/",
    "https://www.dailymail.co.uk/home/index.html",
    "https://www.target.com/",
    "https://www.bestbuy.com/",
    "https://www.homedepot.com/",
    "https://www.lowes.com/",
    "https://www.ikea.com/",
    "https://www.costco.com/",
    "https://www.wayfair.com/",
    "https://www.nike.com/",
    "https://www.adidas.com/",
    "https://www.zappos.com/",
    "https://www.expedia.com/",
    "https://www.kayak.com/",
    "https://www.airbnb.com/",
    "https://www.hotels.com/",
    "https://www.foodnetwork.com/",
    "https://www.epicurious.com/",
    "https://www.simplyrecipes.com/",
    "https://www.sina.com.cn/",
    "https://www.jd.com/",
    "https://www.bilibili.com/",
    "https://weibo.com/",
    "https://www.eastmoney.com/",
    "https://xueqiu.com/",
    "https://36kr.com/",
    "https://www.jianshu.com/",
    "https://www.oschina.net/",
    "https://juejin.cn/",
    "https://www.cnblogs.com/",
    "https://www.csdn.net/",
    "https://www.ithome.com/",
    "https://www.guancha.cn/",
    "https://www.thepaper.cn/",
    "https://www.jiemian.com/",
]


EXTRACT_SCRIPT = r"""
limit => {
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim().slice(0, 500)
  const visible = element => {
    const style = getComputedStyle(element)
    const rect = element.getBoundingClientRect()
    return style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) > 0 && rect.width >= 24 && rect.height >= 18
  }
  const nodes = [...document.querySelectorAll('iframe, img, video, aside, article, main, nav, header, footer, form, button, [role], [id], [class]')]
  const rows = []
  const seen = new Set()
  for (const element of nodes) {
    if (!visible(element)) continue
    const rect = element.getBoundingClientRect()
    const id = clean(element.id)
    const classes = [...element.classList].map(clean).filter(Boolean).slice(0, 12)
    const marker = `${element.tagName}|${id}|${classes.join('.')}|${Math.round(rect.width)}x${Math.round(rect.height)}|${Math.round(rect.x)}x${Math.round(rect.y)}`
    if (seen.has(marker)) continue
    seen.add(marker)
    const descriptor = `${id} ${classes.join(' ')} ${clean(element.getAttribute('aria-label'))} ${clean(element.getAttribute('data-ad-slot'))}`.toLowerCase()
    const suspicious = /(^|[-_\s])(ad|ads|advert|advertisement|sponsor|sponsored|promo|commercial|banner|dfp|gpt)([-_\s]|$)/.test(descriptor) || element.tagName === 'IFRAME'
    const neutral = /^(ARTICLE|MAIN|NAV|HEADER|FOOTER|FORM|BUTTON)$/.test(element.tagName)
    const commonSize = [[300,250],[336,280],[728,90],[970,250],[320,50],[160,600]].some(([w,h]) => Math.abs(rect.width-w) <= 12 && Math.abs(rect.height-h) <= 12)
    let hash = 2166136261
    for (let index = 0; index < marker.length; index += 1) hash = Math.imul(hash ^ marker.charCodeAt(index), 16777619)
    const control = (hash >>> 0) % 20 === 0
    const kind = suspicious ? 'suspicious' : neutral ? 'neutral' : commonSize ? 'ambiguous' : control ? 'control' : ''
    if (!kind) continue
    const directResources = []
    for (const value of [element.src, element.href, element.getAttribute('data-src'), element.getAttribute('data-url')]) {
      if (value && /^https?:/i.test(value)) directResources.push(value)
    }
    const resources = [...directResources]
    for (const child of element.querySelectorAll('img[src], iframe[src], video[src], source[src]')) {
      if (resources.length >= 8) break
      if (child.src && /^https?:/i.test(child.src)) resources.push(child.src)
    }
    const attributes = {}
    for (const name of ['role','aria-label','data-ad','data-ad-slot','data-testid','title']) {
      const value = clean(element.getAttribute(name))
      if (value) attributes[name] = value
    }
    rows.push({
      index: rows.length,
      kind,
      tag: element.tagName,
      text: clean(element.innerText || element.alt || ''),
      html: clean(element.outerHTML),
      element_id: id,
      classes,
      attributes,
      resource_urls: [...new Set(resources)].slice(0, 8),
      direct_resource_urls: [...new Set(directResources)].slice(0, 8),
      resource_ownership_known: true,
      width: Math.round(rect.width * 10) / 10,
      height: Math.round(rect.height * 10) / 10,
      fixed: getComputedStyle(element).position === 'fixed',
      selector: `jev-eval-${rows.length}`
    })
    element.setAttribute('data-jev-eval', `jev-eval-${rows.length}`)
    if (rows.length >= limit) break
  }
  return rows
}
"""


def registrable_host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def overlaps(domain: str, known: set[str]) -> bool:
    return any(domain == item or domain.endswith("." + item) or item.endswith("." + domain) for item in known)


def fingerprint(domain: str, raw: dict[str, Any]) -> str:
    value = {
        "domain": domain,
        "tag": raw["tag"],
        "id": raw["element_id"].lower(),
        "classes": sorted(value.lower() for value in raw["classes"]),
        "resources": [registrable_host(value) for value in raw["resource_urls"]],
        "size": [round(raw["width"] / 20), round(raw["height"] / 20)],
        "text": raw["text"].lower()[:120],
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_rows(
    rows: list[dict[str, Any]],
    per_domain: int,
    seed: int,
    candidate_kinds: tuple[str, ...] = ("suspicious", "control", "neutral", "ambiguous"),
) -> list[dict[str, Any]]:
    generator = random.Random(seed)
    selected = []
    groups = {kind: [] for kind in candidate_kinds}
    for row in rows:
        if row["kind"] in groups:
            groups[row["kind"]].append(row)
    for values in groups.values():
        generator.shuffle(values)
    while len(selected) < per_domain and any(groups.values()):
        for kind in candidate_kinds:
            if groups[kind] and len(selected) < per_domain:
                selected.append(groups[kind].pop())
    return selected


def load_catalog(path: Path, offset: int, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        domain_field = next((value for value in (reader.fieldnames or []) if value.lower() == "domain"), "")
        if not domain_field:
            raise ValueError("catalog CSV requires a Domain column")
        urls = []
        for index, row in enumerate(reader):
            if index < offset:
                continue
            domain = str(row.get(domain_field, "")).lower().strip().rstrip(".")
            if DOMAIN_PATTERN.fullmatch(domain):
                urls.append(f"https://{domain}/")
            if len(urls) >= limit:
                break
    return urls


def collect_page(
    page: Page,
    url: str,
    timeout_ms: int,
    per_domain: int,
    screenshot_dir: Path,
    seed: int,
    candidate_kinds: tuple[str, ...],
    screenshots: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(3500)
    page.evaluate("window.scrollTo(0, Math.min((document.scrollingElement?.scrollHeight || 0) * 0.35, 1800))")
    page.wait_for_timeout(1500)
    final_url = clean_url(page.url)
    domain = registrable_host(final_url)
    rows = page.evaluate(EXTRACT_SCRIPT, 240)
    selected = select_rows(rows, per_domain, seed, candidate_kinds)
    records = []
    for raw in selected:
        digest = fingerprint(domain, raw)
        identifier = f"eval-{digest[:20]}"
        screenshot: Path | None = None
        if screenshots:
            screenshot = screenshot_dir / f"{identifier}.png"
            locator = page.locator(f'[data-jev-eval="{raw["selector"]}"]').first
            try:
                locator.screenshot(path=str(screenshot), timeout=3000)
            except Exception:
                screenshot = None
        resources = [clean_url(value) for value in raw["resource_urls"]]
        resources = [value for value in resources if value]
        direct_resources = [clean_url(value) for value in raw["direct_resource_urls"]]
        direct_resources = [value for value in direct_resources if value]
        candidate = {
            "candidate_id": identifier,
            "page": {
                "url": final_url,
                "title": str(page.title())[:300],
                "top_level_domain": domain,
            },
            "tag": raw["tag"],
            "text": raw["text"][:500],
            "html": raw["html"][:1000],
            "element_id": raw["element_id"][:200],
            "classes": raw["classes"],
            "attributes": raw["attributes"],
            "resource_urls": resources,
            "direct_resource_urls": direct_resources,
            "resource_ownership_known": bool(raw["resource_ownership_known"]),
            "width": raw["width"],
            "height": raw["height"],
            "fixed": raw["fixed"],
            "visible": True,
        }
        records.append(
            {
                "id": identifier,
                "domain": domain,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "candidate": candidate,
                "review_context": {
                    "candidate_kind": raw["kind"],
                    "screenshot": str(screenshot.resolve()) if screenshot is not None else "",
                },
            }
        )
    return records, {
        "requested_url": url,
        "final_url": final_url,
        "domain": domain,
        "status": response.status if response else None,
        "candidates": len(records),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="*")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--edge", type=Path, default=EDGE)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "evaluation-annotation" / "pool.jsonl")
    parser.add_argument("--catalog-csv", type=Path)
    parser.add_argument("--catalog-offset", type=int, default=100)
    parser.add_argument("--catalog-limit", type=int, default=1000)
    parser.add_argument("--candidate-kinds", default="suspicious,control,neutral,ambiguous")
    parser.add_argument("--exclude-data", type=Path, action="append", default=[])
    parser.add_argument("--allow-training-domains", action="store_true")
    parser.add_argument("--no-screenshots", action="store_true")
    parser.add_argument("--revisit-existing-domains", action="store_true")
    return parser.parse_args()


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    valid_kinds = {"suspicious", "control", "neutral", "ambiguous"}
    candidate_kinds = tuple(value.strip() for value in args.candidate_kinds.split(",") if value.strip())
    if not candidate_kinds or set(candidate_kinds) - valid_kinds:
        raise ValueError("--candidate-kinds contains an unsupported value")
    config = load_config(args.config)
    sources = [ROOT / value for value in config.get("auto_training", {}).get("training_sources", [])]
    known = set() if args.allow_training_domains else training_domains(sources)
    for path in args.exclude_data:
        for item in load_jsonl(path):
            domain = str(item.get("domain", item.get("source", ""))).lower().strip()
            if domain:
                known.add(domain)
    if args.urls and args.catalog_csv:
        raise ValueError("positional URLs and --catalog-csv cannot be combined")
    sites = args.urls or (load_catalog(args.catalog_csv, args.catalog_offset, args.catalog_limit) if args.catalog_csv else DEFAULT_SITES)
    output = args.output.resolve()
    screenshot_dir = output.parent / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    report_path = output.parent / "collection.json"
    previous_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    invalid_domains = {
        str(item.get("domain", ""))
        for item in previous_report.get("pages", [])
        if isinstance(item.get("status"), int) and item["status"] >= 400
    }
    records = [item for item in load_jsonl(output) if str(item.get("domain", "")) not in invalid_domains]
    fingerprints = {str(item.get("id", "")) for item in records}
    existing_domains = {str(item.get("domain", "")) for item in records}
    pages = []
    with tempfile.TemporaryDirectory(prefix="jev-evaluation-") as profile:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                profile,
                executable_path=str(args.edge),
                headless=True,
                args=["--no-first-run", "--disable-background-networking"],
                viewport={"width": 1440, "height": 1000},
                locale="zh-CN",
            )
            try:
                for index, url in enumerate(sites):
                    requested_domain = registrable_host(url)
                    if requested_domain in existing_domains and not args.revisit_existing_domains:
                        continue
                    if overlaps(requested_domain, known):
                        pages.append({"requested_url": url, "skipped": "training_domain_overlap"})
                        continue
                    page = context.new_page()
                    try:
                        captured, status = collect_page(
                            page,
                            url,
                            args.timeout_ms,
                            args.per_domain,
                            screenshot_dir,
                            args.seed + index,
                            candidate_kinds,
                            not args.no_screenshots,
                        )
                    except Exception as exc:
                        pages.append({"requested_url": url, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    finally:
                        page.close()
                    pages.append(status)
                    if status["status"] is None or status["status"] >= 400:
                        pages[-1]["skipped"] = "http_error"
                        continue
                    if overlaps(status["domain"], known):
                        pages[-1]["skipped"] = "redirected_to_training_domain"
                        continue
                    for record in captured:
                        digest = record["id"]
                        if digest in fingerprints:
                            continue
                        fingerprints.add(digest)
                        records.append(record)
                        existing_domains.add(record["domain"])
                        if len(records) >= args.target:
                            break
                    write_records(output, records)
                    if len(records) >= args.target:
                        break
            finally:
                context.close()
    write_records(output, records)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "captured": len(records),
        "domains": len({item["domain"] for item in records}),
        "training_domains": len(known),
        "site_source": str(args.catalog_csv.resolve()) if args.catalog_csv else "explicit_urls" if args.urls else "built_in",
        "output": str(output),
        "pages": previous_report.get("pages", []) + pages,
    }
    required_domains = min(30, (min(args.target, 300) + max(1, args.per_domain) - 1) // max(1, args.per_domain))
    report["required_domains"] = required_domains
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(records) >= min(args.target, 300) and report["domains"] >= required_domains else 1


if __name__ == "__main__":
    raise SystemExit(main())
