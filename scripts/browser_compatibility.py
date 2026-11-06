from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import BrowserContext, Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
EXTENSION = ROOT / "browser" / "extension"
DEFAULT_SITES = [
    "https://www.wikipedia.org/",
    "https://developer.mozilla.org/en-US/",
    "https://github.com/",
    "https://www.jd.com/",
    "https://www.bilibili.com/",
    "https://www.sina.com.cn/",
]


@dataclass(frozen=True)
class Snapshot:
    url: str
    final_url: str
    title: str
    text_length: int
    links: int
    visible_links: int
    images: int
    loaded_images: int
    forms: int
    main_regions: int
    console_errors: tuple[str, ...]
    page_errors: tuple[str, ...]
    failed_requests: tuple[dict[str, str], ...]
    status: int | None
    elapsed_ms: int


def collect(context: BrowserContext, url: str, timeout_ms: int) -> Snapshot:
    page = context.new_page()
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[dict[str, str]] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            {"url": request.url, "reason": request.failure or "unknown"}
        ),
    )
    response = None
    elapsed_ms = 0
    try:
        started = time.perf_counter()
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(2500)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        metrics: dict[str, Any] = page.evaluate(
            """
            () => {
              const visible = element => {
                const style = getComputedStyle(element)
                const rect = element.getBoundingClientRect()
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0
              }
              const links = [...document.querySelectorAll('a[href]')]
              const images = [...document.images]
              return {
                title: document.title,
                textLength: (document.body?.innerText || '').trim().length,
                links: links.length,
                visibleLinks: links.filter(visible).length,
                images: images.length,
                loadedImages: images.filter(image => image.complete && image.naturalWidth > 0).length,
                forms: document.forms.length,
                mainRegions: document.querySelectorAll('main, [role="main"], article').length
              }
            }
            """
        )
        return Snapshot(
            url=url,
            final_url=page.url,
            title=metrics["title"],
            text_length=metrics["textLength"],
            links=metrics["links"],
            visible_links=metrics["visibleLinks"],
            images=metrics["images"],
            loaded_images=metrics["loadedImages"],
            forms=metrics["forms"],
            main_regions=metrics["mainRegions"],
            console_errors=tuple(console_errors[:20]),
            page_errors=tuple(page_errors[:20]),
            failed_requests=tuple(failed_requests[:100]),
            status=response.status if response else None,
            elapsed_ms=elapsed_ms,
        )
    finally:
        page.close()


def run_profile(
    executable: Path,
    sites: list[str],
    timeout_ms: int,
    extension: Path | None,
) -> tuple[list[Snapshot], int]:
    args = ["--no-first-run", "--disable-background-networking"]
    if extension:
        args.extend(
            [
                f"--disable-extensions-except={extension}",
                f"--load-extension={extension}",
            ]
        )
    with tempfile.TemporaryDirectory(prefix="jev-compat-") as profile:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                profile,
                executable_path=str(executable),
                headless=True,
                args=args,
                viewport={"width": 1440, "height": 1000},
                locale="zh-CN",
            )
            try:
                snapshots = [collect(context, site, timeout_ms) for site in sites]
                extension_workers = sum(
                    worker.url.startswith("chrome-extension://")
                    for worker in context.service_workers
                )
                return snapshots, extension_workers
            finally:
                context.close()


def ratio(current: int, baseline: int) -> float:
    if baseline <= 0:
        return 1.0 if current >= baseline else 0.0
    return round(current / baseline, 4)


def compare(baseline: Snapshot, current: Snapshot) -> dict[str, Any]:
    failures = []
    text_ratio = ratio(current.text_length, baseline.text_length)
    link_ratio = ratio(current.visible_links, baseline.visible_links)
    image_ratio = ratio(current.loaded_images, baseline.loaded_images)
    if current.status is None or current.status >= 400:
        failures.append("navigation")
    if not current.title:
        failures.append("title")
    if baseline.text_length >= 300 and text_ratio < 0.65:
        failures.append("text")
    if baseline.visible_links >= 10 and link_ratio < 0.55:
        failures.append("links")
    if baseline.loaded_images >= 5 and image_ratio < 0.35:
        failures.append("images")
    if baseline.forms > 0 and current.forms == 0:
        failures.append("forms")
    if baseline.main_regions > 0 and current.main_regions == 0:
        failures.append("main")
    return {
        "url": baseline.url,
        "passed": not failures,
        "failures": failures,
        "blocked_requests": [
            request
            for request in current.failed_requests
            if request["reason"] == "net::ERR_BLOCKED_BY_CLIENT"
        ],
        "ratios": {
            "text": text_ratio,
            "visible_links": link_ratio,
            "loaded_images": image_ratio,
        },
        "baseline": asdict(baseline),
        "extension": asdict(current),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="*")
    parser.add_argument("--edge", type=Path, default=EDGE)
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "compatibility.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sites = args.urls or DEFAULT_SITES
    baseline, baseline_workers = run_profile(
        args.edge, sites, args.timeout_ms, extension=None
    )
    current, extension_workers = run_profile(
        args.edge, sites, args.timeout_ms, extension=args.extension
    )
    comparisons = [compare(left, right) for left, right in zip(baseline, current)]
    payload = {
        "extension_service_workers": extension_workers,
        "baseline_service_workers": baseline_workers,
        "passed": extension_workers > 0 and all(item["passed"] for item in comparisons),
        "sites": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
