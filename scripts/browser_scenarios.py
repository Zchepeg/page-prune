from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat
from playwright.sync_api import BrowserContext, Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
EXTENSION = ROOT / "browser" / "extension"
SCENARIOS = [
    {
        "name": "wikipedia_article",
        "kind": "article",
        "url": "https://en.wikipedia.org/wiki/Advertising",
        "stable_visual": True,
    },
    {
        "name": "mdn_article",
        "kind": "article",
        "url": "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
        "stable_visual": True,
    },
    {
        "name": "github_login",
        "kind": "login",
        "url": "https://github.com/login",
        "stable_visual": True,
    },
    {
        "name": "bilibili_video",
        "kind": "video",
        "url": "https://www.bilibili.com/video/av170001/",
        "stable_visual": False,
    },
    {
        "name": "bilibili_feed",
        "kind": "feed",
        "url": "https://www.bilibili.com/",
        "stable_visual": False,
    },
]


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    kind: str
    url: str
    final_url: str
    status: int | None
    title: str
    text_length: int
    links_before: int
    links_after: int
    images_before: int
    images_after: int
    main_regions: int
    login_forms: int
    password_inputs: int
    video_elements: int
    player_elements: int
    scroll_y: int
    blocked_requests: tuple[str, ...]
    page_errors: tuple[str, ...]
    screenshot: bytes


def page_counts(page: Page) -> dict[str, int]:
    return page.evaluate(
        """
        () => ({
          textLength: (document.body?.innerText || '').trim().length,
          links: document.querySelectorAll('a[href]').length,
          images: document.images.length,
          mainRegions: document.querySelectorAll('main, [role="main"], article').length,
          loginForms: document.querySelectorAll('form').length,
          passwordInputs: document.querySelectorAll('input[type="password"]').length,
          videoElements: document.querySelectorAll('video').length,
          playerElements: document.querySelectorAll('[class*="player" i], [id*="player" i]').length,
          scrollY: Math.round(scrollY)
        })
        """
    )


def exercise(
    context: BrowserContext, scenario: dict[str, Any], timeout_ms: int
) -> ScenarioResult:
    page = context.new_page()
    blocked_requests: list[str] = []
    page_errors: list[str] = []
    page.on(
        "requestfailed",
        lambda request: blocked_requests.append(request.url)
        if request.failure == "net::ERR_BLOCKED_BY_CLIENT"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    response = None
    try:
        response = page.goto(
            scenario["url"], wait_until="domcontentloaded", timeout=timeout_ms
        )
        page.wait_for_timeout(3000)
        before = page_counts(page)
        page.evaluate("scrollTo(0, Math.max(document.body.scrollHeight * 0.75, 900))")
        page.wait_for_timeout(2500)
        after = page_counts(page)
        screenshot = page.screenshot(full_page=False, animations="disabled")
        return ScenarioResult(
            name=scenario["name"],
            kind=scenario["kind"],
            url=scenario["url"],
            final_url=page.url,
            status=response.status if response else None,
            title=page.title(),
            text_length=after["textLength"],
            links_before=before["links"],
            links_after=after["links"],
            images_before=before["images"],
            images_after=after["images"],
            main_regions=after["mainRegions"],
            login_forms=after["loginForms"],
            password_inputs=after["passwordInputs"],
            video_elements=after["videoElements"],
            player_elements=after["playerElements"],
            scroll_y=after["scrollY"],
            blocked_requests=tuple(blocked_requests[:100]),
            page_errors=tuple(page_errors[:20]),
            screenshot=screenshot,
        )
    finally:
        page.close()


def run_profile(
    executable: Path,
    extension: Path | None,
    timeout_ms: int,
) -> tuple[list[ScenarioResult], int]:
    args = ["--no-first-run", "--disable-background-networking"]
    if extension:
        args.extend(
            [
                f"--disable-extensions-except={extension}",
                f"--load-extension={extension}",
            ]
        )
    with tempfile.TemporaryDirectory(prefix="jev-scenario-") as profile:
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
                results = [exercise(context, scenario, timeout_ms) for scenario in SCENARIOS]
                workers = sum(
                    worker.url.startswith("chrome-extension://")
                    for worker in context.service_workers
                )
                return results, workers
            finally:
                context.close()


def image_difference(left: bytes, right: bytes) -> dict[str, float]:
    first = Image.open(io.BytesIO(left)).convert("RGB")
    second = Image.open(io.BytesIO(right)).convert("RGB")
    if first.size != second.size:
        second = second.resize(first.size)
    difference = ImageChops.difference(first, second).convert("L")
    histogram = difference.histogram()
    changed = sum(histogram[21:])
    pixels = first.width * first.height
    return {
        "changed_pixel_ratio": round(changed / pixels, 4),
        "mean_difference": round(ImageStat.Stat(difference).mean[0], 4),
    }


def ratio(current: int, baseline: int) -> float:
    if baseline <= 0:
        return 1.0 if current >= baseline else 0.0
    return round(current / baseline, 4)


def compare(
    baseline: ScenarioResult,
    current: ScenarioResult,
    stable_visual: bool,
    visual_directory: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    if current.status is None or current.status >= 400:
        failures.append("navigation")
    if not current.title:
        failures.append("title")
    if baseline.text_length >= 500 and ratio(current.text_length, baseline.text_length) < 0.65:
        failures.append("text")
    if baseline.main_regions > 0 and current.main_regions == 0:
        failures.append("main")
    if current.kind in {"article", "video", "feed"} and current.scroll_y < 400:
        failures.append("scroll")
    if current.kind == "login" and (
        current.login_forms == 0 or current.password_inputs == 0
    ):
        failures.append("login_form")
    if current.kind == "video" and (
        current.video_elements == 0 or current.player_elements == 0
    ):
        failures.append("video_player")
    if current.kind == "feed" and current.links_after < current.links_before:
        failures.append("feed_links")
    visual_directory.mkdir(parents=True, exist_ok=True)
    baseline_path = visual_directory / f"{current.name}-baseline.png"
    extension_path = visual_directory / f"{current.name}-extension.png"
    baseline_path.write_bytes(baseline.screenshot)
    extension_path.write_bytes(current.screenshot)
    visual = image_difference(baseline.screenshot, current.screenshot)
    if stable_visual and visual["changed_pixel_ratio"] > 0.2:
        failures.append("visual")
    baseline_payload = asdict(baseline)
    current_payload = asdict(current)
    baseline_payload.pop("screenshot")
    current_payload.pop("screenshot")
    return {
        "name": current.name,
        "kind": current.kind,
        "passed": not failures,
        "failures": failures,
        "ratios": {
            "text": ratio(current.text_length, baseline.text_length),
            "links": ratio(current.links_after, baseline.links_after),
            "images": ratio(current.images_after, baseline.images_after),
        },
        "visual": visual,
        "baseline": baseline_payload,
        "extension": current_payload,
        "screenshots": {
            "baseline": str(baseline_path.relative_to(ROOT)),
            "extension": str(extension_path.relative_to(ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", type=Path, default=EDGE)
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "scenarios.json"
    )
    parser.add_argument(
        "--visual-directory", type=Path, default=ROOT / "runs" / "visual"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    baseline, baseline_workers = run_profile(args.edge, None, args.timeout_ms)
    current, extension_workers = run_profile(
        args.edge, args.extension, args.timeout_ms
    )
    comparisons = [
        compare(left, right, scenario["stable_visual"], args.visual_directory)
        for left, right, scenario in zip(baseline, current, SCENARIOS)
    ]
    payload = {
        "extension_service_workers": extension_workers,
        "baseline_service_workers": baseline_workers,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "passed": extension_workers > 0 and all(item["passed"] for item in comparisons),
        "scenarios": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
