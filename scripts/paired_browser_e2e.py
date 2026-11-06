from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
EXTENSION = ROOT / "browser" / "extension"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_security import LocalSessionManager
from service.api import make_handler


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>JEV Paired E2E</title>
<style>
body{font:16px system-ui;margin:40px}aside{display:block;width:360px;height:120px;margin:20px;padding:16px;border:1px solid #999}
</style>
</head>
<body>
<main>
<h1>配对端到端测试</h1>
<aside id="model-target">模型应隐藏这个灰区候选</aside>
<aside id="safe-control">安全控制元素必须保留</aside>
</main>
</body>
</html>"""


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@dataclass
class RecordedDecision:
    candidate_id: str
    element_id: str
    action: str
    source: str
    elapsed_ms: float


class E2EOrchestrator:
    def __init__(self, deferred_source="", defer_seconds=0.0) -> None:
        self.decisions: list[RecordedDecision] = []
        self.deferred_source = deferred_source
        self.deferred_responses = 0
        self.defer_seconds = defer_seconds
        self.target_started_at = None

    def process_many(self, candidates) -> list[dict[str, Any]]:
        results = []
        for candidate in candidates:
            target = candidate.element_id == 'model-target'
            now = time.monotonic()
            if target and self.target_started_at is None:
                self.target_started_at = now
            elapsed = now - self.target_started_at if self.target_started_at is not None else 0.0
            deferred = bool(self.deferred_source and target and (not self.deferred_responses or elapsed < self.defer_seconds))
            if deferred:
                self.deferred_responses += 1
            action = "hide" if candidate.element_id == "model-target" and not deferred else "allow"
            verdict = 'uncertain' if deferred else "block" if action == "hide" else "allow"
            self.decisions.append(
                RecordedDecision(candidate.candidate_id, candidate.element_id, action, self.deferred_source if deferred else 'jev', round(elapsed*1000, 3))
            )
            results.append(
                {
                    "command": {
                        "candidate_id": candidate.candidate_id,
                        "action": action,
                        "reason": "paired_e2e_jev",
                    },
                    "decision": {
                        "candidate_id": candidate.candidate_id,
                        "verdict": verdict,
                        "probabilities": {
                            "block": 0.97 if action == "hide" else 0.02,
                            "allow": 0.02 if action == "hide" else 0.97,
                            "uncertain": 0.01,
                        },
                        "level": "L2",
                        "evidence": ["paired_e2e"],
                        "model_version": "paired-e2e",
                        "question_version": "ad-classification-v1",
                        "diagnostics": {'model_loading_deferred': deferred and self.deferred_source == 'model_starting', 'model_budget_deferred': deferred and self.deferred_source == 'model_budget'},
                    },
                    "source": self.deferred_source if deferred else "jev",
                    "latency_ms": 0.1,
                    "training_capture": {
                        "saved": False,
                        "queued": False,
                        "reason": "e2e_isolated",
                        "example_id": "",
                    },
                    "detector": {"score": 0.6, "route": "jev"},
                }
            )
        return results

    def training_stats(self) -> dict[str, Any]:
        return {"capture": {"enabled": False}, "auto_training": {"enabled": False}}

    def runtime_stats(self):
        return {'backend': 'e2e_stub', 'loaded': True}

    def performance_stats(self):
        return {'scope': 'e2e_stub'}

    def probe_stats(self) -> dict[str, Any]:
        return {"enabled": False}

    def list_probe_reviews(self, limit: int) -> list[dict[str, Any]]:
        return []

    def record_probe(self, value: dict[str, Any]) -> dict[str, Any]:
        return {"saved": False, "reason": "e2e_isolated"}


def start_server(handler) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def prepare_extension(source: Path, destination: Path, port: int) -> Path:
    target = destination / "extension"
    shutil.copytree(source, target)
    background = target / "background.js"
    background.write_text(
        background.read_text(encoding="utf-8").replace(
            'const SERVICE_ROOT = "http://127.0.0.1:8765";',
            f'const SERVICE_ROOT = "http://127.0.0.1:{port}";',
        ),
        encoding="utf-8",
    )
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["host_permissions"] = [f"http://127.0.0.1:{port}/*"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def wait_for_extension_id(context, timeout_ms: int = 10000) -> str:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        workers = [
            worker
            for worker in context.service_workers
            if worker.url.startswith("chrome-extension://")
        ]
        if workers:
            return workers[0].url.split("/")[2]
        remaining = max(1, round((deadline - time.monotonic()) * 1000))
        try:
            worker = context.wait_for_event("serviceworker", timeout=min(remaining, 1000))
            if worker.url.startswith("chrome-extension://"):
                return worker.url.split("/")[2]
        except Exception:
            pass
    raise RuntimeError("extension_service_worker_not_found")


def run(edge: Path, extension_source: Path, output: Path, deferred_source="", defer_seconds=0.0, steady_seconds=1.0) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jev-paired-e2e-") as temporary:
        temporary_path = Path(temporary)
        sessions = LocalSessionManager(
            registry_path=temporary_path / "security" / "paired_clients.json"
        )
        orchestrator = E2EOrchestrator(deferred_source, defer_seconds)
        service, service_thread = start_server(make_handler(orchestrator, sessions))
        page_server, page_thread = start_server(PageHandler)
        extension = prepare_extension(
            extension_source, temporary_path, service.server_port
        )
        try:
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(temporary_path / "profile"),
                    executable_path=str(edge),
                    headless=True,
                    args=[
                        "--no-first-run",
                        "--disable-background-networking",
                        f"--disable-extensions-except={extension}",
                        f"--load-extension={extension}",
                    ],
                    viewport={"width": 1280, "height": 800},
                    locale="zh-CN",
                )
                try:
                    extension_id = wait_for_extension_id(context)
                    options = context.new_page()
                    options.goto(
                        f"chrome-extension://{extension_id}/options.html",
                        wait_until="domcontentloaded",
                    )
                    options.locator("#code").fill(sessions.pairing_code)
                    options.locator("button").click()
                    options.locator("#status").filter(has_text="已绑定").wait_for(
                        timeout=10000
                    )
                    page = context.new_page()
                    page.goto(
                        f"http://127.0.0.1:{page_server.server_port}/",
                        wait_until="domcontentloaded",
                    )
                    target_handle = page.locator('#model-target').element_handle()
                    page.locator("#model-target[data-jev-blocked='paired_e2e_jev']").wait_for(
                        state="attached", timeout=10000 + round(defer_seconds*1000)
                    )
                    same_target_node = target_handle.evaluate("element => element === document.getElementById('model-target')")
                    target_decisions = [decision for decision in orchestrator.decisions if decision.element_id == 'model-target']
                    deferred_actions_allow = all(decision.action == 'allow' for decision in target_decisions if decision.source == deferred_source)
                    first_completion = next((decision for decision in target_decisions if decision.action == 'hide'), None)
                    target_display = page.locator("#model-target").evaluate(
                        "element => getComputedStyle(element).display"
                    )
                    control_display = page.locator("#safe-control").evaluate(
                        "element => getComputedStyle(element).display"
                    )
                    page.wait_for_timeout(round(steady_seconds*1000))
                    initial_hide_count = sum(decision.element_id == 'model-target' and decision.action == 'hide' for decision in orchestrator.decisions)
                    page.locator('#model-target').evaluate("element => element.style.setProperty('display', 'block', 'important')")
                    page.wait_for_function("getComputedStyle(document.getElementById('model-target')).display === 'none'", timeout=5000)
                    page.wait_for_timeout(500)
                    restored_hide_count = sum(decision.element_id == 'model-target' and decision.action == 'hide' for decision in orchestrator.decisions)
                    previous_display_preserved = page.locator('#model-target').get_attribute('data-jev-previous-display') == ''
                    paired_clients = sessions.stats()["paired_clients"]
                    active_sessions = sessions.stats()["active_sessions"]
                    encrypted_decisions = len(orchestrator.decisions)
                    service.shutdown()
                    service.server_close()
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_timeout(2000)
                    offline_target_display = page.locator("#model-target").evaluate(
                        "element => getComputedStyle(element).display"
                    )
                    payload = {
                        "passed": all(
                            [
                                target_display == "none",
                                control_display != "none",
                                offline_target_display != "none",
                                paired_clients == 1,
                                active_sessions >= 1,
                                encrypted_decisions >= 2,
                                same_target_node,
                                initial_hide_count == 1,
                                restored_hide_count == 2,
                                previous_display_preserved,
                                not deferred_source or (orchestrator.deferred_responses >= 1 and len(target_decisions) >= 2 and deferred_actions_allow and target_decisions[-1].source == 'jev' and first_completion is not None and first_completion.elapsed_ms >= defer_seconds*1000),
                            ]
                        ),
                        "extension_id": extension_id,
                        "paired_clients": paired_clients,
                        "active_sessions": active_sessions,
                        "encrypted_decisions": encrypted_decisions,
                        "deferred_source": deferred_source,
                        "deferred_responses": orchestrator.deferred_responses,
                        "simulated_load_seconds": defer_seconds,
                        "same_target_node": same_target_node,
                        "initial_hide_count": initial_hide_count,
                        "steady_seconds": steady_seconds,
                        "restored_hide_count": restored_hide_count,
                        "previous_display_preserved": previous_display_preserved,
                        "target_first_hide_command_ms": first_completion.elapsed_ms if first_completion else None,
                        "scope": "isolated encrypted browser retry test; simulated readiness, not Qwen performance",
                        "target_display": target_display,
                        "control_display": control_display,
                        "offline_target_display": offline_target_display,
                        "decisions": [asdict(decision) for decision in orchestrator.decisions],
                    }
                finally:
                    context.close()
        finally:
            if service_thread.is_alive():
                service.shutdown()
                service.server_close()
            page_server.shutdown()
            page_server.server_close()
            service_thread.join(timeout=2)
            page_thread.join(timeout=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", type=Path, default=EDGE)
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    parser.add_argument('--defer-first', choices=('model_starting', 'model_budget'), default='')
    parser.add_argument('--defer-seconds', type=float, default=0.0)
    parser.add_argument('--steady-seconds', type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "paired-e2e.json"
    )
    args = parser.parse_args()
    if not 0 <= args.defer_seconds <= 45 or (args.defer_seconds and not args.defer_first):
        parser.error('--defer-seconds requires --defer-first and must be between 0 and 45')
    if not 1 <= args.steady_seconds <= 120:
        parser.error('--steady-seconds must be between 1 and 120')
    return args


def main() -> int:
    args = parse_args()
    payload = run(args.edge, args.extension, args.output, args.defer_first, args.defer_seconds, args.steady_seconds)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
