from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev.lightweight_backend import TfidfBackend
from local_security import LocalSessionManager
from scripts.evaluate_fresh_blind import make_evaluator
from scripts.paired_browser_e2e import EDGE, EXTENSION, E2EOrchestrator, PageHandler, prepare_extension, start_server, wait_for_extension_id
from service.api import make_handler

PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Lightweight safety fixture</title>
<style>body{font:16px system-ui;margin:30px}aside{width:360px;height:120px;padding:10px}article,nav,form{padding:12px;min-height:50px}video{display:block;width:400px;height:150px}</style></head><body>
<main id="main"><h1>Local page safety fixture</h1>
<aside id="sponsor-card">Sponsored Advertisement. Paid promotion. Save today with our special offer.</aside>
<article id="story" class="sponsor-notice" aria-label="Sponsored">Normal editorial content must remain readable.</article>
<nav id="navigation" class="sponsor-notice" aria-label="Sponsored"><a id="local-link" href="#destination">Local navigation</a></nav>
<form id="login" class="sponsor-notice" aria-label="Sponsored" onsubmit="event.preventDefault();this.dataset.submitted='true'">
<label>User<input id="user" name="user" data-ad></label><label>Password<input id="password" type="password" data-ad></label><button id="submit" data-ad>Sign in locally</button></form>
<video id="safe-video" controls preload="none" data-ad></video>
<aside id="normal-card">Ordinary reading recommendation</aside><section id="destination">Destination</section>
</main></body></html>'''


class FixtureHandler(PageHandler):
    def do_GET(self):
        body = PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LightweightOrchestrator(E2EOrchestrator):
    def __init__(self, plan, checkpoint, force_action=False):
        super().__init__()
        self.backend = TfidfBackend(checkpoint)
        self.evaluator = make_evaluator(plan, self.backend, self.backend.metadata['state_schema'])
        self.records = []
        self.force_action = force_action

    def process_many(self, candidates):
        from evaluation.benchmark import AblationMode

        results = []
        for candidate in candidates:
            started = time.perf_counter()
            assessment = self.evaluator.ranker.assess(candidate)
            decision, source = self.evaluator._decide(candidate, assessment, AblationMode.CASCADE)
            if self.force_action and candidate.element_id == 'sponsor-card':
                from jev.types import DecisionLevel, Verdict
                decision = replace(decision, verdict=Verdict.BLOCK, probabilities={'block':.97,'allow':.03,'uncertain':0.0}, level=DecisionLevel.L2, model_version='forced-action-fixture-only')
                source = 'forced_fixture'
            action = self.evaluator.policy.apply(candidate, decision)
            self.records.append({'element_id':candidate.element_id, 'source':source, 'action':action.action.value, 'p_ad':decision.probabilities['block'], 'reason':action.reason})
            results.append({'command':{'candidate_id':candidate.candidate_id, 'action':action.action.value, 'reason':action.reason}, 'decision':decision.to_dict(), 'source':source, 'latency_ms':round((time.perf_counter()-started)*1000,3), 'training_capture':{'saved':False,'queued':False,'reason':'isolated_safety_fixture'}, 'detector':{'score':assessment.score,'route':assessment.route}})
        return results


def run(directory, force_action=False):
    if directory.exists():
        raise ValueError('refusing to overwrite safety results')
    plan = json.loads((ROOT/'runs/fresh-blind-review-v1/plan.json').read_text(encoding='utf-8'))
    orchestrator = LightweightOrchestrator(plan, ROOT/plan['models']['tfidf']['checkpoint'], force_action)
    directory.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='jev-lightweight-safety-') as temporary:
        temporary = Path(temporary)
        sessions = LocalSessionManager(registry_path=temporary/'pairs.json')
        service, worker = start_server(make_handler(orchestrator,sessions))
        pages, pages_worker = start_server(FixtureHandler)
        extension = prepare_extension(EXTENSION,temporary,service.server_port)
        try:
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(str(temporary/'profile'),executable_path=str(EDGE),headless=True,args=['--no-first-run','--disable-background-networking',f'--disable-extensions-except={extension}',f'--load-extension={extension}'])
                try:
                    identifier = wait_for_extension_id(context)
                    options = context.new_page()
                    options.goto(f'chrome-extension://{identifier}/options.html')
                    options.locator('#code').fill(sessions.pairing_code)
                    options.locator('button').click()
                    options.locator('#status').filter(has_text='已绑定').wait_for(timeout=10000)
                    page = context.new_page()
                    page.goto(f'http://127.0.0.1:{pages.server_port}/')
                    page.wait_for_timeout(2500)
                    visible = {name:page.locator('#'+name).is_visible() for name in ('main','story','navigation','login','user','password','submit','safe-video','normal-card')}
                    page.locator('#user').fill('fixture-user')
                    page.locator('#password').fill('fixture-password')
                    page.locator('#submit').click()
                    form_works = page.locator('#login').get_attribute('data-submitted') == 'true'
                    page.locator('#local-link').click()
                    navigation_works = page.url.endswith('#destination')
                    video_controls = page.locator('#safe-video').evaluate('element => element.controls')
                    target_hidden = not page.locator('#sponsor-card').is_visible()
                    target_record = next(r for r in reversed(orchestrator.records) if r['element_id']=='sponsor-card')
                    expected_hidden = target_record['action'] in ('hide','block')
                    actual_model_action = any(r['element_id']=='sponsor-card' and r['source']=='model' and r['action'] in ('hide','block') for r in orchestrator.records)
                    page.screenshot(path=str(directory/'active.png'))
                    service.shutdown()
                    service.server_close()
                    restored = page.locator('#sponsor-card').evaluate("element => { const original=element.dataset.jevPreviousDisplay; element.style.setProperty('display',original || 'block'); return original !== undefined; }")
                    page.wait_for_timeout(750)
                    restore_visible = page.locator('#sponsor-card').is_visible()
                    page.reload(wait_until='domcontentloaded')
                    page.wait_for_timeout(1500)
                    offline_visible = page.locator('#sponsor-card').is_visible()
                    report = {'scope':'synthetic functional safety fixture; not independent accuracy or media playback evidence','forced_action_fixture':force_action,'passed':all(visible.values()) and form_works and navigation_works and video_controls and target_hidden == expected_hidden and (not expected_hidden or restored) and restore_visible and offline_visible,'checks':{'protected_visible':visible,'form_submit':form_works,'local_navigation':navigation_works,'video_controls_present':video_controls,'target_hidden':target_hidden,'expected_hidden_from_policy':expected_hidden,'real_model_action':actual_model_action,'positive_action_exercised':expected_hidden,'original_display_stored':restored,'offline_restore_visible':restore_visible,'offline_reload_fail_open':offline_visible},'decisions':orchestrator.records,'production_service_touched':False,'training_enabled':False}
                finally:
                    context.close()
        finally:
            if worker.is_alive():
                service.shutdown()
            service.server_close()
            pages.shutdown()
            pages.server_close()
            worker.join(timeout=2)
            pages_worker.join(timeout=2)
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report['passed']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs/lightweight-page-safety-v1')
    parser.add_argument('--force-action-fixture',action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output.resolve(), args.force_action_fixture) else 1)


if __name__ == '__main__':
    main()
