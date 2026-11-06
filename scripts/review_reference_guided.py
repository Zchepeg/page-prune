from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collector.schemas import Candidate
from collector.state_builder import build_state_payload
from detector.candidate_ranker import CandidateRanker
from evaluation.annotation import append_jsonl, latest_annotations, load_jsonl, pool_status


DEFAULT_RUN = ROOT / "runs" / "reference-guided-v1"
DEFAULT_POOL = ROOT / "data" / "training" / "reference_guided_v1_review.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "training" / "reference_guided_v1_human.jsonl"
PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEV 训练样本复核</title><style>
:root{color-scheme:dark;--bg:#0f1217;--panel:#191f28;--line:#303a48;--text:#edf2f7;--muted:#9eabbc;--accent:#63d6a0}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}.top{position:sticky;top:0;z-index:2;display:flex;gap:16px;align-items:center;padding:14px 22px;background:#141922;border-bottom:1px solid var(--line)}.top strong{font-size:18px}.stats{color:var(--muted)}.export{margin-left:auto}.wrap{display:grid;grid-template-columns:minmax(320px,42%) 1fr;gap:18px;padding:18px;max-width:1500px;margin:auto}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}.preview{display:flex;flex-direction:column;gap:12px;min-height:360px;background:#0b0e12;border-radius:8px;padding:18px}.element{border:1px solid #485569;border-radius:10px;padding:16px;background:#151b23}.element strong{display:block;font-size:18px;margin-bottom:8px}.meta{display:grid;grid-template-columns:105px 1fr;gap:8px 12px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.code{white-space:pre-wrap;max-height:190px;overflow:auto;background:#101318;padding:10px;border-radius:8px;color:#cbd5e0}.machine{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.machine span{color:#f6e05e}.actions{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px}button{border:1px solid var(--line);background:#252e3a;color:var(--text);padding:12px;border-radius:9px;cursor:pointer;font-weight:650}button:hover{border-color:#718096}.ad{background:#6b2737}.non{background:#245c45}.unknown{background:#514629}.nav{display:flex;gap:8px;margin-top:12px}.nav button{flex:1}.current{outline:2px solid var(--accent)}.hint{margin-top:12px;color:var(--muted)}.done{color:var(--accent)}@media(max-width:850px){.wrap{grid-template-columns:1fr}.top{flex-wrap:wrap}.export{margin-left:0}}
</style></head><body><div class="top"><strong>JEV 训练样本复核</strong><span class="stats" id="stats"></span><button class="export" id="export">导出已复核样本</button></div><main class="wrap"><section class="card"><div class="preview"><div class="element"><strong id="element-title"></strong><div id="page"></div></div><div><b>元素文本</b><div class="code" id="text"></div></div><div><b>资源地址</b><div class="code" id="resources"></div></div></div></section><section class="card"><dl class="meta"><dt>进度</dt><dd id="progress"></dd><dt>域名</dt><dd id="domain"></dd><dt>元素</dt><dd id="element"></dd><dt>尺寸</dt><dd id="size"></dd><dt>结构</dt><dd class="code" id="html"></dd><dt>模型信息</dt><dd class="machine"><button id="show-suggestion">0 显示</button><span id="machine"></span></dd></dl><div class="actions"><button class="ad" data-label="ad">1 广告</button><button class="non" data-label="non_ad">2 非广告</button><button class="unknown" data-label="insufficient_evidence">3 证据不足</button></div><div class="nav"><button id="prev">上一个</button><button id="next">跳过</button><button id="unreviewed">下一个未标注</button></div><div class="hint" id="message">快捷键：1 广告、2 非广告、3 证据不足、0 显示模型信息、左右方向键切换。</div></section></main><script>
const token=new URLSearchParams(location.search).get('token');let state={pool:[],annotations:{},status:{}},index=0,suggestionVisible=false;const q=id=>document.getElementById(id);const esc=value=>String(value??'');async function api(path,options={}){const headers={'X-JEV-Token':token,...(options.headers||{})};const response=await fetch(path,{...options,headers});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data}function nextUnreviewed(){for(let offset=1;offset<=state.pool.length;offset++){const next=(index+offset)%state.pool.length;if(!state.annotations[state.pool[next].id]){index=next;return}}index=Math.min(index+1,state.pool.length-1)}function update(){const item=state.pool[index];q('stats').textContent=`已复核 ${state.status.reviewed}/${state.status.total} · 广告 ${state.status.labels.ad} · 非广告 ${state.status.labels.non_ad} · 证据不足 ${state.status.labels.insufficient_evidence}`;if(!item){q('message').textContent='复核队列为空';return}const c=item.candidate,p=item.provenance||{};q('progress').textContent=`${index+1}/${state.pool.length}${state.annotations[item.id]?' · 已标注 '+state.annotations[item.id].label:''}`;q('domain').textContent=item.domain;q('element').textContent=`${c.tag}#${c.element_id}.${(c.classes||[]).join('.')}`;q('element-title').textContent=`${c.tag} · ${c.width} × ${c.height}`;q('page').textContent=c.page?.title||c.page?.url||item.domain;q('size').textContent=`${c.width} × ${c.height}${c.fixed?' · fixed':''}${c.visible?'':' · 不可见'}`;q('text').textContent=esc(c.text);q('resources').textContent=(c.resource_urls||[]).join('\n')||'无';q('html').textContent=esc(c.html);q('machine').textContent=suggestionVisible?`建议 ${item.suggested_label} · 广告概率 ${((p.ad_probability||0)*100).toFixed(1)}% · 集成分歧 ${((p.ensemble_std||0)*100).toFixed(1)}% · 规则 ${(p.rule_score||0).toFixed(2)} · ${(p.rule_evidence||[]).join(', ')}`:'';q('show-suggestion').textContent=suggestionVisible?'0 隐藏':'0 显示';document.querySelectorAll('[data-label]').forEach(button=>button.classList.toggle('current',state.annotations[item.id]?.label===button.dataset.label))}async function load(){state=await api('/api/state');const first=state.pool.findIndex(item=>!state.annotations[item.id]);index=first<0?0:first;update()}async function label(value){const item=state.pool[index];if(!item)return;state=await api('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,label:value})});suggestionVisible=false;nextUnreviewed();update()}document.querySelectorAll('[data-label]').forEach(button=>button.onclick=()=>label(button.dataset.label).catch(error=>q('message').textContent=error.message));q('show-suggestion').onclick=()=>{suggestionVisible=!suggestionVisible;update()};q('prev').onclick=()=>{suggestionVisible=false;index=Math.max(0,index-1);update()};q('next').onclick=()=>{suggestionVisible=false;index=Math.min(state.pool.length-1,index+1);update()};q('unreviewed').onclick=()=>{suggestionVisible=false;nextUnreviewed();update()};q('export').onclick=async()=>{try{const value=await api('/api/export',{method:'POST'});q('message').innerHTML=`<span class="done">已导出 ${value.examples} 条可训练样本</span>`}catch(error){q('message').textContent=error.message}};addEventListener('keydown',event=>{if(event.key==='1')label('ad');if(event.key==='2')label('non_ad');if(event.key==='3')label('insufficient_evidence');if(event.key==='0'){suggestionVisible=!suggestionVisible;update()}if(event.key==='ArrowLeft'){suggestionVisible=false;index=Math.max(0,index-1);update()}if(event.key==='ArrowRight'){suggestionVisible=false;index=Math.min(index+1,state.pool.length-1);update()}});load().catch(error=>q('message').textContent=error.message);
</script></body></html>"""


def export_reviewed(pool_path: Path, annotations_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    pool = load_jsonl(pool_path)
    annotations = latest_annotations(annotations_path)
    ranker = CandidateRanker()
    records = []
    counts = Counter()
    for item in pool:
        annotation = annotations.get(str(item.get("id", "")))
        if not annotation or annotation.get("label") not in {"ad", "non_ad"}:
            continue
        candidate = Candidate.from_dict(item["candidate"])
        assessment = ranker.assess(candidate)
        label = 0 if annotation["label"] == "ad" else 1
        records.append(
            {
                "id": f"reference-human-{item['id']}",
                "source": str(item["domain"]),
                "state": build_state_payload(candidate, assessment.evidence),
                "question": {
                    "instruction": "Classify this candidate web element.",
                    "options": ["ad", "non_ad"],
                    "label": label,
                },
                "provenance": {
                    "label_source": "human_review_reference_guided",
                    "annotation_mode": "human_review",
                    "reference_suggestion": str(item.get("suggested_label", "")),
                    "annotated_at": str(annotation.get("annotated_at", "")),
                },
            }
        )
        counts[annotation["label"]] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    output_path.write_text(content, encoding="utf-8")
    status = pool_status(pool, annotations)
    report = {
        "pool_examples": len(pool),
        "reviewed": status["reviewed"],
        "remaining": status["remaining"],
        "exported_examples": len(records),
        "exported_labels": {"ad": counts["ad"], "non_ad": counts["non_ad"]},
        "excluded_insufficient_evidence": status["labels"]["insufficient_evidence"],
        "domains": len({record["source"] for record in records}),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output_path.resolve()), "report": str(report_path.resolve()), "examples": len(records), **report}


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], settings: dict[str, Any]):
        super().__init__(address, handler)
        self.settings = settings


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: Any) -> None:
        self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def authorized(self) -> bool:
        query = parse_qs(urlsplit(self.path).query)
        supplied = self.headers.get("X-JEV-Token", "") or query.get("token", [""])[0]
        return secrets.compare_digest(supplied, self.server.settings["token"])

    def state(self) -> dict[str, Any]:
        pool = load_jsonl(self.server.settings["pool"])
        annotations = latest_annotations(self.server.settings["annotations"])
        return {"pool": pool, "annotations": annotations, "status": pool_status(pool, annotations)}

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/" and self.authorized():
            self.send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if not self.authorized():
            self.send_json(403, {"error": "invalid token"})
            return
        if path == "/api/state":
            self.send_json(200, self.state())
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.authorized():
            self.send_json(403, {"error": "invalid token"})
            return
        path = urlsplit(self.path).path
        if path == "/api/label":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4096:
                    raise ValueError("invalid request size")
                value = json.loads(self.rfile.read(length))
                identifier = str(value.get("id", ""))
                label = str(value.get("label", ""))
                valid_ids = {str(item.get("id", "")) for item in load_jsonl(self.server.settings["pool"])}
                if identifier not in valid_ids or label not in {"ad", "non_ad", "insufficient_evidence"}:
                    raise ValueError("invalid id or label")
                append_jsonl(
                    self.server.settings["annotations"],
                    {
                        "id": identifier,
                        "label": label,
                        "annotation_mode": "human_review",
                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.send_json(200, self.state())
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            return
        if path == "/api/export":
            self.send_json(
                200,
                export_reviewed(
                    self.server.settings["pool"],
                    self.server.settings["annotations"],
                    self.server.settings["output"],
                    self.server.settings["report"],
                ),
            )
            return
        self.send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review reference-guided training candidates")
    parser.add_argument("action", choices=["serve", "status", "export"], nargs="?", default="serve")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_RUN / "review_annotations.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_RUN / "review_export_report.json")
    parser.add_argument("--port", type=int, default=8877)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = load_jsonl(args.pool)
    annotations = latest_annotations(args.annotations)
    if args.action == "status":
        print(json.dumps(pool_status(pool, annotations), ensure_ascii=False, indent=2))
        return 0
    if args.action == "export":
        print(json.dumps(export_reviewed(args.pool, args.annotations, args.output, args.report), ensure_ascii=False, indent=2))
        return 0
    token = secrets.token_urlsafe(24)
    settings = {
        "token": token,
        "pool": args.pool.resolve(),
        "annotations": args.annotations.resolve(),
        "output": args.output.resolve(),
        "report": args.report.resolve(),
    }
    server = ReviewServer(("127.0.0.1", args.port), Handler, settings)
    print(f"http://127.0.0.1:{args.port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
