from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.annotation import append_jsonl, freeze_pool, latest_annotations, load_jsonl, pool_status


DEFAULT_RUN = ROOT / "runs" / "evaluation-annotation"
PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEV 独立评测标注</title><style>
:root{color-scheme:dark;--bg:#101318;--panel:#1a2029;--line:#303947;--text:#edf2f7;--muted:#9da9b8;--accent:#68d391}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}.top{position:sticky;top:0;z-index:2;display:flex;gap:18px;align-items:center;padding:14px 22px;background:#141922;border-bottom:1px solid var(--line)}.top strong{font-size:18px}.stats{color:var(--muted)}.wrap{display:grid;grid-template-columns:minmax(340px,48%) 1fr;gap:18px;padding:18px;max-width:1500px;margin:auto}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}.shot{display:flex;align-items:center;justify-content:center;min-height:360px;background:#0b0e12;border-radius:8px;overflow:hidden}.shot img{max-width:100%;max-height:620px;object-fit:contain}.empty{max-width:320px;padding:28px;text-align:center;line-height:1.7;color:#e2e8f0;background:#202938;border:1px dashed #718096;border-radius:10px}.meta{display:grid;grid-template-columns:110px 1fr;gap:8px 12px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.code{white-space:pre-wrap;max-height:150px;overflow:auto;background:#101318;padding:10px;border-radius:8px;color:#cbd5e0}.machine{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.machine button{padding:7px 10px}.machine span{color:#f6e05e}.actions{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px}button{border:1px solid var(--line);background:#252e3a;color:var(--text);padding:13px;border-radius:9px;cursor:pointer;font-weight:650}button:hover{border-color:#718096}button:disabled{opacity:.45;cursor:not-allowed}.ad{background:#6b2737}.non{background:#245c45}.unknown{background:#514629}.nav{display:flex;gap:8px;margin-top:12px}.nav button{flex:1}.current{outline:2px solid var(--accent)}.hint{margin-top:12px;color:var(--muted)}.freeze{margin-left:auto}.done{color:var(--accent)}@media(max-width:850px){.wrap{grid-template-columns:1fr}.top{flex-wrap:wrap}.freeze{margin-left:0}}
</style></head><body><div class="top"><strong>JEV 独立评测标注</strong><span class="stats" id="stats"></span><button class="freeze" id="freeze">冻结评测集</button></div><main class="wrap"><section class="card"><div class="shot" id="shot"></div></section><section class="card"><dl class="meta"><dt>进度</dt><dd id="progress"></dd><dt>域名</dt><dd id="domain"></dd><dt>候选类型</dt><dd id="kind"></dd><dt>元素</dt><dd id="element"></dd><dt>尺寸</dt><dd id="size"></dd><dt>文本</dt><dd class="code" id="text"></dd><dt>资源</dt><dd class="code" id="resources"></dd><dt>HTML</dt><dd class="code" id="html"></dd><dt id="machine-label">机器建议</dt><dd class="machine" id="machine-controls"><button id="show-suggestion">0 显示建议</button><button id="accept-suggestion" disabled>Enter 采用建议</button><span id="machine"></span></dd></dl><div class="actions"><button class="ad" data-label="ad">1 广告</button><button class="non" data-label="non_ad">2 非广告</button><button class="unknown" data-label="insufficient_evidence">3 证据不足</button></div><div class="nav"><button id="prev">上一个</button><button id="next">跳过</button><button id="unreviewed">下一个未标注</button></div><div class="hint" id="message">快捷键：1、2、3 独立标注，左右方向键切换。</div></section></main><script>
const token=new URLSearchParams(location.search).get('token');let state={pool:[],annotations:{},prelabels:{},status:{},blind:false},index=0,suggestionVisible=false;
const q=id=>document.getElementById(id);const esc=value=>String(value??'');
async function api(path,options={}){const headers={'X-JEV-Token':token,...(options.headers||{})};const response=await fetch(path,{...options,headers});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data}
function update(){const item=state.pool[index];q('stats').textContent=`已标注 ${state.status.reviewed}/${state.status.total} · ${state.status.reviewed_domains} 个域名 · 广告 ${state.status.labels.ad} · 非广告 ${state.status.labels.non_ad} · 证据不足 ${state.status.labels.insufficient_evidence}`;q('machine-label').style.display=state.blind?'none':'';q('machine-controls').style.display=state.blind?'none':'flex';if(!item){q('message').textContent='候选池为空';return}const c=item.candidate,pre=state.prelabels[item.id];q('progress').textContent=`${index+1}/${state.pool.length}${state.annotations[item.id]?' · 已标注 '+state.annotations[item.id].label:''}`;q('domain').textContent=item.domain;q('kind').textContent=item.review_context.candidate_kind;q('element').textContent=`${c.tag}#${c.element_id}.${c.classes.join('.')}`;q('size').textContent=`${c.width} × ${c.height}`;q('text').textContent=esc(c.text);q('resources').textContent=c.resource_urls.join('\n');q('html').textContent=esc(c.html);q('shot').innerHTML=item.review_context.screenshot_available?`<img src="/image/${encodeURIComponent(item.id)}?token=${encodeURIComponent(token)}" alt="候选截图">`:'<span class="empty">无可用截图，请根据结构证据标注；无法判断时选择“证据不足”。</span>';q('machine').textContent=suggestionVisible&&pre?`${pre.suggested_label} · ${(pre.confidence*100).toFixed(1)}% · ${pre.source} · ${pre.evidence.join(', ')}`:'';q('show-suggestion').textContent=suggestionVisible?'0 隐藏建议':'0 显示建议';q('accept-suggestion').disabled=!(suggestionVisible&&pre);document.querySelectorAll('[data-label]').forEach(button=>button.classList.toggle('current',state.annotations[item.id]?.label===button.dataset.label))}
async function load(){state=await api('/api/state');const first=state.pool.findIndex(item=>!state.annotations[item.id]);index=first<0?0:first;update()}
async function label(value,assisted=false){const item=state.pool[index];if(!item)return;state=await api('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,label:value,assisted})});suggestionVisible=false;nextUnreviewed();update()}
function nextUnreviewed(){for(let offset=1;offset<=state.pool.length;offset++){const next=(index+offset)%state.pool.length;if(!state.annotations[state.pool[next].id]){index=next;return}}index=Math.min(index+1,state.pool.length-1)}
document.querySelectorAll('[data-label]').forEach(button=>button.onclick=()=>label(button.dataset.label).catch(error=>q('message').textContent=error.message));q('show-suggestion').onclick=()=>{suggestionVisible=!suggestionVisible;update()};q('accept-suggestion').onclick=()=>{const pre=state.prelabels[state.pool[index]?.id];if(pre&&suggestionVisible)label(pre.suggested_label,true)};q('prev').onclick=()=>{suggestionVisible=false;index=Math.max(0,index-1);update()};q('next').onclick=()=>{suggestionVisible=false;index=Math.min(state.pool.length-1,index+1);update()};q('unreviewed').onclick=()=>{suggestionVisible=false;nextUnreviewed();update()};q('freeze').onclick=async()=>{try{const value=await api('/api/freeze',{method:'POST'});q('message').innerHTML=`<span class="done">已冻结 ${value.examples} 条：${value.output}</span>`}catch(error){q('message').textContent=`暂不能冻结：${error.message}`}};addEventListener('keydown',event=>{if(event.key==='1')label('ad');if(event.key==='2')label('non_ad');if(event.key==='3')label('insufficient_evidence');if(event.key==='0'){suggestionVisible=!suggestionVisible;update()}if(event.key==='Enter'&&suggestionVisible){const pre=state.prelabels[state.pool[index]?.id];if(pre)label(pre.suggested_label,true)}if(event.key==='ArrowLeft'){suggestionVisible=false;index=Math.max(0,index-1);update()}if(event.key==='ArrowRight'){suggestionVisible=false;index=Math.min(state.pool.length-1,index+1);update()}});load().catch(error=>q('message').textContent=error.message);
</script></body></html>"""


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
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: Any) -> None:
        self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def authorized(self) -> bool:
        query = parse_qs(urlsplit(self.path).query)
        supplied = self.headers.get("X-JEV-Token", "") or query.get("token", [""])[0]
        return secrets.compare_digest(supplied, self.server.settings["token"])

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
        if path.startswith("/image/"):
            identifier = path.removeprefix("/image/")
            item = next((value for value in self.pool() if value.get("id") == identifier), None)
            if not item:
                self.send_json(404, {"error": "image not found"})
                return
            screenshot = Path(str(item.get("review_context", {}).get("screenshot", "")))
            root = self.server.settings["image_root"]
            try:
                resolved = screenshot.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                self.send_json(404, {"error": "image not found"})
                return
            content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            self.send_bytes(200, resolved.read_bytes(), content_type)
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
                assisted = bool(value.get("assisted", False))
                if self.server.settings['blind'] and assisted:
                    raise ValueError('blind review cannot accept machine-assisted labels')
                valid_ids = {str(item.get("id", "")) for item in self.pool()}
                if identifier not in valid_ids or label not in {"ad", "non_ad", "insufficient_evidence"}:
                    raise ValueError("invalid id or label")
                prelabel = self.prelabels().get(identifier, {})
                confirmed = assisted and prelabel.get("suggested_label") == label
                append_jsonl(self.server.settings["annotations"], {
                    "id": identifier,
                    "label": label,
                    "annotation_mode": "human_confirmed_machine_suggestion" if confirmed else "human_independent",
                    "machine_source": prelabel.get("source", "") if confirmed else "",
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                })
                self.send_json(200, self.state())
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            return
        if path == "/api/freeze":
            try:
                result = freeze_pool(
                    self.server.settings["pool"],
                    self.server.settings["annotations"],
                    self.server.settings["output"],
                    self.server.settings["metadata"],
                    required_labels={"ad", "non_ad"} if self.server.settings.get('ground_truth_task') == 'binary_ad_detection' else None,
                    ground_truth_task=self.server.settings.get('ground_truth_task', 'three_class'),
                    allowed_annotation_modes={"human_independent"} if self.server.settings.get('independent_only') else None,
                )
                self.send_json(200, result)
            except ValueError as exc:
                self.send_json(409, {"error": str(exc)})
            return
        self.send_json(404, {"error": "not found"})

    def pool(self) -> list[dict[str, Any]]:
        return load_jsonl(self.server.settings["pool"])

    def prelabels(self) -> dict[str, dict[str, Any]]:
        if self.server.settings["blind"]:
            return {}
        return {str(value.get("id", "")): value for value in load_jsonl(self.server.settings["prelabels"])}

    def state(self) -> dict[str, Any]:
        pool = self.pool()
        annotations = latest_annotations(self.server.settings["annotations"])
        return {
            "pool": pool,
            "annotations": annotations,
            "prelabels": self.prelabels(),
            "status": pool_status(pool, annotations),
            "blind": self.server.settings["blind"],
        }

    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["serve", "status", "freeze"], nargs="?", default="serve")
    default_pool = DEFAULT_RUN / "pool_curated.jsonl"
    parser.add_argument("--pool", type=Path, default=default_pool if default_pool.exists() else DEFAULT_RUN / "pool.jsonl")
    parser.add_argument("--prelabels", type=Path, default=DEFAULT_RUN / "prelabels.jsonl")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_RUN / "annotations.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "independent_test.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "evaluation" / "independent_test.metadata.json")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--ground-truth-task", choices=["three_class", "binary_ad_detection"], default="three_class")
    parser.add_argument("--independent-only", action="store_true")
    parser.add_argument("--blind", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = load_jsonl(args.pool)
    annotations = latest_annotations(args.annotations)
    if args.action == "status":
        print(json.dumps(pool_status(pool, annotations), ensure_ascii=False, indent=2))
        return 0
    if args.action == "freeze":
        try:
            required_labels = {"ad", "non_ad"} if args.ground_truth_task == "binary_ad_detection" else None
            result = freeze_pool(
                args.pool,
                args.annotations,
                args.output,
                args.metadata,
                required_labels=required_labels,
                ground_truth_task=args.ground_truth_task,
                allowed_annotation_modes={"human_independent"} if args.independent_only else None,
            )
        except ValueError as exc:
            print(json.dumps({"frozen": False, "error": str(exc), "status": pool_status(pool, annotations)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps({"frozen": True, **result}, ensure_ascii=False, indent=2))
        return 0
    token = secrets.token_urlsafe(24)
    settings = {
        "token": token,
        "pool": args.pool.resolve(),
        "prelabels": args.prelabels.resolve(),
        "annotations": args.annotations.resolve(),
        "output": args.output.resolve(),
        "metadata": args.metadata.resolve(),
        "image_root": (args.pool.resolve().parent / "screenshots").resolve(),
        "blind": args.blind,
        "ground_truth_task": args.ground_truth_task,
        "independent_only": args.independent_only,
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
