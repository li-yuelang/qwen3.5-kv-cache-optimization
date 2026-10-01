"""
Phase 5：Chatbox Web UI（零三方依赖，仅用标准库 http.server）

把 Phase 5 的 `ContextManager` 接到真实的 `TinyQwenEngine` 上，跑一个可在浏览器里
多轮对话的 chatbox。这个文件**已全部实现**，不是作业的一部分——它的作用是让你
直观看到自己写的 context 管理在真实对话里起了什么作用：

  - 右侧面板实时显示：当前 prompt 的 token 数 / budget、已压缩次数、滚动摘要正文、
    （若开启 Prefix Cache）本轮命中省下的 token 数；
  - 当历史超 budget 触发摘要压缩时，对应消息会标注「已压缩」；
  - 左侧可以新建 / 切换会话，所有会话通过 SessionStore 持久化到磁盘，重启不丢。

运行
----
    uv sync && source .venv/bin/activate
    python chat_app.py                       # 默认 http://127.0.0.1:8000
    python chat_app.py --port 8080
    python chat_app.py --prefix-cache        # 开启 server 端 Prefix Cache（需已合并 Phase 3/4 实现）
    python chat_app.py --max-context 512 --reserve 128 --keep-recent 2

> 这是教学用的最简单实现：单进程、阻塞式生成、非流式。生产环境请用 FastAPI/vLLM 等。
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tiny_inference import ContextManager, GenerationConfig, SessionStore, TinyQwenEngine

# 这些在 main() 里初始化后供 handler 使用（教学用单进程，全局状态足够）。
ENGINE: TinyQwenEngine | None = None
STORE: SessionStore | None = None
SESSIONS: dict[str, ContextManager] = {}
CFG = {
    "system_prompt": "You are a concise, helpful assistant.",
    "max_context": 1024,
    "reserve": 256,
    "keep_recent": 3,
    "max_new_tokens": 256,
}


def _new_manager() -> ContextManager:
    assert ENGINE is not None
    return ContextManager(
        ENGINE.tokenizer,
        system_prompt=CFG["system_prompt"],
        max_context_tokens=CFG["max_context"],
        reserve_for_reply=CFG["reserve"],
        keep_recent_turns=CFG["keep_recent"],
    )


def _get_manager(session_id: str) -> ContextManager:
    assert STORE is not None
    if session_id not in SESSIONS:
        if STORE.exists(session_id):
            SESSIONS[session_id] = STORE.load(session_id, ENGINE.tokenizer)
        else:
            SESSIONS[session_id] = _new_manager()
    return SESSIONS[session_id]


def _summarize_fn(prompt: str) -> str:
    assert ENGINE is not None
    out = ENGINE.generate(
        messages=[{"role": "user", "content": prompt}],
        gen_config=GenerationConfig(max_new_tokens=120, do_sample=False),
    )
    return out["text"]


def _handle_chat(payload: dict) -> dict:
    assert ENGINE is not None and STORE is not None
    session_id = payload.get("session") or "default"
    message = (payload.get("message") or "").strip()
    cm = _get_manager(session_id)

    cm.add_user(message)
    compressed = cm.maybe_compress(_summarize_fn)
    messages = cm.build_messages()
    result = ENGINE.generate(
        messages=messages,
        gen_config=GenerationConfig(max_new_tokens=CFG["max_new_tokens"], do_sample=False),
        benchmark=True,
    )
    reply = result["text"]
    cm.add_assistant(reply)
    STORE.save(session_id, cm)

    metrics = result.get("metrics", {})
    return {
        "reply": reply,
        "compressed": compressed,
        "summary": cm.summary,
        "num_turns": len(cm.turns),
        "num_compressions": cm.num_compressions,
        "turns_folded": cm.turns_folded,
        "prompt_tokens": cm.count_tokens(cm.build_messages()),
        "budget": cm.token_budget(),
        "max_context": cm.max_context_tokens,
        "prefix_hit_tokens": int(metrics.get("prefix_hit_tokens", 0)),
        "decode_tokens_per_s": round(metrics.get("decode_tokens_per_s", 0), 1),
    }


def _history(cm: ContextManager) -> list[dict]:
    out = []
    for t in cm.turns:
        out.append({"role": "user", "content": t.user})
        if t.assistant:
            out.append({"role": "assistant", "content": t.assistant})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 安静一点
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/sessions":
            assert STORE is not None
            self._send_json({"sessions": STORE.list_sessions()})
        elif self.path.startswith("/api/history"):
            sid = self.path.split("session=", 1)[1] if "session=" in self.path else "default"
            cm = _get_manager(sid)
            self._send_json({"history": _history(cm), "summary": cm.summary})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "bad json"}, 400)
            return

        try:
            if self.path == "/api/chat":
                self._send_json(_handle_chat(payload))
            elif self.path == "/api/new":
                sid = payload.get("session") or "default"
                SESSIONS[sid] = _new_manager()
                STORE.save(sid, SESSIONS[sid])
                self._send_json({"ok": True, "session": sid})
            else:
                self._send_json({"error": "not found"}, 404)
        except NotImplementedError as e:
            self._send_json({"error": f"尚未实现：{e}（请先完成 Phase 5 的 TODO）"}, 501)
        except Exception as e:  # noqa: BLE001 教学用，直接把错误回显到前端
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Phase 5 Chatbox</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "PingFang SC", sans-serif; background:#0f1115; color:#e6e6e6; height:100vh; display:flex; }
  #sidebar { width:220px; background:#161922; padding:12px; display:flex; flex-direction:column; gap:8px; border-right:1px solid #262b36; }
  #sidebar h2 { font-size:13px; color:#8b93a7; margin:4px 0; text-transform:uppercase; letter-spacing:.05em; }
  .sess { padding:6px 8px; border-radius:6px; cursor:pointer; font-size:14px; }
  .sess:hover { background:#222736; }
  .sess.active { background:#2d3650; }
  button { background:#3b82f6; color:#fff; border:none; border-radius:6px; padding:8px; cursor:pointer; font-size:14px; }
  button:hover { background:#2f6fd6; }
  #main { flex:1; display:flex; flex-direction:column; }
  #chat { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:72%; padding:10px 14px; border-radius:12px; white-space:pre-wrap; line-height:1.4; }
  .user { align-self:flex-end; background:#3b82f6; color:#fff; }
  .assistant { align-self:flex-start; background:#222736; }
  .tag { font-size:11px; color:#f0a020; margin-left:6px; }
  #inputbar { display:flex; gap:8px; padding:14px; border-top:1px solid #262b36; }
  #inputbar input { flex:1; background:#1b1f2a; border:1px solid #303644; color:#e6e6e6; border-radius:8px; padding:10px; font-size:15px; }
  #panel { width:280px; background:#161922; padding:14px; border-left:1px solid #262b36; font-size:13px; overflow-y:auto; }
  #panel h2 { font-size:13px; color:#8b93a7; text-transform:uppercase; letter-spacing:.05em; }
  .stat { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px dashed #2a2f3c; }
  .bar { height:8px; background:#222736; border-radius:4px; overflow:hidden; margin:6px 0; }
  .bar > div { height:100%; background:#3b82f6; }
  #summary { background:#1b1f2a; border-radius:8px; padding:8px; white-space:pre-wrap; color:#cfd6e4; max-height:240px; overflow-y:auto; }
</style>
</head>
<body>
  <div id="sidebar">
    <h2>会话</h2>
    <div id="sessions"></div>
    <button onclick="newSession()">+ 新建会话</button>
  </div>
  <div id="main">
    <div id="chat"></div>
    <div id="inputbar">
      <input id="box" placeholder="输入消息，回车发送…" onkeydown="if(event.key==='Enter')send()"/>
      <button onclick="send()" id="sendbtn">发送</button>
    </div>
  </div>
  <div id="panel">
    <h2>Context 状态</h2>
    <div class="stat"><span>Prompt tokens</span><span id="ptok">-</span></div>
    <div class="bar"><div id="pbar" style="width:0%"></div></div>
    <div class="stat"><span>budget / 窗口</span><span id="budget">-</span></div>
    <div class="stat"><span>保留轮数</span><span id="turns">-</span></div>
    <div class="stat"><span>压缩次数</span><span id="comps">-</span></div>
    <div class="stat"><span>已折叠轮数</span><span id="folded">-</span></div>
    <div class="stat"><span>Prefix 命中 token</span><span id="hit">-</span></div>
    <div class="stat"><span>decode tok/s</span><span id="speed">-</span></div>
    <h2 style="margin-top:16px">滚动摘要</h2>
    <div id="summary">（暂无）</div>
  </div>
<script>
let current = "default";

async function refreshSessions() {
  const r = await fetch("/api/sessions"); const d = await r.json();
  const el = document.getElementById("sessions"); el.innerHTML = "";
  let list = d.sessions.length ? d.sessions : ["default"];
  for (const s of list) {
    const div = document.createElement("div");
    div.className = "sess" + (s === current ? " active" : "");
    div.textContent = s; div.onclick = () => switchSession(s);
    el.appendChild(div);
  }
}
async function switchSession(s) {
  current = s; await refreshSessions();
  const r = await fetch("/api/history?session=" + encodeURIComponent(s)); const d = await r.json();
  const chat = document.getElementById("chat"); chat.innerHTML = "";
  for (const m of d.history) addMsg(m.role, m.content);
  document.getElementById("summary").textContent = d.summary || "（暂无）";
}
async function newSession() {
  const name = prompt("新会话名称：", "session-" + Date.now()); if (!name) return;
  await fetch("/api/new", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({session:name})});
  current = name; await refreshSessions(); await switchSession(name);
}
function addMsg(role, content, tag) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div"); div.className = "msg " + role;
  div.textContent = content;
  if (tag) { const s = document.createElement("span"); s.className="tag"; s.textContent=tag; div.appendChild(s); }
  chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
}
function setStats(d) {
  document.getElementById("ptok").textContent = d.prompt_tokens;
  document.getElementById("budget").textContent = d.budget + " / " + d.max_context;
  document.getElementById("turns").textContent = d.num_turns;
  document.getElementById("comps").textContent = d.num_compressions;
  document.getElementById("folded").textContent = d.turns_folded;
  document.getElementById("hit").textContent = d.prefix_hit_tokens;
  document.getElementById("speed").textContent = d.decode_tokens_per_s;
  const pct = Math.min(100, Math.round(100 * d.prompt_tokens / d.max_context));
  document.getElementById("pbar").style.width = pct + "%";
  document.getElementById("summary").textContent = d.summary || "（暂无）";
}
async function send() {
  const box = document.getElementById("box"); const text = box.value.trim(); if (!text) return;
  box.value = ""; addMsg("user", text);
  const btn = document.getElementById("sendbtn"); btn.disabled = true; btn.textContent = "生成中…";
  try {
    const r = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({session: current, message: text})});
    const d = await r.json();
    if (d.error) { addMsg("assistant", "⚠️ " + d.error); }
    else { addMsg("assistant", d.reply, d.compressed ? "已压缩历史" : ""); setStats(d); }
  } catch (e) { addMsg("assistant", "⚠️ " + e); }
  finally { btn.disabled = false; btn.textContent = "发送"; }
}
refreshSessions(); switchSession("default");
</script>
</body>
</html>"""


def main() -> None:
    global ENGINE, STORE
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    parser = argparse.ArgumentParser(description="Phase 5 chatbox web UI")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--reserve", type=int, default=256)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--system", default="You are a concise, helpful assistant.")
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="Enable server-side Prefix Cache (Phase 3/4 implementation required, otherwise will error on the first request).",
    )
    parser.add_argument(
        "--ssd-cache-dir",
        default=None,
        help="Directory for SSD-backed prefix cache; requires --prefix-cache.",
    )
    parser.add_argument("--prefix-cache-mem-entries", type=int, default=4)
    parser.add_argument("--sessions-dir", default="./sessions")
    args = parser.parse_args()
    if args.ssd_cache_dir and not args.prefix_cache:
        parser.error("--ssd-cache-dir requires --prefix-cache")

    CFG.update(
        system_prompt=args.system,
        max_context=args.max_context,
        reserve=args.reserve,
        keep_recent=args.keep_recent,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Loading engine {args.model} ...（First time will download the weights）")
    ENGINE = TinyQwenEngine(
        args.model,
        enable_prefix_cache=args.prefix_cache,
        prefix_cache_ssd_dir=args.ssd_cache_dir,
        prefix_cache_mem_entries=args.prefix_cache_mem_entries,
    )
    STORE = SessionStore(args.sessions_dir)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Chatbox is ready → http://{args.host}:{args.port}")
    print("Press Ctrl+C to exit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        server.shutdown()


if __name__ == "__main__":
    main()
