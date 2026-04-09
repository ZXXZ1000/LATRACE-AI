"""
LATRACE Chat-with-Memory Demo
==============================
An end-to-end demo that shows LATRACE's three core memory capabilities:

  1. Memory Ingestion  — POST /ingest/dialog/v1
  2. Memory Retrieval  — POST /retrieval/dialog/v2  (recommended)
  3. TKG Graph View    — D3.js force-directed graph at /graph

Prerequisites
-------------
* LATRACE Memory API running at http://localhost:8000
  (quickstart: docker compose up)
* An OpenAI-compatible LLM API key  (set LLM_API_KEY env var)
* Neo4j running at bolt://localhost:7687  (for /graph view)

Usage
-----
  export LLM_API_KEY=<your-key>
  export LLM_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compat endpoint
  export LLM_MODEL=gpt-4o-mini                    # optional, default gpt-4o-mini
  uv run python examples/chat_with_memory/chat_with_memory.py

Then open:  http://localhost:7860        (chat)
            http://localhost:7860/graph  (memory graph)
"""
from __future__ import annotations

import base64 as _base64
import json
import os
import time
import uuid
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# ── Configuration — all values can be overridden via environment variables ──────
MEMORY_API    = os.getenv("MEMORY_API_URL",  "http://localhost:8000")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL",    "https://api.openai.com/v1")
LLM_API_KEY   = os.getenv("LLM_API_KEY",     "")          # required: set your key
LLM_MODEL     = os.getenv("LLM_MODEL",       "gpt-4o-mini")
TENANT_ID     = os.getenv("DEMO_TENANT_ID",  "chat-demo")
USER_ID       = os.getenv("DEMO_USER_ID",    "demo-user-001")
MEMORY_DOMAIN = "dialog"

MEM_HEADERS   = {"X-Tenant-ID": TENANT_ID, "Content-Type": "application/json"}

# Neo4j HTTP API (used by /graph TKG visualisation)
NEO4J_HTTP_URL = os.getenv("NEO4J_HTTP_URL",  "http://localhost:7474/db/neo4j/tx/commit")
NEO4J_USER     = os.getenv("NEO4J_USER",      "neo4j")
NEO4J_PASS     = os.getenv("NEO4J_PASSWORD",  "neo4j_password")

# ── FastAPI 应用 ──────────────────────────────────────────────────────────────
app = FastAPI()

_conversation: List[Dict[str, str]] = []   # 本地多轮上下文
_pending_turns: List[Dict[str, Any]] = []  # 待写入 LATRACE 的 turns
_flush_count = 0                           # 每批 session_id 计数器
_turn_seq = 0                              # turn_id 序号


def _now_iso() -> str:
    """返回带时区的 ISO8601 时间戳（文档 §9.3 强烈建议）。"""
    return datetime.now(timezone.utc).isoformat()


def _make_turn(role: str, text: str) -> Dict[str, Any]:
    """按文档 §9.3 Turn Schema 构造 turn，包含 turn_id 和 timestamp_iso。"""
    global _turn_seq
    _turn_seq += 1
    return {
        "turn_id": f"t{_turn_seq:06d}",          # 同 session 内唯一可排序
        "role": role,                              # user / assistant
        "text": text,                              # 原话，不改写（§9.4）
        "timestamp_iso": _now_iso(),               # 强烈建议（§9.3）
    }


# ── 记忆检索 ──────────────────────────────────────────────────────────────────

def retrieve_memories(query: str, topk: int = 10) -> Tuple[str, List[Dict]]:
    """
    使用 POST /retrieval/dialog/v2（文档 §5.5 推荐入口）检索记忆。
    返回 (格式化文本, evidence_details列表)。
    """
    try:
        r = httpx.post(
            f"{MEMORY_API}/retrieval/dialog/v2",
            headers=MEM_HEADERS,
            json={
                "query": query,
                "user_tokens": [USER_ID],      # 用户隔离（§4.1）
                "memory_domain": MEMORY_DOMAIN,
                "topk": topk,
                "backend": "tkg",              # TKG 后端（§5.5）
                "tkg_explain": False,          # 不需要图解释链，减少延迟
                "llm_policy": "best_effort",   # 检索不强制依赖 LLM
                "client_meta": {
                    "user_id": USER_ID,
                    "memory_policy": "user",
                    "llm_mode": "platform",
                },
            },
            timeout=15.0,
        )
        if r.status_code != 200:
            return "", []
        data = r.json()
        evidence = data.get("evidence_details", [])
        if not evidence:
            return "", []
        lines = []
        for ev in evidence[:10]:
            text = (ev.get("text") or "").strip()
            score = ev.get("score", 0)
            if not text or score < 0.05:
                continue
            lines.append(f"- {text}")
        return "\n".join(lines), evidence
    except Exception:
        return "", []


def search_all_memories(query: str, topk: int = 20) -> List[Dict]:
    """
    用 POST /search（§5.6）展示记忆面板全量内容，
    包含 fact / Entity / event 各类节点。
    """
    try:
        r = httpx.post(
            f"{MEMORY_API}/search",
            headers=MEM_HEADERS,
            json={
                "query": query,
                "topk": topk,
                "expand_graph": False,
                "filters": {
                    "user_id": [USER_ID],
                    "memory_domain": MEMORY_DOMAIN,
                },
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return []
        hits = r.json().get("hits", [])
        results = []
        for h in hits:
            score = h.get("score", 0)
            entry = h.get("entry", {})
            meta = entry.get("metadata") or {}
            node_type = meta.get("node_type", "")
            contents = entry.get("contents", [])
            text = " ".join(str(c) for c in contents[:2]).strip()
            if not text or node_type == "session_marker":
                continue
            results.append({
                "score": score,
                "node_type": node_type,
                "text": text,
            })
        return results
    except Exception:
        return []


# ── 记忆写入 ──────────────────────────────────────────────────────────────────

def flush_turns_to_memory(turns: List[Dict[str, Any]]) -> str:
    """
    按文档 §5.2 提交对话 turns 到 /ingest/dialog/v1。
    每批使用独立 session_id + commit_id（§5.2 幂等说明），
    避免同 session 二次提交被全量去重丢弃。
    返回 job_id 字符串（失败返回空串）。
    """
    global _flush_count
    if not turns:
        return ""
    _flush_count += 1
    # 独立 session_id 避免去重（同 session_id 二次提交 accepted=0）
    batch_session = f"chat-{USER_ID}-b{_flush_count:04d}-{uuid.uuid4().hex[:6]}"
    commit_id     = f"commit-{batch_session}"    # 幂等键（§5.2 强烈建议）
    try:
        r = httpx.post(
            f"{MEMORY_API}/ingest/dialog/v1",
            headers=MEM_HEADERS,
            json={
                "session_id":    batch_session,
                "commit_id":     commit_id,       # 幂等（§5.2）
                "user_tokens":   [USER_ID],        # 用户身份（§5.2）
                "memory_domain": MEMORY_DOMAIN,
                "turns":         turns,            # 含 turn_id + timestamp_iso
                "llm_policy":    "require",        # 强制 LLM 提取（§5.2）
                "client_meta": {
                    "user_id":        USER_ID,
                    "memory_policy":  "user",
                    "stage2_enabled": True,
                    "stage3_extract": True,
                    "llm_mode":       "platform",
                },
            },
            timeout=5.0,
        )
        if r.status_code in (200, 202):
            return r.json().get("job_id", "")
        return ""
    except Exception:
        return ""


# ── LLM 流式调用 ──────────────────────────────────────────────────────────────

def call_llm_stream(messages: List[Dict[str, str]]):
    """流式调用 OpenAI 兼容 API，逐 token yield。"""
    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":      LLM_MODEL,
                "messages":   messages,
                "stream":     True,
                "max_tokens": 1024,
            },
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


# ── API 端点 ──────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_msg: str = body.get("message", "").strip()
    if not user_msg:
        return {"error": "empty message"}

    # 1. 用 /retrieval/dialog/v2 检索相关历史记忆（§5.5）
    mem_text, evidence = retrieve_memories(user_msg)

    # 2. 构建系统提示（只在有记忆时注入，不预设任何内容）
    system_parts = [
        "你是一个聪明、友好的 AI 助手，具有长期记忆能力。",
        "你能记住用户说过的事情，并在合适时自然地引用。",
        "回答要简洁自然，不要刻意强调「我记得」，直接体现出你知道就好。",
        "使用中文回答。",
    ]
    if mem_text:
        system_parts.append(f"\n【从记忆库检索到的相关信息】\n{mem_text}")
    system_prompt = "\n".join(system_parts)

    # 3. 构建消息列表（最近 10 轮本地上下文）
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_conversation[-10:])
    messages.append({"role": "user", "content": user_msg})

    # 4. 流式调用 LLM
    full_reply: List[str] = []

    def generate():
        for token in call_llm_stream(messages):
            full_reply.append(token)
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        reply_text = "".join(full_reply)

        # 5. 更新本地对话历史
        _conversation.append({"role": "user",      "content": user_msg})
        _conversation.append({"role": "assistant",  "content": reply_text})

        # 6. 按 §9.3 Turn Schema 构造 turns（含 turn_id + timestamp_iso）
        _pending_turns.append(_make_turn("user",      user_msg))
        _pending_turns.append(_make_turn("assistant", reply_text))

        # 7. 每 2 轮（4 条 turn）自动写入记忆
        save_hint = ""
        if len(_pending_turns) >= 4:
            job_id = flush_turns_to_memory(list(_pending_turns))
            _pending_turns.clear()
            if job_id:
                save_hint = f" · 已提交记忆提取 job={job_id[:10]}…"

        # 8. 发送完成信号（含记忆状态）
        ev_count = len([e for e in evidence if e.get("score", 0) >= 0.05])
        if ev_count:
            mem_info = f"检索到 {ev_count} 条相关记忆{save_hint}"
        else:
            mem_info = f"暂无匹配记忆（对话写入后约 20s 可检索）{save_hint}"
        yield f"data: {json.dumps({'done': True, 'mem_info': mem_info}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/save_memory")
async def save_memory_now():
    """手动立即保存当前未提交的 turns。"""
    if not _pending_turns:
        return {"ok": False, "message": "没有待写入内容（已自动提交）"}
    job_id = flush_turns_to_memory(list(_pending_turns))
    _pending_turns.clear()
    if job_id:
        return {"ok": True,  "message": f"已提交 job={job_id[:12]}…，约 20 秒后可检索"}
    return {"ok": False, "message": "提交失败，请检查 Memory API 服务"}


@app.get("/api/memories")
async def list_memories(q: str = "用户 信息 爱好 工作"):
    """
    记忆面板：用 /search 展示当前用户的全部记忆条目（§5.6）。
    同时也用 /retrieval/dialog/v2 获取语义证据。
    """
    hits = search_all_memories(q, topk=20)
    _, evidence = retrieve_memories(q, topk=15)
    return {
        "hits":     hits,
        "evidence": evidence,
    }


@app.get("/api/status")
async def status():
    """检查 Memory API 健康状态。"""
    try:
        r = httpx.get(f"{MEMORY_API}/health", headers=MEM_HEADERS, timeout=5.0)
        deps = r.json().get("dependencies", {})
        vectors_ok = deps.get("vectors", {}).get("status") == "ok"
        graph_ok   = deps.get("graph",   {}).get("status") == "ok"
        has_memory = len(search_all_memories("用户", topk=1)) > 0
        return {
            "vectors": vectors_ok,
            "graph":   graph_ok,
            "has_memory": has_memory,
        }
    except Exception:
        return {"vectors": False, "graph": False, "has_memory": False}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# ── 前端页面 ──────────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LATRACE 记忆对话</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f0f13; color: #e8e8f0; height: 100vh; display: flex; flex-direction: column; }

  .header { background: #1a1a24; border-bottom: 1px solid #2a2a3a;
            padding: 12px 20px; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  .header-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #6c63ff, #3ec6e0);
                 border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
  .header-title h1 { font-size: 15px; font-weight: 600; }
  .header-title p  { font-size: 11px; color: #7777a0; margin-top: 2px; }
  .header-actions  { margin-left: auto; display: flex; gap: 8px; align-items: center; }

  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #333355;
                display: inline-block; margin-right: 4px; transition: background .4s; }
  .status-dot.ok  { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
  .status-dot.err { background: #ef4444; }

  .btn { font-size: 11px; padding: 5px 12px; border-radius: 20px; cursor: pointer;
         border: 1px solid #333350; background: #1e1e2c; color: #9999cc;
         transition: all .2s; white-space: nowrap; }
  .btn:hover { background: #2a2a3e; border-color: #5555aa; color: #ccccee; }
  .btn.primary { background: #252545; border-color: #4444aa; color: #aaaaee; }

  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px;
              scroll-behavior: smooth; }
  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-thumb { background: #333345; border-radius: 4px; }

  .welcome { background: linear-gradient(135deg, #1a1a28, #202030);
             border: 1px solid #2a2a3a; border-radius: 16px; padding: 28px;
             text-align: center; margin: auto; max-width: 500px; }
  .welcome .icon { font-size: 44px; margin-bottom: 14px; }
  .welcome h2 { font-size: 17px; margin-bottom: 8px; color: #c8c8e8; }
  .welcome p  { font-size: 13px; color: #6666a0; line-height: 1.8; }
  .welcome .info-grid { margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .info-card { background: #18182a; border: 1px solid #2a2a3a; border-radius: 10px;
               padding: 12px; text-align: left; }
  .info-card .ic-icon { font-size: 18px; margin-bottom: 6px; }
  .info-card .ic-title { font-size: 11px; font-weight: 600; color: #8888cc; margin-bottom: 3px; }
  .info-card .ic-desc  { font-size: 11px; color: #5555a0; line-height: 1.6; }

  .msg { display: flex; gap: 10px; max-width: 780px; animation: fadeUp .2s ease; }
  .msg.user      { flex-direction: row-reverse; margin-left: auto; }
  .msg.assistant { margin-right: auto; }
  @keyframes fadeUp { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }

  .avatar { width: 32px; height: 32px; border-radius: 10px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center; font-size: 16px; }
  .avatar.user      { background: linear-gradient(135deg, #6c63ff, #a855f7); }
  .avatar.assistant { background: linear-gradient(135deg, #0ea5e9, #22d3ee); }

  .msg-body { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
  .bubble { max-width: 620px; padding: 12px 16px; border-radius: 16px;
            font-size: 14px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
  .msg.user .bubble      { background: #25184a; border: 1px solid #3a2a7a; border-radius: 16px 4px 16px 16px; }
  .msg.assistant .bubble { background: #141e2c; border: 1px solid #1e2e3e; border-radius: 4px 16px 16px 16px; }

  .mem-tag { font-size: 10px; color: #4a5a7a; display: flex; align-items: center; gap: 4px;
             padding: 0 4px; }
  .mem-tag.has-mem { color: #5577aa; }

  .thinking { display: flex; gap: 5px; padding: 4px 0; }
  .thinking span { width: 7px; height: 7px; background: #3366aa; border-radius: 50%;
                   animation: bounce .8s ease infinite; }
  .thinking span:nth-child(2) { animation-delay: .15s; }
  .thinking span:nth-child(3) { animation-delay: .30s; }
  @keyframes bounce { 0%,80%,100%{transform:scale(.6)} 40%{transform:scale(1)} }

  .input-area { padding: 14px 20px; background: #0f0f18; border-top: 1px solid #1e1e2e; flex-shrink: 0; }
  .input-wrap { display: flex; gap: 10px; background: #181824; border: 1px solid #2a2a3e;
                border-radius: 14px; padding: 10px 14px; max-width: 880px; margin: 0 auto;
                transition: border-color .2s; }
  .input-wrap:focus-within { border-color: #4444aa; }
  #userInput { flex: 1; background: none; border: none; color: #e0e0f0; font-size: 14px;
               resize: none; outline: none; max-height: 120px; line-height: 1.6; font-family: inherit; }
  #userInput::placeholder { color: #444466; }
  #sendBtn { background: linear-gradient(135deg, #4a42cc, #22aacc); border: none;
             border-radius: 10px; width: 36px; height: 36px; cursor: pointer; flex-shrink: 0;
             display: flex; align-items: center; justify-content: center; align-self: flex-end;
             margin-bottom: 2px; transition: opacity .2s; }
  #sendBtn:disabled { opacity: .35; cursor: not-allowed; }
  #sendBtn svg { width: 16px; height: 16px; fill: white; }
  .input-tip { text-align: center; font-size: 11px; color: #33334a; margin-top: 8px; }

  /* 记忆侧边栏 */
  .mem-panel { position: fixed; right: 0; top: 0; bottom: 0; width: 340px;
               background: #13131c; border-left: 1px solid #1e1e2e;
               transform: translateX(100%); transition: transform .3s ease;
               z-index: 100; display: flex; flex-direction: column; }
  .mem-panel.open { transform: translateX(0); box-shadow: -8px 0 32px #00000066; }
  .mem-header { padding: 14px 16px; border-bottom: 1px solid #1e1e2e;
                display: flex; align-items: center; justify-content: space-between; }
  .mem-header h3 { font-size: 13px; color: #8888cc; font-weight: 600; }
  .close-btn { cursor: pointer; color: #444466; font-size: 20px; line-height: 1; padding: 2px 6px; }
  .close-btn:hover { color: #aaaacc; }

  .mem-tabs { display: flex; border-bottom: 1px solid #1e1e2e; }
  .mem-tab { flex: 1; padding: 8px; font-size: 11px; color: #555580; cursor: pointer;
             text-align: center; border-bottom: 2px solid transparent; transition: all .2s; }
  .mem-tab.active { color: #8888cc; border-bottom-color: #5555aa; background: #18182a; }
  .mem-tab:hover:not(.active) { background: #16161e; color: #7777aa; }

  .mem-content { flex: 1; overflow-y: auto; padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
  .mem-content::-webkit-scrollbar { width: 3px; }
  .mem-content::-webkit-scrollbar-thumb { background: #222235; border-radius: 3px; }

  .mem-item { background: #18182a; border: 1px solid #222235; border-radius: 10px;
              padding: 10px 12px; }
  .mem-item-type { font-size: 9px; background: #1e1e38; color: #6666aa; padding: 2px 6px;
                   border-radius: 4px; display: inline-block; margin-bottom: 5px; }
  .mem-item-text { font-size: 12px; color: #8888bb; line-height: 1.6; }
  .mem-item-score { font-size: 10px; color: #333360; margin-top: 4px; }
  .mem-empty { color: #333355; font-size: 12px; text-align: center; padding: 40px 20px; line-height: 1.8; }

  .mem-footer { padding: 10px 14px; border-top: 1px solid #1e1e2e; }
  .mem-footer button { width: 100%; padding: 8px; background: #18182a; border: 1px solid #222235;
                        border-radius: 8px; color: #666699; cursor: pointer; font-size: 12px;
                        transition: all .2s; }
  .mem-footer button:hover { background: #1e1e38; color: #8888cc; }
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">🧠</div>
  <div class="header-title">
    <h1>LATRACE 记忆对话</h1>
    <p id="statusText"><span class="status-dot" id="statusDot"></span>检测中...</p>
  </div>
  <div class="header-actions">
    <div class="btn" onclick="window.open('/graph','_blank')">🕸️ 图谱</div>
    <div class="btn" onclick="openMemPanel()">📚 记忆库</div>
    <div class="btn primary" onclick="saveMemory()" id="saveBtn">💾 保存</div>
  </div>
</div>

<div class="messages" id="messages">
  <div class="welcome" id="welcome">
    <div class="icon">🧠</div>
    <h2>带记忆的 AI 对话</h2>
    <p>像和真人交谈一样，我会记住你说过的每件事。<br>关闭再打开，依然记得你。</p>
    <div class="info-grid">
      <div class="info-card">
        <div class="ic-icon">🔍</div>
        <div class="ic-title">每次对话前</div>
        <div class="ic-desc">自动从记忆库检索相关信息，注入对话上下文</div>
      </div>
      <div class="info-card">
        <div class="ic-icon">✍️</div>
        <div class="ic-title">每 2 轮自动保存</div>
        <div class="ic-desc">LLM 从对话中提取关键信息写入向量库和图谱</div>
      </div>
      <div class="info-card">
        <div class="ic-icon">💾</div>
        <div class="ic-title">手动保存</div>
        <div class="ic-desc">点击「保存」立即提交当前对话，约 20 秒后可检索</div>
      </div>
      <div class="info-card">
        <div class="ic-icon">📚</div>
        <div class="ic-title">查看记忆库</div>
        <div class="ic-desc">点击「记忆库」查看所有已提取的事实和实体</div>
      </div>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="userInput" placeholder="说点什么..." rows="1"
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button id="sendBtn" onclick="sendMessage()">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
  <div class="input-tip">Enter 发送 · Shift+Enter 换行 · 每 2 轮自动写入记忆</div>
</div>

<div class="mem-panel" id="memPanel">
  <div class="mem-header">
    <h3>🧠 记忆库</h3>
    <span class="close-btn" onclick="closeMemPanel()">×</span>
  </div>
  <div class="mem-tabs">
    <div class="mem-tab active" onclick="switchTab('evidence')" id="tab-evidence">语义证据</div>
    <div class="mem-tab" onclick="switchTab('nodes')" id="tab-nodes">节点列表</div>
  </div>
  <div class="mem-content" id="memContent">
    <div class="mem-empty">点击刷新加载记忆</div>
  </div>
  <div class="mem-footer">
    <button onclick="loadMemories()">🔄 刷新记忆库</button>
  </div>
</div>

<script>
let isStreaming = false;
let currentTab = 'evidence';
let memData = {hits: [], evidence: []};

// 启动时检查服务状态
(async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    const ok = d.vectors && d.graph;
    dot.className = 'status-dot ' + (ok ? 'ok' : 'err');
    if (ok) {
      txt.innerHTML = '<span class="status-dot ok"></span>' +
        (d.has_memory ? '已找到历史记忆 · 继续聊' : '服务就绪 · 开始聊天');
    } else {
      txt.innerHTML = '<span class="status-dot err"></span>服务异常，请检查 Docker';
    }
  } catch {
    document.getElementById('statusText').textContent = '无法连接到 Memory API';
  }
})();

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function scrollBottom() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function removeWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}

function addThinking() {
  removeWelcome();
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg assistant'; d.id = 'thinking';
  d.innerHTML = '<div class="avatar assistant">🤖</div><div class="msg-body"><div class="bubble"><div class="thinking"><span></span><span></span><span></span></div></div></div>';
  msgs.appendChild(d); scrollBottom();
}

function addUserMsg(text) {
  removeWelcome();
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg user';
  d.innerHTML = `<div class="avatar user">👤</div><div class="msg-body"><div class="bubble">${esc(text)}</div></div>`;
  msgs.appendChild(d); scrollBottom();
}

function createAssistantMsg() {
  const msgs = document.getElementById('messages');
  const wrapper = document.createElement('div');
  wrapper.className = 'msg assistant';
  const bubble  = document.createElement('div'); bubble.className = 'bubble';
  const memTag  = document.createElement('div'); memTag.className = 'mem-tag';
  const avatar  = document.createElement('div'); avatar.className = 'avatar assistant'; avatar.textContent = '🤖';
  const body    = document.createElement('div'); body.className = 'msg-body';
  body.appendChild(bubble); body.appendChild(memTag);
  wrapper.appendChild(avatar); wrapper.appendChild(body);
  msgs.appendChild(wrapper);
  return {bubble, memTag};
}

async function sendMessage() {
  if (isStreaming) return;
  const input = document.getElementById('userInput');
  const text  = input.value.trim();
  if (!text) return;

  input.value = ''; input.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;
  isStreaming = true;

  addUserMsg(text);
  addThinking();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });

    document.getElementById('thinking')?.remove();
    const {bubble, memTag} = createAssistantMsg();

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', fullText = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.token) { fullText += d.token; bubble.textContent = fullText; scrollBottom(); }
          if (d.done) {
            const hasMem = d.mem_info && d.mem_info.startsWith('检索到');
            memTag.className = 'mem-tag' + (hasMem ? ' has-mem' : '');
            memTag.textContent = '🧠 ' + (d.mem_info || '');
          }
        } catch {}
      }
    }
  } catch (e) {
    document.getElementById('thinking')?.remove();
    const {bubble} = createAssistantMsg();
    bubble.textContent = '出错了：' + e.message;
  }

  isStreaming = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('userInput').focus();
}

async function saveMemory() {
  const btn = document.getElementById('saveBtn');
  btn.textContent = '提交中...';
  try {
    const r = await fetch('/api/save_memory', {method: 'POST'});
    const d = await r.json();
    btn.textContent = d.ok ? '✅ 已提交' : '⚠️ ' + d.message.slice(0, 10);
  } catch {
    btn.textContent = '❌ 失败';
  }
  setTimeout(() => btn.textContent = '💾 保存', 3000);
}

function openMemPanel()  { document.getElementById('memPanel').classList.add('open'); loadMemories(); }
function closeMemPanel() { document.getElementById('memPanel').classList.remove('open'); }

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-evidence').className = 'mem-tab' + (tab === 'evidence' ? ' active' : '');
  document.getElementById('tab-nodes').className    = 'mem-tab' + (tab === 'nodes'    ? ' active' : '');
  renderMemPanel();
}

function renderMemPanel() {
  const container = document.getElementById('memContent');
  container.innerHTML = '';

  if (currentTab === 'evidence') {
    const ev = memData.evidence || [];
    const filtered = ev.filter(e => e.score >= 0.05 && e.text);
    if (!filtered.length) {
      container.innerHTML = '<div class="mem-empty">暂无语义证据<br><small>对话并保存后，LLM 会从中提取关键信息</small></div>';
      return;
    }
    filtered.forEach(e => {
      const d = document.createElement('div'); d.className = 'mem-item';
      const src = (e.source || 'unknown').replace('event_search_', '').replace('_vec', '');
      d.innerHTML = `<div class="mem-item-type">${esc(src)}</div>
                     <div class="mem-item-text">${esc(e.text)}</div>
                     <div class="mem-item-score">相关度 ${(e.score * 100).toFixed(0)}%</div>`;
      container.appendChild(d);
    });
  } else {
    const hits = memData.hits || [];
    if (!hits.length) {
      container.innerHTML = '<div class="mem-empty">暂无记忆节点<br><small>开始对话后自动积累</small></div>';
      return;
    }
    hits.forEach(h => {
      const d = document.createElement('div'); d.className = 'mem-item';
      d.innerHTML = `<div class="mem-item-type">${esc(h.node_type || 'memory')}</div>
                     <div class="mem-item-text">${esc(h.text)}</div>`;
      container.appendChild(d);
    });
  }
}

async function loadMemories() {
  document.getElementById('memContent').innerHTML = '<div class="mem-empty">加载中...</div>';
  try {
    const r = await fetch('/api/memories?q=用户+信息+爱好+工作+生活');
    memData = await r.json();
    renderMemPanel();
  } catch {
    document.getElementById('memContent').innerHTML = '<div class="mem-empty">加载失败</div>';
  }
}
</script>
</body>
</html>
"""


# ── Neo4j 图谱 API ────────────────────────────────────────────────────────────

def _neo4j_query(cypher: str) -> dict:
    """调用 Neo4j HTTP Transactional API，返回原始 results 结构。"""
    creds = _base64.b64encode(f"{NEO4J_USER}:{NEO4J_PASS}".encode()).decode()
    r = httpx.post(
        NEO4J_HTTP_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/json",
            "Accept":        "application/json;charset=UTF-8",
        },
        json={"statements": [{"statement": cypher, "resultDataContents": ["graph"]}]},
        timeout=10.0,
    )
    return r.json()


@app.get("/api/graph-data")
async def graph_data():
    """
    从 Neo4j 读取 TKG 图数据，返回 D3.js 力导向图所需的 {nodes, links, stats} 格式。
    包含 Event / Entity / Knowledge / UtteranceEvidence / MediaSegment / MemoryNode 节点。
    """
    cypher = """
    MATCH (n)
    WHERE (n:Event OR n:Entity OR n:Knowledge OR n:UtteranceEvidence
           OR n:MediaSegment OR n:MemoryNode)
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE (m:Event OR m:Entity OR m:Knowledge OR m:UtteranceEvidence
           OR m:MediaSegment OR m:MemoryNode)
    RETURN n, r, m
    LIMIT 500
    """
    try:
        result = _neo4j_query(cypher)
        errors = result.get("errors", [])
        if errors:
            return {"error": errors[0].get("message", "neo4j error"), "nodes": [], "links": []}

        nodes_map: Dict[str, Dict] = {}
        links: List[Dict] = []
        seen_rels: set = set()

        for row_data in result.get("results", [{}])[0].get("data", []):
            graph = row_data.get("graph", {})
            for node in graph.get("nodes", []):
                nid = node["id"]
                if nid not in nodes_map:
                    labels = node.get("labels", ["Unknown"])
                    props  = node.get("properties", {})
                    label_type = labels[0] if labels else "Unknown"
                    # 按优先级选取显示文本
                    display = (
                        props.get("summary") or props.get("name") or
                        props.get("content") or props.get("text") or
                        props.get("raw_text") or props.get("id", nid)
                    )
                    nodes_map[nid] = {
                        "id":    nid,
                        "type":  label_type,
                        "label": str(display)[:80],
                        "props": {
                            k: str(v)[:300] for k, v in props.items()
                            if k in {"id", "summary", "name", "content", "text",
                                     "raw_text", "tenant_id", "importance",
                                     "topic_path", "created_at", "memory_domain"}
                        },
                    }
            for rel in graph.get("relationships", []):
                rid = rel["id"]
                if rid not in seen_rels:
                    seen_rels.add(rid)
                    links.append({
                        "id":     rid,
                        "source": rel["startNode"],
                        "target": rel["endNode"],
                        "type":   rel["type"],
                    })

        return {
            "nodes": list(nodes_map.values()),
            "links": links,
            "stats": {
                "node_count": len(nodes_map),
                "link_count": len(links),
            },
        }
    except Exception as e:
        return {"error": str(e), "nodes": [], "links": []}


@app.get("/graph", response_class=HTMLResponse)
async def graph_page():
    return GRAPH_PAGE


# ── 图谱可视化前端页面 ────────────────────────────────────────────────────────
GRAPH_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LATRACE 记忆图谱</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0b0b10; color: #e0e0f0; height: 100vh; overflow: hidden;
       display: flex; flex-direction: column; }

.topbar { background: #13131c; border-bottom: 1px solid #1e1e2e;
          padding: 10px 18px; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.topbar-icon { font-size: 22px; }
.topbar-title { font-size: 14px; font-weight: 700; color: #c0c0e0; }
.topbar-sub   { font-size: 11px; color: #44446a; margin-left: 6px; }
.topbar-right { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.refresh-txt  { font-size: 10px; color: #33335a; }

.btn { font-size: 11px; padding: 5px 13px; border-radius: 20px; cursor: pointer;
       border: 1px solid #2a2a3e; background: #1a1a28; color: #8888bb;
       transition: all .2s; white-space: nowrap; text-decoration: none;
       display: inline-flex; align-items: center; gap: 4px; }
.btn:hover { background: #222236; border-color: #4444aa; color: #ccccee; }
.btn.active { background: #252545; border-color: #5555cc; color: #9999ff; }
.btn-icon { font-size: 13px; }

.main { flex: 1; display: flex; overflow: hidden; }

/* ── Sidebar ── */
.sidebar { width: 200px; background: #10101a; border-right: 1px solid #1a1a28;
           padding: 14px 12px; overflow-y: auto; flex-shrink: 0;
           display: flex; flex-direction: column; gap: 16px; }
.sidebar h4 { font-size: 10px; font-weight: 600; color: #44446a;
              text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }

.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.stat-box { background: #14142a; border: 1px solid #1e1e32; border-radius: 8px;
            padding: 8px 6px; text-align: center; }
.stat-num   { font-size: 20px; font-weight: 700; color: #6677cc; }
.stat-label { font-size: 10px; color: #33335a; margin-top: 2px; }

.filter-item { display: flex; align-items: center; gap: 6px; padding: 5px 4px;
               border-radius: 6px; cursor: pointer; transition: background .15s; }
.filter-item:hover { background: #18182a; }
.filter-dot   { width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }
.filter-label { font-size: 11px; color: #8888aa; flex: 1; }
.filter-count { font-size: 10px; color: #33334a; }
input[type=checkbox] { accent-color: #5566cc; cursor: pointer; width: 13px; height: 13px; }

.rel-row { display: flex; justify-content: space-between; align-items: center;
           padding: 3px 2px; }
.rel-name { font-size: 10px; color: #555580; }
.rel-cnt  { font-size: 10px; color: #33334a; }

/* ── Canvas ── */
.canvas { flex: 1; position: relative; overflow: hidden; }
#graphSvg { width: 100%; height: 100%; }

/* ── Detail Panel ── */
.detail-panel { width: 270px; background: #10101a; border-left: 1px solid #1a1a28;
                overflow-y: auto; flex-shrink: 0; display: none; flex-direction: column; }
.detail-panel.open { display: flex; }
.detail-header { padding: 14px 14px 0; display: flex; align-items: flex-start; justify-content: space-between; }
.detail-badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
                font-size: 11px; font-weight: 600; }
.detail-close { cursor: pointer; color: #333355; font-size: 22px; line-height: 1;
                padding: 0 4px; transition: color .2s; }
.detail-close:hover { color: #aaaacc; }
.detail-text  { font-size: 13px; color: #b0b0d0; line-height: 1.65; padding: 10px 14px; }
.detail-props { padding: 0 14px 14px; display: flex; flex-direction: column; gap: 7px; }
.prop-box  { background: #14142a; border: 1px solid #1e1e32; border-radius: 8px; padding: 8px 10px; }
.prop-key  { font-size: 10px; color: #44446a; text-transform: uppercase; letter-spacing:.04em; margin-bottom: 3px; }
.prop-val  { font-size: 12px; color: #8888bb; line-height: 1.5; word-break: break-all; }

/* ── SVG elements ── */
.link line { stroke-opacity: .5; }
.link-label { font-size: 9px; fill: #333355; pointer-events: none; }
.node circle { stroke-width: 2; cursor: pointer; }
.node circle:hover { filter: brightness(1.25); stroke-width: 3; }
.node.selected circle { stroke: #ffffff !important; stroke-width: 3; }
.node-label { font-size: 10px; fill: #9999cc; pointer-events: none;
              text-shadow: 0 1px 4px #000a; }

/* ── Tooltip ── */
.tooltip { position: absolute; background: #1a1a2c; border: 1px solid #2a2a3e;
           border-radius: 8px; padding: 8px 12px; font-size: 12px; color: #c0c0e0;
           pointer-events: none; max-width: 220px; z-index: 200; display: none;
           line-height: 1.5; box-shadow: 0 4px 16px #00000088; }

/* ── Loading overlay ── */
.loading { position: absolute; inset: 0; display: flex; align-items: center;
           justify-content: center; background: #0b0b1088; z-index: 50; }
.loading-inner { background: #13131c; border: 1px solid #2a2a3a; border-radius: 12px;
                 padding: 24px 36px; text-align: center; }
.spinner { width: 36px; height: 36px; border: 3px solid #222235;
           border-top-color: #5566cc; border-radius: 50%;
           animation: spin .8s linear infinite; margin: 0 auto 12px; }
@keyframes spin { to { transform: rotate(360deg); } }

.pulse { animation: pulse 1.2s ease infinite; }
@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }

.empty-hint { position: absolute; inset: 0; display: flex; align-items: center;
              justify-content: center; pointer-events: none; }
.empty-inner { text-align: center; color: #333355; }
.empty-inner .icon { font-size: 48px; margin-bottom: 12px; }
.empty-inner p { font-size: 13px; }
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-icon">🕸️</span>
  <span class="topbar-title">LATRACE 记忆图谱</span>
  <span class="topbar-sub">TKG · Neo4j 实时数据</span>
  <div class="topbar-right">
    <span class="refresh-txt" id="refreshTxt">-</span>
    <button class="btn" id="autoBtn" onclick="toggleAuto()">
      <span class="btn-icon">⏱</span>自动刷新
    </button>
    <button class="btn" onclick="loadGraph()">
      <span class="btn-icon">🔄</span>刷新
    </button>
    <a class="btn" href="/" target="_blank">
      <span class="btn-icon">💬</span>对话窗口
    </a>
  </div>
</div>

<div class="main">
  <!-- Sidebar -->
  <div class="sidebar">
    <div>
      <h4>统计</h4>
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-num" id="sNodes">-</div>
          <div class="stat-label">节点</div>
        </div>
        <div class="stat-box">
          <div class="stat-num" id="sLinks">-</div>
          <div class="stat-label">关系</div>
        </div>
      </div>
    </div>
    <div>
      <h4>节点类型过滤</h4>
      <div id="filterList"></div>
    </div>
    <div>
      <h4>关系类型</h4>
      <div id="relList"></div>
    </div>
  </div>

  <!-- Graph canvas -->
  <div class="canvas" id="canvasWrap">
    <svg id="graphSvg"></svg>
    <div class="tooltip" id="tooltip"></div>
    <div class="loading" id="loading">
      <div class="loading-inner">
        <div class="spinner"></div>
        <div style="font-size:13px;color:#6677aa">加载图谱数据...</div>
      </div>
    </div>
    <div class="empty-hint" id="emptyHint" style="display:none">
      <div class="empty-inner">
        <div class="icon">🕸️</div>
        <p>暂无图谱数据<br><small style="font-size:11px;color:#222240">先在对话窗口聊几句，等记忆写入后刷新</small></p>
      </div>
    </div>
  </div>

  <!-- Detail panel -->
  <div class="detail-panel" id="detailPanel">
    <div class="detail-header">
      <span class="detail-badge" id="detailBadge">-</span>
      <span class="detail-close" onclick="closeDetail()">×</span>
    </div>
    <div class="detail-text" id="detailText">-</div>
    <div class="detail-props" id="detailProps"></div>
  </div>
</div>

<script>
// ── 颜色/尺寸配置 ──────────────────────────────────────────────────────────────
const TYPE_COLORS = {
  Event:             '#4f7fff',
  Entity:            '#22c55e',
  Knowledge:         '#f59e0b',
  UtteranceEvidence: '#6b7280',
  MediaSegment:      '#14b8a6',
  MemoryNode:        '#a855f7',
  Semantic:          '#c084fc',
};
const TYPE_RADIUS = {
  Event: 20, Entity: 18, Knowledge: 16,
  UtteranceEvidence: 10, MediaSegment: 13,
  MemoryNode: 14, Semantic: 12,
};
const PROP_LABELS = {
  id: 'ID', summary: '摘要', name: '名称', content: '内容', text: '文本',
  raw_text: '原始文本', topic_path: '主题路径', importance: '重要性',
  tenant_id: '租户', memory_domain: '记忆域', created_at: '创建时间',
};
function getColor(t) { return TYPE_COLORS[t] || '#555577'; }
function getRadius(t) { return TYPE_RADIUS[t] || 12; }

// ── 状态 ──────────────────────────────────────────────────────────────────────
let allNodes = [], allLinks = [];
let visibleTypes = new Set();
let simulation   = null;
let isAutoRefresh = false;
let autoTimer     = null;
let svg, g, zoomBehavior;
let selectedId = null;

// ── SVG 初始化 ─────────────────────────────────────────────────────────────────
function initSvg() {
  svg = d3.select('#graphSvg');
  svg.selectAll('*').remove();

  // 箭头 marker，每种节点类型一个颜色
  const defs = svg.append('defs');
  const markerTypes = [...Object.keys(TYPE_COLORS), 'default'];
  markerTypes.forEach(t => {
    const color = TYPE_COLORS[t] || '#555577';
    defs.append('marker')
      .attr('id', `arrow-${t}`)
      .attr('viewBox', '0 -4 8 8').attr('refX', 8).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
        .attr('d', 'M0,-4L8,0L0,4')
        .attr('fill', color).attr('opacity', 0.65);
  });

  zoomBehavior = d3.zoom()
    .scaleExtent([0.05, 6])
    .on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoomBehavior);

  g = svg.append('g');
}

// ── 渲染图 ─────────────────────────────────────────────────────────────────────
function renderGraph(nodes, links) {
  const W = document.getElementById('canvasWrap').clientWidth;
  const H = document.getElementById('canvasWrap').clientHeight;

  if (simulation) { simulation.stop(); }
  g.selectAll('*').remove();

  if (!nodes.length) {
    document.getElementById('emptyHint').style.display = 'flex';
    return;
  }
  document.getElementById('emptyHint').style.display = 'none';

  // ── 关系线 ──
  const linkG  = g.append('g').attr('class', 'links');
  const linkEl = linkG.selectAll('g.link').data(links).enter().append('g').attr('class', 'link');

  const lines = linkEl.append('line')
    .attr('stroke', d => {
      const t = (typeof d.source === 'object' ? d.source.type : null) || 'default';
      return (TYPE_COLORS[t] || '#555577') + '88';
    })
    .attr('stroke-width', 1.5)
    .attr('marker-end', d => {
      const t = (typeof d.source === 'object' ? d.source.type : null) || 'default';
      return `url(#arrow-${TYPE_COLORS[t] ? t : 'default'})`;
    });

  const relLabels = linkEl.append('text')
    .attr('class', 'link-label')
    .attr('text-anchor', 'middle')
    .text(d => d.type);

  // ── 节点 ──
  const nodeG  = g.append('g').attr('class', 'nodes');
  const nodeEl = nodeG.selectAll('g.node').data(nodes, d => d.id).enter()
    .append('g').attr('class', 'node')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('click',     (e, d) => { e.stopPropagation(); selectNode(d, nodeEl); showDetail(d); })
    .on('mouseover', (e, d) => showTooltip(e, d))
    .on('mouseout',  ()     => hideTooltip());

  nodeEl.append('circle')
    .attr('r', d => getRadius(d.type))
    .attr('fill',   d => getColor(d.type) + 'bb')
    .attr('stroke', d => getColor(d.type));

  nodeEl.append('text')
    .attr('class', 'node-label')
    .attr('dy', d => getRadius(d.type) + 13)
    .attr('text-anchor', 'middle')
    .text(d => {
      const lbl = d.label || d.id;
      return lbl.length > 14 ? lbl.slice(0, 14) + '…' : lbl;
    });

  // ── 力模拟 ──
  simulation = d3.forceSimulation(nodes)
    .force('link',    d3.forceLink(links).id(d => d.id).distance(130).strength(0.5))
    .force('charge',  d3.forceManyBody().strength(-320))
    .force('center',  d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide(d => getRadius(d.type) + 22))
    .on('tick', () => {
      lines
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => edgeX(d))
        .attr('y2', d => edgeY(d));
      relLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2 - 5);
      nodeEl.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

  // click on background clears selection
  svg.on('click', () => { selectedId = null; nodeEl.classed('selected', false); });

  // 初始缩放让图谱适中显示
  svg.transition().duration(600)
    .call(zoomBehavior.transform,
      d3.zoomIdentity.translate(W * 0.05, H * 0.05).scale(0.9));
}

function edgeX(d) {
  const r = getRadius(d.target.type) + 8;
  const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
  return d.target.x - (dx / dist) * r;
}
function edgeY(d) {
  const r = getRadius(d.target.type) + 8;
  const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
  return d.target.y - (dy / dist) * r;
}

function selectNode(d, nodeEl) {
  selectedId = d.id;
  nodeEl.classed('selected', n => n.id === selectedId);
}

// ── Tooltip ────────────────────────────────────────────────────────────────────
function showTooltip(e, d) {
  const wrap = document.getElementById('canvasWrap');
  const rect  = wrap.getBoundingClientRect();
  const tt = document.getElementById('tooltip');
  tt.style.display = 'block';
  tt.style.left = (e.clientX - rect.left + 16) + 'px';
  tt.style.top  = (e.clientY - rect.top  - 10) + 'px';
  tt.innerHTML = `<strong style="color:${getColor(d.type)}">${esc(d.type)}</strong><br>${esc(d.label)}`;
}
function hideTooltip() { document.getElementById('tooltip').style.display = 'none'; }

// ── Detail Panel ───────────────────────────────────────────────────────────────
function showDetail(d) {
  const panel = document.getElementById('detailPanel');
  panel.classList.add('open');

  const badge = document.getElementById('detailBadge');
  badge.textContent = d.type;
  badge.style.cssText = `background:${getColor(d.type)}22;color:${getColor(d.type)};border:1px solid ${getColor(d.type)}44`;

  document.getElementById('detailText').textContent = d.label;

  const propsEl = document.getElementById('detailProps');
  propsEl.innerHTML = '';
  Object.entries(d.props || {}).forEach(([k, v]) => {
    if (!v || v === 'None') return;
    const div = document.createElement('div');
    div.className = 'prop-box';
    div.innerHTML = `<div class="prop-key">${esc(PROP_LABELS[k] || k)}</div>
                     <div class="prop-val">${esc(v)}</div>`;
    propsEl.appendChild(div);
  });
}
function closeDetail() { document.getElementById('detailPanel').classList.remove('open'); }

// ── 过滤 ───────────────────────────────────────────────────────────────────────
function applyFilter() {
  const filteredNodes = allNodes.filter(n => visibleTypes.has(n.type));
  const filteredIds   = new Set(filteredNodes.map(n => n.id));
  const filteredLinks = allLinks.filter(l => {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    return filteredIds.has(s) && filteredIds.has(t);
  });
  // 重建对象副本避免 D3 mutation 污染
  const nc = filteredNodes.map(n => ({ ...n }));
  const lc = filteredLinks.map(l => ({
    ...l,
    source: typeof l.source === 'object' ? l.source.id : l.source,
    target: typeof l.target === 'object' ? l.target.id : l.target,
  }));
  renderGraph(nc, lc);
}
function toggleType(type, checked) {
  if (checked) visibleTypes.add(type); else visibleTypes.delete(type);
  applyFilter();
}

// ── 更新侧边栏 ─────────────────────────────────────────────────────────────────
function updateStats(s) {
  document.getElementById('sNodes').textContent = s?.node_count ?? '-';
  document.getElementById('sLinks').textContent = s?.link_count ?? '-';
}
function updateFilters(nodes) {
  const counts = {};
  nodes.forEach(n => { counts[n.type] = (counts[n.type] || 0) + 1; });
  const fl = document.getElementById('filterList');
  fl.innerHTML = '';
  Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([type, cnt]) => {
    const div = document.createElement('div');
    div.className = 'filter-item';
    const chk = visibleTypes.has(type) ? 'checked' : '';
    div.innerHTML = `
      <input type="checkbox" ${chk} onchange="toggleType('${type}', this.checked)">
      <div class="filter-dot" style="background:${getColor(type)}"></div>
      <span class="filter-label">${esc(type)}</span>
      <span class="filter-count">${cnt}</span>`;
    fl.appendChild(div);
  });
}
function updateRelList(links) {
  const counts = {};
  links.forEach(l => { counts[l.type] = (counts[l.type] || 0) + 1; });
  document.getElementById('relList').innerHTML =
    Object.entries(counts).sort((a, b) => b[1] - a[1])
      .map(([t, c]) => `<div class="rel-row"><span class="rel-name">${esc(t)}</span><span class="rel-cnt">${c}</span></div>`)
      .join('');
}

// ── 加载数据 ───────────────────────────────────────────────────────────────────
async function loadGraph() {
  document.getElementById('loading').style.display = 'flex';
  document.getElementById('refreshTxt').className = 'refresh-txt pulse';
  document.getElementById('refreshTxt').textContent = '加载中…';
  try {
    const r    = await fetch('/api/graph-data');
    const data = await r.json();
    if (data.error) throw new Error(data.error);

    allNodes = data.nodes;
    allLinks = data.links;

    // 初始化所有类型为可见
    visibleTypes = new Set(allNodes.map(n => n.type));

    updateStats(data.stats);
    updateFilters(allNodes);
    updateRelList(allLinks);

    // 渲染（创建副本防止 D3 mutation）
    const nc = allNodes.map(n => ({ ...n }));
    const lc = allLinks.map(l => ({ ...l }));
    renderGraph(nc, lc);

    const now = new Date().toLocaleTimeString('zh-CN');
    document.getElementById('refreshTxt').textContent = `${now} 更新`;
    document.getElementById('refreshTxt').className = 'refresh-txt';
  } catch (e) {
    document.getElementById('refreshTxt').textContent = '加载失败: ' + e.message;
    document.getElementById('refreshTxt').className = 'refresh-txt';
  } finally {
    document.getElementById('loading').style.display = 'none';
  }
}

// ── 自动刷新 ───────────────────────────────────────────────────────────────────
function toggleAuto() {
  isAutoRefresh = !isAutoRefresh;
  const btn = document.getElementById('autoBtn');
  if (isAutoRefresh) {
    btn.classList.add('active');
    btn.innerHTML = '<span class="btn-icon">⏱</span>自动刷新 ON';
    autoTimer = setInterval(loadGraph, 15000);
  } else {
    btn.classList.remove('active');
    btn.innerHTML = '<span class="btn-icon">⏱</span>自动刷新';
    clearInterval(autoTimer);
  }
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── 启动 ───────────────────────────────────────────────────────────────────────
initSvg();
loadGraph();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not LLM_API_KEY:
        print("WARNING: LLM_API_KEY is not set. Chat responses will fail.")
        print("         export LLM_API_KEY=<your-openai-compatible-key>")
        print()
    port = int(os.getenv("DEMO_PORT", "7860"))
    print("=" * 60)
    print("  LATRACE Chat-with-Memory Demo")
    print("=" * 60)
    print(f"  Chat:         http://localhost:{port}")
    print(f"  Memory Graph: http://localhost:{port}/graph")
    print(f"  User ID:      {USER_ID}")
    print(f"  LLM Model:    {LLM_MODEL}")
    print(f"  Memory API:   {MEMORY_API}")
    print(f"  Ingest:       POST /ingest/dialog/v1")
    print(f"  Retrieve:     POST /retrieval/dialog/v2")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
