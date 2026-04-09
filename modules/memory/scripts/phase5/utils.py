from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        items.append(json.loads(raw))
    return items


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_headers(tenant_id: str, api_key: Optional[str]) -> Dict[str, str]:
    headers = {"X-Tenant-ID": tenant_id}
    if api_key:
        headers["X-API-Token"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def call_api(
    base_url: str,
    path: str,
    *,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float = 20.0,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

