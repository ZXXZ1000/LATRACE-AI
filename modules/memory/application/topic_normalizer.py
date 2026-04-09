from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


def _norm_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: List[str] = []
    seen: set[str] = set()
    for it in items:
        s = str(it).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _join_text(parts: Sequence[Optional[str]]) -> str:
    chunks: List[str] = []
    for p in parts:
        if p and str(p).strip():
            chunks.append(str(p).strip())
    return " ".join(chunks)


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    if not text:
        return False
    for kw in keywords:
        if kw and kw in text:
            return True
    return False


def _derive_keywords(text: str, *, max_items: int = 8) -> List[str]:
    if not text:
        return []
    stop_en = {
        "the", "and", "with", "from", "this", "that", "have", "has", "had", "about",
        "today", "yesterday", "tomorrow", "really", "very", "just", "thing", "things",
        "like", "want", "need", "plan", "work", "meeting", "project",
    }
    stop_zh = {
        "今天", "昨天", "明天", "现在", "最近", "然后", "因为", "感觉", "一个", "我们",
        "你们", "他们", "就是", "还是", "但是", "这个", "那个", "事情", "东西", "问题",
        "有点", "有些", "其实",
    }
    out: List[str] = []
    seen: set[str] = set()

    # English-like tokens
    for m in re.findall(r"[A-Za-z][A-Za-z0-9_+\\-]{2,}", text):
        kw = m.lower()
        if kw in stop_en or kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
        if len(out) >= max_items:
            return out

    # CJK sequences: split into meaningful chunks
    for m in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if m in stop_zh or m in seen:
            continue
        if len(m) <= 4:
            cand = [m]
        else:
            cand = [m[:2], m[2:4], m[-2:]]
        for c in cand:
            if c in stop_zh or c in seen:
                continue
            seen.add(c)
            out.append(c)
            if len(out) >= max_items:
                return out

    return out


def _keywords_hit(keywords: Sequence[str], needles: Sequence[str]) -> bool:
    if not keywords:
        return False
    kw_set = set([k for k in keywords if k])
    for n in needles:
        if n in kw_set:
            return True
    return False


def _all_keywords_hit(keywords: Sequence[str], needles: Sequence[str]) -> bool:
    if not needles:
        return True
    kw_set = set([k for k in keywords if k])
    return all(n in kw_set for n in needles if n)


def _env_flag(name: str, default: str = "false") -> bool:
    raw = str(os.getenv(name, default)).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _topic_registry_enabled() -> bool:
    return _env_flag("MEMORY_TOPIC_REGISTRY_ENABLED", "true")


def _topic_registry_override_topic_id() -> bool:
    return _env_flag("MEMORY_TOPIC_REGISTRY_OVERRIDE_TOPIC_ID", "true")


def _topic_registry_max_per_scope() -> int:
    try:
        return max(32, int(os.getenv("MEMORY_TOPIC_REGISTRY_MAX_PER_SCOPE", "2000") or 2000))
    except Exception:
        return 2000


def _canonicalize_topic_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.casefold()
    text = re.sub(r"[\s\-_]+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _topic_text_basis(event: Dict[str, Any]) -> str:
    for key in ("topic_id_raw", "topic_id", "summary", "desc"):
        val = _canonicalize_topic_token(event.get(key))
        if val:
            return val
    kws = [_canonicalize_topic_token(x) for x in (event.get("keywords") or [])]
    kws = [x for x in kws if x]
    return " ".join(kws[:6]).strip()


def build_topic_canonical_key(event: Dict[str, Any]) -> str:
    basis = _topic_text_basis(event)
    topic_path = _canonicalize_topic_token(event.get("topic_path")) if not basis else ""
    kw_part = ""
    if not basis:
        kws = [_canonicalize_topic_token(x) for x in (event.get("keywords") or [])]
        kws = sorted(dict.fromkeys([k for k in kws if k]))[:6]
        kw_part = ",".join(kws)
    return f"v1|path={topic_path}|basis={basis}|kw={kw_part}"


@dataclass
class TopicVocab:
    version: str
    topic_paths: set[str]
    tags: set[str]
    tags_version: str
    rules: List[Dict[str, Any]]
    synonyms: Dict[str, str]


@dataclass(frozen=True)
class TopicNormalization:
    """Normalized topic text payload for read/query-time reuse."""

    topic_text: str
    topic_id: str
    topic_path: str
    tags: Tuple[str, ...]
    keywords: Tuple[str, ...]
    tags_vocab_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_text": self.topic_text,
            "topic_id": self.topic_id,
            "topic_path": self.topic_path,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "tags_vocab_version": self.tags_vocab_version,
        }


class TopicRegistry:
    """In-memory topic registry with deterministic canonical key (no fuzzy merge yet)."""

    def __init__(self, *, max_per_scope: int = 2000) -> None:
        self.max_per_scope = max(32, int(max_per_scope or 2000))
        self._lock = threading.RLock()
        self._scopes: Dict[str, "OrderedDict[str, str]"] = {}

    def clear(self) -> None:
        with self._lock:
            self._scopes.clear()

    def _scope_key(self, *, tenant_id: Optional[str], memory_domain: Optional[str]) -> str:
        t = str(tenant_id or "").strip() or "*"
        d = str(memory_domain or "").strip() or "*"
        return f"{t}::{d}"

    def resolve(
        self,
        *,
        canonical_key: str,
        tenant_id: Optional[str] = None,
        memory_domain: Optional[str] = None,
    ) -> Tuple[str, bool]:
        digest = hashlib.sha1(canonical_key.encode("utf-8")).hexdigest()[:16]
        topic_id = f"tpk_{digest}"
        scope = self._scope_key(tenant_id=tenant_id, memory_domain=memory_domain)
        with self._lock:
            bucket = self._scopes.setdefault(scope, OrderedDict())
            if canonical_key in bucket:
                val = bucket.pop(canonical_key)
                bucket[canonical_key] = val
                return val, True
            bucket[canonical_key] = topic_id
            while len(bucket) > self.max_per_scope:
                bucket.popitem(last=False)
            return topic_id, False


class TopicNormalizer:
    def __init__(self, vocab_dir: Optional[Path] = None) -> None:
        self._vocab_dir = vocab_dir or self._default_vocab_dir()
        self._vocab: Optional[TopicVocab] = None

    @staticmethod
    def _default_vocab_dir() -> Path:
        # repo_root/MOYAN_AGENT_INFRA/modules/memory/vocab
        root = Path(__file__).resolve().parents[3]
        return root / "modules" / "memory" / "vocab"

    def _load_vocab(self) -> TopicVocab:
        if self._vocab is not None:
            return self._vocab
        vocab_dir = self._vocab_dir
        topic_path_fp = vocab_dir / "topic_path.yaml"
        tags_fp = vocab_dir / "tags.yaml"
        rules_fp = vocab_dir / "normalization_rules.yaml"
        synonyms_fp = vocab_dir / "synonyms.yaml"

        def _read_yaml(fp: Path) -> Dict[str, Any]:
            try:
                with fp.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}

        topic_data = _read_yaml(topic_path_fp)
        tag_data = _read_yaml(tags_fp)
        rules_data = _read_yaml(rules_fp)
        syn_data = _read_yaml(synonyms_fp)

        topic_paths = set([str(p.get("path")).strip() for p in topic_data.get("paths", []) if p.get("path")])
        tags = set([str(t).strip() for t in tag_data.get("tags", []) if str(t).strip()])
        rules = list(rules_data.get("rules") or [])
        # sort by priority desc
        rules.sort(key=lambda r: int(r.get("priority") or 0), reverse=True)

        synonyms: Dict[str, str] = {}
        for item in syn_data.get("synonyms", []) or []:
            canonical = str(item.get("canonical") or "").strip()
            if not canonical:
                continue
            for v in item.get("variants") or []:
                vv = str(v).strip()
                if vv:
                    synonyms[vv] = canonical

        self._vocab = TopicVocab(
            version=str(topic_data.get("version") or "v1"),
            topic_paths=topic_paths,
            tags=tags,
            tags_version=str(tag_data.get("version") or "v1"),
            rules=rules,
            synonyms=synonyms,
        )
        return self._vocab

    def _fallback_bucket(self, keywords: Sequence[str], text: str, vocab: TopicVocab) -> str:
        # lightweight heuristic buckets
        learning_keys = ["学习", "课程", "上课", "作业", "考试", "复习", "阅读", "读书"]
        work_keys = ["工作", "项目", "会议", "任务", "需求", "老板", "同事"]
        lifestyle_keys = ["生活", "日常", "吃饭", "做饭", "睡觉", "锻炼", "运动", "旅行", "家庭"]
        high_entropy_keys = ["随便聊", "杂谈", "乱七八糟", "胡扯", "闲聊"]

        if _keywords_hit(keywords, high_entropy_keys) or _contains_any(text, high_entropy_keys):
            return "_uncategorized/high_entropy"
        if _keywords_hit(keywords, learning_keys) or _contains_any(text, learning_keys):
            return "_uncategorized/learning"
        if _keywords_hit(keywords, work_keys) or _contains_any(text, work_keys):
            return "_uncategorized/work"
        if _keywords_hit(keywords, lifestyle_keys) or _contains_any(text, lifestyle_keys):
            return "_uncategorized/lifestyle"
        return "_uncategorized/general"

    def normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        vocab = self._load_vocab()
        summary = str(event.get("summary") or "").strip()
        desc = str(event.get("desc") or "").strip()
        topic_id = str(event.get("topic_id") or "").strip()
        keywords = _norm_list(event.get("keywords"))
        tags = _norm_list(event.get("tags"))
        text = _join_text([summary, desc, topic_id] + keywords)
        if not keywords:
            keywords = _derive_keywords(text)
            text = _join_text([summary, desc, topic_id] + keywords)

        topic_path = str(event.get("topic_path") or "").strip() or None
        if topic_path and (topic_path not in vocab.topic_paths or topic_path.startswith("_uncategorized/")):
            topic_path = None

        # rule-based normalization
        if not topic_path:
            for rule in vocab.rules:
                match = rule.get("match") or {}
                places = _norm_list(match.get("places"))
                any_keys = _norm_list(match.get("keywords_any"))
                all_keys = _norm_list(match.get("keywords_all"))
                if places and not (_keywords_hit(keywords, places) or _contains_any(text, places)):
                    continue
                if any_keys and not (_keywords_hit(keywords, any_keys) or _contains_any(text, any_keys)):
                    continue
                if all_keys and not _all_keywords_hit(keywords, all_keys):
                    continue
                out = rule.get("output") or {}
                cand = str(out.get("topic_path") or "").strip()
                if cand and cand in vocab.topic_paths:
                    topic_path = cand
                # tags from rule
                rule_tags = _norm_list(out.get("tags"))
                if rule_tags:
                    tags = rule_tags
                break

        # synonym-based mapping
        if not topic_path:
            # check topic_id first, then keywords
            if topic_id and topic_id in vocab.synonyms:
                cand = vocab.synonyms.get(topic_id)
                if cand in vocab.topic_paths:
                    topic_path = cand
            if not topic_path:
                for kw in keywords:
                    cand = vocab.synonyms.get(kw)
                    if cand and cand in vocab.topic_paths:
                        topic_path = cand
                        break
            if not topic_path and text:
                for var, cand in vocab.synonyms.items():
                    if var and var in text and cand in vocab.topic_paths:
                        topic_path = cand
                        break

        if not topic_path:
            topic_path = self._fallback_bucket(keywords, text, vocab)

        # filter tags by vocab
        tags = [t for t in tags if t in vocab.tags]

        out = dict(event)
        out["topic_path"] = topic_path
        out["tags"] = tags or None
        out["keywords"] = keywords or None
        if tags or topic_path:
            out["tags_vocab_version"] = vocab.tags_version
        return out


def get_normalization_mode() -> str:
    return str(os.getenv("MEMORY_TOPIC_NORMALIZATION_MODE", "sync")).strip().lower()


def _default_queue_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "modules" / "memory" / "outputs" / "topic_normalization_queue.jsonl"


def enqueue_deferred_event(event: Dict[str, Any], *, queue_path: Optional[Path] = None) -> bool:
    try:
        path = queue_path or Path(
            os.getenv("MEMORY_TOPIC_NORMALIZATION_QUEUE_PATH", str(_default_queue_path()))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_id": event.get("id") or event.get("event_id"),
            "tenant_id": event.get("tenant_id"),
            "user_id": event.get("user_id"),
            "memory_domain": event.get("memory_domain"),
            "summary": event.get("summary"),
            "desc": event.get("desc"),
            "topic_id": event.get("topic_id"),
            "topic_path": event.get("topic_path"),
            "tags": event.get("tags"),
            "keywords": event.get("keywords"),
            "time_bucket": event.get("time_bucket"),
            "tags_vocab_version": event.get("tags_vocab_version"),
            "source_turn_ids": event.get("source_turn_ids"),
            "time_hint": event.get("time_hint"),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def apply_topic_registry(
    event: Dict[str, Any],
    *,
    tenant_id: Optional[str] = None,
    memory_domain: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(event)
    raw_topic_id = str(out.get("topic_id") or "").strip() or None
    if raw_topic_id:
        out.setdefault("topic_id_raw", raw_topic_id)
    if not _topic_registry_enabled():
        return out

    canonical_key = build_topic_canonical_key(out)
    topic_id, cache_hit = get_topic_registry().resolve(
        canonical_key=canonical_key,
        tenant_id=tenant_id or out.get("tenant_id"),
        memory_domain=memory_domain or out.get("memory_domain"),
    )
    out["topic_registry_key"] = canonical_key
    out["topic_registry_source"] = "cache_hit" if cache_hit else "cache_miss"
    if _topic_registry_override_topic_id():
        out["topic_id"] = topic_id
    return out


def normalize_events(
    events: List[Dict[str, Any]],
    *,
    mode: Optional[str] = None,
    queue_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    norm = get_topic_normalizer()
    norm_mode = (mode or get_normalization_mode()).strip().lower()
    if norm_mode not in {"sync", "async"}:
        norm_mode = "sync"
    out: List[Dict[str, Any]] = []
    for ev in events:
        normalized = norm.normalize_event(ev)
        normalized = apply_topic_registry(normalized)
        out.append(normalized)
        if norm_mode == "async":
            tp = str(normalized.get("topic_path") or "").strip()
            if tp.startswith("_uncategorized/") and (normalized.get("id") or normalized.get("event_id")):
                enqueue_deferred_event(normalized, queue_path=queue_path)
    return out


_NORMALIZER: Optional[TopicNormalizer] = None
_TOPIC_REGISTRY: Optional[TopicRegistry] = None


def get_topic_normalizer() -> TopicNormalizer:
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = TopicNormalizer()
    return _NORMALIZER


def get_topic_registry() -> TopicRegistry:
    global _TOPIC_REGISTRY
    max_per_scope = _topic_registry_max_per_scope()
    if _TOPIC_REGISTRY is None:
        _TOPIC_REGISTRY = TopicRegistry(max_per_scope=max_per_scope)
    elif _TOPIC_REGISTRY.max_per_scope != max_per_scope:
        _TOPIC_REGISTRY = TopicRegistry(max_per_scope=max_per_scope)
    return _TOPIC_REGISTRY


def _canonicalize_topic_text(topic_text: Any) -> str:
    text = str(topic_text or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=1024)
def _normalize_topic_text_cached(
    canonical_text: str,
    registry_enabled: bool,
    registry_override: bool,
) -> TopicNormalization:
    normalizer = get_topic_normalizer()
    normalized = normalizer.normalize_event({"summary": canonical_text, "topic_id": canonical_text})
    if registry_enabled:
        normalized = apply_topic_registry(normalized)
        if not registry_override:
            raw_topic_id = str(normalized.get("topic_id_raw") or "").strip()
            if raw_topic_id:
                normalized["topic_id"] = raw_topic_id

    topic_id = str(normalized.get("topic_id") or canonical_text).strip() or canonical_text
    topic_path = str(normalized.get("topic_path") or "_uncategorized/general").strip() or "_uncategorized/general"
    tags = tuple(str(x).strip() for x in (normalized.get("tags") or []) if str(x).strip())
    keywords = tuple(str(x).strip() for x in (normalized.get("keywords") or []) if str(x).strip())
    tags_vocab_version = str(normalized.get("tags_vocab_version") or "").strip() or None
    return TopicNormalization(
        topic_text=canonical_text,
        topic_id=topic_id,
        topic_path=topic_path,
        tags=tags,
        keywords=keywords,
        tags_vocab_version=tags_vocab_version,
    )


def normalize_topic_text(topic_text: str) -> TopicNormalization:
    """Normalize free-text topic input for query-time APIs (LRU cached)."""
    return _normalize_topic_text_cached(
        _canonicalize_topic_text(topic_text),
        _topic_registry_enabled(),
        _topic_registry_override_topic_id(),
    )


# Expose cache controls for tests/debugging while keeping canonicalization wrapper.
normalize_topic_text.cache_info = _normalize_topic_text_cached.cache_info  # type: ignore[attr-defined]
normalize_topic_text.cache_clear = _normalize_topic_text_cached.cache_clear  # type: ignore[attr-defined]
