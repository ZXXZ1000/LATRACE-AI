from __future__ import annotations

from typing import Any, Dict, List

from modules.memory.contracts.memory_models import MemoryEntry


class InMemVectorStore:
    """A minimal in-memory vector-store facade for testing and development.

    - Stores MemoryEntry objects keyed by id.
    - Provides a naive text-based scoring for search (token overlap).
    - Exposes health info and a dump method for tests.
    """

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        self._entries: Dict[str, MemoryEntry] = {}

    async def upsert_vectors(self, entries: List[MemoryEntry]) -> None:
        for e in entries:
            if not e.id:
                continue
            # Store a shallow copy to avoid external mutation
            self._entries[e.id] = e.model_copy(deep=True)

    def _passes_filters(self, e: MemoryEntry, filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        pub = filters.get("published")
        if pub is True and e.published is False:
            return False
        if pub is False and e.published is not False:
            return False
        # tenant_id exact match (hard boundary when provided)
        tenant = filters.get("tenant_id")
        if tenant is not None and str(e.metadata.get("tenant_id")) != str(tenant):
            return False
        if (mods := filters.get("modality")):
            modset = set(mods)
            if e.modality not in modset:
                # Treat structured payloads as searchable under the text modality for in-mem tests.
                if not (e.modality == "structured" and "text" in modset):
                    return False
        if (mtypes := filters.get("memory_type")) and e.kind not in set(mtypes):
            return False
        if (srcs := filters.get("source")) and e.metadata.get("source") not in set(srcs):
            return False
        # user_id (list) match: any/all
        uflt = filters.get("user_id")
        if uflt:
            # entry user ids can be str or list; normalize to list[str]
            e_users_raw = e.metadata.get("user_id")
            if e_users_raw is None:
                return False
            if isinstance(e_users_raw, list):
                e_users = set(str(x) for x in e_users_raw)
            else:
                e_users = {str(e_users_raw)}
            f_users = set(str(x) for x in (uflt or []))
            mode = (filters.get("user_match") or "any").lower()
            if mode == "all":
                if not f_users.issubset(e_users):
                    return False
            else:  # any
                if not e_users.intersection(f_users):
                    return False
        # memory_domain exact match
        dom = filters.get("memory_domain")
        if dom is not None and str(e.metadata.get("memory_domain")) != str(dom):
            return False
        # run_id exact match
        rid = filters.get("run_id")
        if rid is not None and str(e.metadata.get("run_id")) != str(rid):
            return False
        # memory_scope exact match
        mscope = filters.get("memory_scope")
        if mscope is not None and str(e.metadata.get("memory_scope")) != str(mscope):
            return False
        # entities intersection (expects list in metadata.entities)
        ents = filters.get("entities")
        if ents:
            eents = e.metadata.get("entities") or []
            try:
                if not set(eents).intersection(set(ents)):
                    return False
            except Exception:
                return False
        # character_id: OR across provided ids; compare as strings
        chars = filters.get("character_id")
        if chars:
            cid = e.metadata.get("character_id")
            if cid is None:
                return False
            # normalize to set[str]
            if isinstance(chars, list):
                cset = set(str(x) for x in chars)
            else:
                cset = {str(chars)}
            if str(cid) not in cset:
                return False
        # time_range with ISO8601 timestamps (string) or epoch seconds
        tr = filters.get("time_range") or {}
        if isinstance(tr, dict) and ("gte" in tr or "lte" in tr):
            ts = e.metadata.get("timestamp") or e.metadata.get("created_at")
            def _to_epoch(val) -> float | None:
                from datetime import datetime
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    try:
                        return datetime.fromisoformat(val).timestamp()
                    except Exception:
                        return None
                return None
            tsv = _to_epoch(ts)
            if tsv is None:
                return False
            if tr.get("gte") is not None:
                gv = _to_epoch(tr.get("gte"))
                if gv is not None and tsv < gv:
                    return False
            if tr.get("lte") is not None:
                lv = _to_epoch(tr.get("lte"))
                if lv is not None and tsv > lv:
                    return False
        # character_id match (OR)
        chars = filters.get("character_id")
        if chars:
            cid = e.metadata.get("character_id")
            if cid is None:
                return False
            if isinstance(chars, list):
                cset = set(str(x) for x in chars)
            else:
                cset = {str(chars)}
            if str(cid) not in cset:
                return False
        # time_range/clip_id/entities etc. kept for future extension
        return True

    def _score(self, query: str, entry: MemoryEntry) -> float:
        """Naive token-overlap score with bigram support for CJK texts.

        - If spaces present, split by spaces.
        - Otherwise, use character bigrams; single char falls back to itself.
        """
        if not query or not entry.contents:
            return 0.0

        def _bigrams(s: str) -> list[str]:
            if len(s) == 1:
                return [s]
            return [s[i : i + 2] for i in range(len(s) - 1)]

        def _tok(s: str) -> list[str]:
            s = (s or "").strip().lower()
            if not s:
                return []
            if any(ch.isspace() for ch in s):
                out: list[str] = []
                for term in (t for t in s.split() if t):
                    out.append(term)
                    out.extend(_bigrams(term))
                    # also include unigrams to improve recall for short CJK terms
                    if len(term) > 1:
                        out.extend(list(term))
                return out
            out = _bigrams(s)
            if len(s) > 1:
                out.extend(list(s))
            return out

        q_list = _tok(query)
        q_tokens = set(q_list)
        q_lower = (query or "").strip().lower()
        best = 0.0
        q_terms = [t for t in q_lower.split() if t] if q_lower else []
        for c in entry.contents:
            cstr = str(c)
            c_lower = cstr.strip().lower()
            ctoks = set(_tok(cstr))
            # quick substring match (both directions) and per-term matching for whitespace queries
            if q_lower:
                if q_lower in c_lower:
                    return max(best, float(len(q_lower)))
                if c_lower and c_lower in q_lower:
                    best = max(best, float(len(c_lower)))
                for term in q_terms:
                    if term and term in c_lower:
                        best = max(best, float(len(term)))
            inter = q_tokens.intersection(ctoks)
            base = float(len(inter))
            # prioritize earlier query tokens (e.g., first token carries higher weight)
            pos_weight = 0.0
            n = len(q_list)
            for idx, t in enumerate(q_list):
                if t in ctoks:
                    pos_weight += float(n - idx)
            score = base + 0.1 * pos_weight
            if score > best:
                best = score
        return float(best)

    async def search_vectors(
        self,
        query: str,
        filters: Dict[str, Any],
        topk: int,
        threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        candidates = []
        for e in self._entries.values():
            # skip soft-deleted
            if e.metadata.get("is_deleted") is True:
                continue
            if not self._passes_filters(e, filters or {}):
                continue
            s = self._score(query, e)
            # if character filter命中但文本无匹配，给一个基础分保障命中
            if filters.get("character_id") and s <= 0.0:
                s = 1.0
            if threshold is not None and s < threshold:
                continue
            candidates.append({"id": e.id, "score": s, "payload": e})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[: max(1, topk)]

    async def fetch_text_corpus(self, filters: Dict[str, Any], *, limit: int = 500) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry_id in sorted(self._entries.keys()):
            e = self._entries[entry_id]
            if e.metadata.get("is_deleted") is True:
                continue
            if not self._passes_filters(e, filters or {}):
                continue
            out.append({"id": e.id, "payload": e.model_copy(deep=True)})
            if len(out) >= max(1, int(limit)):
                break
        return out

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok", "entries": len(self._entries)}

    # Testing helper
    def dump(self) -> Dict[str, MemoryEntry]:
        return dict(self._entries)

    # CRUD helpers for service
    async def get(self, entry_id: str) -> MemoryEntry | None:
        e = self._entries.get(entry_id)
        return e.model_copy(deep=True) if e else None

    async def delete_ids(self, ids: List[str]) -> None:
        for i in ids:
            self._entries.pop(i, None)

    async def count_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        total = 0
        for entry in self._entries.values():
            metadata = dict(entry.metadata or {})
            if str(metadata.get("tenant_id") or "") != str(tenant_id):
                continue
            if api_key_id is not None and str(metadata.get("api_key_id") or "") != str(api_key_id):
                continue
            total += 1
        return total

    async def list_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        out: List[str] = []
        for entry_id, entry in self._entries.items():
            metadata = dict(entry.metadata or {})
            if str(metadata.get("tenant_id") or "") != str(tenant_id):
                continue
            if api_key_id is not None and str(metadata.get("api_key_id") or "") != str(api_key_id):
                continue
            out.append(str(entry_id))
        return out

    async def list_entry_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        return await self.list_ids_by_filter(tenant_id=tenant_id, api_key_id=api_key_id)

    async def delete_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        to_delete: List[str] = []
        for entry_id, entry in self._entries.items():
            metadata = dict(entry.metadata or {})
            if str(metadata.get("tenant_id") or "") != str(tenant_id):
                continue
            if api_key_id is not None and str(metadata.get("api_key_id") or "") != str(api_key_id):
                continue
            to_delete.append(str(entry_id))
        for entry_id in to_delete:
            self._entries.pop(entry_id, None)
        return len(to_delete)

    async def set_published(self, ids: List[str], published: bool) -> int:
        updated = 0
        for i in ids or []:
            e = self._entries.get(str(i))
            if e is None:
                continue
            e.published = bool(published)
            self._entries[str(i)] = e
            updated += 1
        return updated
