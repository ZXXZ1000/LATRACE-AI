from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from modules.media_graph_compiler.types import OperatorAssetRef


class LocalOperatorAssetStore:
    """Simple JSON-backed asset store for stage outputs."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            root_dir = ".artifacts/media_graph_compiler"
        self._root = Path(root_dir)

    def save_asset(
        self,
        *,
        asset_id: str,
        asset_type: str,
        payload: Mapping[str, Any],
        source_id: Optional[str] = None,
        created_by_run_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OperatorAssetRef:
        out_dir = self._root / asset_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asset_id}.json"
        out_path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return OperatorAssetRef(
            asset_id=asset_id,
            asset_type=asset_type,
            source_id=source_id,
            file_path=str(out_path),
            created_by_run_id=created_by_run_id,
            metadata=dict(metadata or {}),
        )

    def load_asset(self, asset_ref: OperatorAssetRef) -> Dict[str, Any]:
        if asset_ref.file_path:
            return json.loads(Path(asset_ref.file_path).read_text(encoding="utf-8"))
        raise NotImplementedError("blob-backed assets are not yet implemented")
