from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = REPO_ROOT / "modules" / "memory" / "config" / "memory.config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _set_dot_path(root: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in str(dotted).split(".") if p]
    if not parts:
        raise ValueError("override path must be non-empty")
    cur: Dict[str, Any] = root
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def materialize(base_cfg: Dict[str, Any], experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = yaml.safe_load(yaml.safe_dump(base_cfg, allow_unicode=True)) or {}
    overrides = experiment_cfg.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("experiment overrides must be a mapping")
    for path, value in overrides.items():
        _set_dot_path(merged, str(path), value)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a full config from baseline + experiment delta.")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Baseline config YAML path.")
    parser.add_argument("--experiment", required=True, help="Experiment delta YAML path.")
    parser.add_argument("--output", help="Optional output YAML path. Prints to stdout if omitted.")
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    exp_path = Path(args.experiment).resolve()
    merged = materialize(_load_yaml(base_path), _load_yaml(exp_path))
    rendered = yaml.safe_dump(merged, allow_unicode=True, sort_keys=False)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote materialized config to {out_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
