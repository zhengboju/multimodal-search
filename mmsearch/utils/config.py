"""Tiny YAML config system: `_base_` inheritance, `a.b.c=value` CLI overrides, env-var expansion."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base:
        bases = base if isinstance(base, list) else [base]
        merged: dict = {}
        for b in bases:
            merged = deep_merge(merged, _load_raw((path.parent / b).resolve()))
        data = deep_merge(merged, data)
    return data


def set_by_dotted(d: dict, dotted: str, value: Any) -> None:
    """Set `a.b.c=value`; numeric parts index into lists (`suite.0.args.1=x`)."""
    parts = dotted.split(".")
    cur: Any = d
    for i, p in enumerate(parts[:-1]):
        if isinstance(cur, list):
            cur = cur[int(p)]
        else:
            cur = cur.setdefault(p, [] if parts[i + 1].isdigit() and not isinstance(cur.get(p), dict) else {})
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Cfg:
    from .paths import storage_dir

    os.environ.setdefault("MMS_STORAGE", str(storage_dir()))  # so configs can say ${MMS_STORAGE}/runs/...
    data = _load_raw(Path(path)) if path else {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        set_by_dotted(data, k.strip(), yaml.safe_load(v))
    return _wrap(data)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), f, allow_unicode=True, sort_keys=False)
