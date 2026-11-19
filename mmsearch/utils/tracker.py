"""Experiment tracking: always writes local JSONL; mirrors to wandb when MMS_WANDB=1 and wandb is installed."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import save_config
from .misc import get_logger, write_json
from .paths import repo_root

log = get_logger(__name__)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class Tracker:
    def __init__(self, run_dir: str | Path, config: dict | None = None, name: str | None = None, enabled: bool = True):
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self._wandb = None
        self._summary: dict[str, Any] = {}
        if not enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            save_config(config, self.run_dir / "config.yaml")
        write_json({"git_commit": git_commit(), "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}, self.run_dir / "meta.json")
        self._fh = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        if os.environ.get("MMS_WANDB") == "1":
            try:
                import wandb

                self._wandb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "mmsearch"),
                    name=name or self.run_dir.name,
                    config=config,
                    dir=str(self.run_dir),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("wandb disabled: %r", e)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        rec = {"step": step, "time": time.time(), **metrics}
        self._fh.write(json.dumps(rec, default=float) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._summary.update(metrics)
        write_json(self._summary, self.run_dir / "summary.json")
        if self._wandb is not None:
            self._wandb.summary.update(metrics)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()
