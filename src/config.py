"""Configuration loading."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_DIR / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ws = Path(os.path.expanduser(cfg.get("agent", {}).get("workspace", "~/hearth/workspace")))
    ws.mkdir(parents=True, exist_ok=True)
    cfg["_workspace"] = ws

    data_dir = Path(os.path.expanduser(cfg.get("agent", {}).get("data_dir", "~/.hearth")))
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg["_data_dir"] = data_dir

    cfg["_project_dir"] = PROJECT_DIR
    return cfg
