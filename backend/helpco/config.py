"""Configuration: packaged defaults, overridden by an optional helpco.toml."""
from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self.data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    @property
    def save_dir(self) -> Path:
        p = Path(self.get("world.save_dir"))
        if not p.is_absolute() and self.path is not None:
            p = self.path.parent / p
        return p

    def llm_provider(self) -> str:
        provider = self.get("llm.provider", "auto")
        if provider == "auto":
            return "openrouter" if os.environ.get(self.get("llm.api_key_env", "")) else "mock"
        return provider


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    data = tomllib.loads(DEFAULTS_PATH.read_text())
    cfg_path = None
    if path is None and Path("helpco.toml").exists():
        path = "helpco.toml"
    if path is not None:
        cfg_path = Path(path).resolve()
        data = _merge(data, tomllib.loads(cfg_path.read_text()))
    if overrides:
        data = _merge(data, overrides)
    return Config(data=data, path=cfg_path)
