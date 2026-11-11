"""Configuration and artifact helpers shared by every independent step."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    from .configuration import read_config, validate_config

    config_path = Path(path).resolve()
    config = read_config(config_path)
    validate_config(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    for name, value in config["paths"].items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        config["paths"][name] = str(candidate.resolve())
    return config


def experiment_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["runs_dir"]) / config["experiment_name"]


def step_dir(config: dict[str, Any], number: int, name: str) -> Path:
    return experiment_dir(config) / f"{number:02d}_{name}"


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def make_new_directory(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. Remove it or change experiment_name."
        )
    path.mkdir(parents=True)
    return path


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        return item

    path.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(name)
