from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    weights: str
    imgsz: int
    conf: float
    description: str


@dataclass(frozen=True)
class AppConfig:
    person_class_id: int
    models: dict[str, ModelConfig]


def _resolve_weights(weights: str) -> str:
    candidate = PROJECT_ROOT / weights
    if candidate.exists():
        return str(candidate)
    return weights


def load_app_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_models = raw.get("models") or {}
    if not raw_models:
        raise ValueError(f"No models configured in {path}")

    models: dict[str, ModelConfig] = {}
    for key, value in raw_models.items():
        models[key] = ModelConfig(
            key=key,
            name=str(value.get("name", key)),
            weights=_resolve_weights(str(value["weights"])),
            imgsz=int(value.get("imgsz", 640)),
            conf=float(value.get("conf", 0.35)),
            description=str(value.get("description", "")),
        )

    return AppConfig(
        person_class_id=int(raw.get("person_class_id", 0)),
        models=models,
    )


def describe_models(config: AppConfig) -> str:
    lines = []
    for key, model in config.models.items():
        lines.append(
            f"{key}: {model.name} | weights={model.weights} | "
            f"imgsz={model.imgsz} | conf={model.conf}"
        )
    return "\n".join(lines)
