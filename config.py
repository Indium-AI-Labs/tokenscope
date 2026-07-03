from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONFIG_FILE_NAMES = (".tokenscoperc", "tokenscope.json")


@dataclass
class TokenScopeConfig:
    """Persistent user configuration loaded from ``.tokenscoperc`` or ``tokenscope.json``."""

    default_tokenizer: str | None = None
    default_compare_tokenizer: str | None = None
    default_budget: int | None = None
    default_export_format: str = "json"
    default_hub_model: str | None = None
    hub_cache_dir: str | None = None
    encode_special_tokens: bool = False
    default_corpus_path: str | None = None
    default_batch_path: str | None = None


def load_config(workspace: Path | None = None) -> TokenScopeConfig:
    """Search for a config file in *workspace* (defaults to CWD), then ``~``.

    Returns a default config if nothing is found.
    """
    search_dirs = [workspace or Path.cwd(), Path.home()]
    for directory in search_dirs:
        for name in CONFIG_FILE_NAMES:
            path = directory / name
            if path.is_file():
                try:
                    return _parse_config(path)
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
    return TokenScopeConfig()


def save_config(config: TokenScopeConfig, path: Path | None = None) -> None:
    """Persist *config* to *path* (defaults to CWD / ``.tokenscoperc``)."""
    target = path or (Path.cwd() / CONFIG_FILE_NAMES[0])
    target.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_config(path: Path) -> TokenScopeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    return TokenScopeConfig(
        default_tokenizer=_optional_str(payload.get("default_tokenizer")),
        default_compare_tokenizer=_optional_str(payload.get("default_compare_tokenizer")),
        default_budget=_optional_positive_int(payload.get("default_budget")),
        default_export_format=_export_format_choice(payload.get("default_export_format")),
        default_hub_model=_optional_str(payload.get("default_hub_model")),
        hub_cache_dir=_optional_str(payload.get("hub_cache_dir")),
        encode_special_tokens=bool(payload.get("encode_special_tokens", False)),
        default_corpus_path=_optional_str(payload.get("default_corpus_path")),
        default_batch_path=_optional_str(payload.get("default_batch_path")),
    )


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
            return parsed if parsed > 0 else None
        except ValueError:
            return None
    return None


def _export_format_choice(value: Any) -> str:
    if isinstance(value, str) and value.lower() in ("json", "csv", "md", "html"):
        return value.lower()
    return "json"
