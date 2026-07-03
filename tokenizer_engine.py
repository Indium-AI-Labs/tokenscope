from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer, ByteLevelBPETokenizer

from version import app_version_label


class TokenizerLoadError(ValueError):
    """Raised when a local tokenizer path cannot be loaded."""


class ChatTemplateError(ValueError):
    """Raised when a tokenizer chat template cannot be rendered."""


@dataclass(frozen=True)
class TokenSpan:
    index: int
    token: str
    token_id: int
    offset_start: int
    offset_end: int
    text: str
    byte_repr: str


@dataclass(frozen=True)
class TokenStats:
    vocab_size: int
    token_count: int
    character_count: int
    chars_per_token: float
    compression_ratio: float
    unique_token_count: int
    most_frequent_token_id: int | None
    avg_token_length: float


@dataclass(frozen=True)
class TokenizationResult:
    input_text: str
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    spans: tuple[TokenSpan, ...]
    stats: TokenStats

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "spans": [asdict(span) for span in self.spans],
            "stats": asdict(self.stats),
        }


@dataclass
class MergeNode:
    text: str
    rank: int | None = None
    left: "MergeNode | None" = None
    right: "MergeNode | None" = None


class TokenizerEngine:
    CONFIG_FILE_NAMES = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        source_path: Path,
        name: str,
        tokenizer_type: str,
        raw_config: dict[str, Any] | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        merge_ranks: dict[tuple[str, str], int] | None = None,
        loaded_tokenizer_file: Path | None = None,
        config_files_found: Sequence[Path] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_path = source_path
        self.name = name
        self.tokenizer_type = tokenizer_type
        self.raw_config = raw_config or {}
        self.tokenizer_config = tokenizer_config or {}
        self.vocab = tokenizer.get_vocab()
        self.vocab_size = tokenizer.get_vocab_size()
        self.merge_ranks = merge_ranks or {}
        self.loaded_tokenizer_file = loaded_tokenizer_file.resolve() if loaded_tokenizer_file else None
        self.config_files_found = tuple(path.resolve() for path in (config_files_found or ()))
        self._vocab_by_id = sorted(self.vocab.items(), key=lambda item: item[1])

    @classmethod
    def load(cls, path_value: str | Path) -> "TokenizerEngine":
        source = Path(path_value).expanduser()
        if not source.exists():
            raise TokenizerLoadError(f"Path does not exist: {source}")

        if source.is_file():
            tokenizer_file = source
            root = source.parent
        else:
            root = source
            tokenizer_file = cls._find_tokenizer_json(root)

        config = cls._read_optional_json(root / "tokenizer_config.json")

        if tokenizer_file is None and config:
            config_file = config.get("tokenizer_file")
            if isinstance(config_file, str):
                candidate = (root / config_file).resolve()
                if candidate.exists():
                    tokenizer_file = candidate

        raw_tokenizer = cls._read_optional_json(tokenizer_file) if tokenizer_file else None
        tokenizer = cls._load_tokenizer(root, tokenizer_file)
        tokenizer_type = cls._detect_type(tokenizer, raw_tokenizer)
        name = cls._detect_name(root, config, raw_tokenizer)
        merge_ranks = cls._extract_merge_ranks(raw_tokenizer, root)
        loaded_tokenizer_file = cls._detect_loaded_tokenizer_file(root, tokenizer_file)
        config_files_found = cls._discover_config_files(root, tokenizer_file)

        return cls(
            tokenizer=tokenizer,
            source_path=root.resolve(),
            name=name,
            tokenizer_type=tokenizer_type,
            raw_config=raw_tokenizer,
            tokenizer_config=config,
            merge_ranks=merge_ranks,
            loaded_tokenizer_file=loaded_tokenizer_file,
            config_files_found=config_files_found,
        )

    @classmethod
    def load_from_hub(
        cls,
        model_id: str,
        cache_dir: str | Path | None = None,
    ) -> "TokenizerEngine":
        """Download tokenizer files from HuggingFace Hub and load them locally.

        Requires the ``huggingface_hub`` package. The download is cached so
        subsequent loads of the same *model_id* are instant.
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TokenizerLoadError(
                "huggingface_hub is required for Hub downloads. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        allow_patterns = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "config.json",
        ]

        logger = logging.getLogger("tokenscope")
        logger.info("Downloading tokenizer from Hub: %s", model_id)

        try:
            local_dir = snapshot_download(
                repo_id=model_id,
                allow_patterns=allow_patterns,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception as exc:
            raise TokenizerLoadError(
                f"Failed to download tokenizer from Hub ({model_id}): {exc}"
            ) from exc

        return cls.load(local_dir)

    @staticmethod
    def _find_tokenizer_json(root: Path) -> Path | None:
        direct = root / "tokenizer.json"
        if direct.exists():
            return direct
        matches = sorted(root.glob("**/tokenizer.json"))
        return matches[0] if matches else None

    @staticmethod
    def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TokenizerLoadError(f"Invalid JSON in {path}: {exc}") from exc

    @staticmethod
    def _load_tokenizer(root: Path, tokenizer_file: Path | None) -> Tokenizer:
        if tokenizer_file is not None:
            try:
                return Tokenizer.from_file(str(tokenizer_file))
            except Exception as exc:  # tokenizers raises several Rust-backed exceptions.
                raise TokenizerLoadError(f"Could not load tokenizer file {tokenizer_file}: {exc}") from exc

        vocab_txt = root / "vocab.txt"
        if vocab_txt.exists():
            return BertWordPieceTokenizer(str(vocab_txt), lowercase=False)

        vocab_json = root / "vocab.json"
        merges_txt = root / "merges.txt"
        if vocab_json.exists() and merges_txt.exists():
            return ByteLevelBPETokenizer(str(vocab_json), str(merges_txt))

        config = root / "tokenizer_config.json"
        if config.exists():
            raise TokenizerLoadError(
                "Found tokenizer_config.json but no loadable tokenizer.json, "
                "tokenizer_file entry, vocab.txt, or vocab.json/merges.txt pair."
            )

        raise TokenizerLoadError(
            "Missing tokenizer files. Expected tokenizer.json, tokenizer_config.json, "
            "vocab.txt, or vocab.json plus merges.txt."
        )

    @classmethod
    def _detect_loaded_tokenizer_file(cls, root: Path, tokenizer_file: Path | None) -> Path | None:
        if tokenizer_file is not None:
            return tokenizer_file
        if (root / "vocab.txt").exists():
            return root / "vocab.txt"
        if (root / "vocab.json").exists():
            return root / "vocab.json"
        return None

    @classmethod
    def _discover_config_files(cls, root: Path, tokenizer_file: Path | None) -> tuple[Path, ...]:
        found: list[Path] = []
        if tokenizer_file is not None and tokenizer_file.exists():
            found.append(tokenizer_file)
        for name in cls.CONFIG_FILE_NAMES:
            path = root / name
            if path.exists() and path not in found:
                found.append(path)
        return tuple(found)

    @staticmethod
    def _detect_type(tokenizer: Tokenizer, raw_tokenizer: dict[str, Any] | None) -> str:
        model = (raw_tokenizer or {}).get("model", {})
        model_type = model.get("type")
        if isinstance(model_type, str) and model_type:
            return model_type
        return type(tokenizer.model).__name__

    @staticmethod
    def _detect_name(
        root: Path,
        config: dict[str, Any] | None,
        raw_tokenizer: dict[str, Any] | None,
    ) -> str:
        for source in (config, raw_tokenizer):
            if not source:
                continue
            for key in ("name_or_path", "tokenizer_class", "name"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return Path(value).name
        return root.name or "local-tokenizer"

    @classmethod
    def _extract_merge_ranks(
        cls,
        raw_tokenizer: dict[str, Any] | None,
        root: Path,
    ) -> dict[tuple[str, str], int]:
        merges: Iterable[Any] | None = None
        model = (raw_tokenizer or {}).get("model", {})
        if model.get("type") == "BPE":
            merges = model.get("merges")

        if merges is None and (root / "merges.txt").exists():
            lines = (root / "merges.txt").read_text(encoding="utf-8").splitlines()
            merges = [line for line in lines if line and not line.startswith("#")]

        ranks: dict[tuple[str, str], int] = {}
        if not merges:
            return ranks

        for rank, item in enumerate(merges):
            pair: tuple[str, str] | None = None
            if isinstance(item, str):
                parts = item.split()
                if len(parts) >= 2:
                    pair = (parts[0], parts[1])
            elif isinstance(item, Sequence) and len(item) >= 2:
                pair = (str(item[0]), str(item[1]))
            if pair is not None and pair not in ranks:
                ranks[pair] = rank
        return ranks

    @property
    def is_bpe(self) -> bool:
        return self.tokenizer_type.lower() == "bpe" or bool(self.merge_ranks)

    @property
    def is_byte_level(self) -> bool:
        raw_pre_tokenizer = (self.raw_config or {}).get("pre_tokenizer")
        if self._raw_component_has_type(raw_pre_tokenizer, "ByteLevel"):
            return True
        raw_decoder = (self.raw_config or {}).get("decoder")
        if self._raw_component_has_type(raw_decoder, "ByteLevel"):
            return True
        pre_tokenizer = self.tokenizer.pre_tokenizer
        decoder = self.tokenizer.decoder
        return (
            type(pre_tokenizer).__name__ == "ByteLevel"
            or type(decoder).__name__ == "ByteLevel"
        )

    @property
    def header_label(self) -> str:
        return (
            f"{app_version_label()} | {self.name} | "
            f"{self.tokenizer_type} | vocab: {self.vocab_size}"
        )

    @property
    def chat_template(self) -> str | None:
        return self._normalize_chat_template(
            self.tokenizer_config.get("chat_template")
            or self.raw_config.get("chat_template")
        )

    @staticmethod
    def _normalize_chat_template(value: Any) -> str | None:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            template = value.get("template")
            return template if isinstance(template, str) and template else None
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("name") == "default":
                    template = item.get("template")
                    if isinstance(template, str) and template:
                        return template
            for item in value:
                if isinstance(item, dict):
                    template = item.get("template")
                    if isinstance(template, str) and template:
                        return template
                elif isinstance(item, str) and item:
                    return item
        return None

    def render_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        template = self.chat_template
        if not template:
            raise ChatTemplateError("Tokenizer does not define a chat_template.")

        environment = SandboxedEnvironment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        environment.globals["raise_exception"] = self._raise_chat_template_exception
        environment.globals["strftime_now"] = lambda _format: ""

        context: dict[str, Any] = {
            "messages": list(messages),
            "add_generation_prompt": add_generation_prompt,
            **self.special_token_variables(),
        }
        try:
            return environment.from_string(template).render(**context)
        except ChatTemplateError:
            raise
        except Exception as exc:
            raise ChatTemplateError(f"Chat template render failed: {exc}") from exc

    def special_token_variables(self) -> dict[str, str | None]:
        values: dict[str, str | None] = {}
        for source in (self.tokenizer_config, self.raw_config):
            for key, value in source.items():
                if key.endswith("_token") and isinstance(value, str):
                    values[key] = value
                elif key.endswith("_token") and isinstance(value, dict):
                    content = value.get("content")
                    if isinstance(content, str):
                        values[key] = content
        for token_id, token in self.tokenizer.get_added_tokens_decoder().items():
            content = str(getattr(token, "content", str(token)))
            key = content.strip("[]<>").lower().replace("-", "_")
            if key:
                values.setdefault(f"{key}_token", content)
        return values

    @staticmethod
    def _raise_chat_template_exception(message: str) -> None:
        raise ChatTemplateError(message)

    def encode(self, text: str, *, encode_special_tokens: bool = False) -> TokenizationResult:
        if not text:
            return self._empty_result()

        previous_encode_special = getattr(self.tokenizer, "encode_special_tokens", None)
        if previous_encode_special is not None:
            self.tokenizer.encode_special_tokens = encode_special_tokens
        try:
            encoding = self.tokenizer.encode(text)
        finally:
            if previous_encode_special is not None:
                self.tokenizer.encode_special_tokens = previous_encode_special
        tokens = tuple(encoding.tokens)
        ids = tuple(int(token_id) for token_id in encoding.ids)
        offsets = tuple(encoding.offsets)

        spans: list[TokenSpan] = []
        for index, (token, token_id) in enumerate(zip(tokens, ids, strict=True)):
            start, end = offsets[index] if index < len(offsets) else (0, 0)
            span_text = self._safe_slice(text, start, end) or token
            spans.append(
                TokenSpan(
                    index=index,
                    token=token,
                    token_id=token_id,
                    offset_start=start,
                    offset_end=end,
                    text=span_text,
                    byte_repr=self._byte_repr(span_text),
                )
            )

        return TokenizationResult(
            input_text=text,
            tokens=tokens,
            token_ids=ids,
            spans=tuple(spans),
            stats=self._stats(text, tokens, ids),
        )

    def _empty_result(self) -> TokenizationResult:
        return TokenizationResult(
            input_text="",
            tokens=(),
            token_ids=(),
            spans=(),
            stats=TokenStats(
                vocab_size=self.vocab_size,
                token_count=0,
                character_count=0,
                chars_per_token=0.0,
                compression_ratio=0.0,
                unique_token_count=0,
                most_frequent_token_id=None,
                avg_token_length=0.0,
            ),
        )

    @staticmethod
    def _safe_slice(text: str, start: int, end: int) -> str:
        if start < 0 or end < start:
            return ""
        try:
            return text[start:end]
        except IndexError:
            return ""

    @staticmethod
    def _byte_repr(text: str) -> str:
        return " ".join(f"{byte:02x}" for byte in text.encode("utf-8", errors="replace"))

    def _stats(self, text: str, tokens: Sequence[str], ids: Sequence[int]) -> TokenStats:
        token_count = len(ids)
        character_count = len(text)
        token_lengths = np.array([len(token) for token in tokens], dtype=np.float64)
        counts = Counter(ids)
        most_frequent = counts.most_common(1)[0][0] if counts else None
        return TokenStats(
            vocab_size=self.vocab_size,
            token_count=token_count,
            character_count=character_count,
            chars_per_token=(character_count / token_count) if token_count else 0.0,
            compression_ratio=(token_count / character_count) if character_count else 0.0,
            unique_token_count=len(counts),
            most_frequent_token_id=most_frequent,
            avg_token_length=float(np.mean(token_lengths)) if token_lengths.size else 0.0,
        )

    def search_vocab(self, query: str, limit: int = 200) -> list[tuple[str, int]]:
        needle = query.casefold()
        if not needle:
            return self._vocab_by_id[: min(limit, len(self._vocab_by_id))]
        matches: list[tuple[str, int]] = []
        for token, token_id in self._vocab_by_id:
            if needle in token.casefold():
                matches.append((token, token_id))
                if len(matches) >= limit:
                    break
        return matches

    def token_to_id(self, token: str) -> int | None:
        token_id = self.tokenizer.token_to_id(token)
        return int(token_id) if token_id is not None else None

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def benchmark(self, text: str, *, iterations: int = 10) -> dict[str, object]:
        """Time tokenization of *text* over *iterations* runs.

        Returns a dict suitable for constructing a ``BenchmarkResult``.
        """
        if not text:
            return {
                "source": "primary",
                "tokenizer_name": self.name,
                "input_chars": 0,
                "output_tokens": 0,
                "iterations": 0,
                "total_seconds": 0.0,
                "mean_seconds": 0.0,
                "tokens_per_second": 0.0,
                "chars_per_second": 0.0,
            }

        # Warmup run (excluded from timing).
        warmup = self.encode(text)
        token_count = warmup.stats.token_count
        char_count = len(text)

        start = time.perf_counter()
        for _ in range(iterations):
            self.encode(text)
        elapsed = time.perf_counter() - start

        mean = elapsed / iterations if iterations else elapsed
        return {
            "source": "primary",
            "tokenizer_name": self.name,
            "input_chars": char_count,
            "output_tokens": token_count,
            "iterations": iterations,
            "total_seconds": elapsed,
            "mean_seconds": mean,
            "tokens_per_second": (token_count / mean) if mean > 0 else 0.0,
            "chars_per_second": (char_count / mean) if mean > 0 else 0.0,
        }

    def render_bpe_merge_tree(self, result: TokenizationResult | None, max_tokens: int = 24) -> str:
        if not self.is_bpe:
            return "Merge tree only available for BPE tokenizers."
        if not result or not result.spans:
            return "Type text to inspect BPE merges."

        lines: list[str] = []
        for span in result.spans[:max_tokens]:
            lines.extend(self._render_single_span_lines(span))
            lines.append("")

        if lines and lines[-1] == "":
            lines.pop()

        if len(result.spans) > max_tokens:
            lines.append(f"... {len(result.spans) - max_tokens} more tokens omitted")
        return "\n".join(lines)

    def render_single_token_merge_tree(self, span: TokenSpan | None) -> str:
        if not self.is_bpe:
            return "Merge tree only available for BPE tokenizers."
        if span is None:
            return "No token selected."
        return "\n".join(self._render_single_span_lines(span))

    def _render_single_span_lines(self, span: TokenSpan) -> list[str]:
        lines = [f"[{span.index}] {span.token!r} id={span.token_id}"]
        node, exact = self._build_bpe_tree(span.token, span.text)
        if node is None:
            lines.append("  no merge history found")
            return lines
        self._render_node(node, lines, prefix="  ")
        if not exact:
            lines.append("  note: token contains segments not present in merge ranks")
        return lines

    def _build_bpe_tree(self, token: str, span_text: str | None = None) -> tuple[MergeNode | None, bool]:
        if not token:
            return None, False

        nodes = [MergeNode(symbol) for symbol in self._initial_symbols(token, span_text)]
        if len(nodes) == 1:
            return nodes[0], True

        exact = True
        while len(nodes) > 1:
            best_index: int | None = None
            best_rank: int | None = None
            for index in range(len(nodes) - 1):
                rank = self.merge_ranks.get((nodes[index].text, nodes[index + 1].text))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index

            if best_index is None:
                exact = False
                left = nodes.pop(0)
                right = nodes.pop(0)
                nodes.insert(0, MergeNode(left.text + right.text, None, left, right))
                continue

            left = nodes[best_index]
            right = nodes[best_index + 1]
            nodes[best_index : best_index + 2] = [
                MergeNode(left.text + right.text, best_rank, left, right)
            ]

        return nodes[0], exact

    def _initial_symbols(self, token: str, span_text: str | None = None) -> list[str]:
        if token.startswith("<") and token.endswith(">"):
            return [token]
        if self.is_byte_level:
            token_symbols = list(token)
            if self._can_reconstruct_token_from_symbols(token_symbols, token):
                return token_symbols
            if span_text:
                return self.byte_level_symbols(span_text)
        return list(token)

    @classmethod
    def byte_level_symbols(cls, text: str) -> list[str]:
        byte_encoder = cls._byte_encoder()
        return [byte_encoder[byte] for byte in text.encode("utf-8", errors="replace")]

    @staticmethod
    def _can_reconstruct_token_from_symbols(symbols: Sequence[str], token: str) -> bool:
        return "".join(symbols) == token

    @staticmethod
    def _raw_component_has_type(component: Any, type_name: str) -> bool:
        if isinstance(component, dict):
            if component.get("type") == type_name:
                return True
            return any(TokenizerEngine._raw_component_has_type(value, type_name) for value in component.values())
        if isinstance(component, list):
            return any(TokenizerEngine._raw_component_has_type(value, type_name) for value in component)
        return False

    @staticmethod
    def _byte_encoder() -> dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(161, 173))
            + list(range(174, 256))
        )
        cs = bs[:]
        n = 0
        for byte in range(256):
            if byte not in bs:
                bs.append(byte)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, (chr(value) for value in cs), strict=True))

    @classmethod
    def _render_node(cls, node: MergeNode, lines: list[str], prefix: str) -> None:
        rank = f" rank={node.rank}" if node.rank is not None else ""
        lines.append(f"{prefix}+- {node.text!r}{rank}")
        if node.left is not None:
            cls._render_node(node.left, lines, prefix + "|  ")
        if node.right is not None:
            cls._render_node(node.right, lines, prefix + "   ")
