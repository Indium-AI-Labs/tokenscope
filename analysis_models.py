from __future__ import annotations

import csv
import html
import io
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from rich.cells import cell_len

from compare_engine import CompareResult
from tokenizer_engine import ChatTemplateError, TokenSpan, TokenizationResult, TokenizerEngine

SearchMode = Literal["text", "token", "id"]
ExportFormat = Literal["json", "csv", "md", "html"]

SUPPORTED_CORPUS_SUFFIXES = {".txt", ".md", ".jsonl", ".json", ".csv"}
RECENT_TOKENIZER_FILE = ".tokenscope_recent.json"
PRICING_PROFILE_FILE = ".tokenscope_pricing.json"
MAX_RECENT_TOKENIZERS = 10
PROJECT_SCHEMA_VERSION = 2
REGRESSION_SCHEMA_VERSION = 1
ZERO_WIDTH_CODEPOINTS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
}


@dataclass(frozen=True)
class TokenInspection:
    source: str
    index: int
    text_span: str
    token_string: str
    token_id: int
    offset_start: int
    offset_end: int
    byte_repr: str
    display_width: int
    frequency_in_input: int
    in_vocab: bool
    merge_tree: str


@dataclass(frozen=True)
class DecodeRoundTrip:
    source: str
    original_text: str
    decoded_text: str
    exact_match: bool
    first_difference: int | None


@dataclass(frozen=True)
class SpecialTokenInfo:
    token_id: int
    token: str
    special: bool | None
    normalized: bool | None
    single_word: bool | None
    lstrip: bool | None
    rstrip: bool | None


@dataclass(frozen=True)
class PromptBudget:
    source: str
    limit: int
    used_tokens: int
    remaining_tokens: int
    percent_used: float


@dataclass(frozen=True)
class TokenSearchMatch:
    source: str
    index: int
    token: str
    token_id: int
    text: str
    offset_start: int
    offset_end: int


@dataclass(frozen=True)
class CorpusLineResult:
    file_path: str
    line_number: int
    preview: str
    token_count: int
    char_count: int


@dataclass(frozen=True)
class CorpusAnalysisResult:
    source_path: str
    total_files: int
    skipped_files: int
    unreadable_files: int
    total_chars: int
    total_tokens: int
    chars_per_token: float
    top_token_ids: tuple[tuple[int, int], ...]
    top_token_strings: tuple[tuple[str, int], ...]
    longest_lines: tuple[CorpusLineResult, ...]


@dataclass(frozen=True)
class TokenizerMetadata:
    model_type: str
    source_path: str
    vocab_size: int
    normalizer: str
    pre_tokenizer: str
    decoder: str
    post_processor: str
    padding: str
    truncation: str
    config_files_found: tuple[str, ...]
    loaded_tokenizer_file: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatTemplateRender:
    source: str
    has_template: bool
    rendered_text: str
    token_count: int
    remaining_tokens: int | None
    percent_used: float | None
    error: str | None = None


@dataclass(frozen=True)
class ChatBudgetResult:
    messages: tuple[ChatMessage, ...]
    add_generation_prompt: bool
    primary: ChatTemplateRender | None
    compare: ChatTemplateRender | None


@dataclass(frozen=True)
class PipelinePreToken:
    text: str
    offset_start: int
    offset_end: int


@dataclass(frozen=True)
class PipelineDebugResult:
    source: str
    input_text: str
    normalized_text: str
    normalizer_error: str | None
    pre_tokens: tuple[PipelinePreToken, ...]
    pre_tokenizer_error: str | None
    token_count: int
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchPromptRow:
    file_path: str
    line_number: int
    preview: str
    char_count: int
    primary_tokens: int
    compare_tokens: int | None
    token_delta: int | None
    budget_exceeded: bool


@dataclass(frozen=True)
class BatchPromptAnalysisResult:
    source_path: str
    total_prompts: int
    total_files: int
    skipped_files: int
    unreadable_files: int
    primary_total_tokens: int
    compare_total_tokens: int | None
    token_delta: int | None
    avg_tokens: float
    p50_tokens: float
    p95_tokens: float
    max_tokens: int
    budget_failures: int
    longest_prompts: tuple[BatchPromptRow, ...]


@dataclass(frozen=True)
class CorpusCompareRow:
    file_path: str
    line_number: int
    preview: str
    char_count: int
    primary_tokens: int
    compare_tokens: int
    token_delta: int


@dataclass(frozen=True)
class CorpusCompareResult:
    source_path: str
    total_files: int
    skipped_files: int
    unreadable_files: int
    total_chars: int
    primary_total_tokens: int
    compare_total_tokens: int
    token_delta: int
    primary_chars_per_token: float
    compare_chars_per_token: float
    biggest_savings: tuple[CorpusCompareRow, ...]
    biggest_regressions: tuple[CorpusCompareRow, ...]


@dataclass(frozen=True)
class PricingProfile:
    name: str
    input_per_million: float
    output_per_million: float
    estimated_output_tokens: int


def default_pricing_profile() -> PricingProfile:
    return PricingProfile("custom", 0.0, 0.0, 0)


@dataclass(frozen=True)
class ProjectState:
    version: int
    tokenizer_path: str | None
    compare_tokenizer_path: str | None
    input_text: str
    encode_special_tokens: bool
    export_format: str
    budget_limit: int | None
    chat_messages: tuple[ChatMessage, ...]
    add_generation_prompt: bool
    corpus_path: str | None
    batch_path: str | None
    active_tab: str
    selected_source: str
    pricing_profile: PricingProfile = field(default_factory=default_pricing_profile)


@dataclass(frozen=True)
class TokenizerDiffItem:
    kind: str
    key: str
    primary: str
    compare: str


@dataclass(frozen=True)
class TokenizerDiffResult:
    primary_name: str
    compare_name: str
    metadata_differences: int
    component_differences: int
    vocab_primary_only: int
    vocab_compare_only: int
    vocab_id_differences: int
    added_token_differences: int
    special_token_differences: int
    merge_primary_only: int
    merge_compare_only: int
    merge_rank_differences: int
    items: tuple[TokenizerDiffItem, ...]


@dataclass(frozen=True)
class PackingSegment:
    source: str
    index: int
    text: str
    token_count: int
    kept: bool


@dataclass(frozen=True)
class PackingResult:
    strategy: str
    limit: int
    source_tokens: int
    kept_tokens: int
    dropped_tokens: int
    remaining_tokens: int
    kept_text: str
    dropped_text: str
    segments: tuple[PackingSegment, ...]


@dataclass(frozen=True)
class RegressionCase:
    name: str
    input_text: str
    encode_special_tokens: bool
    expected_ids: tuple[int, ...] | None
    expected_tokens: tuple[str, ...] | None


@dataclass(frozen=True)
class RegressionCaseResult:
    name: str
    passed: bool
    input_text: str
    actual_ids: tuple[int, ...]
    actual_tokens: tuple[str, ...]
    expected_ids: tuple[int, ...] | None
    expected_tokens: tuple[str, ...] | None
    mismatch: str


@dataclass(frozen=True)
class RegressionSuiteResult:
    suite_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    cases: tuple[RegressionCaseResult, ...]


@dataclass(frozen=True)
class UnicodeCharacterInfo:
    index: int
    character: str
    codepoint: str
    name: str
    category: str
    utf8_bytes: str
    is_whitespace: bool
    is_control: bool
    is_zero_width: bool
    is_combining_mark: bool
    nfc_changes: bool
    nfkc_changes: bool


@dataclass(frozen=True)
class UnicodeInspectionResult:
    input_text: str
    character_count: int
    zero_width_count: int
    control_count: int
    combining_mark_count: int
    nfc_changed: bool
    nfkc_changed: bool
    characters: tuple[UnicodeCharacterInfo, ...]


@dataclass(frozen=True)
class TextUnit:
    source: str
    line_number: int
    text: str


@dataclass(frozen=True)
class RAGChunk:
    source: str
    chunk_index: int
    text: str
    token_count: int
    overflow: bool


@dataclass(frozen=True)
class RAGChunkingResult:
    max_tokens: int
    overlap_tokens: int
    mode: str
    source_units: int
    chunk_count: int
    total_tokens: int
    avg_tokens: float
    p50_tokens: float
    p95_tokens: float
    max_chunk_tokens: int
    overflow_chunks: int
    wasted_budget: int
    chunks: tuple[RAGChunk, ...]


@dataclass(frozen=True)
class HistogramBucket:
    label: str
    count: int
    bar: str


@dataclass(frozen=True)
class DistributionResult:
    source: str
    count: int
    min_tokens: int
    p50_tokens: float
    p90_tokens: float
    p95_tokens: float
    p99_tokens: float
    max_tokens: int
    avg_tokens: float
    budget_limit: int | None
    budget_failures: int
    budget_failure_rate: float
    histogram: tuple[HistogramBucket, ...]


@dataclass(frozen=True)
class CostEstimate:
    profile_name: str
    input_tokens: int
    estimated_output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass(frozen=True)
class RepairSuggestion:
    severity: str
    issue: str
    suggestion: str


@dataclass(frozen=True)
class TokenizerRepairResult:
    tokenizer_path: str
    suggestion_count: int
    suggestions: tuple[RepairSuggestion, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    source: str
    tokenizer_name: str
    input_chars: int
    output_tokens: int
    iterations: int
    total_seconds: float
    mean_seconds: float
    tokens_per_second: float
    chars_per_second: float


@dataclass(frozen=True)
class BenchmarkComparison:
    primary: BenchmarkResult
    compare: BenchmarkResult | None
    speedup: float | None


def inspect_token(
    source: str,
    engine: TokenizerEngine | None,
    result: TokenizationResult | None,
    index: int,
) -> TokenInspection | None:
    if engine is None or result is None or not result.spans:
        return None
    bounded = max(0, min(index, len(result.spans) - 1))
    span = result.spans[bounded]
    frequencies = Counter(result.token_ids)
    return TokenInspection(
        source=source,
        index=span.index,
        text_span=span.text,
        token_string=span.token,
        token_id=span.token_id,
        offset_start=span.offset_start,
        offset_end=span.offset_end,
        byte_repr=span.byte_repr,
        display_width=cell_len(_visible(span.text or span.token)),
        frequency_in_input=frequencies[span.token_id],
        in_vocab=engine.token_to_id(span.token) == span.token_id,
        merge_tree=engine.render_single_token_merge_tree(span),
    )


def decode_round_trip(
    source: str,
    engine: TokenizerEngine | None,
    result: TokenizationResult | None,
) -> DecodeRoundTrip | None:
    if engine is None or result is None:
        return None
    decoded = engine.decode(result.token_ids, skip_special_tokens=False)
    return DecodeRoundTrip(
        source=source,
        original_text=result.input_text,
        decoded_text=decoded,
        exact_match=decoded == result.input_text,
        first_difference=first_difference_offset(result.input_text, decoded),
    )


def first_difference_offset(left: str, right: str) -> int | None:
    if left == right:
        return None
    for index, (left_char, right_char) in enumerate(zip(left, right, strict=False)):
        if left_char != right_char:
            return index
    return min(len(left), len(right))


def extract_special_tokens(engine: TokenizerEngine | None) -> tuple[SpecialTokenInfo, ...]:
    if engine is None:
        return ()

    by_id: dict[int, SpecialTokenInfo] = {}
    decoder = engine.tokenizer.get_added_tokens_decoder()
    for token_id, token in decoder.items():
        info = _special_info_from_added_token(int(token_id), token)
        by_id[info.token_id] = info

    for item in _iter_raw_added_tokens(engine.raw_config):
        token_id = item.get("id")
        content = item.get("content")
        if not isinstance(token_id, int) or not isinstance(content, str):
            continue
        current = by_id.get(token_id)
        by_id[token_id] = SpecialTokenInfo(
            token_id=token_id,
            token=current.token if current else content,
            special=_bool_or_none(item.get("special"), current.special if current else None),
            normalized=_bool_or_none(item.get("normalized"), current.normalized if current else None),
            single_word=_bool_or_none(item.get("single_word"), current.single_word if current else None),
            lstrip=_bool_or_none(item.get("lstrip"), current.lstrip if current else None),
            rstrip=_bool_or_none(item.get("rstrip"), current.rstrip if current else None),
        )

    return tuple(sorted(by_id.values(), key=lambda info: info.token_id))


def calculate_prompt_budget(
    source: str,
    result: TokenizationResult | None,
    limit: int,
) -> PromptBudget | None:
    if result is None or limit <= 0:
        return None
    used = result.stats.token_count
    return PromptBudget(
        source=source,
        limit=limit,
        used_tokens=used,
        remaining_tokens=limit - used,
        percent_used=(used / limit) * 100.0,
    )


def search_tokens(
    source: str,
    result: TokenizationResult | None,
    query: str,
    mode: SearchMode,
) -> tuple[TokenSearchMatch, ...]:
    if result is None or not query:
        return ()

    matches: list[TokenSearchMatch] = []
    if mode == "id":
        try:
            needle_id = int(query.strip())
        except ValueError:
            return ()
        for span in result.spans:
            if span.token_id == needle_id:
                matches.append(_match_from_span(source, span))
        return tuple(matches)

    needle = query.casefold()
    for span in result.spans:
        haystack = span.text if mode == "text" else span.token
        if needle in haystack.casefold():
            matches.append(_match_from_span(source, span))
    return tuple(matches)


def analyze_corpus_path(path_value: str | Path, engine: TokenizerEngine) -> CorpusAnalysisResult:
    source = Path(path_value).expanduser().resolve()
    files = list(_iter_corpus_files(source))
    token_counts: Counter[int] = Counter()
    token_string_counts: Counter[str] = Counter()
    longest: list[CorpusLineResult] = []
    total_chars = 0
    total_tokens = 0
    skipped_files = 0
    unreadable_files = 0
    processed_files = 0

    if source.is_file() and source.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
        skipped_files += 1

    for file_path in files:
        documents, unreadable = _read_corpus_file(file_path)
        if unreadable:
            unreadable_files += 1
            continue
        processed_files += 1
        for line_number, text in documents:
            result = engine.encode(text)
            char_count = len(text)
            token_count = result.stats.token_count
            total_chars += char_count
            total_tokens += token_count
            token_counts.update(result.token_ids)
            token_string_counts.update(result.tokens)
            if text or token_count:
                longest.append(
                    CorpusLineResult(
                        file_path=str(file_path),
                        line_number=line_number,
                        preview=_preview(text),
                        token_count=token_count,
                        char_count=char_count,
                    )
                )

    if source.is_dir():
        skipped_files += _count_unsupported_files(source)

    longest.sort(key=lambda line: line.token_count, reverse=True)
    return CorpusAnalysisResult(
        source_path=str(source),
        total_files=processed_files,
        skipped_files=skipped_files,
        unreadable_files=unreadable_files,
        total_chars=total_chars,
        total_tokens=total_tokens,
        chars_per_token=(total_chars / total_tokens) if total_tokens else 0.0,
        top_token_ids=tuple(token_counts.most_common(10)),
        top_token_strings=tuple(token_string_counts.most_common(10)),
        longest_lines=tuple(longest[:10]),
    )


def tokenizer_metadata(engine: TokenizerEngine | None) -> TokenizerMetadata | None:
    if engine is None:
        return None
    raw = engine.raw_config or {}
    return TokenizerMetadata(
        model_type=str(raw.get("model", {}).get("type") or engine.tokenizer_type),
        source_path=str(engine.source_path),
        vocab_size=engine.vocab_size,
        normalizer=_component_label(raw.get("normalizer"), engine.tokenizer.normalizer),
        pre_tokenizer=_component_label(raw.get("pre_tokenizer"), engine.tokenizer.pre_tokenizer),
        decoder=_component_label(raw.get("decoder"), engine.tokenizer.decoder),
        post_processor=_component_label(raw.get("post_processor"), engine.tokenizer.post_processor),
        padding=_jsonish(engine.tokenizer.padding),
        truncation=_jsonish(engine.tokenizer.truncation),
        config_files_found=tuple(str(path) for path in engine.config_files_found),
        loaded_tokenizer_file=str(engine.loaded_tokenizer_file or ""),
    )


def analyze_chat_budget(
    messages: Sequence[ChatMessage],
    primary_engine: TokenizerEngine | None,
    compare_engine: TokenizerEngine | None = None,
    *,
    add_generation_prompt: bool = False,
    budget_limit: int | None = None,
    encode_special_tokens: bool = False,
) -> ChatBudgetResult:
    return ChatBudgetResult(
        messages=tuple(messages),
        add_generation_prompt=add_generation_prompt,
        primary=_render_chat_source(
            "primary",
            primary_engine,
            messages,
            add_generation_prompt,
            budget_limit,
            encode_special_tokens,
        ),
        compare=_render_chat_source(
            "compare",
            compare_engine,
            messages,
            add_generation_prompt,
            budget_limit,
            encode_special_tokens,
        ),
    )


def pipeline_debug(
    source: str,
    engine: TokenizerEngine | None,
    text: str,
    *,
    encode_special_tokens: bool = False,
) -> PipelineDebugResult | None:
    if engine is None:
        return None

    normalized = text
    normalizer_error: str | None = None
    normalizer = engine.tokenizer.normalizer
    if normalizer is not None:
        try:
            normalized = normalizer.normalize_str(text)
        except Exception as exc:
            normalizer_error = str(exc)

    pre_tokens: list[PipelinePreToken] = []
    pre_tokenizer_error: str | None = None
    pre_tokenizer = engine.tokenizer.pre_tokenizer
    if pre_tokenizer is not None:
        try:
            for piece, offsets in pre_tokenizer.pre_tokenize_str(normalized):
                start, end = offsets
                pre_tokens.append(PipelinePreToken(piece, int(start), int(end)))
        except Exception as exc:
            pre_tokenizer_error = str(exc)

    result = engine.encode(text, encode_special_tokens=encode_special_tokens)
    return PipelineDebugResult(
        source=source,
        input_text=text,
        normalized_text=normalized,
        normalizer_error=normalizer_error,
        pre_tokens=tuple(pre_tokens),
        pre_tokenizer_error=pre_tokenizer_error,
        token_count=result.stats.token_count,
        tokens=result.tokens,
        token_ids=result.token_ids,
    )


def analyze_batch_prompts(
    path_value: str | Path,
    primary_engine: TokenizerEngine,
    compare_engine: TokenizerEngine | None = None,
    *,
    budget_limit: int | None = None,
    encode_special_tokens: bool = False,
) -> BatchPromptAnalysisResult:
    source = Path(path_value).expanduser().resolve()
    files = list(_iter_corpus_files(source))
    rows: list[BatchPromptRow] = []
    skipped_files = 0
    unreadable_files = 0
    processed_files = 0
    primary_counts: list[int] = []
    compare_total = 0

    if source.is_file() and source.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
        skipped_files += 1

    for file_path in files:
        documents, unreadable = _read_corpus_file(file_path)
        if unreadable:
            unreadable_files += 1
            continue
        processed_files += 1
        for line_number, text in documents:
            if not text:
                continue
            primary = primary_engine.encode(text, encode_special_tokens=encode_special_tokens)
            compare_count: int | None = None
            delta: int | None = None
            if compare_engine is not None:
                compare = compare_engine.encode(text, encode_special_tokens=encode_special_tokens)
                compare_count = compare.stats.token_count
                compare_total += compare_count
                delta = compare_count - primary.stats.token_count
            primary_counts.append(primary.stats.token_count)
            rows.append(
                BatchPromptRow(
                    file_path=str(file_path),
                    line_number=line_number,
                    preview=_preview(text),
                    char_count=len(text),
                    primary_tokens=primary.stats.token_count,
                    compare_tokens=compare_count,
                    token_delta=delta,
                    budget_exceeded=bool(budget_limit and primary.stats.token_count > budget_limit),
                )
            )

    if source.is_dir():
        skipped_files += _count_unsupported_files(source)

    rows.sort(key=lambda row: row.primary_tokens, reverse=True)
    primary_total = sum(primary_counts)
    return BatchPromptAnalysisResult(
        source_path=str(source),
        total_prompts=len(rows),
        total_files=processed_files,
        skipped_files=skipped_files,
        unreadable_files=unreadable_files,
        primary_total_tokens=primary_total,
        compare_total_tokens=compare_total if compare_engine is not None else None,
        token_delta=(compare_total - primary_total) if compare_engine is not None else None,
        avg_tokens=(primary_total / len(primary_counts)) if primary_counts else 0.0,
        p50_tokens=_percentile(primary_counts, 50),
        p95_tokens=_percentile(primary_counts, 95),
        max_tokens=max(primary_counts) if primary_counts else 0,
        budget_failures=sum(1 for row in rows if row.budget_exceeded),
        longest_prompts=tuple(rows[:20]),
    )


def compare_corpus_path(
    path_value: str | Path,
    primary_engine: TokenizerEngine,
    compare_engine: TokenizerEngine,
    *,
    encode_special_tokens: bool = False,
) -> CorpusCompareResult:
    source = Path(path_value).expanduser().resolve()
    files = list(_iter_corpus_files(source))
    rows: list[CorpusCompareRow] = []
    total_chars = 0
    primary_total = 0
    compare_total = 0
    skipped_files = 0
    unreadable_files = 0
    processed_files = 0

    if source.is_file() and source.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
        skipped_files += 1

    for file_path in files:
        documents, unreadable = _read_corpus_file(file_path)
        if unreadable:
            unreadable_files += 1
            continue
        processed_files += 1
        for line_number, text in documents:
            if not text:
                continue
            primary = primary_engine.encode(text, encode_special_tokens=encode_special_tokens)
            compare = compare_engine.encode(text, encode_special_tokens=encode_special_tokens)
            primary_count = primary.stats.token_count
            compare_count = compare.stats.token_count
            primary_total += primary_count
            compare_total += compare_count
            total_chars += len(text)
            rows.append(
                CorpusCompareRow(
                    file_path=str(file_path),
                    line_number=line_number,
                    preview=_preview(text),
                    char_count=len(text),
                    primary_tokens=primary_count,
                    compare_tokens=compare_count,
                    token_delta=compare_count - primary_count,
                )
            )

    if source.is_dir():
        skipped_files += _count_unsupported_files(source)

    savings = sorted(rows, key=lambda row: row.token_delta)[:10]
    regressions = sorted(rows, key=lambda row: row.token_delta, reverse=True)[:10]
    return CorpusCompareResult(
        source_path=str(source),
        total_files=processed_files,
        skipped_files=skipped_files,
        unreadable_files=unreadable_files,
        total_chars=total_chars,
        primary_total_tokens=primary_total,
        compare_total_tokens=compare_total,
        token_delta=compare_total - primary_total,
        primary_chars_per_token=(total_chars / primary_total) if primary_total else 0.0,
        compare_chars_per_token=(total_chars / compare_total) if compare_total else 0.0,
        biggest_savings=tuple(savings),
        biggest_regressions=tuple(regressions),
    )


def save_project_state(project: ProjectState, path_value: str | Path) -> None:
    path = Path(path_value).expanduser()
    path.write_text(json.dumps(asdict(project), indent=2, ensure_ascii=False), encoding="utf-8")


def load_project_state(path_value: str | Path) -> ProjectState:
    path = Path(path_value).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Project file must contain a JSON object.")
    messages = tuple(
        ChatMessage(str(item.get("role", "user")), str(item.get("content", "")))
        for item in payload.get("chat_messages", [])
        if isinstance(item, dict)
    )
    return ProjectState(
        version=int(payload.get("version", PROJECT_SCHEMA_VERSION)),
        tokenizer_path=_optional_str(payload.get("tokenizer_path")),
        compare_tokenizer_path=_optional_str(payload.get("compare_tokenizer_path")),
        input_text=str(payload.get("input_text", "")),
        encode_special_tokens=bool(payload.get("encode_special_tokens", False)),
        export_format=str(payload.get("export_format", "json")),
        budget_limit=_optional_positive_int(payload.get("budget_limit")),
        chat_messages=messages,
        add_generation_prompt=bool(payload.get("add_generation_prompt", True)),
        corpus_path=_optional_str(payload.get("corpus_path")),
        batch_path=_optional_str(payload.get("batch_path")),
        active_tab=str(payload.get("active_tab", "token-table-tab")),
        selected_source=str(payload.get("selected_source", "primary")),
        pricing_profile=_pricing_profile_from_payload(payload.get("pricing_profile")) or default_pricing_profile(),
    )


def diff_tokenizers(
    primary_engine: TokenizerEngine | None,
    compare_engine: TokenizerEngine | None,
    *,
    limit: int = 200,
) -> TokenizerDiffResult | None:
    if primary_engine is None or compare_engine is None:
        return None

    items: list[TokenizerDiffItem] = []

    metadata_fields = {
        "name": (primary_engine.name, compare_engine.name),
        "type": (primary_engine.tokenizer_type, compare_engine.tokenizer_type),
        "vocab_size": (primary_engine.vocab_size, compare_engine.vocab_size),
        "chat_template": (bool(primary_engine.chat_template), bool(compare_engine.chat_template)),
    }
    metadata_differences = _append_diff_items(items, "metadata", metadata_fields, limit)

    primary_raw = primary_engine.raw_config or {}
    compare_raw = compare_engine.raw_config or {}
    component_fields = {
        "normalizer": (
            _component_label(primary_raw.get("normalizer"), primary_engine.tokenizer.normalizer),
            _component_label(compare_raw.get("normalizer"), compare_engine.tokenizer.normalizer),
        ),
        "pre_tokenizer": (
            _component_label(primary_raw.get("pre_tokenizer"), primary_engine.tokenizer.pre_tokenizer),
            _component_label(compare_raw.get("pre_tokenizer"), compare_engine.tokenizer.pre_tokenizer),
        ),
        "decoder": (
            _component_label(primary_raw.get("decoder"), primary_engine.tokenizer.decoder),
            _component_label(compare_raw.get("decoder"), compare_engine.tokenizer.decoder),
        ),
        "post_processor": (
            _component_label(primary_raw.get("post_processor"), primary_engine.tokenizer.post_processor),
            _component_label(compare_raw.get("post_processor"), compare_engine.tokenizer.post_processor),
        ),
    }
    component_differences = _append_diff_items(items, "component", component_fields, limit)

    primary_vocab = primary_engine.vocab
    compare_vocab = compare_engine.vocab
    primary_tokens = set(primary_vocab)
    compare_tokens = set(compare_vocab)
    vocab_primary_only = len(primary_tokens - compare_tokens)
    vocab_compare_only = len(compare_tokens - primary_tokens)
    vocab_id_differences = 0
    for token in sorted(primary_tokens & compare_tokens):
        primary_id = primary_vocab[token]
        compare_id = compare_vocab[token]
        if primary_id != compare_id:
            vocab_id_differences += 1
            if len(items) < limit:
                items.append(TokenizerDiffItem("vocab_id", token, str(primary_id), str(compare_id)))
    for kind, tokens, left_label, right_label in (
        ("vocab_primary_only", sorted(primary_tokens - compare_tokens), "present", "missing"),
        ("vocab_compare_only", sorted(compare_tokens - primary_tokens), "missing", "present"),
    ):
        for token in tokens[: max(0, limit - len(items))]:
            items.append(TokenizerDiffItem(kind, token, left_label, right_label))

    primary_added = _specials_by_token(primary_engine)
    compare_added = _specials_by_token(compare_engine)
    added_token_differences = _append_map_diffs(items, "added_token", primary_added, compare_added, limit)
    primary_special = {key: value for key, value in primary_added.items() if "special=True" in value}
    compare_special = {key: value for key, value in compare_added.items() if "special=True" in value}
    special_token_differences = _count_map_diffs(primary_special, compare_special)

    primary_merges = primary_engine.merge_ranks
    compare_merges = compare_engine.merge_ranks
    primary_merge_keys = set(primary_merges)
    compare_merge_keys = set(compare_merges)
    merge_primary_only = len(primary_merge_keys - compare_merge_keys)
    merge_compare_only = len(compare_merge_keys - primary_merge_keys)
    merge_rank_differences = 0
    for pair in sorted(primary_merge_keys & compare_merge_keys):
        if primary_merges[pair] != compare_merges[pair]:
            merge_rank_differences += 1
            if len(items) < limit:
                items.append(
                    TokenizerDiffItem(
                        "merge_rank",
                        " ".join(pair),
                        str(primary_merges[pair]),
                        str(compare_merges[pair]),
                    )
                )

    return TokenizerDiffResult(
        primary_name=primary_engine.name,
        compare_name=compare_engine.name,
        metadata_differences=metadata_differences,
        component_differences=component_differences,
        vocab_primary_only=vocab_primary_only,
        vocab_compare_only=vocab_compare_only,
        vocab_id_differences=vocab_id_differences,
        added_token_differences=added_token_differences,
        special_token_differences=special_token_differences,
        merge_primary_only=merge_primary_only,
        merge_compare_only=merge_compare_only,
        merge_rank_differences=merge_rank_differences,
        items=tuple(items[:limit]),
    )


def simulate_packing(
    engine: TokenizerEngine | None,
    text: str,
    *,
    budget_limit: int | None,
    strategy: str = "head_tail",
    segments: Sequence[tuple[str, str]] | None = None,
    encode_special_tokens: bool = False,
) -> PackingResult | None:
    if engine is None or not budget_limit or budget_limit <= 0:
        return None
    source_result = engine.encode(text, encode_special_tokens=encode_special_tokens)
    source_tokens = source_result.stats.token_count
    limit = max(0, budget_limit)
    if source_tokens <= limit:
        return PackingResult(
            strategy=strategy,
            limit=limit,
            source_tokens=source_tokens,
            kept_tokens=source_tokens,
            dropped_tokens=0,
            remaining_tokens=limit - source_tokens,
            kept_text=text,
            dropped_text="",
            segments=(PackingSegment("input", 0, _preview(text), source_tokens, True),),
        )

    if strategy == "head":
        kept_ids = source_result.token_ids[:limit]
    elif strategy == "tail":
        kept_ids = source_result.token_ids[-limit:]
    elif strategy in {"segments", "context_pack"} and segments:
        return _pack_segments(engine, segments, limit, strategy, encode_special_tokens)
    else:
        head_count = limit // 2
        tail_count = limit - head_count
        kept_ids = source_result.token_ids[:head_count] + source_result.token_ids[-tail_count:]

    kept_text = engine.decode(kept_ids, skip_special_tokens=False)
    return PackingResult(
        strategy=strategy,
        limit=limit,
        source_tokens=source_tokens,
        kept_tokens=len(kept_ids),
        dropped_tokens=max(0, source_tokens - len(kept_ids)),
        remaining_tokens=max(0, limit - len(kept_ids)),
        kept_text=kept_text,
        dropped_text=f"{source_tokens - len(kept_ids):,} tokens dropped",
        segments=(
            PackingSegment("input", 0, _preview(kept_text), len(kept_ids), True),
            PackingSegment("input", 1, "dropped middle" if strategy == "head_tail" else "dropped", source_tokens - len(kept_ids), False),
        ),
    )


def load_regression_suite(path_value: str | Path) -> tuple[str, tuple[RegressionCase, ...]]:
    path = Path(path_value).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Regression suite must contain a JSON object.")
    cases: list[RegressionCase] = []
    for index, item in enumerate(payload.get("cases", []), start=1):
        if not isinstance(item, dict):
            continue
        expected_ids = item.get("expected_ids")
        expected_tokens = item.get("expected_tokens")
        cases.append(
            RegressionCase(
                name=str(item.get("name") or f"case-{index}"),
                input_text=str(item.get("input", item.get("input_text", ""))),
                encode_special_tokens=bool(item.get("encode_special_tokens", False)),
                expected_ids=tuple(int(value) for value in expected_ids) if isinstance(expected_ids, list) else None,
                expected_tokens=tuple(str(value) for value in expected_tokens) if isinstance(expected_tokens, list) else None,
            )
        )
    return str(payload.get("name", path.stem)), tuple(cases)


def run_regression_suite(
    suite_name: str,
    cases: Sequence[RegressionCase],
    engine: TokenizerEngine,
) -> RegressionSuiteResult:
    results: list[RegressionCaseResult] = []
    for case in cases:
        actual = engine.encode(case.input_text, encode_special_tokens=case.encode_special_tokens)
        id_ok = case.expected_ids is None or case.expected_ids == actual.token_ids
        token_ok = case.expected_tokens is None or case.expected_tokens == actual.tokens
        mismatch = ""
        if not id_ok:
            mismatch = "token ids differ"
        elif not token_ok:
            mismatch = "token strings differ"
        results.append(
            RegressionCaseResult(
                name=case.name,
                passed=id_ok and token_ok,
                input_text=case.input_text,
                actual_ids=actual.token_ids,
                actual_tokens=actual.tokens,
                expected_ids=case.expected_ids,
                expected_tokens=case.expected_tokens,
                mismatch=mismatch,
            )
        )
    passed = sum(1 for result in results if result.passed)
    return RegressionSuiteResult(
        suite_name=suite_name,
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        cases=tuple(results),
    )


def regression_case_from_result(name: str, result: TokenizationResult, *, encode_special_tokens: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "input": result.input_text,
        "encode_special_tokens": encode_special_tokens,
        "expected_ids": list(result.token_ids),
        "expected_tokens": list(result.tokens),
    }


def inspect_unicode(text: str, *, limit: int = 500) -> UnicodeInspectionResult:
    nfc = unicodedata.normalize("NFC", text)
    nfkc = unicodedata.normalize("NFKC", text)
    characters: list[UnicodeCharacterInfo] = []
    for index, character in enumerate(text[:limit]):
        category = unicodedata.category(character)
        characters.append(
            UnicodeCharacterInfo(
                index=index,
                character=_visible_char(character),
                codepoint=f"U+{ord(character):04X}",
                name=unicodedata.name(character, "<unnamed>"),
                category=category,
                utf8_bytes=" ".join(f"{byte:02x}" for byte in character.encode("utf-8", errors="replace")),
                is_whitespace=character.isspace(),
                is_control=category.startswith("C"),
                is_zero_width=character in ZERO_WIDTH_CODEPOINTS,
                is_combining_mark=unicodedata.combining(character) != 0,
                nfc_changes=unicodedata.normalize("NFC", character) != character,
                nfkc_changes=unicodedata.normalize("NFKC", character) != character,
            )
        )
    return UnicodeInspectionResult(
        input_text=text,
        character_count=len(text),
        zero_width_count=sum(1 for character in text if character in ZERO_WIDTH_CODEPOINTS),
        control_count=sum(1 for character in text if unicodedata.category(character).startswith("C")),
        combining_mark_count=sum(1 for character in text if unicodedata.combining(character) != 0),
        nfc_changed=nfc != text,
        nfkc_changed=nfkc != text,
        characters=tuple(characters),
    )


def collect_text_units(path_value: str | Path) -> tuple[TextUnit, ...]:
    units: list[TextUnit] = []
    for file_path in _iter_corpus_files(Path(path_value).expanduser().resolve()):
        documents, unreadable = _read_corpus_file(file_path)
        if unreadable:
            continue
        for line_number, text in documents:
            if text:
                units.append(TextUnit(str(file_path), line_number, text))
    return tuple(units)


def analyze_rag_chunks(
    engine: TokenizerEngine | None,
    units: Sequence[TextUnit] | Sequence[str],
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
    mode: str = "token",
    encode_special_tokens: bool = False,
    limit: int = 100,
) -> RAGChunkingResult | None:
    if engine is None or max_tokens <= 0:
        return None
    normalized_units = _normalize_text_units(units)
    chunks: list[RAGChunk] = []
    total_tokens = 0
    for unit in normalized_units:
        result = engine.encode(unit.text, encode_special_tokens=encode_special_tokens)
        ids = list(result.token_ids)
        total_tokens += len(ids)
        if not ids:
            continue
        step = max(1, max_tokens - max(0, min(overlap_tokens, max_tokens - 1)))
        if mode == "separator":
            pieces = [piece.strip() for piece in unit.text.split("\n\n") if piece.strip()] or [unit.text]
            for piece in pieces:
                piece_result = engine.encode(piece, encode_special_tokens=encode_special_tokens)
                chunks.append(
                    RAGChunk(
                        source=f"{unit.source}:{unit.line_number}",
                        chunk_index=len(chunks),
                        text=_preview(piece),
                        token_count=piece_result.stats.token_count,
                        overflow=piece_result.stats.token_count > max_tokens,
                    )
                )
            continue
        for start in range(0, len(ids), step):
            part_ids = ids[start : start + max_tokens]
            chunks.append(
                RAGChunk(
                    source=f"{unit.source}:{unit.line_number}",
                    chunk_index=len(chunks),
                    text=_preview(engine.decode(part_ids, skip_special_tokens=False)),
                    token_count=len(part_ids),
                    overflow=len(part_ids) > max_tokens,
                )
            )
            if start + max_tokens >= len(ids):
                break

    counts = [chunk.token_count for chunk in chunks]
    wasted = sum(max(0, max_tokens - count) for count in counts)
    return RAGChunkingResult(
        max_tokens=max_tokens,
        overlap_tokens=max(0, overlap_tokens),
        mode=mode,
        source_units=len(normalized_units),
        chunk_count=len(chunks),
        total_tokens=total_tokens,
        avg_tokens=(sum(counts) / len(counts)) if counts else 0.0,
        p50_tokens=_percentile(counts, 50),
        p95_tokens=_percentile(counts, 95),
        max_chunk_tokens=max(counts) if counts else 0,
        overflow_chunks=sum(1 for chunk in chunks if chunk.overflow),
        wasted_budget=wasted,
        chunks=tuple(sorted(chunks, key=lambda chunk: chunk.token_count, reverse=True)[:limit]),
    )


def distribution_from_counts(
    source: str,
    counts: Sequence[int],
    *,
    budget_limit: int | None = None,
) -> DistributionResult:
    values = [int(value) for value in counts if value >= 0]
    budget_failures = sum(1 for value in values if budget_limit and value > budget_limit)
    return DistributionResult(
        source=source,
        count=len(values),
        min_tokens=min(values) if values else 0,
        p50_tokens=_percentile(values, 50),
        p90_tokens=_percentile(values, 90),
        p95_tokens=_percentile(values, 95),
        p99_tokens=_percentile(values, 99),
        max_tokens=max(values) if values else 0,
        avg_tokens=(sum(values) / len(values)) if values else 0.0,
        budget_limit=budget_limit,
        budget_failures=budget_failures,
        budget_failure_rate=(budget_failures / len(values) * 100.0) if values else 0.0,
        histogram=tuple(_histogram(values)),
    )


def distribution_from_batch(
    result: BatchPromptAnalysisResult | None,
    *,
    budget_limit: int | None = None,
) -> DistributionResult | None:
    if result is None:
        return None
    return distribution_from_counts(
        "batch",
        [row.primary_tokens for row in result.longest_prompts],
        budget_limit=budget_limit,
    )


def distribution_from_corpus(
    result: CorpusAnalysisResult | None,
    *,
    budget_limit: int | None = None,
) -> DistributionResult | None:
    if result is None:
        return None
    return distribution_from_counts(
        "corpus",
        [row.token_count for row in result.longest_lines],
        budget_limit=budget_limit,
    )


def estimate_token_cost(
    profile: PricingProfile,
    *,
    input_tokens: int,
    estimated_output_tokens: int | None = None,
) -> CostEstimate:
    output_tokens = profile.estimated_output_tokens if estimated_output_tokens is None else estimated_output_tokens
    input_cost = input_tokens / 1_000_000 * profile.input_per_million
    output_cost = output_tokens / 1_000_000 * profile.output_per_million
    return CostEstimate(
        profile_name=profile.name,
        input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


def load_pricing_profiles(workspace: Path | None = None) -> tuple[PricingProfile, ...]:
    path = (workspace or Path.cwd()) / PRICING_PROFILE_FILE
    if not path.exists():
        return (default_pricing_profile(),)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (default_pricing_profile(),)
    profiles: list[PricingProfile] = []
    seen_names: set[str] = set()
    values = payload if isinstance(payload, list) else payload.get("profiles", []) if isinstance(payload, dict) else []
    for item in values:
        profile = _pricing_profile_from_payload(item)
        if profile is None or profile.name in seen_names:
            continue
        profiles.append(profile)
        seen_names.add(profile.name)
    return tuple(profiles or (default_pricing_profile(),))


def save_pricing_profiles(profiles: Sequence[PricingProfile], workspace: Path | None = None) -> None:
    path = (workspace or Path.cwd()) / PRICING_PROFILE_FILE
    path.write_text(
        json.dumps({"profiles": [asdict(profile) for profile in profiles]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def suggest_tokenizer_repairs(engine: TokenizerEngine | None) -> TokenizerRepairResult | None:
    if engine is None:
        return None
    suggestions: list[RepairSuggestion] = []
    if not engine.chat_template:
        suggestions.append(
            RepairSuggestion(
                "warning",
                "missing chat_template",
                "Add a tokenizer_config.json chat_template before using chat budget workflows.",
            )
        )
    config_file = engine.tokenizer_config.get("tokenizer_file")
    if isinstance(config_file, str) and not (engine.source_path / config_file).exists():
        suggestions.append(
            RepairSuggestion(
                "error",
                "tokenizer_file path does not exist",
                f"Update tokenizer_config.json tokenizer_file or add {config_file}.",
            )
        )
    for info in extract_special_tokens(engine):
        if engine.token_to_id(info.token) is None:
            suggestions.append(
                RepairSuggestion(
                    "warning",
                    "special token missing from vocab",
                    f"Add {info.token!r} to the tokenizer vocabulary or remove it from added tokens.",
                )
            )
    for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
        value = engine.tokenizer_config.get(key)
        if isinstance(value, str) and engine.token_to_id(value) is None:
            suggestions.append(
                RepairSuggestion(
                    "info",
                    f"{key} not found in vocab",
                    f"Verify whether {value!r} should be present as an added or special token.",
                )
            )
    if not engine.config_files_found:
        suggestions.append(
            RepairSuggestion(
                "info",
                "no companion config files found",
                "Add tokenizer_config.json and special_tokens_map.json for easier downstream loading.",
            )
        )
    return TokenizerRepairResult(
        tokenizer_path=str(engine.source_path),
        suggestion_count=len(suggestions),
        suggestions=tuple(suggestions),
    )


def write_repair_preview(result: TokenizerRepairResult, path_value: str | Path) -> None:
    Path(path_value).expanduser().write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_recent_tokenizers(workspace: Path | None = None) -> list[str]:
    path = _recent_file(workspace)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    values: list[str] = []
    for item in payload:
        if isinstance(item, str) and item not in values:
            values.append(item)
    return values[:MAX_RECENT_TOKENIZERS]


def add_recent_tokenizer(path_value: str | Path, workspace: Path | None = None) -> list[str]:
    path = str(Path(path_value).expanduser().resolve())
    recent = [item for item in load_recent_tokenizers(workspace) if item != path]
    recent.insert(0, path)
    recent = recent[:MAX_RECENT_TOKENIZERS]
    _recent_file(workspace).write_text(json.dumps(recent, indent=2), encoding="utf-8")
    return recent


def build_export_payload(
    primary_engine: TokenizerEngine,
    primary_result: TokenizationResult,
    compare_engine: TokenizerEngine | None = None,
    comparison: CompareResult | None = None,
    budget_limit: int | None = None,
    corpus_result: CorpusAnalysisResult | None = None,
    chat_budget: ChatBudgetResult | None = None,
    batch_result: BatchPromptAnalysisResult | None = None,
    corpus_compare_result: CorpusCompareResult | None = None,
    pipeline_result: PipelineDebugResult | None = None,
    project_state: ProjectState | None = None,
    tokenizer_diff: TokenizerDiffResult | None = None,
    packing_result: PackingResult | None = None,
    regression_result: RegressionSuiteResult | None = None,
    unicode_result: UnicodeInspectionResult | None = None,
    rag_result: RAGChunkingResult | None = None,
    distribution_result: DistributionResult | None = None,
    cost_estimate: CostEstimate | None = None,
    repair_result: TokenizerRepairResult | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tokenizer": _engine_metadata_dict(primary_engine),
        "tokenization": primary_result.to_export_dict(),
    }
    if comparison is not None and compare_engine is not None:
        payload["comparison"] = {
            "primary_tokenizer": _engine_metadata_dict(primary_engine),
            "compare_tokenizer": _engine_metadata_dict(compare_engine),
            **comparison.to_export_dict(),
        }
    if budget_limit is not None and budget_limit > 0:
        budgets = [calculate_prompt_budget("primary", primary_result, budget_limit)]
        if comparison is not None:
            budgets.append(calculate_prompt_budget("compare", comparison.compare, budget_limit))
        payload["budget"] = [asdict(item) for item in budgets if item is not None]
    if corpus_result is not None:
        payload["corpus"] = asdict(corpus_result)
    if chat_budget is not None:
        payload["chat_budget"] = asdict(chat_budget)
    if batch_result is not None:
        payload["batch"] = asdict(batch_result)
    if corpus_compare_result is not None:
        payload["corpus_compare"] = asdict(corpus_compare_result)
    if pipeline_result is not None:
        payload["pipeline"] = asdict(pipeline_result)
    if project_state is not None:
        payload["project"] = asdict(project_state)
    if tokenizer_diff is not None:
        payload["tokenizer_diff"] = asdict(tokenizer_diff)
    if packing_result is not None:
        payload["packing"] = asdict(packing_result)
    if regression_result is not None:
        payload["regression"] = asdict(regression_result)
    if unicode_result is not None:
        payload["unicode"] = asdict(unicode_result)
    if rag_result is not None:
        payload["rag_chunking"] = asdict(rag_result)
    if distribution_result is not None:
        payload["distribution"] = asdict(distribution_result)
    if cost_estimate is not None:
        payload["cost"] = asdict(cost_estimate)
    if repair_result is not None:
        payload["repair"] = asdict(repair_result)
    return payload


def format_export(payload: dict[str, Any], export_format: ExportFormat) -> str:
    if export_format == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if export_format == "csv":
        return _format_csv_export(payload)
    if export_format == "md":
        return _format_markdown_export(payload)
    if export_format == "html":
        return _format_html_export(payload)
    raise ValueError(f"Unsupported export format: {export_format}")


def export_extension(export_format: ExportFormat) -> str:
    return "md" if export_format == "md" else export_format


def _format_csv_export(payload: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    writer.writerow(["section", "field", "value"])
    tokenizer = payload.get("tokenizer", {})
    if isinstance(tokenizer, dict):
        for key, value in tokenizer.items():
            writer.writerow(["tokenizer", key, value])

    writer.writerow([])
    writer.writerow(["section", "index", "token_string", "token_id", "offset_start", "offset_end", "text", "byte_repr"])
    tokenization = payload.get("tokenization", {})
    if isinstance(tokenization, dict):
        for span in tokenization.get("spans", []):
            if isinstance(span, dict):
                writer.writerow(
                    [
                        "primary_tokens",
                        span.get("index", ""),
                        span.get("token", ""),
                        span.get("token_id", ""),
                        span.get("offset_start", ""),
                        span.get("offset_end", ""),
                        span.get("text", ""),
                        span.get("byte_repr", ""),
                    ]
                )

    comparison = payload.get("comparison", {})
    if isinstance(comparison, dict):
        writer.writerow([])
        writer.writerow(["section", "range", "text", "primary_token", "primary_token_id", "compare_token", "compare_token_id", "status"])
        for row in comparison.get("rows", []):
            if isinstance(row, dict):
                writer.writerow(
                    [
                        "compare_diff",
                        f"{row.get('offset_start', '')}:{row.get('offset_end', '')}",
                        row.get("text", ""),
                        row.get("primary_token", ""),
                        row.get("primary_token_id", ""),
                        row.get("compare_token", ""),
                        row.get("compare_token_id", ""),
                        row.get("status", ""),
                    ]
                )

    budgets = payload.get("budget", [])
    if isinstance(budgets, list) and budgets:
        writer.writerow([])
        writer.writerow(["section", "source", "limit", "used_tokens", "remaining_tokens", "percent_used"])
        for budget in budgets:
            if isinstance(budget, dict):
                writer.writerow(
                    [
                        "budget",
                        budget.get("source", ""),
                        budget.get("limit", ""),
                        budget.get("used_tokens", ""),
                        budget.get("remaining_tokens", ""),
                        f"{float(budget.get('percent_used', 0.0)):.2f}",
                    ]
                )

    corpus = payload.get("corpus")
    if isinstance(corpus, dict):
        writer.writerow([])
        writer.writerow(["section", "field", "value"])
        for key in ("source_path", "total_files", "skipped_files", "unreadable_files", "total_chars", "total_tokens", "chars_per_token"):
            writer.writerow(["corpus_summary", key, corpus.get(key, "")])
        writer.writerow([])
        writer.writerow(["section", "file_path", "line_number", "token_count", "char_count", "preview"])
        for line in corpus.get("longest_lines", []):
            if isinstance(line, dict):
                writer.writerow(
                    [
                        "corpus_longest_lines",
                        line.get("file_path", ""),
                        line.get("line_number", ""),
                        line.get("token_count", ""),
                        line.get("char_count", ""),
                        line.get("preview", ""),
                    ]
                )

    _write_flat_csv_section(writer, "chat_budget", payload.get("chat_budget"))
    _write_flat_csv_section(writer, "batch", payload.get("batch"))
    _write_flat_csv_section(writer, "corpus_compare", payload.get("corpus_compare"))
    _write_flat_csv_section(writer, "pipeline", payload.get("pipeline"))
    _write_flat_csv_section(writer, "project", payload.get("project"))
    _write_flat_csv_section(writer, "tokenizer_diff", payload.get("tokenizer_diff"))
    _write_flat_csv_section(writer, "packing", payload.get("packing"))
    _write_flat_csv_section(writer, "regression", payload.get("regression"))
    _write_flat_csv_section(writer, "unicode", payload.get("unicode"))
    _write_flat_csv_section(writer, "rag_chunking", payload.get("rag_chunking"))
    _write_flat_csv_section(writer, "distribution", payload.get("distribution"))
    _write_flat_csv_section(writer, "cost", payload.get("cost"))
    _write_flat_csv_section(writer, "repair", payload.get("repair"))
    return buffer.getvalue()


def _format_markdown_export(payload: dict[str, Any]) -> str:
    lines: list[str] = ["# TokenScope Export", ""]
    tokenizer = payload.get("tokenizer", {})
    if isinstance(tokenizer, dict):
        lines.extend(["## Tokenizer", "", "| Field | Value |", "| --- | --- |"])
        for key, value in tokenizer.items():
            lines.append(f"| {_escape_md(str(key))} | {_escape_md(str(value))} |")
        lines.append("")

    tokenization = payload.get("tokenization", {})
    if isinstance(tokenization, dict):
        lines.extend(["## Token Table", "", "| Index | Token | ID | Offsets | Text | Bytes |", "| ---: | --- | ---: | --- | --- | --- |"])
        for span in tokenization.get("spans", []):
            if isinstance(span, dict):
                lines.append(
                    "| "
                    f"{span.get('index', '')} | "
                    f"{_escape_md(str(span.get('token', '')))} | "
                    f"{span.get('token_id', '')} | "
                    f"{span.get('offset_start', '')}:{span.get('offset_end', '')} | "
                    f"{_escape_md(_visible(str(span.get('text', ''))))} | "
                    f"{_escape_md(str(span.get('byte_repr', '')))} |"
                )
        lines.append("")

    comparison = payload.get("comparison", {})
    if isinstance(comparison, dict):
        lines.extend(["## Compare Diff", "", "| Range | Text | Primary | Compare | Status |", "| --- | --- | --- | --- | --- |"])
        for row in comparison.get("rows", []):
            if isinstance(row, dict):
                primary = f"{row.get('primary_token', '')} / {row.get('primary_token_id', '')}"
                compare = f"{row.get('compare_token', '')} / {row.get('compare_token_id', '')}"
                lines.append(
                    "| "
                    f"{row.get('offset_start', '')}:{row.get('offset_end', '')} | "
                    f"{_escape_md(_visible(str(row.get('text', ''))))} | "
                    f"{_escape_md(primary)} | "
                    f"{_escape_md(compare)} | "
                    f"{_escape_md(str(row.get('status', '')))} |"
                )
        lines.append("")

    budgets = payload.get("budget", [])
    if isinstance(budgets, list) and budgets:
        lines.extend(["## Budget", "", "| Source | Limit | Used | Remaining | Used % |", "| --- | ---: | ---: | ---: | ---: |"])
        for budget in budgets:
            if isinstance(budget, dict):
                lines.append(
                    "| "
                    f"{budget.get('source', '')} | "
                    f"{budget.get('limit', '')} | "
                    f"{budget.get('used_tokens', '')} | "
                    f"{budget.get('remaining_tokens', '')} | "
                    f"{float(budget.get('percent_used', 0.0)):.2f} |"
                )
        lines.append("")

    corpus = payload.get("corpus")
    if isinstance(corpus, dict):
        lines.extend(["## Corpus", "", "| Field | Value |", "| --- | --- |"])
        for key in ("source_path", "total_files", "skipped_files", "unreadable_files", "total_chars", "total_tokens", "chars_per_token"):
            lines.append(f"| {_escape_md(key)} | {_escape_md(str(corpus.get(key, '')))} |")
        lines.extend(["", "### Longest Lines", "", "| File | Line | Tokens | Chars | Preview |", "| --- | ---: | ---: | ---: | --- |"])
        for line in corpus.get("longest_lines", []):
            if isinstance(line, dict):
                lines.append(
                    "| "
                    f"{_escape_md(str(line.get('file_path', '')))} | "
                    f"{line.get('line_number', '')} | "
                    f"{line.get('token_count', '')} | "
                    f"{line.get('char_count', '')} | "
                    f"{_escape_md(_visible(str(line.get('preview', ''))))} |"
                )
        lines.append("")

    _append_markdown_dict(lines, "Chat Budget", payload.get("chat_budget"))
    _append_markdown_dict(lines, "Batch", payload.get("batch"))
    _append_markdown_dict(lines, "Corpus Compare", payload.get("corpus_compare"))
    _append_markdown_dict(lines, "Pipeline", payload.get("pipeline"))
    _append_markdown_dict(lines, "Project", payload.get("project"))
    _append_markdown_dict(lines, "Tokenizer Diff", payload.get("tokenizer_diff"))
    _append_markdown_dict(lines, "Packing", payload.get("packing"))
    _append_markdown_dict(lines, "Regression", payload.get("regression"))
    _append_markdown_dict(lines, "Unicode", payload.get("unicode"))
    _append_markdown_dict(lines, "RAG Chunking", payload.get("rag_chunking"))
    _append_markdown_dict(lines, "Distribution", payload.get("distribution"))
    _append_markdown_dict(lines, "Cost", payload.get("cost"))
    _append_markdown_dict(lines, "Repair", payload.get("repair"))
    return "\n".join(lines)


def _format_html_export(payload: dict[str, Any]) -> str:
    sections: list[str] = []
    for title, value in (
        ("Tokenizer", payload.get("tokenizer")),
        ("Tokenization", payload.get("tokenization")),
        ("Comparison", payload.get("comparison")),
        ("Budget", payload.get("budget")),
        ("Corpus", payload.get("corpus")),
        ("Chat Budget", payload.get("chat_budget")),
        ("Batch", payload.get("batch")),
        ("Corpus Compare", payload.get("corpus_compare")),
        ("Pipeline", payload.get("pipeline")),
        ("Project", payload.get("project")),
        ("Tokenizer Diff", payload.get("tokenizer_diff")),
        ("Packing", payload.get("packing")),
        ("Regression", payload.get("regression")),
        ("Unicode", payload.get("unicode")),
        ("RAG Chunking", payload.get("rag_chunking")),
        ("Distribution", payload.get("distribution")),
        ("Cost", payload.get("cost")),
        ("Repair", payload.get("repair")),
    ):
        if value:
            sections.append(f"<section><h2>{html.escape(title)}</h2>{_html_value(value)}</section>")
    body = "\n".join(sections) or "<section><p>No export data.</p></section>"
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>TokenScope Export</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem;background:#0d1117;color:#c9d1d9;}"
        "section{margin:0 0 1.5rem 0;padding:1rem;border:1px solid #30363d;border-radius:8px;background:#161b22;}"
        "h1,h2{color:#f0f6fc}table{border-collapse:collapse;width:100%;font-size:0.9rem;}"
        "td,th{border:1px solid #30363d;padding:0.35rem;vertical-align:top;}th{background:#21262d;}"
        "pre{white-space:pre-wrap;background:#0d1117;border:1px solid #30363d;padding:0.75rem;overflow:auto;}"
        "</style></head><body><h1>TokenScope Export</h1>"
        f"{body}</body></html>\n"
    )


def _render_chat_source(
    source: str,
    engine: TokenizerEngine | None,
    messages: Sequence[ChatMessage],
    add_generation_prompt: bool,
    budget_limit: int | None,
    encode_special_tokens: bool,
) -> ChatTemplateRender | None:
    if engine is None:
        return None
    try:
        rendered = engine.render_chat_template(
            [asdict(message) for message in messages],
            add_generation_prompt=add_generation_prompt,
        )
        tokenized = engine.encode(rendered, encode_special_tokens=encode_special_tokens)
        remaining = (budget_limit - tokenized.stats.token_count) if budget_limit else None
        percent = (tokenized.stats.token_count / budget_limit * 100.0) if budget_limit else None
        return ChatTemplateRender(
            source=source,
            has_template=True,
            rendered_text=rendered,
            token_count=tokenized.stats.token_count,
            remaining_tokens=remaining,
            percent_used=percent,
            error=None,
        )
    except ChatTemplateError as exc:
        return ChatTemplateRender(
            source=source,
            has_template=bool(engine.chat_template),
            rendered_text="",
            token_count=0,
            remaining_tokens=None,
            percent_used=None,
            error=str(exc),
        )


def _percentile(values: Sequence[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _pricing_profile_from_payload(value: Any) -> PricingProfile | None:
    if not isinstance(value, dict):
        return None
    try:
        name = str(value.get("name", "custom")).strip() or "custom"
        input_per_million = max(0.0, float(value.get("input_per_million", 0.0)))
        output_per_million = max(0.0, float(value.get("output_per_million", 0.0)))
        estimated_output_tokens = max(0, int(value.get("estimated_output_tokens", 0)))
    except (TypeError, ValueError):
        return None
    return PricingProfile(
        name=name,
        input_per_million=input_per_million,
        output_per_million=output_per_million,
        estimated_output_tokens=estimated_output_tokens,
    )


def _append_diff_items(
    items: list[TokenizerDiffItem],
    kind: str,
    fields: dict[str, tuple[Any, Any]],
    limit: int,
) -> int:
    count = 0
    for key, (primary, compare) in fields.items():
        if primary == compare:
            continue
        count += 1
        if len(items) < limit:
            items.append(TokenizerDiffItem(kind, key, _jsonish(primary), _jsonish(compare)))
    return count


def _specials_by_token(engine: TokenizerEngine) -> dict[str, str]:
    values: dict[str, str] = {}
    for info in extract_special_tokens(engine):
        values[info.token] = (
            f"id={info.token_id};special={info.special};normalized={info.normalized};"
            f"single_word={info.single_word};lstrip={info.lstrip};rstrip={info.rstrip}"
        )
    return values


def _count_map_diffs(primary: dict[str, str], compare: dict[str, str]) -> int:
    keys = set(primary) | set(compare)
    return sum(1 for key in keys if primary.get(key) != compare.get(key))


def _append_map_diffs(
    items: list[TokenizerDiffItem],
    kind: str,
    primary: dict[str, str],
    compare: dict[str, str],
    limit: int,
) -> int:
    count = 0
    for key in sorted(set(primary) | set(compare)):
        primary_value = primary.get(key, "missing")
        compare_value = compare.get(key, "missing")
        if primary_value == compare_value:
            continue
        count += 1
        if len(items) < limit:
            items.append(TokenizerDiffItem(kind, key, primary_value, compare_value))
    return count


def _pack_segments(
    engine: TokenizerEngine,
    segments: Sequence[tuple[str, str]],
    limit: int,
    strategy: str,
    encode_special_tokens: bool,
) -> PackingResult:
    packed: list[PackingSegment] = []
    kept_texts: list[str] = []
    dropped_texts: list[str] = []
    used = 0
    total = 0
    for index, (source, text) in enumerate(segments):
        result = engine.encode(text, encode_special_tokens=encode_special_tokens)
        count = result.stats.token_count
        total += count
        keep = used + count <= limit
        if keep:
            used += count
            kept_texts.append(text)
        else:
            dropped_texts.append(text)
        packed.append(PackingSegment(source, index, _preview(text), count, keep))
    return PackingResult(
        strategy=strategy,
        limit=limit,
        source_tokens=total,
        kept_tokens=used,
        dropped_tokens=max(0, total - used),
        remaining_tokens=max(0, limit - used),
        kept_text="\n".join(kept_texts),
        dropped_text="\n".join(_preview(text) for text in dropped_texts),
        segments=tuple(packed),
    )


def _visible_char(character: str) -> str:
    if character == " ":
        return "<space>"
    if character == "\t":
        return "<tab>"
    if character == "\n":
        return "<newline>"
    if character == "\r":
        return "<carriage-return>"
    if character in ZERO_WIDTH_CODEPOINTS:
        return "<zero-width>"
    return character


def _normalize_text_units(units: Sequence[TextUnit] | Sequence[str]) -> tuple[TextUnit, ...]:
    normalized: list[TextUnit] = []
    for index, item in enumerate(units):
        if isinstance(item, TextUnit):
            normalized.append(item)
        else:
            normalized.append(TextUnit("input", index + 1, str(item)))
    return tuple(normalized)


def _histogram(values: Sequence[int], *, width: int = 20, buckets: int = 8) -> list[HistogramBucket]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [HistogramBucket(str(minimum), len(values), "#" * width)]
    span = maximum - minimum + 1
    bucket_size = max(1, int((span + buckets - 1) / buckets))
    counts: Counter[int] = Counter()
    for value in values:
        bucket_start = minimum + ((value - minimum) // bucket_size) * bucket_size
        counts[bucket_start] += 1
    max_count = max(counts.values()) if counts else 1
    output: list[HistogramBucket] = []
    for bucket_start in sorted(counts):
        bucket_end = min(maximum, bucket_start + bucket_size - 1)
        count = counts[bucket_start]
        bar_width = max(1, int(count / max_count * width))
        label = str(bucket_start) if bucket_start == bucket_end else f"{bucket_start}-{bucket_end}"
        output.append(HistogramBucket(label, count, "#" * bar_width))
    return output


def _write_flat_csv_section(csv_writer: Any, section_name: str, value: Any) -> None:
    if not value:
        return
    csv_writer.writerow([])
    csv_writer.writerow(["section", "field", "value"])
    for key, item in _flatten_export_value(value):
        csv_writer.writerow([section_name, key, item])


def _append_markdown_dict(lines: list[str], title: str, value: Any) -> None:
    if not value:
        return
    lines.extend([f"## {title}", "", "| Field | Value |", "| --- | --- |"])
    for key, item in _flatten_export_value(value):
        lines.append(f"| {_escape_md(key)} | {_escape_md(str(item))} |")
    lines.append("")


def _flatten_export_value(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_export_value(item, next_prefix)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            yield from _flatten_export_value(item, next_prefix)
    else:
        yield prefix, _visible(str(value))


def _html_value(value: Any) -> str:
    if isinstance(value, dict):
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{_html_value(item)}</td></tr>"
            for key, item in value.items()
        )
        return f"<table>{rows}</table>"
    if isinstance(value, (list, tuple)):
        if not value:
            return "<em>none</em>"
        if all(isinstance(item, dict) for item in value):
            keys = sorted({key for item in value for key in item.keys()})
            header = "".join(f"<th>{html.escape(str(key))}</th>" for key in keys)
            rows = ""
            for item in value:
                rows += "<tr>" + "".join(_html_cell(item.get(key, "")) for key in keys) + "</tr>"
            return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
        return "<ul>" + "".join(f"<li>{_html_value(item)}</li>" for item in value) + "</ul>"
    text = str(value)
    if "\n" in text or len(text) > 160:
        return f"<pre>{html.escape(text)}</pre>"
    return html.escape(text)


def _html_cell(value: Any) -> str:
    return f"<td>{_html_value(value)}</td>"


def _special_info_from_added_token(token_id: int, token: Any) -> SpecialTokenInfo:
    return SpecialTokenInfo(
        token_id=token_id,
        token=str(getattr(token, "content", str(token))),
        special=getattr(token, "special", None),
        normalized=getattr(token, "normalized", None),
        single_word=getattr(token, "single_word", None),
        lstrip=getattr(token, "lstrip", None),
        rstrip=getattr(token, "rstrip", None),
    )


def _iter_raw_added_tokens(raw_config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    added_tokens = raw_config.get("added_tokens", [])
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if isinstance(item, dict):
                yield item


def _bool_or_none(value: Any, fallback: bool | None) -> bool | None:
    return value if isinstance(value, bool) else fallback


def _match_from_span(source: str, span: TokenSpan) -> TokenSearchMatch:
    return TokenSearchMatch(
        source=source,
        index=span.index,
        token=span.token,
        token_id=span.token_id,
        text=span.text,
        offset_start=span.offset_start,
        offset_end=span.offset_end,
    )


def _iter_corpus_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in SUPPORTED_CORPUS_SUFFIXES:
            yield source
        return
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_CORPUS_SUFFIXES:
            yield path


def _count_unsupported_files(source: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
            count += 1
    return count


def _read_corpus_file(file_path: Path) -> tuple[list[tuple[int, str]], bool]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [], True

    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(index, line) for index, line in enumerate(text.splitlines(), start=1)], False
    if suffix == ".csv":
        return _read_csv_lines(text), False
    if suffix == ".jsonl":
        return _read_jsonl_lines(text), False
    if suffix == ".json":
        return _read_json_lines(text), False
    return [(1, text)], False


def _read_csv_lines(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for index, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        rows.append((index, " ".join(cell for cell in row if cell)))
    return rows


def _read_jsonl_lines(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            rows.append((index, stripped))
            continue
        rows.append((index, " ".join(_iter_json_strings(payload))))
    return rows


def _read_json_lines(text: str) -> list[tuple[int, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [(index, line) for index, line in enumerate(text.splitlines(), start=1)]
    values = list(_iter_json_strings(payload))
    return [(1, " ".join(values))]


def _iter_json_strings(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_json_strings(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_json_strings(value)
    elif payload is not None:
        yield str(payload)


def _preview(text: str, limit: int = 120) -> str:
    visible = _visible(text.strip())
    return visible if len(visible) <= limit else visible[: limit - 1] + "..."


def _component_label(raw_component: Any, runtime_component: Any) -> str:
    if isinstance(raw_component, dict):
        component_type = raw_component.get("type")
        return str(component_type or raw_component)
    if raw_component is not None:
        return str(raw_component)
    if runtime_component is None:
        return "none"
    return type(runtime_component).__name__


def _jsonish(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _engine_metadata_dict(engine: TokenizerEngine) -> dict[str, object]:
    metadata = tokenizer_metadata(engine)
    if metadata is None:
        return {}
    return asdict(metadata)


def _recent_file(workspace: Path | None) -> Path:
    return (workspace or Path.cwd()) / RECENT_TOKENIZER_FILE


def _escape_md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _visible(value: str) -> str:
    return value.replace("\n", "\\n").replace("\t", "\\t")
