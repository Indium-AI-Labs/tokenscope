from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Select, Static, TabPane, TabbedContent

from analysis_models import (
    BatchPromptAnalysisResult,
    BenchmarkComparison,
    BenchmarkResult,
    ChatBudgetResult,
    ChatMessage,
    CorpusAnalysisResult,
    CorpusCompareResult,
    CostEstimate,
    DistributionResult,
    ExportFormat,
    PackingResult,
    PipelineDebugResult,
    PricingProfile,
    ProjectState,
    RAGChunkingResult,
    RegressionSuiteResult,
    SearchMode,
    TokenizerDiffResult,
    TokenizerRepairResult,
    TokenSearchMatch,
    UnicodeInspectionResult,
    analyze_chat_budget,
    analyze_rag_chunks,
    calculate_prompt_budget,
    decode_round_trip,
    diff_tokenizers,
    distribution_from_batch,
    distribution_from_corpus,
    estimate_token_cost,
    extract_special_tokens,
    inspect_token,
    inspect_unicode,
    load_pricing_profiles,
    load_regression_suite,
    pipeline_debug,
    regression_case_from_result,
    run_regression_suite,
    search_tokens,
    simulate_packing,
    suggest_tokenizer_repairs,
    save_pricing_profiles,
    write_repair_preview,
    tokenizer_metadata,
)
from compare_engine import CompareResult
from tokenizer_engine import TokenizationResult, TokenizerEngine


class MergeTreeWidget(Vertical):
    TAB_IDS = (
        "token-table-tab",
        "compare-diff-tab",
        "inspector-tab",
        "decode-tab",
        "special-tokens-tab",
        "budget-tab",
        "chat-budget-tab",
        "corpus-tab",
        "corpus-compare-tab",
        "batch-tab",
        "pipeline-tab",
        "project-tab",
        "tokenizer-diff-tab",
        "packing-tab",
        "regression-tab",
        "unicode-tab",
        "rag-tab",
        "distribution-tab",
        "cost-tab",
        "repair-tab",
        "metadata-tab",
        "search-tab",
        "merge-tree-tab",
        "vocab-search-tab",
        "benchmark-tab",
    )

    BUDGET_OPTIONS = (
        ("4k", "4096"),
        ("8k", "8192"),
        ("16k", "16384"),
        ("32k", "32768"),
        ("128k", "131072"),
        ("Custom", "custom"),
    )

    class TokenSelected(Message):
        def __init__(self, source: str, index: int) -> None:
            self.source = source
            self.index = index
            super().__init__()

    class SearchChanged(Message):
        def __init__(self, source: str, matches: tuple[TokenSearchMatch, ...]) -> None:
            self.source = source
            self.matches = matches
            super().__init__()

    class BudgetChanged(Message):
        def __init__(self, limit: int | None) -> None:
            self.limit = limit
            super().__init__()

    class ExportFormatChanged(Message):
        def __init__(self, export_format: ExportFormat) -> None:
            self.export_format = export_format
            super().__init__()

    class CorpusBrowseRequested(Message):
        pass

    class BatchBrowseRequested(Message):
        pass

    class EncodeSpecialToggleRequested(Message):
        pass

    class ProjectSaveRequested(Message):
        def __init__(self, path: str) -> None:
            self.path = path
            super().__init__()

    class ProjectLoadRequested(Message):
        def __init__(self, path: str) -> None:
            self.path = path
            super().__init__()

    def __init__(
        self,
        *args,
        export_format: ExportFormat = "json",
        budget_limit: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.primary_engine: TokenizerEngine | None = None
        self.compare_engine: TokenizerEngine | None = None
        self.primary_result: TokenizationResult | None = None
        self.compare_result: TokenizationResult | None = None
        self.comparison: CompareResult | None = None
        self.corpus_result: CorpusAnalysisResult | None = None
        self.corpus_compare_result: CorpusCompareResult | None = None
        self.batch_result: BatchPromptAnalysisResult | None = None
        self.regression_result: RegressionSuiteResult | None = None
        self.selected_source = "primary"
        self.selected_indices = {"primary": 0, "compare": 0}
        self.chat_messages: list[ChatMessage] = [
            ChatMessage("system", "You are a helpful assistant."),
            ChatMessage("user", "Hello"),
        ]
        self.chat_selected_index = 0
        self.add_generation_prompt = True
        self.search_mode: SearchMode = "text"
        self.search_query = ""
        self.search_matches: dict[str, tuple[TokenSearchMatch, ...]] = {"primary": (), "compare": ()}
        self.active_search_index = -1
        self.budget_limit: int | None = budget_limit
        self.export_format: ExportFormat = export_format
        self.encode_special_tokens = False
        self.project_path = "tokenscope_project.json"
        self.packing_strategy = "head_tail"
        self.regression_path = "tokenscope_regression.json"
        self.rag_max_tokens = 256
        self.rag_overlap_tokens = 32
        self.rag_mode = "token"
        self.cost_profiles = list(load_pricing_profiles(Path.cwd()))
        self.cost_profile = self.cost_profiles[0]
        self._active_index = 0
        self._updating_controls = False

    def compose(self):
        with Horizontal(id="bottom-controls"):
            yield Select(
                [("Primary tokenizer", "primary"), ("Compare tokenizer", "compare")],
                id="source-select",
                allow_blank=False,
                value="primary",
                compact=True,
            )
            yield Select(
                [
                    ("JSON export", "json"),
                    ("CSV export", "csv"),
                    ("Markdown export", "md"),
                    ("HTML export", "html"),
                ],
                id="export-format-select",
                allow_blank=False,
                value=self.export_format,
                compact=True,
            )
        with TabbedContent(initial=self.TAB_IDS[0], id="bottom-tabs"):
            with TabPane("Token Table", id=self.TAB_IDS[0]):
                yield DataTable(id="token-table")
            with TabPane("Compare Diff", id=self.TAB_IDS[1]):
                yield DataTable(id="compare-table")
            with TabPane("Inspector", id=self.TAB_IDS[2]):
                yield Static("Select a token to inspect.", id="inspector-details")
            with TabPane("Decode", id=self.TAB_IDS[3]):
                yield Static("Type text to compare decode output.", id="decode-details")
            with TabPane("Special Tokens", id=self.TAB_IDS[4]):
                yield Static("", id="special-mode")
                yield Button("Toggle encode special tokens", id="toggle-special")
                yield DataTable(id="special-table")
            with TabPane("Budget", id=self.TAB_IDS[5]):
                with Horizontal(id="budget-controls"):
                    yield Select(
                        self.BUDGET_OPTIONS,
                        id="budget-select",
                        allow_blank=False,
                        value=self._initial_budget_select_value(),
                        compact=True,
                    )
                    yield Input(placeholder="Custom token limit", id="budget-input")
                yield Static("Set a budget to inspect prompt usage.", id="budget-summary")
            with TabPane("Chat Budget", id=self.TAB_IDS[6]):
                with Horizontal(id="chat-controls"):
                    yield Select(
                        [("system", "system"), ("user", "user"), ("assistant", "assistant"), ("tool", "tool")],
                        id="chat-role",
                        allow_blank=False,
                        value="user",
                        compact=True,
                    )
                    yield Input(placeholder="Chat message content", id="chat-content")
                    yield Button("Add", id="chat-add")
                    yield Button("Update", id="chat-update")
                    yield Button("Delete", id="chat-delete")
                    yield Button("Up", id="chat-up")
                    yield Button("Down", id="chat-down")
                    yield Button("Gen prompt", id="chat-toggle-generation")
                yield DataTable(id="chat-messages")
                yield Static("No chat template rendered yet.", id="chat-summary")
            with TabPane("Corpus", id=self.TAB_IDS[7]):
                yield Button("Open corpus file/folder", id="open-corpus")
                yield Static("Open a local .txt, .md, .jsonl, .json, or .csv corpus.", id="corpus-summary")
                yield DataTable(id="corpus-lines")
            with TabPane("Corpus Compare", id=self.TAB_IDS[8]):
                yield Static("Load a compare tokenizer and corpus to compare corpus tokenization.", id="corpus-compare-summary")
                yield DataTable(id="corpus-compare-table")
            with TabPane("Batch", id=self.TAB_IDS[9]):
                yield Button("Open batch file/folder", id="open-batch")
                yield Static("Open a local prompt file or folder for batch analysis.", id="batch-summary")
                yield DataTable(id="batch-table")
            with TabPane("Pipeline", id=self.TAB_IDS[10]):
                yield Static("Type text to inspect tokenizer pipeline stages.", id="pipeline-summary")
                yield DataTable(id="pipeline-table")
            with TabPane("Project", id=self.TAB_IDS[11]):
                with Horizontal(id="project-controls"):
                    yield Input(value=self.project_path, placeholder="Project JSON path", id="project-path")
                    yield Button("Load", id="project-load")
                    yield Button("Save", id="project-save")
                yield Static("Save or load a TokenScope project.", id="project-summary")
            with TabPane("Tokenizer Diff", id=self.TAB_IDS[12]):
                yield Static("Load a compare tokenizer to inspect structural differences.", id="tokenizer-diff-summary")
                yield DataTable(id="tokenizer-diff-table")
            with TabPane("Packing", id=self.TAB_IDS[13]):
                with Horizontal(id="packing-controls"):
                    yield Select(
                        [("Head + tail", "head_tail"), ("Head only", "head"), ("Tail only", "tail"), ("Context pack", "context_pack")],
                        id="packing-strategy",
                        allow_blank=False,
                        value=self.packing_strategy,
                        compact=True,
                    )
                yield Static("Set a budget to simulate truncation and packing.", id="packing-summary")
                yield DataTable(id="packing-table")
            with TabPane("Regression", id=self.TAB_IDS[14]):
                with Horizontal(id="regression-controls"):
                    yield Input(value=self.regression_path, placeholder="Regression suite JSON path", id="regression-path")
                    yield Button("Run", id="regression-run")
                    yield Button("Add current", id="regression-add-current")
                yield Static("Load a tokenizer and run a regression suite.", id="regression-summary")
                yield DataTable(id="regression-table")
            with TabPane("Unicode", id=self.TAB_IDS[15]):
                yield Static("Type text to inspect Unicode and invisible characters.", id="unicode-summary")
                yield DataTable(id="unicode-table")
            with TabPane("RAG Chunking", id=self.TAB_IDS[16]):
                with Horizontal(id="rag-controls"):
                    yield Input(value=str(self.rag_max_tokens), placeholder="Max tokens", id="rag-max-tokens")
                    yield Input(value=str(self.rag_overlap_tokens), placeholder="Overlap tokens", id="rag-overlap-tokens")
                    yield Select(
                        [("Token window", "token"), ("Separator", "separator")],
                        id="rag-mode",
                        allow_blank=False,
                        value=self.rag_mode,
                        compact=True,
                    )
                yield Static("Analyze chunk sizes for the active text or loaded corpus/batch previews.", id="rag-summary")
                yield DataTable(id="rag-table")
            with TabPane("Distribution", id=self.TAB_IDS[17]):
                yield Static("Analyze token-count distributions for loaded corpus or batch data.", id="distribution-summary")
                yield DataTable(id="distribution-table")
            with TabPane("Cost", id=self.TAB_IDS[18]):
                with Horizontal(id="cost-controls"):
                    yield Select(
                        self._cost_profile_options(),
                        id="cost-profile-select",
                        allow_blank=False,
                        value=self.cost_profile.name,
                        compact=True,
                    )
                    yield Input(value=self.cost_profile.name, placeholder="Profile name", id="cost-profile-name")
                    yield Input(
                        value=self._format_number(self.cost_profile.input_per_million),
                        placeholder="Input cost / 1M",
                        id="cost-input-price",
                    )
                    yield Input(
                        value=self._format_number(self.cost_profile.output_per_million),
                        placeholder="Output cost / 1M",
                        id="cost-output-price",
                    )
                    yield Input(
                        value=str(self.cost_profile.estimated_output_tokens),
                        placeholder="Output tokens",
                        id="cost-output-tokens",
                    )
                    yield Button("Save", id="cost-profile-save")
                    yield Button("Delete", id="cost-profile-delete")
                yield Static("Enter local pricing to estimate token costs.", id="cost-summary")
            with TabPane("Repair", id=self.TAB_IDS[19]):
                yield Button("Write patch preview", id="repair-write-preview")
                yield Static("Load a tokenizer to inspect repair suggestions.", id="repair-summary")
                yield DataTable(id="repair-table")
            with TabPane("Metadata", id=self.TAB_IDS[20]):
                yield Static("Load a tokenizer to inspect metadata.", id="metadata-details")
            with TabPane("Search", id=self.TAB_IDS[21]):
                with Horizontal(id="token-search-controls"):
                    yield Select(
                        [("Text substring", "text"), ("Token substring", "token"), ("Exact token ID", "id")],
                        id="token-search-mode",
                        allow_blank=False,
                        value="text",
                        compact=True,
                    )
                    yield Input(placeholder="Search current tokenization", id="token-search")
                    yield Button("Previous", id="search-prev")
                    yield Button("Next", id="search-next")
                yield Static("No search query.", id="search-status")
                yield DataTable(id="search-results")
            with TabPane("Merge Tree", id=self.TAB_IDS[22]):
                yield Static("Load a tokenizer to inspect BPE merges.", id="merge-tree")
            with TabPane("Vocab Search", id=self.TAB_IDS[23]):
                yield Input(placeholder="Search vocabulary substring", id="vocab-search")
                yield DataTable(id="vocab-table")
            with TabPane("Benchmark", id=self.TAB_IDS[24]):
                yield Button("Run benchmark", id="benchmark-run")
                yield Static("Load a tokenizer and type text to benchmark tokenization speed.", id="benchmark-summary")
                yield DataTable(id="benchmark-table")

    def on_mount(self) -> None:
        self.query_one("#source-select", Select).add_class("hidden")

        token_table = self.query_one("#token-table", DataTable)
        token_table.cursor_type = "row"
        token_table.add_columns("sel", "match", "index", "token_string", "token_id", "offsets", "text", "byte_repr")

        compare_table = self.query_one("#compare-table", DataTable)
        compare_table.cursor_type = "row"
        compare_table.add_columns("range", "text", "primary", "compare", "status")

        special_table = self.query_one("#special-table", DataTable)
        special_table.cursor_type = "row"
        special_table.add_columns("token_string", "token_id", "special", "normalized", "single_word", "lstrip", "rstrip")

        corpus_lines = self.query_one("#corpus-lines", DataTable)
        corpus_lines.cursor_type = "row"
        corpus_lines.add_columns("file", "line", "tokens", "chars", "preview")

        chat_messages = self.query_one("#chat-messages", DataTable)
        chat_messages.cursor_type = "row"
        chat_messages.add_columns("sel", "role", "content")

        corpus_compare = self.query_one("#corpus-compare-table", DataTable)
        corpus_compare.cursor_type = "row"
        corpus_compare.add_columns("kind", "file", "line", "primary", "compare", "delta", "preview")

        batch_table = self.query_one("#batch-table", DataTable)
        batch_table.cursor_type = "row"
        batch_table.add_columns("file", "line", "primary", "compare", "delta", "budget", "preview")

        pipeline_table = self.query_one("#pipeline-table", DataTable)
        pipeline_table.cursor_type = "row"
        pipeline_table.add_columns("stage", "index", "text", "offsets", "token_id")

        tokenizer_diff = self.query_one("#tokenizer-diff-table", DataTable)
        tokenizer_diff.cursor_type = "row"
        tokenizer_diff.add_columns("kind", "key", "primary", "compare")

        packing_table = self.query_one("#packing-table", DataTable)
        packing_table.cursor_type = "row"
        packing_table.add_columns("source", "index", "tokens", "kept", "text")

        regression_table = self.query_one("#regression-table", DataTable)
        regression_table.cursor_type = "row"
        regression_table.add_columns("case", "passed", "mismatch", "actual_ids", "actual_tokens")

        unicode_table = self.query_one("#unicode-table", DataTable)
        unicode_table.cursor_type = "row"
        unicode_table.add_columns("index", "char", "codepoint", "name", "category", "bytes", "flags")

        rag_table = self.query_one("#rag-table", DataTable)
        rag_table.cursor_type = "row"
        rag_table.add_columns("source", "chunk", "tokens", "overflow", "preview")

        distribution_table = self.query_one("#distribution-table", DataTable)
        distribution_table.cursor_type = "row"
        distribution_table.add_columns("bucket", "count", "bar")

        repair_table = self.query_one("#repair-table", DataTable)
        repair_table.cursor_type = "row"
        repair_table.add_columns("severity", "issue", "suggestion")

        search_results = self.query_one("#search-results", DataTable)
        search_results.cursor_type = "row"
        search_results.add_columns("source", "index", "token_string", "token_id", "offsets", "text")

        vocab_table = self.query_one("#vocab-table", DataTable)
        vocab_table.cursor_type = "row"
        vocab_table.add_columns("token_string", "token_id")

        benchmark_table = self.query_one("#benchmark-table", DataTable)
        benchmark_table.cursor_type = "row"
        benchmark_table.add_columns("metric", "primary", "compare")

        self.set_export_format(self.export_format)
        self._sync_budget_controls()
        self._sync_chat_form()
        self._sync_cost_controls()
        self._update_all()

    def _initial_budget_select_value(self) -> str:
        if self.budget_limit is None:
            return "custom"
        value = str(self.budget_limit)
        known_values = {item[1] for item in self.BUDGET_OPTIONS}
        return value if value in known_values else "custom"

    def _cost_profile_options(self) -> tuple[tuple[str, str], ...]:
        if not self.cost_profiles:
            self.cost_profiles = [PricingProfile("custom", 0.0, 0.0, 0)]
        return tuple((profile.name, profile.name) for profile in self.cost_profiles)

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:g}"

    def set_engine(self, engine: TokenizerEngine | None) -> None:
        self.set_engines(engine, None)

    def set_engines(
        self,
        primary_engine: TokenizerEngine | None,
        compare_engine: TokenizerEngine | None,
    ) -> None:
        self.primary_engine = primary_engine
        self.compare_engine = compare_engine
        if compare_engine is None:
            self.selected_source = "primary"
        self._sync_source_selector()
        self._update_all()

    def update_result(self, result: TokenizationResult | None) -> None:
        self.update_results(result, None, None)

    def update_results(
        self,
        primary_result: TokenizationResult | None,
        compare_result: TokenizationResult | None,
        comparison: CompareResult | None,
    ) -> None:
        self.primary_result = primary_result
        self.compare_result = compare_result
        self.comparison = comparison
        self._clamp_selected_indices()
        self._update_search_state(post_message=False)
        self._update_all()

    def set_selected_token(self, source: str, index: int) -> None:
        if source not in ("primary", "compare"):
            return
        if source == "compare" and self.compare_engine is None:
            return
        result = self._result_for(source)
        if result is None or not result.spans:
            self.selected_indices[source] = 0
            return
        bounded = max(0, min(index, len(result.spans) - 1))
        self.selected_indices[source] = bounded
        self.selected_source = source
        self._sync_source_selector()
        self._update_token_table()
        self._update_inspector()
        self._update_merge_tree()

    def set_budget_limit(self, limit: int | None) -> None:
        self.budget_limit = limit if limit and limit > 0 else None
        self._sync_budget_controls()
        self._update_budget()
        self._update_packing()
        self._update_distribution()
        self._update_cost()

    def set_export_format(self, export_format: ExportFormat) -> None:
        self.export_format = export_format
        self._updating_controls = True
        try:
            self.query_one("#export-format-select", Select).value = export_format
        finally:
            self._updating_controls = False

    def set_encode_special_tokens(self, enabled: bool) -> None:
        self.encode_special_tokens = enabled
        self._update_special_tokens()
        self._update_chat_budget()
        self._update_pipeline()
        self._update_packing()
        self._update_rag()

    def set_corpus_result(self, result: CorpusAnalysisResult | None) -> None:
        self.corpus_result = result
        self._update_corpus()
        self._update_rag()
        self._update_distribution()
        self._update_cost()

    def set_corpus_compare_result(self, result: CorpusCompareResult | None) -> None:
        self.corpus_compare_result = result
        self._update_corpus_compare()

    def set_batch_result(self, result: BatchPromptAnalysisResult | None) -> None:
        self.batch_result = result
        self._update_batch()
        self._update_rag()
        self._update_distribution()
        self._update_cost()

    def current_chat_budget(self) -> ChatBudgetResult | None:
        if self.primary_engine is None:
            return None
        return analyze_chat_budget(
            self.chat_messages,
            self.primary_engine,
            self.compare_engine,
            add_generation_prompt=self.add_generation_prompt,
            budget_limit=self.budget_limit,
            encode_special_tokens=self.encode_special_tokens,
        )

    def current_pipeline_debug(self) -> PipelineDebugResult | None:
        return pipeline_debug(
            self.selected_source,
            self._selected_engine(),
            self._selected_result().input_text if self._selected_result() is not None else "",
            encode_special_tokens=self.encode_special_tokens,
        )

    def current_tokenizer_diff(self) -> TokenizerDiffResult | None:
        return diff_tokenizers(self.primary_engine, self.compare_engine)

    def current_packing(self) -> PackingResult | None:
        result = self._selected_result()
        text = result.input_text if result is not None else ""
        segments = None
        if self.packing_strategy == "context_pack":
            segments = self._packing_segments()
        return simulate_packing(
            self._selected_engine(),
            text,
            budget_limit=self.budget_limit,
            strategy=self.packing_strategy,
            segments=segments,
            encode_special_tokens=self.encode_special_tokens,
        )

    def current_unicode(self) -> UnicodeInspectionResult | None:
        result = self._selected_result()
        if result is None:
            return None
        return inspect_unicode(result.input_text)

    def current_rag(self) -> RAGChunkingResult | None:
        return analyze_rag_chunks(
            self._selected_engine(),
            self._rag_units(),
            max_tokens=self.rag_max_tokens,
            overlap_tokens=self.rag_overlap_tokens,
            mode=self.rag_mode,
            encode_special_tokens=self.encode_special_tokens,
        )

    def current_distribution(self) -> DistributionResult | None:
        if self.batch_result is not None:
            return distribution_from_batch(self.batch_result, budget_limit=self.budget_limit)
        return distribution_from_corpus(self.corpus_result, budget_limit=self.budget_limit)

    def current_cost(self) -> CostEstimate | None:
        tokens = 0
        if self.primary_result is not None:
            tokens += self.primary_result.stats.token_count
        if self.batch_result is not None:
            tokens += self.batch_result.primary_total_tokens
        if self.corpus_result is not None:
            tokens += self.corpus_result.total_tokens
        return estimate_token_cost(self.cost_profile, input_tokens=tokens)

    def current_repair(self) -> TokenizerRepairResult | None:
        return suggest_tokenizer_repairs(self._selected_engine())

    def set_project_state(self, project: ProjectState) -> None:
        self.project_path = self.project_path or "tokenscope_project.json"
        self.chat_messages = list(project.chat_messages) or self.chat_messages
        self.chat_selected_index = 0
        self.add_generation_prompt = project.add_generation_prompt
        self.selected_source = project.selected_source if project.selected_source in ("primary", "compare") else "primary"
        self._upsert_cost_profile(project.pricing_profile, persist=False)
        self._sync_chat_form()
        self._sync_cost_controls()
        self._update_chat_budget()
        self._update_cost()
        try:
            self.query_one("#bottom-tabs", TabbedContent).active = project.active_tab
        except Exception:
            pass

    def focus_token_search(self) -> None:
        tabs = self.query_one("#bottom-tabs", TabbedContent)
        tabs.active = "search-tab"
        self.query_one("#token-search", Input).focus()

    def focus_budget_input(self) -> None:
        tabs = self.query_one("#bottom-tabs", TabbedContent)
        tabs.active = "budget-tab"
        self.query_one("#budget-input", Input).focus()

    def cycle_tab(self) -> None:
        tabs = self.query_one("#bottom-tabs", TabbedContent)
        active = getattr(tabs, "active", self.TAB_IDS[self._active_index])
        try:
            self._active_index = self.TAB_IDS.index(active)
        except ValueError:
            self._active_index = 0
        self._active_index = (self._active_index + 1) % len(self.TAB_IDS)
        tabs.active = self.TAB_IDS[self._active_index]
        active_tab = self.TAB_IDS[self._active_index]
        if active_tab == "vocab-search-tab":
            self.query_one("#vocab-search", Input).focus()
        elif active_tab == "search-tab":
            self.query_one("#token-search", Input).focus()
        elif active_tab == "budget-tab":
            self.query_one("#budget-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._updating_controls:
            return
        if event.input.id == "vocab-search":
            event.stop()
            self._update_vocab_table(event.value)
            return
        if event.input.id == "budget-input":
            event.stop()
            self._handle_budget_input(event.value)
            return
        if event.input.id == "token-search":
            event.stop()
            self.search_query = event.value
            self._update_search_state()
            self._update_token_table()
            return
        if event.input.id == "chat-content":
            event.stop()
            return
        if event.input.id == "project-path":
            event.stop()
            self.project_path = event.value.strip() or "tokenscope_project.json"
            self._update_project()
            return
        if event.input.id == "regression-path":
            event.stop()
            self.regression_path = event.value.strip() or "tokenscope_regression.json"
            return
        if event.input.id == "rag-max-tokens":
            event.stop()
            self.rag_max_tokens = self._positive_int(event.value, self.rag_max_tokens)
            self._update_rag()
            return
        if event.input.id == "rag-overlap-tokens":
            event.stop()
            self.rag_overlap_tokens = max(0, self._positive_int(event.value, self.rag_overlap_tokens))
            self._update_rag()
            return
        if event.input.id in {"cost-profile-name", "cost-input-price", "cost-output-price", "cost-output-tokens"}:
            event.stop()
            self._sync_cost_profile_from_inputs()
            self._update_cost()
            return

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._updating_controls:
            return
        if event.select.id == "source-select":
            event.stop()
            if event.value in ("primary", "compare"):
                self.selected_source = str(event.value)
                self._update_vocab_table(self.query_one("#vocab-search", Input).value)
                self._update_token_table()
                self._update_inspector()
                self._update_merge_tree()
                self._update_special_tokens()
                self._update_metadata()
                self._update_pipeline()
                self._update_chat_budget()
                self._update_search_state()
                self._update_packing()
                self._update_unicode()
                self._update_rag()
                self._update_cost()
                self._update_repair()
            return
        if event.select.id == "budget-select":
            event.stop()
            if event.value == "custom":
                self.query_one("#budget-input", Input).focus()
                return
            self.set_budget_limit(int(str(event.value)))
            self._update_chat_budget()
            self._update_packing()
            self._update_distribution()
            self._update_cost()
            self.post_message(self.BudgetChanged(self.budget_limit))
            return
        if event.select.id == "token-search-mode":
            event.stop()
            if event.value in ("text", "token", "id"):
                self.search_mode = str(event.value)  # type: ignore[assignment]
                self._update_search_state()
                self._update_token_table()
            return
        if event.select.id == "chat-role":
            event.stop()
            return
        if event.select.id == "export-format-select":
            event.stop()
            if event.value in ("json", "csv", "md", "html"):
                self.export_format = str(event.value)  # type: ignore[assignment]
                self.post_message(self.ExportFormatChanged(self.export_format))
            return
        if event.select.id == "packing-strategy":
            event.stop()
            self.packing_strategy = str(event.value)
            self._update_packing()
            return
        if event.select.id == "rag-mode":
            event.stop()
            self.rag_mode = str(event.value)
            self._update_rag()
            return
        if event.select.id == "cost-profile-select":
            event.stop()
            self._apply_cost_profile(str(event.value))
            return

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-corpus":
            event.stop()
            self.post_message(self.CorpusBrowseRequested())
        elif event.button.id == "open-batch":
            event.stop()
            self.post_message(self.BatchBrowseRequested())
        elif event.button.id == "toggle-special":
            event.stop()
            self.post_message(self.EncodeSpecialToggleRequested())
        elif event.button.id and event.button.id.startswith("chat-"):
            event.stop()
            self._handle_chat_button(event.button.id)
        elif event.button.id == "search-next":
            event.stop()
            self._navigate_search(1)
        elif event.button.id == "search-prev":
            event.stop()
            self._navigate_search(-1)
        elif event.button.id == "project-save":
            event.stop()
            self.post_message(self.ProjectSaveRequested(self.project_path))
            self._update_project(f"Requested save to {self.project_path}.")
        elif event.button.id == "project-load":
            event.stop()
            self.post_message(self.ProjectLoadRequested(self.project_path))
            self._update_project(f"Requested load from {self.project_path}.")
        elif event.button.id == "regression-run":
            event.stop()
            self._run_regression_suite()
        elif event.button.id == "regression-add-current":
            event.stop()
            self._write_current_regression_case()
        elif event.button.id == "repair-write-preview":
            event.stop()
            result = self.current_repair()
            if result is not None:
                write_repair_preview(result, "tokenscope_tokenizer_patch.json")
                self._update_repair("Wrote tokenscope_tokenizer_patch.json.")
        elif event.button.id == "cost-profile-save":
            event.stop()
            self._save_cost_profile()
        elif event.button.id == "cost-profile-delete":
            event.stop()
            self._delete_cost_profile()
        elif event.button.id == "benchmark-run":
            event.stop()
            await self._run_benchmark()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "token-table":
            event.stop()
            self._select_row(event.cursor_row)
        elif event.data_table.id == "search-results":
            event.stop()
            self._select_search_result(event.cursor_row)
        elif event.data_table.id == "chat-messages":
            event.stop()
            self._select_chat_row(event.cursor_row)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "token-table":
            return
        event.stop()
        self._select_row(event.cursor_row)

    def _update_all(self) -> None:
        self._sync_source_selector()
        self._update_token_table()
        self._update_compare_table()
        self._update_inspector()
        self._update_decode()
        self._update_special_tokens()
        self._update_budget()
        self._update_chat_budget()
        self._update_corpus()
        self._update_corpus_compare()
        self._update_batch()
        self._update_pipeline()
        self._update_project()
        self._update_tokenizer_diff()
        self._update_packing()
        self._update_regression()
        self._update_unicode()
        self._update_rag()
        self._update_distribution()
        self._update_cost()
        self._update_repair()
        self._update_metadata()
        self._update_search_tables()
        self._update_merge_tree()
        self._update_vocab_table(self.query_one("#vocab-search", Input).value)

    def _sync_source_selector(self) -> None:
        selector = self.query_one("#source-select", Select)
        self._updating_controls = True
        try:
            if self.compare_engine is None:
                selector.add_class("hidden")
                selector.value = "primary"
                self.selected_source = "primary"
            else:
                selector.remove_class("hidden")
                selector.value = self.selected_source
        finally:
            self._updating_controls = False

    def _sync_budget_controls(self) -> None:
        self._updating_controls = True
        try:
            budget_input = self.query_one("#budget-input", Input)
            budget_select = self.query_one("#budget-select", Select)
            if self.budget_limit is None:
                budget_input.value = ""
                budget_select.value = "custom"
                return
            budget_input.value = str(self.budget_limit)
            value = str(self.budget_limit)
            known_values = {item[1] for item in self.BUDGET_OPTIONS}
            budget_select.value = value if value in known_values else "custom"
        finally:
            self._updating_controls = False

    def _sync_chat_form(self) -> None:
        self._updating_controls = True
        try:
            role = self.query_one("#chat-role", Select)
            content = self.query_one("#chat-content", Input)
            if self.chat_messages and 0 <= self.chat_selected_index < len(self.chat_messages):
                message = self.chat_messages[self.chat_selected_index]
                role.value = message.role
                content.value = message.content
            else:
                role.value = "user"
                content.value = ""
        finally:
            self._updating_controls = False

    def _sync_cost_controls(self) -> None:
        if self._profile_by_name(self.cost_profile.name) is None:
            self.cost_profiles.append(self.cost_profile)
        self._updating_controls = True
        try:
            profile_select = self.query_one("#cost-profile-select", Select)
            profile_select.set_options(self._cost_profile_options())
            profile_select.value = self.cost_profile.name
            self.query_one("#cost-profile-name", Input).value = self.cost_profile.name
            self.query_one("#cost-input-price", Input).value = self._format_number(
                self.cost_profile.input_per_million
            )
            self.query_one("#cost-output-price", Input).value = self._format_number(
                self.cost_profile.output_per_million
            )
            self.query_one("#cost-output-tokens", Input).value = str(
                self.cost_profile.estimated_output_tokens
            )
        finally:
            self._updating_controls = False

    def _apply_cost_profile(self, profile_name: str) -> None:
        profile = self._profile_by_name(profile_name)
        if profile is None:
            return
        self.cost_profile = profile
        self._sync_cost_controls()
        self._update_cost()

    def _save_cost_profile(self) -> None:
        profile = self._sync_cost_profile_from_inputs()
        self._upsert_cost_profile(profile, persist=True)
        self._sync_cost_controls()
        self._update_cost(f"Saved pricing profile {profile.name}.")

    def _delete_cost_profile(self) -> None:
        profile_name = self.cost_profile.name
        remaining = [profile for profile in self.cost_profiles if profile.name != profile_name]
        if not remaining:
            remaining = [PricingProfile("custom", 0.0, 0.0, 0)]
        self.cost_profiles = remaining
        self.cost_profile = self.cost_profiles[0]
        self._persist_cost_profiles()
        self._sync_cost_controls()
        self._update_cost(f"Deleted pricing profile {profile_name}.")

    def _profile_by_name(self, profile_name: str) -> PricingProfile | None:
        for profile in self.cost_profiles:
            if profile.name == profile_name:
                return profile
        return None

    def _upsert_cost_profile(self, profile: PricingProfile, *, persist: bool) -> None:
        for index, existing in enumerate(self.cost_profiles):
            if existing.name == profile.name:
                self.cost_profiles[index] = profile
                break
        else:
            self.cost_profiles.append(profile)
        self.cost_profile = profile
        if persist:
            self._persist_cost_profiles()

    def _persist_cost_profiles(self) -> None:
        save_pricing_profiles(self.cost_profiles, Path.cwd())

    def _handle_chat_button(self, button_id: str) -> None:
        if button_id == "chat-toggle-generation":
            self.add_generation_prompt = not self.add_generation_prompt
            self._update_chat_budget()
            return

        role_value = self.query_one("#chat-role", Select).value
        role = str(role_value) if role_value in ("system", "user", "assistant", "tool") else "user"
        content = self.query_one("#chat-content", Input).value

        if button_id == "chat-add":
            self.chat_messages.append(ChatMessage(role, content))
            self.chat_selected_index = len(self.chat_messages) - 1
        elif button_id == "chat-update" and self.chat_messages:
            self.chat_messages[self.chat_selected_index] = ChatMessage(role, content)
        elif button_id == "chat-delete" and self.chat_messages:
            self.chat_messages.pop(self.chat_selected_index)
            self.chat_selected_index = max(0, min(self.chat_selected_index, len(self.chat_messages) - 1))
        elif button_id == "chat-up" and self.chat_selected_index > 0:
            index = self.chat_selected_index
            self.chat_messages[index - 1], self.chat_messages[index] = (
                self.chat_messages[index],
                self.chat_messages[index - 1],
            )
            self.chat_selected_index -= 1
        elif button_id == "chat-down" and self.chat_selected_index < len(self.chat_messages) - 1:
            index = self.chat_selected_index
            self.chat_messages[index + 1], self.chat_messages[index] = (
                self.chat_messages[index],
                self.chat_messages[index + 1],
            )
            self.chat_selected_index += 1

        self._sync_chat_form()
        self._update_chat_budget()

    def _select_chat_row(self, row: int) -> None:
        if row < 0 or row >= len(self.chat_messages):
            return
        self.chat_selected_index = row
        self._sync_chat_form()
        self._update_chat_budget()

    def _run_regression_suite(self) -> None:
        engine = self.primary_engine
        if engine is None:
            self.query_one("#regression-summary", Static).update("Load a primary tokenizer before running regressions.")
            return
        try:
            suite_name, cases = load_regression_suite(self.regression_path)
            self.regression_result = run_regression_suite(suite_name, cases, engine)
        except Exception as exc:
            self.query_one("#regression-summary", Static).update(f"Regression failed: {exc}")
            return
        self._update_regression()

    def _write_current_regression_case(self) -> None:
        result = self.primary_result
        if result is None:
            self.query_one("#regression-summary", Static).update("Type text before adding a regression case.")
            return
        path = Path(self.regression_path).expanduser()
        payload: dict[str, object]
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                payload = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        cases = payload.get("cases")
        if not isinstance(cases, list):
            cases = []
        cases.append(
            regression_case_from_result(
                f"case-{len(cases) + 1}",
                result,
                encode_special_tokens=self.encode_special_tokens,
            )
        )
        payload["version"] = 1
        payload["name"] = str(payload.get("name") or path.stem)
        payload["cases"] = cases
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.query_one("#regression-summary", Static).update(f"Added current input to {path}.")

    def _sync_cost_profile_from_inputs(self) -> PricingProfile:
        profile_name = self.query_one("#cost-profile-name", Input).value.strip() or "custom"
        input_price = max(0.0, self._float_input("#cost-input-price"))
        output_price = max(0.0, self._float_input("#cost-output-price"))
        output_tokens = self._positive_int(self.query_one("#cost-output-tokens", Input).value, 0)
        self.cost_profile = PricingProfile(profile_name, input_price, output_price, output_tokens)
        return self.cost_profile

    def _packing_segments(self) -> tuple[tuple[str, str], ...]:
        segments: list[tuple[str, str]] = []
        if self.primary_result is not None and self.primary_result.input_text:
            segments.append(("input", self.primary_result.input_text))
        segments.extend((f"chat:{message.role}", message.content) for message in self.chat_messages if message.content)
        if self.corpus_result is not None:
            segments.extend((f"corpus:{line.line_number}", line.preview) for line in self.corpus_result.longest_lines)
        if self.batch_result is not None:
            segments.extend((f"batch:{row.line_number}", row.preview) for row in self.batch_result.longest_prompts)
        return tuple(segments)

    def _rag_units(self):
        result = self._selected_result()
        units = []
        if result is not None and result.input_text:
            units.append(result.input_text)
        if self.corpus_result is not None:
            units.extend(line.preview for line in self.corpus_result.longest_lines)
        if self.batch_result is not None:
            units.extend(row.preview for row in self.batch_result.longest_prompts)
        return units or [""]

    def _positive_int(self, value: str, fallback: int) -> int:
        try:
            parsed = int(value)
        except ValueError:
            return fallback
        return parsed if parsed > 0 else fallback

    def _float_input(self, selector: str) -> float:
        try:
            return float(self.query_one(selector, Input).value)
        except ValueError:
            return 0.0

    def _selected_engine(self) -> TokenizerEngine | None:
        return self._engine_for(self.selected_source)

    def _selected_result(self) -> TokenizationResult | None:
        return self._result_for(self.selected_source)

    def _engine_for(self, source: str) -> TokenizerEngine | None:
        if source == "compare" and self.compare_engine is not None:
            return self.compare_engine
        return self.primary_engine

    def _result_for(self, source: str) -> TokenizationResult | None:
        if source == "compare" and self.compare_result is not None:
            return self.compare_result
        return self.primary_result

    def _clamp_selected_indices(self) -> None:
        for source in ("primary", "compare"):
            result = self._result_for(source)
            if result is None or not result.spans:
                self.selected_indices[source] = 0
            else:
                self.selected_indices[source] = max(
                    0,
                    min(self.selected_indices[source], len(result.spans) - 1),
                )

    def _select_row(self, row: int) -> None:
        result = self._selected_result()
        if result is None or not result.spans or row < 0 or row >= len(result.spans):
            return
        source = self.selected_source
        self.selected_indices[source] = row
        self._update_inspector()
        self._update_merge_tree()
        self.post_message(self.TokenSelected(source, row))

    def _select_search_result(self, row: int) -> None:
        matches = self.search_matches.get(self.selected_source, ())
        if row < 0 or row >= len(matches):
            return
        self.active_search_index = row
        match = matches[row]
        self.set_selected_token(match.source, match.index)
        self.post_message(self.TokenSelected(match.source, match.index))

    def _navigate_search(self, delta: int) -> None:
        matches = self.search_matches.get(self.selected_source, ())
        if not matches:
            return
        self.active_search_index = (self.active_search_index + delta) % len(matches)
        match = matches[self.active_search_index]
        self.set_selected_token(match.source, match.index)
        self.post_message(self.TokenSelected(match.source, match.index))

    def _handle_budget_input(self, value: str) -> None:
        stripped = value.strip()
        limit: int | None = None
        if stripped:
            try:
                limit = int(stripped)
            except ValueError:
                limit = None
        self.budget_limit = limit if limit and limit > 0 else None
        self._updating_controls = True
        try:
            self.query_one("#budget-select", Select).value = "custom"
        finally:
            self._updating_controls = False
        self._update_budget()
        self._update_chat_budget()
        self._update_packing()
        self._update_distribution()
        self._update_cost()
        self.post_message(self.BudgetChanged(self.budget_limit))

    def _update_token_table(self) -> None:
        table = self.query_one("#token-table", DataTable)
        table.clear()
        result = self._selected_result()
        if result is None:
            return
        selected = self.selected_indices.get(self.selected_source, 0)
        match_indices = {match.index for match in self.search_matches.get(self.selected_source, ())}
        for span in result.spans:
            table.add_row(
                ">" if span.index == selected else "",
                "*" if span.index in match_indices else "",
                str(span.index),
                span.token,
                str(span.token_id),
                f"{span.offset_start}:{span.offset_end}",
                self._visible(span.text),
                span.byte_repr,
            )
        if result.spans:
            table.move_cursor(row=selected, column=0, animate=False)

    def _update_compare_table(self) -> None:
        table = self.query_one("#compare-table", DataTable)
        table.clear()
        if self.comparison is None:
            table.add_row("", "", "", "", "Load a compare tokenizer to see differences.")
            return
        for row in self.comparison.rows:
            primary = self._format_token(row.primary_token, row.primary_token_id)
            compare = self._format_token(row.compare_token, row.compare_token_id)
            table.add_row(
                f"{row.offset_start}:{row.offset_end}",
                self._visible(row.text),
                primary,
                compare,
                row.status,
            )

    def _update_inspector(self) -> None:
        detail = self.query_one("#inspector-details", Static)
        inspection = inspect_token(
            self.selected_source,
            self._selected_engine(),
            self._selected_result(),
            self.selected_indices.get(self.selected_source, 0),
        )
        if inspection is None:
            detail.update(Text("Select a token to inspect.", style="dim"))
            return
        table = Table.grid(expand=True)
        table.add_column("field", style="dim")
        table.add_column("value", overflow="fold")
        for key, value in asdict(inspection).items():
            if key == "merge_tree":
                continue
            table.add_row(key, self._visible(str(value)))
        detail.update(Group(table, Text(), Text(inspection.merge_tree)))

    def _update_decode(self) -> None:
        detail = self.query_one("#decode-details", Static)
        reports = [
            decode_round_trip("primary", self.primary_engine, self.primary_result),
            decode_round_trip("compare", self.compare_engine, self.compare_result),
        ]
        reports = [report for report in reports if report is not None]
        if not reports:
            detail.update(Text("Type text to compare decode output.", style="dim"))
            return
        table = Table.grid(expand=True)
        table.add_column("field", style="dim")
        table.add_column("value", overflow="fold")
        for report in reports:
            table.add_row(f"{report.source} exact", "yes" if report.exact_match else "no")
            table.add_row(
                f"{report.source} first diff",
                "n/a" if report.first_difference is None else str(report.first_difference),
            )
            table.add_row(f"{report.source} original", self._visible(report.original_text))
            table.add_row(f"{report.source} decoded", self._visible(report.decoded_text))
            table.add_row("", "")
        detail.update(table)

    def _update_special_tokens(self) -> None:
        self.query_one("#special-mode", Static).update(
            f"Encode special tokens: {'on' if self.encode_special_tokens else 'off'}"
        )
        table = self.query_one("#special-table", DataTable)
        table.clear()
        for info in extract_special_tokens(self._selected_engine()):
            table.add_row(
                info.token,
                str(info.token_id),
                self._bool_label(info.special),
                self._bool_label(info.normalized),
                self._bool_label(info.single_word),
                self._bool_label(info.lstrip),
                self._bool_label(info.rstrip),
            )

    def _update_budget(self) -> None:
        summary = self.query_one("#budget-summary", Static)
        if self.budget_limit is None:
            summary.update(Text("Set a token limit to inspect prompt usage.", style="dim"))
            return
        budgets = [
            calculate_prompt_budget("primary", self.primary_result, self.budget_limit),
            calculate_prompt_budget("compare", self.compare_result, self.budget_limit),
        ]
        budgets = [budget for budget in budgets if budget is not None]
        if not budgets:
            summary.update(Text("Type text to inspect prompt usage.", style="dim"))
            return
        table = Table.grid(expand=True)
        table.add_column("source", style="dim")
        table.add_column("used", justify="right")
        table.add_column("remaining", justify="right")
        table.add_column("used %", justify="right")
        for budget in budgets:
            table.add_row(
                budget.source,
                f"{budget.used_tokens:,} / {budget.limit:,}",
                f"{budget.remaining_tokens:,}",
                f"{budget.percent_used:.2f}%",
            )
        summary.update(table)

    def _update_chat_budget(self) -> None:
        table = self.query_one("#chat-messages", DataTable)
        table.clear()
        for index, message in enumerate(self.chat_messages):
            table.add_row(
                ">" if index == self.chat_selected_index else "",
                message.role,
                self._visible(message.content),
            )
        if self.chat_messages:
            table.move_cursor(row=self.chat_selected_index, column=0, animate=False)

        summary = self.query_one("#chat-summary", Static)
        result = self.current_chat_budget()
        if result is None:
            summary.update(Text("Load a tokenizer to render chat templates.", style="dim"))
            return

        detail = Table.grid(expand=True)
        detail.add_column("field", style="dim")
        detail.add_column("value", overflow="fold")
        detail.add_row("add_generation_prompt", "yes" if result.add_generation_prompt else "no")
        for render in (result.primary, result.compare):
            if render is None:
                continue
            prefix = render.source
            if render.error:
                detail.add_row(f"{prefix} error", render.error)
                continue
            detail.add_row(f"{prefix} tokens", f"{render.token_count:,}")
            if render.remaining_tokens is not None:
                detail.add_row(f"{prefix} remaining", f"{render.remaining_tokens:,}")
            if render.percent_used is not None:
                detail.add_row(f"{prefix} used", f"{render.percent_used:.2f}%")
            detail.add_row(f"{prefix} rendered", self._visible(render.rendered_text))
        summary.update(detail)

    def _update_corpus(self) -> None:
        summary = self.query_one("#corpus-summary", Static)
        table = self.query_one("#corpus-lines", DataTable)
        table.clear()
        if self.corpus_result is None:
            summary.update("Open a local .txt, .md, .jsonl, .json, or .csv corpus.")
            return
        result = self.corpus_result
        summary.update(
            " | ".join(
                [
                    f"Files: {result.total_files:,}",
                    f"Skipped: {result.skipped_files:,}",
                    f"Unreadable: {result.unreadable_files:,}",
                    f"Chars: {result.total_chars:,}",
                    f"Tokens: {result.total_tokens:,}",
                    f"Chars/token: {result.chars_per_token:.2f}",
                    f"Top IDs: {self._top_pairs(result.top_token_ids)}",
                    f"Top tokens: {self._top_pairs(result.top_token_strings)}",
                ]
            )
        )
        for line in result.longest_lines:
            table.add_row(
                line.file_path,
                str(line.line_number),
                str(line.token_count),
                str(line.char_count),
                line.preview,
            )

    def _update_corpus_compare(self) -> None:
        summary = self.query_one("#corpus-compare-summary", Static)
        table = self.query_one("#corpus-compare-table", DataTable)
        table.clear()
        result = self.corpus_compare_result
        if result is None:
            summary.update("Load a compare tokenizer and corpus to compare corpus tokenization.")
            return
        summary.update(
            " | ".join(
                [
                    f"Files: {result.total_files:,}",
                    f"Primary tokens: {result.primary_total_tokens:,}",
                    f"Compare tokens: {result.compare_total_tokens:,}",
                    f"Delta: {result.token_delta:+,}",
                    f"Primary chars/token: {result.primary_chars_per_token:.2f}",
                    f"Compare chars/token: {result.compare_chars_per_token:.2f}",
                ]
            )
        )
        for kind, rows in (("saving", result.biggest_savings), ("regression", result.biggest_regressions)):
            for row in rows:
                table.add_row(
                    kind,
                    row.file_path,
                    str(row.line_number),
                    str(row.primary_tokens),
                    str(row.compare_tokens),
                    f"{row.token_delta:+}",
                    row.preview,
                )

    def _update_batch(self) -> None:
        summary = self.query_one("#batch-summary", Static)
        table = self.query_one("#batch-table", DataTable)
        table.clear()
        result = self.batch_result
        if result is None:
            summary.update("Open a local prompt file or folder for batch analysis.")
            return
        summary.update(
            " | ".join(
                [
                    f"Prompts: {result.total_prompts:,}",
                    f"Files: {result.total_files:,}",
                    f"Primary tokens: {result.primary_total_tokens:,}",
                    f"Avg: {result.avg_tokens:.2f}",
                    f"P50: {result.p50_tokens:.2f}",
                    f"P95: {result.p95_tokens:.2f}",
                    f"Max: {result.max_tokens:,}",
                    f"Budget failures: {result.budget_failures:,}",
                    f"Delta: {result.token_delta:+,}" if result.token_delta is not None else "Delta: n/a",
                ]
            )
        )
        for row in result.longest_prompts:
            table.add_row(
                row.file_path,
                str(row.line_number),
                str(row.primary_tokens),
                "n/a" if row.compare_tokens is None else str(row.compare_tokens),
                "n/a" if row.token_delta is None else f"{row.token_delta:+}",
                "yes" if row.budget_exceeded else "no",
                row.preview,
            )

    def _update_pipeline(self) -> None:
        summary = self.query_one("#pipeline-summary", Static)
        table = self.query_one("#pipeline-table", DataTable)
        table.clear()
        result = self.current_pipeline_debug()
        if result is None:
            summary.update("Type text to inspect tokenizer pipeline stages.")
            return
        summary.update(
            f"{result.source} | input chars: {len(result.input_text):,} | tokens: {result.token_count:,}"
        )
        table.add_row("input", "", self._visible(result.input_text), "", "")
        table.add_row(
            "normalizer",
            "",
            self._visible(result.normalized_text),
            "",
            result.normalizer_error or "",
        )
        if result.pre_tokenizer_error:
            table.add_row("pre-tokenizer error", "", result.pre_tokenizer_error, "", "")
        for index, pre_token in enumerate(result.pre_tokens):
            table.add_row(
                "pre-tokenizer",
                str(index),
                self._visible(pre_token.text),
                f"{pre_token.offset_start}:{pre_token.offset_end}",
                "",
            )
        for index, (token, token_id) in enumerate(zip(result.tokens, result.token_ids, strict=False)):
            table.add_row("model", str(index), token, "", str(token_id))

    def _update_project(self, message: str | None = None) -> None:
        summary = self.query_one("#project-summary", Static)
        detail = message or (
            f"Path: {self.project_path} | chat messages: {len(self.chat_messages)} | "
            f"source: {self.selected_source} | export: {self.export_format}"
        )
        summary.update(detail)

    def _update_tokenizer_diff(self) -> None:
        summary = self.query_one("#tokenizer-diff-summary", Static)
        table = self.query_one("#tokenizer-diff-table", DataTable)
        table.clear()
        result = self.current_tokenizer_diff()
        if result is None:
            summary.update("Load a compare tokenizer to inspect structural differences.")
            return
        summary.update(
            " | ".join(
                [
                    f"{result.primary_name} vs {result.compare_name}",
                    f"metadata: {result.metadata_differences}",
                    f"components: {result.component_differences}",
                    f"vocab primary-only: {result.vocab_primary_only}",
                    f"vocab compare-only: {result.vocab_compare_only}",
                    f"id diffs: {result.vocab_id_differences}",
                    f"merge rank diffs: {result.merge_rank_differences}",
                ]
            )
        )
        for item in result.items:
            table.add_row(item.kind, item.key, self._visible(item.primary), self._visible(item.compare))

    def _update_packing(self) -> None:
        summary = self.query_one("#packing-summary", Static)
        table = self.query_one("#packing-table", DataTable)
        table.clear()
        result = self.current_packing()
        if result is None:
            summary.update("Set a budget to simulate truncation and packing.")
            return
        summary.update(
            " | ".join(
                [
                    f"Strategy: {result.strategy}",
                    f"Limit: {result.limit:,}",
                    f"Source: {result.source_tokens:,}",
                    f"Kept: {result.kept_tokens:,}",
                    f"Dropped: {result.dropped_tokens:,}",
                    f"Remaining: {result.remaining_tokens:,}",
                ]
            )
        )
        for segment in result.segments:
            table.add_row(
                segment.source,
                str(segment.index),
                str(segment.token_count),
                "yes" if segment.kept else "no",
                self._visible(segment.text),
            )

    def _update_regression(self) -> None:
        summary = self.query_one("#regression-summary", Static)
        table = self.query_one("#regression-table", DataTable)
        table.clear()
        result = self.regression_result
        if result is None:
            summary.update("Run a regression suite JSON file against the primary tokenizer.")
            return
        summary.update(
            f"{result.suite_name} | passed: {result.passed_cases:,}/{result.total_cases:,} | failed: {result.failed_cases:,}"
        )
        for case in result.cases:
            table.add_row(
                case.name,
                "yes" if case.passed else "no",
                case.mismatch,
                " ".join(str(item) for item in case.actual_ids),
                " ".join(case.actual_tokens),
            )

    def _update_unicode(self) -> None:
        summary = self.query_one("#unicode-summary", Static)
        table = self.query_one("#unicode-table", DataTable)
        table.clear()
        result = self.current_unicode()
        if result is None:
            summary.update("Type text to inspect Unicode and invisible characters.")
            return
        summary.update(
            " | ".join(
                [
                    f"Chars: {result.character_count:,}",
                    f"Zero-width: {result.zero_width_count:,}",
                    f"Control: {result.control_count:,}",
                    f"Combining: {result.combining_mark_count:,}",
                    f"NFC changes: {'yes' if result.nfc_changed else 'no'}",
                    f"NFKC changes: {'yes' if result.nfkc_changed else 'no'}",
                ]
            )
        )
        for character in result.characters:
            flags = ",".join(
                flag
                for flag, enabled in (
                    ("space", character.is_whitespace),
                    ("control", character.is_control),
                    ("zero-width", character.is_zero_width),
                    ("combining", character.is_combining_mark),
                    ("nfc", character.nfc_changes),
                    ("nfkc", character.nfkc_changes),
                )
                if enabled
            )
            table.add_row(
                str(character.index),
                character.character,
                character.codepoint,
                character.name,
                character.category,
                character.utf8_bytes,
                flags,
            )

    def _update_rag(self) -> None:
        summary = self.query_one("#rag-summary", Static)
        table = self.query_one("#rag-table", DataTable)
        table.clear()
        result = self.current_rag()
        if result is None:
            summary.update("Load a tokenizer and set a positive max-token chunk size.")
            return
        summary.update(
            " | ".join(
                [
                    f"Chunks: {result.chunk_count:,}",
                    f"Units: {result.source_units:,}",
                    f"Avg: {result.avg_tokens:.2f}",
                    f"P95: {result.p95_tokens:.2f}",
                    f"Max: {result.max_chunk_tokens:,}",
                    f"Overflow: {result.overflow_chunks:,}",
                    f"Wasted: {result.wasted_budget:,}",
                ]
            )
        )
        for chunk in result.chunks:
            table.add_row(
                chunk.source,
                str(chunk.chunk_index),
                str(chunk.token_count),
                "yes" if chunk.overflow else "no",
                self._visible(chunk.text),
            )

    def _update_distribution(self) -> None:
        summary = self.query_one("#distribution-summary", Static)
        table = self.query_one("#distribution-table", DataTable)
        table.clear()
        result = self.current_distribution()
        if result is None:
            summary.update("Load corpus or batch data to inspect token-count distributions.")
            return
        summary.update(
            " | ".join(
                [
                    f"Source: {result.source}",
                    f"Count: {result.count:,}",
                    f"P50: {result.p50_tokens:.2f}",
                    f"P95: {result.p95_tokens:.2f}",
                    f"P99: {result.p99_tokens:.2f}",
                    f"Max: {result.max_tokens:,}",
                    f"Budget failures: {result.budget_failures:,} ({result.budget_failure_rate:.2f}%)",
                ]
            )
        )
        for bucket in result.histogram:
            table.add_row(bucket.label, str(bucket.count), bucket.bar)

    def _update_cost(self, message: str | None = None) -> None:
        summary = self.query_one("#cost-summary", Static)
        estimate = self.current_cost()
        if estimate is None:
            summary.update("Enter local pricing to estimate token costs.")
            return
        details = " | ".join(
            [
                f"Profile: {estimate.profile_name}",
                f"Input tokens: {estimate.input_tokens:,}",
                f"Output tokens: {estimate.estimated_output_tokens:,}",
                f"Input cost: {estimate.input_cost:.6f}",
                f"Output cost: {estimate.output_cost:.6f}",
                f"Total: {estimate.total_cost:.6f}",
            ]
        )
        summary.update(f"{message} | {details}" if message else details)

    def _update_repair(self, message: str | None = None) -> None:
        summary = self.query_one("#repair-summary", Static)
        table = self.query_one("#repair-table", DataTable)
        table.clear()
        result = self.current_repair()
        if result is None:
            summary.update(message or "Load a tokenizer to inspect repair suggestions.")
            return
        summary.update(message or f"{result.suggestion_count:,} repair suggestions for {result.tokenizer_path}")
        for suggestion in result.suggestions:
            table.add_row(suggestion.severity, suggestion.issue, suggestion.suggestion)

    def _update_metadata(self) -> None:
        detail = self.query_one("#metadata-details", Static)
        metadata = tokenizer_metadata(self._selected_engine())
        if metadata is None:
            detail.update(Text("Load a tokenizer to inspect metadata.", style="dim"))
            return
        table = Table.grid(expand=True)
        table.add_column("field", style="dim")
        table.add_column("value", overflow="fold")
        for key, value in asdict(metadata).items():
            if isinstance(value, tuple):
                value = ", ".join(value) if value else "none"
            table.add_row(key, self._visible(str(value)))
        detail.update(table)

    def _update_search_state(self, *, post_message: bool = True) -> None:
        matches = search_tokens(
            self.selected_source,
            self._selected_result(),
            self.search_query,
            self.search_mode,
        )
        self.search_matches[self.selected_source] = matches
        self.active_search_index = 0 if matches else -1
        self._update_search_tables()
        if post_message:
            self.post_message(self.SearchChanged(self.selected_source, matches))

    def _update_search_tables(self) -> None:
        status = self.query_one("#search-status", Static)
        table = self.query_one("#search-results", DataTable)
        table.clear()
        matches = self.search_matches.get(self.selected_source, ())
        if not self.search_query:
            status.update("No search query.")
        elif not matches:
            status.update("No matches.")
        else:
            status.update(f"{len(matches):,} matches in {self.selected_source}.")
        for match in matches:
            table.add_row(
                match.source,
                str(match.index),
                match.token,
                str(match.token_id),
                f"{match.offset_start}:{match.offset_end}",
                self._visible(match.text),
            )

    def _update_merge_tree(self) -> None:
        tree = self.query_one("#merge-tree", Static)
        engine = self._selected_engine()
        if engine is None:
            tree.update("Load a tokenizer to inspect BPE merges.")
            return
        result = self._selected_result()
        selected = self.selected_indices.get(self.selected_source, 0)
        if result is not None and result.spans:
            tree.update(engine.render_single_token_merge_tree(result.spans[selected]))
        else:
            tree.update(engine.render_bpe_merge_tree(result))

    def _update_vocab_table(self, query: str) -> None:
        table = self.query_one("#vocab-table", DataTable)
        table.clear()
        engine = self._selected_engine()
        if engine is None:
            return
        for token, token_id in engine.search_vocab(query):
            table.add_row(token, str(token_id))

    async def _run_benchmark(self) -> None:
        summary = self.query_one("#benchmark-summary", Static)
        table = self.query_one("#benchmark-table", DataTable)
        table.clear()

        result = self._selected_result()
        if result is None or not result.input_text:
            summary.update("Type text before running a benchmark.")
            return
        if self.primary_engine is None:
            summary.update("Load a tokenizer before running a benchmark.")
            return

        summary.update("Running benchmark...")
        text = result.input_text
        primary_data = await asyncio.to_thread(self.primary_engine.benchmark, text, iterations=10)
        primary_result = BenchmarkResult(**primary_data)  # type: ignore[arg-type]

        compare_result: BenchmarkResult | None = None
        if self.compare_engine is not None:
            compare_data = await asyncio.to_thread(self.compare_engine.benchmark, text, iterations=10)
            compare_data["source"] = "compare"
            compare_result = BenchmarkResult(**compare_data)  # type: ignore[arg-type]

        speedup: float | None = None
        if compare_result is not None and compare_result.mean_seconds > 0:
            speedup = compare_result.mean_seconds / primary_result.mean_seconds if primary_result.mean_seconds > 0 else None

        comparison = BenchmarkComparison(
            primary=primary_result,
            compare=compare_result,
            speedup=speedup,
        )

        self._update_benchmark(comparison)

    def _update_benchmark(self, comparison: BenchmarkComparison) -> None:
        summary = self.query_one("#benchmark-summary", Static)
        table = self.query_one("#benchmark-table", DataTable)
        table.clear()

        primary = comparison.primary
        compare = comparison.compare

        def _fmt(value: float, unit: str = "") -> str:
            if value >= 1_000_000:
                return f"{value / 1_000_000:.2f}M{unit}"
            if value >= 1_000:
                return f"{value / 1_000:.2f}K{unit}"
            return f"{value:.2f}{unit}"

        metrics = [
            ("Tokenizer", primary.tokenizer_name, compare.tokenizer_name if compare else "n/a"),
            ("Input chars", f"{primary.input_chars:,}", f"{compare.input_chars:,}" if compare else "n/a"),
            ("Output tokens", f"{primary.output_tokens:,}", f"{compare.output_tokens:,}" if compare else "n/a"),
            ("Iterations", str(primary.iterations), str(compare.iterations) if compare else "n/a"),
            ("Total time", f"{primary.total_seconds:.4f}s", f"{compare.total_seconds:.4f}s" if compare else "n/a"),
            ("Mean time", f"{primary.mean_seconds * 1000:.3f}ms", f"{compare.mean_seconds * 1000:.3f}ms" if compare else "n/a"),
            ("Tokens/sec", _fmt(primary.tokens_per_second, "/s"), _fmt(compare.tokens_per_second, "/s") if compare else "n/a"),
            ("Chars/sec", _fmt(primary.chars_per_second, "/s"), _fmt(compare.chars_per_second, "/s") if compare else "n/a"),
        ]

        if comparison.speedup is not None:
            metrics.append(("Speedup", f"{comparison.speedup:.2f}×", "baseline"))

        for metric, primary_val, compare_val in metrics:
            table.add_row(metric, primary_val, compare_val)

        summary_text = (
            f"{primary.tokenizer_name}: {_fmt(primary.tokens_per_second, ' tok/s')} | "
            f"{primary.mean_seconds * 1000:.3f}ms/encode"
        )
        if compare:
            summary_text += (
                f" | {compare.tokenizer_name}: {_fmt(compare.tokens_per_second, ' tok/s')} | "
                f"{compare.mean_seconds * 1000:.3f}ms/encode"
            )
        summary.update(summary_text)

    @staticmethod
    def _format_token(token: str | None, token_id: int | None) -> str:
        if token is None or token_id is None:
            return "n/a"
        return f"{token!r} / {token_id}"

    @staticmethod
    def _bool_label(value: bool | None) -> str:
        if value is None:
            return "n/a"
        return "yes" if value else "no"

    @staticmethod
    def _top_pairs(values: Iterable[tuple[object, int]]) -> str:
        pairs = list(values)[:5]
        if not pairs:
            return "n/a"
        return ", ".join(f"{value}:{count}" for value, count in pairs)

    @staticmethod
    def _visible(value: str) -> str:
        return value.replace("\n", "\\n").replace("\t", "\\t")
