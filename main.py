from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Static

from analysis_models import (
    BatchPromptAnalysisResult,
    CorpusAnalysisResult,
    CorpusCompareResult,
    ExportFormat,
    PROJECT_SCHEMA_VERSION,
    PricingProfile,
    ProjectState,
    RegressionSuiteResult,
    TokenSearchMatch,
    add_recent_tokenizer,
    analyze_batch_prompts,
    analyze_corpus_path,
    analyze_rag_chunks,
    build_export_payload,
    collect_text_units,
    compare_corpus_path,
    diff_tokenizers,
    export_extension,
    estimate_token_cost,
    format_export,
    inspect_unicode,
    load_project_state,
    load_regression_suite,
    load_recent_tokenizers,
    run_regression_suite,
    save_project_state,
    simulate_packing,
    suggest_tokenizer_repairs,
)
from compare_engine import CompareResult, build_compare_result
from config import load_config
from theme import APP_CSS
from tokenizer_engine import TokenizationResult, TokenizerEngine, TokenizerLoadError
from version import __version__, app_version_label
from widgets.folder_browser import FolderBrowser
from widgets.input_bar import DebouncedTextInput
from widgets.merge_tree import MergeTreeWidget
from widgets.stats_panel import StatsPanel
from widgets.token_view import TokenView

BrowserMode = Literal["primary", "compare", "corpus", "batch"]


class TokenscopeApp(App[None]):
    CSS = APP_CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+l", "clear_input", "Clear input"),
        Binding("ctrl+o", "open_tokenizer", "Open tokenizer"),
        Binding("ctrl+p", "open_compare_tokenizer", "Open compare tokenizer"),
        Binding("ctrl+x", "clear_compare_tokenizer", "Clear compare tokenizer"),
        Binding("tab", "cycle_bottom_tab", "Cycle bottom tab"),
        Binding("ctrl+s", "save_export", "Save export"),
        Binding("[", "select_previous_token", "Previous token"),
        Binding("]", "select_next_token", "Next token"),
        Binding("ctrl+f", "focus_token_search", "Search tokens"),
        Binding("ctrl+b", "focus_budget_input", "Prompt budget"),
        Binding("ctrl+t", "toggle_encode_special_tokens", "Toggle special encoding"),
        Binding("ctrl+y", "copy_to_clipboard", "Copy token IDs"),
        Binding("backspace", "browser_parent", "Parent folder", show=False),
        Binding("escape", "cancel_browser", "Cancel browser", show=False),
    ]

    def __init__(
        self,
        tokenizer_path: str | None = None,
        compare_tokenizer_path: str | None = None,
        corpus_path: str | None = None,
        batch_path: str | None = None,
        budget_limit: int | None = None,
        export_format: ExportFormat = "json",
        project_state: ProjectState | None = None,
        encode_special_tokens: bool = False,
    ) -> None:
        super().__init__()
        if project_state is not None:
            tokenizer_path = tokenizer_path or project_state.tokenizer_path
            compare_tokenizer_path = compare_tokenizer_path or project_state.compare_tokenizer_path
            corpus_path = corpus_path or project_state.corpus_path
            batch_path = batch_path or project_state.batch_path
            budget_limit = budget_limit if budget_limit is not None else project_state.budget_limit
            if export_format == "json" and project_state.export_format in ("json", "csv", "md", "html"):
                export_format = project_state.export_format  # type: ignore[assignment]
        self.initial_tokenizer_path = tokenizer_path
        self.initial_project_state = project_state
        self.initial_input_text = project_state.input_text if project_state is not None else ""
        self.pending_compare_tokenizer_path = compare_tokenizer_path
        self.pending_corpus_path = corpus_path
        self.pending_batch_path = batch_path
        self.current_corpus_path: str | None = None
        self.current_batch_path: str | None = None
        self.primary_engine: TokenizerEngine | None = None
        self.compare_engine: TokenizerEngine | None = None
        self.primary_result: TokenizationResult | None = None
        self.compare_result: TokenizationResult | None = None
        self.comparison: CompareResult | None = None
        self.corpus_result: CorpusAnalysisResult | None = None
        self.corpus_compare_result: CorpusCompareResult | None = None
        self.batch_result: BatchPromptAnalysisResult | None = None
        self.browser_mode: BrowserMode = "primary"
        self._load_task: asyncio.Task[None] | None = None
        self._encode_task: asyncio.Task[None] | None = None
        self._corpus_task: asyncio.Task[None] | None = None
        self._batch_task: asyncio.Task[None] | None = None
        self._corpus_compare_task: asyncio.Task[None] | None = None
        self._encode_sequence = 0
        self.encode_special_tokens = (
            project_state.encode_special_tokens
            if project_state is not None
            else encode_special_tokens
        )
        self.selected_token_indices = {"primary": 0, "compare": 0}
        self.search_matches: dict[str, tuple[TokenSearchMatch, ...]] = {"primary": (), "compare": ()}
        self.budget_limit = budget_limit
        self.export_format: ExportFormat = export_format
        self.recent_tokenizers = load_recent_tokenizers(Path.cwd())

    def compose(self) -> ComposeResult:
        yield Static(f"{app_version_label()} | no tokenizer loaded", id="header")

        with Container(id="browser-screen"):
            yield FolderBrowser(id="folder-browser")

        with Vertical(id="main-layout"):
            yield DebouncedTextInput(
                placeholder="Type text here",
                id="text-input",
                debounce_seconds=0.08,
            )
            with Horizontal(id="content-row"):
                yield TokenView(id="primary-token-view")
                yield TokenView(id="compare-token-view")
                yield StatsPanel(id="stats-panel")
            yield MergeTreeWidget(
                id="bottom-panel",
                export_format=self.export_format,
                budget_limit=self.budget_limit,
            )

    def on_mount(self) -> None:
        self.query_one("#main-layout", Vertical).add_class("hidden")
        self.query_one("#compare-token-view", TokenView).add_class("hidden")
        bottom = self.query_one("#bottom-panel", MergeTreeWidget)
        bottom.set_budget_limit(self.budget_limit)
        bottom.set_export_format(self.export_format)
        bottom.set_encode_special_tokens(self.encode_special_tokens)
        if self.initial_project_state is not None:
            bottom.set_project_state(self.initial_project_state)
        self.query_one("#text-input", DebouncedTextInput).value = self.initial_input_text
        if self.initial_tokenizer_path:
            self._open_browser("primary", root=self.initial_tokenizer_path)
            self._request_tokenizer_load(self.initial_tokenizer_path, "primary", from_cli=True)
        else:
            self._open_browser("primary")

    def on_folder_browser_folder_selected(self, event: FolderBrowser.FolderSelected) -> None:
        event.stop()
        if self.browser_mode == "corpus":
            self._hide_browser()
            self.query_one("#main-layout", Vertical).remove_class("hidden")
            self._request_corpus_analysis(event.path)
            return
        if self.browser_mode == "batch":
            self._hide_browser()
            self.query_one("#main-layout", Vertical).remove_class("hidden")
            self._request_batch_analysis(event.path)
            return
        self._request_tokenizer_load(str(event.path), self.browser_mode, from_cli=False)

    async def on_folder_browser_hub_download_requested(self, event: FolderBrowser.HubDownloadRequested) -> None:
        event.stop()
        browser = self.query_one("#folder-browser", FolderBrowser)
        if self.browser_mode not in ("primary", "compare"):
            browser.set_status("Hub downloads are only available when loading tokenizers.")
            return
        browser.set_status(f"Downloading from Hub: {event.model_id}...")
        browser.set_loading(True)
        try:
            config = load_config()
            engine = await asyncio.to_thread(
                TokenizerEngine.load_from_hub,
                event.model_id,
                cache_dir=config.hub_cache_dir
            )
            self._request_tokenizer_load(str(engine.source_path), self.browser_mode, from_cli=False)
        except Exception as exc:
            browser.set_status(f"Hub download failed: {exc}")
        finally:
            browser.set_loading(False)

    def on_debounced_text_input_debounced(self, event: DebouncedTextInput.Debounced) -> None:
        event.stop()
        self._schedule_encode(event.value)

    def _request_tokenizer_load(
        self,
        path_value: str,
        target: BrowserMode,
        *,
        from_cli: bool,
    ) -> None:
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()

        if not from_cli or target == "primary" or self.primary_engine is None:
            self._show_browser_loading(f"Loading {path_value}")
        elif target == "compare":
            self.notify(f"Loading compare tokenizer: {path_value}")

        self._load_task = asyncio.create_task(
            self._load_tokenizer(path_value, target, from_cli=from_cli)
        )

    async def _load_tokenizer(
        self,
        path_value: str,
        target: BrowserMode,
        *,
        from_cli: bool,
    ) -> None:
        try:
            engine = await asyncio.to_thread(TokenizerEngine.load, path_value)
        except asyncio.CancelledError:
            return
        except (TokenizerLoadError, OSError, ValueError) as exc:
            self._show_load_error(str(exc), target, from_cli=from_cli)
            return
        except Exception as exc:
            self._show_load_error(f"Unexpected tokenizer load error: {exc}", target, from_cli=from_cli)
            return

        if target == "primary":
            self.primary_engine = engine
            self.primary_result = None
            self.selected_token_indices["primary"] = 0
        else:
            self.compare_engine = engine
            self.compare_result = None
            self.selected_token_indices["compare"] = 0

        self.recent_tokenizers = add_recent_tokenizer(engine.source_path, Path.cwd())

        self._hide_browser()
        self.query_one("#main-layout", Vertical).remove_class("hidden")
        self.query_one("#text-input", DebouncedTextInput).focus()
        self._sync_engine_dependent_widgets()
        self._schedule_encode(self.query_one("#text-input", DebouncedTextInput).value)
        self._update_header()

        if target == "primary" and self.pending_compare_tokenizer_path:
            compare_path = self.pending_compare_tokenizer_path
            self.pending_compare_tokenizer_path = None
            self._request_tokenizer_load(compare_path, "compare", from_cli=True)

        if target == "primary" and self.pending_corpus_path:
            corpus_path = self.pending_corpus_path
            self.pending_corpus_path = None
            self._request_corpus_analysis(corpus_path)

        if target == "primary" and self.pending_batch_path:
            batch_path = self.pending_batch_path
            self.pending_batch_path = None
            self._request_batch_analysis(batch_path)

        if target == "compare" and self.current_corpus_path:
            self._request_corpus_compare(self.current_corpus_path)

        if target == "compare" and self.current_batch_path:
            self._request_batch_analysis(self.current_batch_path)

    def _show_load_error(self, message: str, target: BrowserMode, *, from_cli: bool) -> None:
        if target == "compare" and from_cli and self.primary_engine is not None:
            self.notify(f"Compare tokenizer failed: {message}", severity="error")
            self._hide_browser()
            self.query_one("#main-layout", Vertical).remove_class("hidden")
            return

        browser = self.query_one("#folder-browser", FolderBrowser)
        browser.set_loading(False)
        browser.set_status(f"[red]Error:[/red] {message}")
        self.query_one("#browser-screen", Container).remove_class("hidden")
        if self.primary_engine is None:
            self.query_one("#main-layout", Vertical).add_class("hidden")
        browser.focus_tree()

    def _show_browser_loading(self, message: str) -> None:
        browser = self.query_one("#folder-browser", FolderBrowser)
        browser.set_status(message)
        browser.set_loading(True)
        self.query_one("#browser-screen", Container).remove_class("hidden")

    def _open_browser(self, mode: BrowserMode, root: str | Path | None = None) -> None:
        self.browser_mode = mode
        if mode == "primary":
            title = "Select primary tokenizer folder"
            help_text = "Enter selects a folder. Backspace moves to the parent folder."
            default_root = root or (self.primary_engine.source_path if self.primary_engine else Path.cwd())
            allow_files = False
            recent_paths = self.recent_tokenizers
        elif mode == "compare":
            title = "Select compare tokenizer folder"
            help_text = "Enter selects a second tokenizer for side-by-side comparison. Esc cancels."
            default_root = root or (self.compare_engine.source_path if self.compare_engine else Path.cwd())
            allow_files = False
            recent_paths = self.recent_tokenizers
        else:
            if mode == "corpus":
                title = "Select corpus file or folder"
                help_text = "Enter selects a local .txt, .md, .jsonl, .json, .csv file or a folder."
            else:
                title = "Select batch prompt file or folder"
                help_text = "Enter selects a local prompt file or folder for batch analysis."
            default_root = root or Path.cwd()
            allow_files = True
            recent_paths = []

        browser = self.query_one("#folder-browser", FolderBrowser)
        browser.configure(
            title,
            help_text,
            default_root,
            allow_files=allow_files,
            allow_hub=mode in ("primary", "compare"),
            recent_paths=recent_paths,
        )
        self.query_one("#browser-screen", Container).remove_class("hidden")
        self.query_one("#main-layout", Vertical).add_class("hidden")
        browser.focus_tree()

    def _hide_browser(self) -> None:
        browser = self.query_one("#folder-browser", FolderBrowser)
        browser.set_loading(False)
        self.query_one("#browser-screen", Container).add_class("hidden")

    def _schedule_encode(self, text: str) -> None:
        if self.primary_engine is None:
            return
        self._encode_sequence += 1
        sequence = self._encode_sequence
        primary_engine = self.primary_engine
        compare_engine = self.compare_engine
        encode_special_tokens = self.encode_special_tokens
        if self._encode_task and not self._encode_task.done():
            self._encode_task.cancel()
        self._encode_task = asyncio.create_task(
            self._encode_text(sequence, text, primary_engine, compare_engine, encode_special_tokens)
        )

    async def _encode_text(
        self,
        sequence: int,
        text: str,
        primary_engine: TokenizerEngine,
        compare_engine: TokenizerEngine | None,
        encode_special_tokens: bool,
    ) -> None:
        try:
            primary_result, compare_result, comparison = await asyncio.to_thread(
                self._encode_all,
                text,
                primary_engine,
                compare_engine,
                encode_special_tokens,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.notify(f"Tokenization failed: {exc}", severity="error")
            return
        if sequence != self._encode_sequence:
            return
        self.primary_result = primary_result
        self.compare_result = compare_result
        self.comparison = comparison
        self._render_results()

    @staticmethod
    def _encode_all(
        text: str,
        primary_engine: TokenizerEngine,
        compare_engine: TokenizerEngine | None,
        encode_special_tokens: bool = False,
    ) -> tuple[TokenizationResult, TokenizationResult | None, CompareResult | None]:
        primary_result = primary_engine.encode(text, encode_special_tokens=encode_special_tokens)
        if compare_engine is None:
            return primary_result, None, None
        compare_result = compare_engine.encode(text, encode_special_tokens=encode_special_tokens)
        return primary_result, compare_result, build_compare_result(primary_result, compare_result)

    def _render_results(self) -> None:
        try:
            primary_view = self.query_one("#primary-token-view", TokenView)
        except NoMatches:
            return
        self._clamp_selected_tokens()
        primary_matches = {match.index for match in self.search_matches.get("primary", ())}
        compare_matches = {match.index for match in self.search_matches.get("compare", ())}
        primary_view.update_tokens(
            self.primary_result,
            selected_index=self.selected_token_indices["primary"],
            match_indices=primary_matches,
        )
        compare_view = self.query_one("#compare-token-view", TokenView)
        if self.compare_engine is None:
            compare_view.add_class("hidden")
            compare_view.update_tokens(None)
            self.query_one("#stats-panel", StatsPanel).update_stats(self.primary_result)
        else:
            compare_view.remove_class("hidden")
            compare_view.update_tokens(
                self.compare_result,
                selected_index=self.selected_token_indices["compare"],
                match_indices=compare_matches,
            )
            self.query_one("#stats-panel", StatsPanel).update_comparison(self.comparison)

        self.query_one("#bottom-panel", MergeTreeWidget).update_results(
            self.primary_result,
            self.compare_result,
            self.comparison,
        )

    def _clamp_selected_tokens(self) -> None:
        for source, result in (("primary", self.primary_result), ("compare", self.compare_result)):
            if result is None or not result.spans:
                self.selected_token_indices[source] = 0
            else:
                self.selected_token_indices[source] = max(
                    0,
                    min(self.selected_token_indices[source], len(result.spans) - 1),
                )

    def _sync_engine_dependent_widgets(self) -> None:
        self.query_one("#bottom-panel", MergeTreeWidget).set_engines(
            self.primary_engine,
            self.compare_engine,
        )
        self.query_one("#bottom-panel", MergeTreeWidget).set_budget_limit(self.budget_limit)
        self.query_one("#bottom-panel", MergeTreeWidget).set_export_format(self.export_format)
        self.query_one("#bottom-panel", MergeTreeWidget).set_encode_special_tokens(
            self.encode_special_tokens
        )
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_result(self.corpus_result)
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_compare_result(
            self.corpus_compare_result
        )
        self.query_one("#bottom-panel", MergeTreeWidget).set_batch_result(self.batch_result)
        self.query_one("#stats-panel", StatsPanel).set_budget_limit(self.budget_limit)
        if self.compare_engine is None:
            self.query_one("#compare-token-view", TokenView).add_class("hidden")
        else:
            self.query_one("#compare-token-view", TokenView).remove_class("hidden")

    def _update_header(self) -> None:
        if self.primary_engine is None:
            label = f"{app_version_label()} | no tokenizer loaded"
        else:
            label = f"{app_version_label()} | primary: {self._engine_label(self.primary_engine)}"
            if self.compare_engine is not None:
                label += f" | compare: {self._engine_label(self.compare_engine)}"
            if self.encode_special_tokens:
                label += " | encode specials: on"
            budget_badge = self._budget_badge()
            if budget_badge:
                label += f" | {budget_badge}"
        self.query_one("#header", Static).update(label)

    def _budget_badge(self) -> str:
        """Return a color-coded budget badge for the header, or empty string."""
        if self.budget_limit is None or self.budget_limit <= 0:
            return ""
        if self.primary_result is None:
            return ""
        used = self.primary_result.stats.token_count
        pct = (used / self.budget_limit) * 100.0
        remaining = self.budget_limit - used
        if pct > 100:
            return f"[bold red]OVER BUDGET {used:,}/{self.budget_limit:,} ({pct:.0f}%)[/bold red]"
        if pct >= 95:
            return f"[red]budget: {used:,}/{self.budget_limit:,} ({pct:.0f}%)[/red]"
        if pct >= 80:
            return f"[yellow]budget: {used:,}/{self.budget_limit:,} ({pct:.0f}%)[/yellow]"
        return f"[green]budget: {used:,}/{self.budget_limit:,} ({pct:.0f}%)[/green]"

    @staticmethod
    def _engine_label(engine: TokenizerEngine) -> str:
        return f"{engine.name} | {engine.tokenizer_type} | vocab: {engine.vocab_size}"

    def action_clear_input(self) -> None:
        if self.query_one("#main-layout", Vertical).has_class("hidden"):
            return
        input_widget = self.query_one("#text-input", DebouncedTextInput)
        input_widget.value = ""
        input_widget.emit_now()

    def on_merge_tree_widget_token_selected(self, event: MergeTreeWidget.TokenSelected) -> None:
        event.stop()
        self.selected_token_indices[event.source] = event.index
        self._render_results()

    def on_merge_tree_widget_search_changed(self, event: MergeTreeWidget.SearchChanged) -> None:
        event.stop()
        self.search_matches[event.source] = event.matches
        self._render_results()

    def on_merge_tree_widget_budget_changed(self, event: MergeTreeWidget.BudgetChanged) -> None:
        event.stop()
        self.budget_limit = event.limit
        self.query_one("#stats-panel", StatsPanel).set_budget_limit(self.budget_limit)
        self._render_results()
        self._update_header()

    def on_merge_tree_widget_export_format_changed(
        self,
        event: MergeTreeWidget.ExportFormatChanged,
    ) -> None:
        event.stop()
        self.export_format = event.export_format

    def on_merge_tree_widget_corpus_browse_requested(
        self,
        event: MergeTreeWidget.CorpusBrowseRequested,
    ) -> None:
        event.stop()
        self.action_open_corpus()

    def on_merge_tree_widget_batch_browse_requested(
        self,
        event: MergeTreeWidget.BatchBrowseRequested,
    ) -> None:
        event.stop()
        self.action_open_batch()

    def on_merge_tree_widget_project_save_requested(
        self,
        event: MergeTreeWidget.ProjectSaveRequested,
    ) -> None:
        event.stop()
        save_project_state(self._current_project_state(), event.path)
        self.notify(f"Saved project {Path(event.path).name}")

    def on_merge_tree_widget_project_load_requested(
        self,
        event: MergeTreeWidget.ProjectLoadRequested,
    ) -> None:
        event.stop()
        try:
            project = load_project_state(event.path)
        except Exception as exc:
            self.notify(f"Project load failed: {exc}", severity="error")
            return
        self._apply_project_state(project)
        self.notify(f"Loaded project {Path(event.path).name}")

    def on_merge_tree_widget_encode_special_toggle_requested(
        self,
        event: MergeTreeWidget.EncodeSpecialToggleRequested,
    ) -> None:
        event.stop()
        self.action_toggle_encode_special_tokens()

    def action_open_tokenizer(self) -> None:
        self._open_browser("primary")

    def action_open_compare_tokenizer(self) -> None:
        if self.primary_engine is None:
            self.notify("Load a primary tokenizer first.", severity="warning")
            return
        self._open_browser("compare")

    def action_open_corpus(self) -> None:
        if self.primary_engine is None:
            self.notify("Load a primary tokenizer before analyzing a corpus.", severity="warning")
            return
        self._open_browser("corpus")

    def action_open_batch(self) -> None:
        if self.primary_engine is None:
            self.notify("Load a primary tokenizer before analyzing batch prompts.", severity="warning")
            return
        self._open_browser("batch")

    def action_clear_compare_tokenizer(self) -> None:
        self.compare_engine = None
        self.compare_result = None
        self.comparison = None
        self.corpus_compare_result = None
        self.selected_token_indices["compare"] = 0
        self.search_matches["compare"] = ()
        self._sync_engine_dependent_widgets()
        self._update_header()
        self._schedule_encode(self.query_one("#text-input", DebouncedTextInput).value)
        self.notify("Compare tokenizer cleared.")

    def action_select_previous_token(self) -> None:
        self._move_selected_token(-1)

    def action_select_next_token(self) -> None:
        self._move_selected_token(1)

    def action_focus_token_search(self) -> None:
        if self.query_one("#main-layout", Vertical).has_class("hidden"):
            return
        self.query_one("#bottom-panel", MergeTreeWidget).focus_token_search()

    def action_focus_budget_input(self) -> None:
        if self.query_one("#main-layout", Vertical).has_class("hidden"):
            return
        self.query_one("#bottom-panel", MergeTreeWidget).focus_budget_input()

    def action_toggle_encode_special_tokens(self) -> None:
        if self.primary_engine is None:
            return
        self.encode_special_tokens = not self.encode_special_tokens
        self.query_one("#bottom-panel", MergeTreeWidget).set_encode_special_tokens(
            self.encode_special_tokens
        )
        self._update_header()
        self._schedule_encode(self.query_one("#text-input", DebouncedTextInput).value)
        if self.current_corpus_path and self.compare_engine is not None:
            self._request_corpus_compare(self.current_corpus_path)
        if self.current_batch_path:
            self._request_batch_analysis(self.current_batch_path)
        state = "on" if self.encode_special_tokens else "off"
        self.notify(f"Encode special tokens: {state}")

    def action_copy_to_clipboard(self) -> None:
        """Copy current token IDs to the system clipboard."""
        if self.primary_result is None:
            self.notify("Nothing to copy yet.", severity="warning")
            return
        ids_text = " ".join(str(tid) for tid in self.primary_result.token_ids)
        try:
            import pyperclip  # type: ignore[import-untyped]
            pyperclip.copy(ids_text)
            self.notify(f"Copied {len(self.primary_result.token_ids)} token IDs to clipboard.")
        except ImportError:
            self.notify("pyperclip not installed. Install with: pip install pyperclip", severity="warning")
        except Exception as exc:
            self.notify(f"Clipboard copy failed: {exc}", severity="error")

    def _move_selected_token(self, delta: int) -> None:
        if self.query_one("#main-layout", Vertical).has_class("hidden"):
            return
        bottom = self.query_one("#bottom-panel", MergeTreeWidget)
        source = bottom.selected_source
        if source == "compare" and self.compare_result is None:
            source = "primary"
        result = self.compare_result if source == "compare" else self.primary_result
        if result is None or not result.spans:
            return
        next_index = (self.selected_token_indices[source] + delta) % len(result.spans)
        self.selected_token_indices[source] = next_index
        bottom.set_selected_token(source, next_index)
        self._render_results()

    def action_browser_parent(self) -> None:
        if self.query_one("#browser-screen", Container).has_class("hidden"):
            return
        self.query_one("#folder-browser", FolderBrowser).parent_root()

    def action_cancel_browser(self) -> None:
        if self.query_one("#browser-screen", Container).has_class("hidden"):
            return
        if self.primary_engine is None:
            self.query_one("#folder-browser", FolderBrowser).set_status(
                "Select a primary tokenizer folder to continue."
            )
            return
        self._hide_browser()
        self.query_one("#main-layout", Vertical).remove_class("hidden")
        self.query_one("#text-input", DebouncedTextInput).focus()

    def action_cycle_bottom_tab(self) -> None:
        if self.query_one("#main-layout", Vertical).has_class("hidden"):
            return
        self.query_one("#bottom-panel", MergeTreeWidget).cycle_tab()

    def _current_project_state(self) -> ProjectState:
        bottom = self.query_one("#bottom-panel", MergeTreeWidget)
        tabs = bottom.query_one("#bottom-tabs")
        tokenizer_path = str(self.primary_engine.source_path) if self.primary_engine is not None else self.initial_tokenizer_path
        compare_path = str(self.compare_engine.source_path) if self.compare_engine is not None else self.pending_compare_tokenizer_path
        return ProjectState(
            version=PROJECT_SCHEMA_VERSION,
            tokenizer_path=tokenizer_path,
            compare_tokenizer_path=compare_path,
            input_text=self.query_one("#text-input", DebouncedTextInput).value,
            encode_special_tokens=self.encode_special_tokens,
            export_format=self.export_format,
            budget_limit=self.budget_limit,
            chat_messages=tuple(bottom.chat_messages),
            add_generation_prompt=bottom.add_generation_prompt,
            corpus_path=self.current_corpus_path or self.pending_corpus_path,
            batch_path=self.current_batch_path or self.pending_batch_path,
            active_tab=str(getattr(tabs, "active", "token-table-tab")),
            selected_source=bottom.selected_source,
            pricing_profile=bottom.cost_profile,
        )

    def _apply_project_state(self, project: ProjectState) -> None:
        self.initial_project_state = project
        self.initial_input_text = project.input_text
        self.pending_compare_tokenizer_path = project.compare_tokenizer_path
        self.pending_corpus_path = project.corpus_path
        self.pending_batch_path = project.batch_path
        self.budget_limit = project.budget_limit
        self.export_format = project.export_format if project.export_format in ("json", "csv", "md", "html") else "json"
        self.encode_special_tokens = project.encode_special_tokens
        text_input = self.query_one("#text-input", DebouncedTextInput)
        text_input.value = project.input_text
        bottom = self.query_one("#bottom-panel", MergeTreeWidget)
        bottom.set_budget_limit(self.budget_limit)
        bottom.set_export_format(self.export_format)
        bottom.set_encode_special_tokens(self.encode_special_tokens)
        bottom.set_project_state(project)
        if project.tokenizer_path:
            self._request_tokenizer_load(project.tokenizer_path, "primary", from_cli=True)
        elif self.primary_engine is not None:
            self._schedule_encode(project.input_text)

    def _request_corpus_analysis(self, path_value: str | Path) -> None:
        if self.primary_engine is None:
            self.pending_corpus_path = str(path_value)
            return
        if self._corpus_task and not self._corpus_task.done():
            self._corpus_task.cancel()
        self.current_corpus_path = str(path_value)
        self.corpus_result = None
        self.corpus_compare_result = None
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_result(None)
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_compare_result(None)
        self.notify(f"Analyzing corpus: {path_value}")
        self._corpus_task = asyncio.create_task(
            self._analyze_corpus(path_value, self.primary_engine)
        )

    async def _analyze_corpus(self, path_value: str | Path, engine: TokenizerEngine) -> None:
        try:
            result = await asyncio.to_thread(analyze_corpus_path, path_value, engine)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.notify(f"Corpus analysis failed: {exc}", severity="error")
            return
        self.corpus_result = result
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_result(result)
        if self.compare_engine is not None:
            self._request_corpus_compare(path_value)
        self.notify(
            f"Analyzed {result.total_files:,} files, {result.total_tokens:,} tokens."
        )

    def _request_corpus_compare(self, path_value: str | Path) -> None:
        if self.primary_engine is None or self.compare_engine is None:
            return
        if self._corpus_compare_task and not self._corpus_compare_task.done():
            self._corpus_compare_task.cancel()
        self.corpus_compare_result = None
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_compare_result(None)
        self._corpus_compare_task = asyncio.create_task(
            self._compare_corpus(path_value, self.primary_engine, self.compare_engine)
        )

    async def _compare_corpus(
        self,
        path_value: str | Path,
        primary_engine: TokenizerEngine,
        compare_engine: TokenizerEngine,
    ) -> None:
        try:
            result = await asyncio.to_thread(
                compare_corpus_path,
                path_value,
                primary_engine,
                compare_engine,
                encode_special_tokens=self.encode_special_tokens,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.notify(f"Corpus compare failed: {exc}", severity="error")
            return
        self.corpus_compare_result = result
        self.query_one("#bottom-panel", MergeTreeWidget).set_corpus_compare_result(result)

    def _request_batch_analysis(self, path_value: str | Path) -> None:
        if self.primary_engine is None:
            self.pending_batch_path = str(path_value)
            return
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
        self.current_batch_path = str(path_value)
        self.batch_result = None
        self.query_one("#bottom-panel", MergeTreeWidget).set_batch_result(None)
        self.notify(f"Analyzing batch prompts: {path_value}")
        self._batch_task = asyncio.create_task(
            self._analyze_batch(path_value, self.primary_engine, self.compare_engine)
        )

    async def _analyze_batch(
        self,
        path_value: str | Path,
        primary_engine: TokenizerEngine,
        compare_engine: TokenizerEngine | None,
    ) -> None:
        try:
            result = await asyncio.to_thread(
                analyze_batch_prompts,
                path_value,
                primary_engine,
                compare_engine,
                budget_limit=self.budget_limit,
                encode_special_tokens=self.encode_special_tokens,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.notify(f"Batch analysis failed: {exc}", severity="error")
            return
        self.batch_result = result
        self.query_one("#bottom-panel", MergeTreeWidget).set_batch_result(result)
        self.notify(f"Analyzed {result.total_prompts:,} prompts.")

    def action_save_export(self) -> None:
        if self.primary_result is None or self.primary_engine is None:
            self.notify("Nothing to export yet.", severity="warning")
            return

        output = Path.cwd() / f"tokenscope_export.{export_extension(self.export_format)}"
        bottom = self.query_one("#bottom-panel", MergeTreeWidget)
        payload = build_export_payload(
            self.primary_engine,
            self.primary_result,
            self.compare_engine,
            self.comparison,
            self.budget_limit,
            self.corpus_result,
            bottom.current_chat_budget(),
            self.batch_result,
            self.corpus_compare_result,
            bottom.current_pipeline_debug(),
            project_state=self._current_project_state(),
            tokenizer_diff=bottom.current_tokenizer_diff(),
            packing_result=bottom.current_packing(),
            regression_result=bottom.regression_result,
            unicode_result=bottom.current_unicode(),
            rag_result=bottom.current_rag(),
            distribution_result=bottom.current_distribution(),
            cost_estimate=bottom.current_cost(),
            repair_result=bottom.current_repair(),
        )
        output.write_text(format_export(payload, self.export_format), encoding="utf-8")
        self.notify(f"Saved {output.name}")

    @staticmethod
    def _tokenizer_metadata(engine: TokenizerEngine) -> dict[str, object]:
        return {
            "name": engine.name,
            "type": engine.tokenizer_type,
            "vocab_size": engine.vocab_size,
            "source_path": str(engine.source_path),
        }


def run_headless_analyze(args: argparse.Namespace) -> int:
    primary_engine = TokenizerEngine.load(args.tokenizer)
    compare_engine = TokenizerEngine.load(args.compare_tokenizer) if args.compare_tokenizer else None
    text = _read_headless_input(args)
    primary_result, compare_result, comparison = TokenscopeApp._encode_all(
        text,
        primary_engine,
        compare_engine,
        False,
    )

    corpus_result = analyze_corpus_path(args.corpus_path, primary_engine) if args.corpus_path else None
    corpus_compare_result = (
        compare_corpus_path(args.corpus_path, primary_engine, compare_engine)
        if args.corpus_path and compare_engine is not None
        else None
    )
    batch_result = (
        analyze_batch_prompts(
            args.batch_path,
            primary_engine,
            compare_engine,
            budget_limit=args.budget,
        )
        if args.batch_path
        else None
    )
    tokenizer_diff = diff_tokenizers(primary_engine, compare_engine)
    packing_result = simulate_packing(
        primary_engine,
        text,
        budget_limit=args.budget,
        strategy="head_tail",
    )
    unicode_result = inspect_unicode(text)
    regression_result: RegressionSuiteResult | None = None
    if args.regression_suite:
        suite_name, cases = load_regression_suite(args.regression_suite)
        regression_result = run_regression_suite(suite_name, cases, primary_engine)
    units = collect_text_units(args.corpus_path) if args.corpus_path else []
    rag_result = (
        analyze_rag_chunks(
            primary_engine,
            units or [text],
            max_tokens=args.rag_max_tokens,
            overlap_tokens=args.rag_overlap_tokens,
        )
        if args.rag_max_tokens
        else None
    )
    input_tokens = primary_result.stats.token_count
    if batch_result is not None:
        input_tokens += batch_result.primary_total_tokens
    if corpus_result is not None:
        input_tokens += corpus_result.total_tokens
    cost_estimate = estimate_token_cost(
        PricingProfile(
            "cli",
            args.input_cost_per_million,
            args.output_cost_per_million,
            args.estimated_output_tokens,
        ),
        input_tokens=input_tokens,
    )
    repair_result = suggest_tokenizer_repairs(primary_engine)

    payload = build_export_payload(
        primary_engine,
        primary_result,
        compare_engine,
        comparison,
        args.budget,
        corpus_result,
        None,
        batch_result,
        corpus_compare_result,
        None,
        tokenizer_diff=tokenizer_diff,
        packing_result=packing_result,
        regression_result=regression_result,
        unicode_result=unicode_result,
        rag_result=rag_result,
        cost_estimate=cost_estimate,
        repair_result=repair_result,
    )
    rendered = format_export(payload, args.export_format)
    if args.export_path:
        Path(args.export_path).expanduser().write_text(rendered, encoding="utf-8")
    else:
        output_bytes = rendered.encode("utf-8")
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(output_bytes)
            if not rendered.endswith("\n"):
                sys.stdout.buffer.write(b"\n")
        else:
            sys.stdout.write(rendered.encode("utf-8", errors="replace").decode("utf-8"))
            if not rendered.endswith("\n"):
                sys.stdout.write("\n")

    failed = False
    if args.fail_on_budget and args.budget:
        failed = primary_result.stats.token_count > args.budget
        failed = failed or bool(batch_result and batch_result.budget_failures)
    if args.fail_on_regression:
        failed = failed or bool(comparison and comparison.summary.token_count_delta > 0)
        failed = failed or bool(corpus_compare_result and corpus_compare_result.token_delta > 0)
        failed = failed or bool(batch_result and batch_result.token_delta and batch_result.token_delta > 0)
        failed = failed or bool(regression_result and regression_result.failed_cases)
    return 2 if failed else 0


def _read_headless_input(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).expanduser().read_text(encoding="utf-8")
    if args.input:
        return str(args.input)
    # Read from stdin if piped (not a TTY) or --stdin was explicitly set.
    if getattr(args, "stdin", False) or (not sys.stdin.isatty()):
        return sys.stdin.read()
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline tokenizer explorer TUI.")
    parser.add_argument("--version", action="version", version=f"tokenscope {__version__}")
    parser.add_argument("--tokenizer", help="Local tokenizer directory or tokenizer.json path.")
    parser.add_argument(
        "--hub",
        help="HuggingFace Hub model ID to download and load (e.g., meta-llama/Llama-3.1-8B).",
    )
    parser.add_argument(
        "--compare-tokenizer",
        help="Optional local tokenizer directory or tokenizer.json path to compare.",
    )
    parser.add_argument(
        "--file",
        dest="corpus_path",
        help="Optional local corpus file or folder to analyze after tokenizer load.",
    )
    parser.add_argument(
        "--batch",
        dest="batch_path",
        help="Optional local prompt file or folder to analyze after tokenizer load.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="Initial prompt budget token limit.",
    )
    parser.add_argument(
        "--export-format",
        choices=("json", "csv", "md", "html"),
        default=None,
        help="Default Ctrl+S export format.",
    )
    parser.add_argument(
        "--project",
        help="Optional TokenScope project JSON file to load at startup.",
    )

    subparsers = parser.add_subparsers(dest="command")
    init_cfg = subparsers.add_parser("init-config", help="Initialize a default .tokenscoperc configuration file.")
    init_cfg.add_argument("--path", help="Custom path/filename to write the configuration file to.")

    analyze = subparsers.add_parser("analyze", help="Run headless tokenizer analysis.")
    analyze.add_argument("--tokenizer", help="Local tokenizer directory or tokenizer.json path.")
    analyze.add_argument(
        "--hub",
        help="HuggingFace Hub model ID to download and load.",
    )
    analyze.add_argument("--compare-tokenizer", help="Optional tokenizer to compare.")
    analyze.add_argument("--input", default="", help="Inline text to tokenize.")
    analyze.add_argument("--input-file", help="Local UTF-8 text file to tokenize.")
    analyze.add_argument("--stdin", action="store_true", help="Read input text from stdin.")
    analyze.add_argument("--file", dest="corpus_path", help="Optional local corpus file or folder.")
    analyze.add_argument("--batch", dest="batch_path", help="Optional local prompt file or folder.")
    analyze.add_argument("--budget", type=int, help="Prompt budget token limit.")
    analyze.add_argument("--export", dest="export_path", help="Output report path. Prints to stdout when omitted.")
    analyze.add_argument(
        "--export-format",
        choices=("json", "csv", "md", "html"),
        default=None,
        help="Output report format.",
    )
    analyze.add_argument("--fail-on-budget", action="store_true", help="Exit nonzero when budget is exceeded.")
    analyze.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when compare/regression checks fail.")
    analyze.add_argument("--regression-suite", help="Optional regression suite JSON file.")
    analyze.add_argument("--rag-max-tokens", type=int, help="Run RAG chunking analysis with this chunk token limit.")
    analyze.add_argument("--rag-overlap-tokens", type=int, default=0, help="Token overlap for RAG chunking.")
    analyze.add_argument("--input-cost-per-million", type=float, default=0.0, help="Input-token cost per 1M tokens.")
    analyze.add_argument("--output-cost-per-million", type=float, default=0.0, help="Output-token cost per 1M tokens.")
    analyze.add_argument("--estimated-output-tokens", type=int, default=0, help="Estimated output tokens for cost analysis.")
    return parser.parse_args()


def _resolve_tokenizer_path(
    args: argparse.Namespace,
    config=None,
) -> str | None:
    """Resolve --tokenizer or --hub into a local path."""
    if getattr(args, "tokenizer", None):
        return args.tokenizer
    hub_id = getattr(args, "hub", None)
    if hub_id:
        config = config or load_config()
        engine = TokenizerEngine.load_from_hub(
            hub_id,
            cache_dir=config.hub_cache_dir,
        )
        return str(engine.source_path)
    return None


def main() -> None:
    args = parse_args()

    if args.command == "init-config":
        from config import TokenScopeConfig, save_config
        target_path = Path(args.path) if args.path else None
        try:
            save_config(TokenScopeConfig(), target_path)
            dest = target_path or (Path.cwd() / ".tokenscoperc")
            print(f"Successfully initialized default configuration file at {dest}")
        except Exception as exc:
            print(f"Error: Failed to write configuration file: {exc}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)

    # Load config file defaults.
    config = load_config()

    if args.command == "analyze":
        if not args.tokenizer and not args.hub:
            if config.default_tokenizer:
                args.tokenizer = config.default_tokenizer
            elif config.default_hub_model:
                args.hub = config.default_hub_model
        args.compare_tokenizer = args.compare_tokenizer or config.default_compare_tokenizer
        args.corpus_path = args.corpus_path or config.default_corpus_path
        args.batch_path = args.batch_path or config.default_batch_path
        args.budget = args.budget if args.budget is not None else config.default_budget
        args.export_format = args.export_format or config.default_export_format or "json"
        # Resolve hub before headless analysis.
        tokenizer_path = _resolve_tokenizer_path(args, config)
        if tokenizer_path:
            args.tokenizer = tokenizer_path
        if not args.tokenizer:
            print("Error: --tokenizer or --hub is required for analyze mode.", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(run_headless_analyze(args))

    project_state = load_project_state(args.project) if args.project else None
    if args.project:
        assert project_state is not None
        args.tokenizer = args.tokenizer or project_state.tokenizer_path
        args.compare_tokenizer = args.compare_tokenizer or project_state.compare_tokenizer_path
        args.corpus_path = args.corpus_path or project_state.corpus_path
        args.batch_path = args.batch_path or project_state.batch_path
        args.budget = args.budget if args.budget is not None else project_state.budget_limit

    # Apply config defaults where CLI didn't specify.
    tokenizer_path = args.tokenizer
    hub_id = getattr(args, "hub", None)
    if not tokenizer_path and hub_id:
        engine = TokenizerEngine.load_from_hub(hub_id, cache_dir=config.hub_cache_dir)
        tokenizer_path = str(engine.source_path)
    if not tokenizer_path and config.default_tokenizer:
        tokenizer_path = config.default_tokenizer
    if not tokenizer_path and config.default_hub_model:
        try:
            engine = TokenizerEngine.load_from_hub(
                config.default_hub_model,
                cache_dir=config.hub_cache_dir,
            )
            tokenizer_path = str(engine.source_path)
        except Exception:
            pass

    export_format = args.export_format or config.default_export_format or "json"
    budget = args.budget if args.budget is not None else config.default_budget
    compare_tokenizer = getattr(args, "compare_tokenizer", None) or config.default_compare_tokenizer
    corpus_path = getattr(args, "corpus_path", None) or config.default_corpus_path
    batch_path = getattr(args, "batch_path", None) or config.default_batch_path

    TokenscopeApp(
        tokenizer_path=tokenizer_path,
        compare_tokenizer_path=compare_tokenizer,
        corpus_path=corpus_path,
        batch_path=batch_path,
        budget_limit=budget,
        export_format=export_format,
        project_state=project_state,
        encode_special_tokens=config.encode_special_tokens,
    ).run()


if __name__ == "__main__":
    main()
