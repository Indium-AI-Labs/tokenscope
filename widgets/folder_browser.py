from __future__ import annotations

from pathlib import Path
from typing import Iterable

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, DirectoryTree, Input, Label, LoadingIndicator, Static


class BrowserDirectoryTree(DirectoryTree):
    show_files = False
    allowed_suffixes: set[str] = set()

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        filtered: list[Path] = []
        for path in paths:
            if self._safe_is_dir(path):
                filtered.append(path)
            elif self.show_files and path.suffix.lower() in self.allowed_suffixes:
                filtered.append(path)
        return filtered


class FolderBrowser(Vertical):
    class FolderSelected(Message):
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    class HubDownloadRequested(Message):
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            super().__init__()

    DEFAULT_ALLOWED_SUFFIXES = {".txt", ".md", ".jsonl", ".json", ".csv"}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._recent_paths: list[Path] = []

    def compose(self):
        yield Label("Select tokenizer folder", id="browser-title")
        yield Static("", id="browser-help")
        with Horizontal(id="hub-controls"):
            yield Input(placeholder="HuggingFace Hub model ID (e.g., gpt2)", id="hub-model-id")
            yield Button("Load from Hub", id="hub-load")
        yield Static("", id="browser-root")
        yield DataTable(id="recent-tokenizers")
        yield BrowserDirectoryTree(Path.cwd(), id="folder-tree")
        yield Static("", id="browser-status")
        yield LoadingIndicator(id="browser-loading")

    def on_mount(self) -> None:
        self.query_one("#browser-loading", LoadingIndicator).add_class("hidden")
        recent = self.query_one("#recent-tokenizers", DataTable)
        recent.cursor_type = "row"
        recent.add_columns("recent tokenizer folders")
        recent.add_class("hidden")

    def configure(
        self,
        title: str,
        help_text: str,
        root: Path | str | None = None,
        *,
        allow_files: bool = False,
        allow_hub: bool = True,
        allowed_suffixes: set[str] | None = None,
        recent_paths: Iterable[str | Path] | None = None,
    ) -> None:
        root_path = Path(root or Path.cwd()).expanduser()
        if root_path.is_file():
            root_path = root_path.parent
        if not root_path.exists():
            root_path = Path.cwd()
        self.query_one("#browser-title", Label).update(title)
        self.query_one("#browser-help", Static).update(help_text)
        tree = self.query_one("#folder-tree", BrowserDirectoryTree)
        tree.show_files = allow_files
        tree.allowed_suffixes = allowed_suffixes or self.DEFAULT_ALLOWED_SUFFIXES
        hub_controls = self.query_one("#hub-controls", Horizontal)
        if allow_hub:
            hub_controls.remove_class("hidden")
        else:
            hub_controls.add_class("hidden")
            self.query_one("#hub-model-id", Input).value = ""
        self.set_recent_paths(recent_paths or [])
        self.set_root(root_path)
        self.set_status("")
        self.set_loading(False)

    def set_recent_paths(self, paths: Iterable[str | Path]) -> None:
        self._recent_paths = [Path(path).expanduser() for path in paths]
        table = self.query_one("#recent-tokenizers", DataTable)
        table.clear()
        if not self._recent_paths:
            table.add_class("hidden")
            return
        table.remove_class("hidden")
        for path in self._recent_paths:
            table.add_row(str(path))

    def set_root(self, root: Path | str) -> None:
        root_path = Path(root).expanduser().resolve()
        tree = self.query_one("#folder-tree", BrowserDirectoryTree)
        tree.path = root_path
        self.query_one("#browser-root", Static).update(f"Root: {root_path}")

    def parent_root(self) -> None:
        tree = self.query_one("#folder-tree", BrowserDirectoryTree)
        current = Path(tree.path).expanduser().resolve()
        parent = current.parent
        if parent != current:
            self.set_root(parent)

    def focus_tree(self) -> None:
        recent = self.query_one("#recent-tokenizers", DataTable)
        if not recent.has_class("hidden") and self._recent_paths:
            recent.focus()
            return
        self.query_one("#folder-tree", BrowserDirectoryTree).focus()

    def set_status(self, message: str) -> None:
        self.query_one("#browser-status", Static).update(message)

    def set_loading(self, loading: bool) -> None:
        loader = self.query_one("#browser-loading", LoadingIndicator)
        if loading:
            loader.remove_class("hidden")
        else:
            loader.add_class("hidden")

    def on_directory_tree_directory_selected(
        self,
        event: DirectoryTree.DirectorySelected,
    ) -> None:
        event.stop()
        self.post_message(self.FolderSelected(event.path))

    def on_directory_tree_file_selected(
        self,
        event: DirectoryTree.FileSelected,
    ) -> None:
        event.stop()
        self.post_message(self.FolderSelected(event.path))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "recent-tokenizers":
            return
        event.stop()
        if 0 <= event.cursor_row < len(self._recent_paths):
            self.post_message(self.FolderSelected(self._recent_paths[event.cursor_row]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hub-load":
            event.stop()
            model_id = self.query_one("#hub-model-id", Input).value.strip()
            if model_id:
                self.post_message(self.HubDownloadRequested(model_id))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "hub-model-id":
            event.stop()
            model_id = event.value.strip()
            if model_id:
                self.post_message(self.HubDownloadRequested(model_id))
