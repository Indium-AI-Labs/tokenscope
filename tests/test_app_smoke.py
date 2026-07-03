from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers

from analysis_models import add_recent_tokenizer
from main import TokenscopeApp
from widgets.folder_browser import FolderBrowser
from widgets.input_bar import DebouncedTextInput
from widgets.merge_tree import MergeTreeWidget


def make_tokenizer_dir(root: Path, corpus: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=64, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save(str(root / "tokenizer.json"))
    return root


def make_special_tokenizer_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "hello": 2,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([AddedToken("[PAD]", special=True, normalized=False)])
    tokenizer.save(str(root / "tokenizer.json"))
    return root


def make_chat_tokenizer_dir(root: Path) -> Path:
    make_tokenizer_dir(root, ["hello world assistant user system"])
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "chat_template": (
                    "{{ bos_token }}{% for message in messages %}"
                    "{{ message['role'] }}: {{ message['content'] }}\\n"
                    "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
                ),
            }
        ),
        encoding="utf-8",
    )
    return root


async def wait_for(predicate, pilot, attempts: int = 40) -> None:
    for _ in range(attempts):
        await pilot.pause(0.1)
        if predicate():
            return
    raise AssertionError("Timed out waiting for app state.")


class AppSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_version_reports_shared_version(self) -> None:
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("tokenscope 0.2.0", completed.stdout)

    async def test_startup_browser_appears_without_cli_tokenizer(self) -> None:
        app = TokenscopeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            self.assertFalse(app.query_one("#browser-screen").has_class("hidden"))
            self.assertTrue(app.query_one("#main-layout").has_class("hidden"))

    async def test_config_encode_special_tokens_initializes_app(self) -> None:
        app = TokenscopeApp(encode_special_tokens=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            self.assertTrue(app.encode_special_tokens)
            self.assertTrue(app.query_one("#bottom-panel", MergeTreeWidget).encode_special_tokens)

    async def test_hub_download_handler_requests_tokenizer_load(self) -> None:
        app = TokenscopeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            mock_engine = MagicMock()
            mock_engine.source_path = Path("cached") / "gpt2"
            with (
                patch("main.TokenizerEngine.load_from_hub", return_value=mock_engine) as mock_load,
                patch.object(app, "_request_tokenizer_load") as mock_request,
            ):
                await app.on_folder_browser_hub_download_requested(
                    FolderBrowser.HubDownloadRequested("gpt2")
                )

            mock_load.assert_called_once_with("gpt2", cache_dir=None)
            mock_request.assert_called_once_with(str(mock_engine.source_path), "primary", from_cli=False)

    async def test_hub_controls_hidden_for_corpus_browser(self) -> None:
        app = TokenscopeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            app._open_browser("corpus")
            self.assertTrue(app.query_one("#hub-controls").has_class("hidden"))

            with patch("main.TokenizerEngine.load_from_hub") as mock_load:
                await app.on_folder_browser_hub_download_requested(
                    FolderBrowser.HubDownloadRequested("gpt2")
                )

            mock_load.assert_not_called()

    async def test_folder_browser_selection_loads_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            app = TokenscopeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.2)
                app.post_message(FolderBrowser.FolderSelected(root))
                await wait_for(lambda: app.primary_engine is not None, pilot)
                self.assertEqual(app.primary_engine.source_path, root.resolve())
                self.assertTrue(app.query_one("#browser-screen").has_class("hidden"))

    async def test_cli_primary_bypasses_browser_after_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            app = TokenscopeApp(tokenizer_path=str(root))
            async with app.run_test(size=(120, 40)) as pilot:
                await wait_for(lambda: app.primary_engine is not None, pilot)
                self.assertTrue(app.query_one("#browser-screen").has_class("hidden"))
                self.assertFalse(app.query_one("#main-layout").has_class("hidden"))

    async def test_cli_compare_loads_compare_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            compare = make_tokenizer_dir(Path(tmp) / "compare", ["hello tokenscope world"])
            app = TokenscopeApp(tokenizer_path=str(primary), compare_tokenizer_path=str(compare))
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(lambda: app.compare_engine is not None, pilot, attempts=60)
                text_input = app.query_one("#text-input", DebouncedTextInput)
                text_input.value = "hello world"
                text_input.emit_now()
                await wait_for(lambda: app.comparison is not None, pilot)
                self.assertFalse(app.query_one("#compare-token-view").has_class("hidden"))

    async def test_clear_compare_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            compare = make_tokenizer_dir(Path(tmp) / "compare", ["hello tokenscope world"])
            app = TokenscopeApp(tokenizer_path=str(primary), compare_tokenizer_path=str(compare))
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(lambda: app.compare_engine is not None, pilot, attempts=60)
                app.action_clear_compare_tokenizer()
                await pilot.pause(0.2)
                self.assertIsNone(app.compare_engine)
                self.assertTrue(app.query_one("#compare-token-view").has_class("hidden"))

    async def test_invalid_cli_compare_keeps_primary_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            invalid = Path(tmp) / "missing"
            app = TokenscopeApp(tokenizer_path=str(primary), compare_tokenizer_path=str(invalid))
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(lambda: app.primary_engine is not None, pilot)
                await pilot.pause(0.5)
                self.assertIsNone(app.compare_engine)
                self.assertFalse(app.query_one("#main-layout").has_class("hidden"))

    async def test_compare_export_contains_comparison_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            primary = make_tokenizer_dir(workspace / "primary", ["hello world"])
            compare = make_tokenizer_dir(workspace / "compare", ["hello tokenscope world"])
            app = TokenscopeApp(tokenizer_path=str(primary), compare_tokenizer_path=str(compare))
            original_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                async with app.run_test(size=(150, 45)) as pilot:
                    await wait_for(lambda: app.compare_engine is not None, pilot, attempts=60)
                    text_input = app.query_one("#text-input", DebouncedTextInput)
                    text_input.value = "hello world"
                    text_input.emit_now()
                    await wait_for(lambda: app.comparison is not None, pilot)
                    app.action_save_export()
                    await pilot.pause(0.2)
                payload = json.loads((workspace / "tokenscope_export.json").read_text(encoding="utf-8"))
                self.assertIn("comparison", payload)
                self.assertIn("primary_tokenizer", payload["comparison"])
                self.assertIn("compare_tokenizer", payload["comparison"])
                self.assertIn("summary", payload["comparison"])
                self.assertIn("rows", payload["comparison"])
            finally:
                os.chdir(original_cwd)

    async def test_new_tabs_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            app = TokenscopeApp(tokenizer_path=str(root))
            async with app.run_test(size=(160, 50)) as pilot:
                await wait_for(lambda: app.primary_engine is not None, pilot)
                bottom = app.query_one("#bottom-panel", MergeTreeWidget)
                for tab_id in MergeTreeWidget.TAB_IDS:
                    self.assertIsNotNone(bottom.query_one(f"#{tab_id}"))

    async def test_keyboard_navigation_updates_selected_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_tokenizer_dir(Path(tmp) / "primary", ["hello world"])
            app = TokenscopeApp(tokenizer_path=str(root))
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(
                    lambda: app.primary_engine is not None
                    and not app.query_one("#main-layout").has_class("hidden"),
                    pilot,
                )
                text_input = app.query_one("#text-input", DebouncedTextInput)
                text_input.value = "hello world"
                text_input.emit_now()
                await wait_for(lambda: app.primary_result is not None and len(app.primary_result.spans) >= 2, pilot)
                app.action_select_next_token()
                await wait_for(lambda: app.selected_token_indices["primary"] == 1, pilot)

    async def test_special_token_toggle_reencodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_special_tokenizer_dir(Path(tmp) / "primary")
            app = TokenscopeApp(tokenizer_path=str(root))
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(lambda: app.primary_engine is not None, pilot)
                text_input = app.query_one("#text-input", DebouncedTextInput)
                text_input.value = "[PAD] hello"
                text_input.emit_now()
                await wait_for(
                    lambda: app.primary_result is not None and app.primary_result.input_text == "[PAD] hello",
                    pilot,
                )
                before = app.primary_result.token_ids
                app.action_toggle_encode_special_tokens()
                await wait_for(lambda: app.primary_result is not None and app.primary_result.token_ids != before, pilot)
                self.assertTrue(app.encode_special_tokens)

    async def test_cli_corpus_path_runs_background_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = make_tokenizer_dir(workspace / "primary", ["hello world alpha beta"])
            corpus = workspace / "corpus"
            corpus.mkdir()
            (corpus / "sample.txt").write_text("hello world\nalpha beta", encoding="utf-8")
            app = TokenscopeApp(tokenizer_path=str(root), corpus_path=str(corpus))
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(lambda: app.corpus_result is not None, pilot, attempts=80)
                self.assertEqual(app.corpus_result.total_files, 1)
                self.assertGreater(app.corpus_result.total_tokens, 0)

    async def test_export_format_selector_controls_save_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = make_tokenizer_dir(workspace / "primary", ["hello world"])
            app = TokenscopeApp(tokenizer_path=str(root), export_format="csv")
            original_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                async with app.run_test(size=(150, 45)) as pilot:
                    await wait_for(lambda: app.primary_engine is not None, pilot)
                    text_input = app.query_one("#text-input", DebouncedTextInput)
                    text_input.value = "hello world"
                    text_input.emit_now()
                    await wait_for(
                        lambda: app.primary_result is not None and app.primary_result.input_text == "hello world",
                        pilot,
                    )
                    self.assertEqual(app.query_one("#export-format-select").value, "csv")
                    app.action_save_export()
                    await pilot.pause(0.1)
                self.assertTrue((workspace / "tokenscope_export.csv").exists())
            finally:
                os.chdir(original_cwd)

    async def test_html_export_contains_new_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = make_chat_tokenizer_dir(workspace / "primary")
            batch = workspace / "batch.txt"
            batch.write_text("hello world", encoding="utf-8")
            app = TokenscopeApp(tokenizer_path=str(root), batch_path=str(batch), export_format="html")
            original_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                async with app.run_test(size=(170, 55)) as pilot:
                    await wait_for(lambda: app.batch_result is not None, pilot, attempts=80)
                    text_input = app.query_one("#text-input", DebouncedTextInput)
                    text_input.value = "hello world"
                    text_input.emit_now()
                    await wait_for(
                        lambda: app.primary_result is not None and app.primary_result.input_text == "hello world",
                        pilot,
                    )
                    app.action_save_export()
                    await pilot.pause(0.2)
                exported = (workspace / "tokenscope_export.html").read_text(encoding="utf-8")
                self.assertIn("Chat Budget", exported)
                self.assertIn("Batch", exported)
                self.assertIn("Pipeline", exported)
            finally:
                os.chdir(original_cwd)

    async def test_cli_batch_path_runs_background_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = make_tokenizer_dir(workspace / "primary", ["hello world alpha beta"])
            batch = workspace / "batch"
            batch.mkdir()
            (batch / "prompts.txt").write_text(
                "hello world alpha beta zzz qqq rrr sss ttt\nalpha beta zzz qqq rrr sss ttt",
                encoding="utf-8",
            )
            app = TokenscopeApp(tokenizer_path=str(root), batch_path=str(batch), budget_limit=1)
            async with app.run_test(size=(150, 45)) as pilot:
                await wait_for(lambda: app.batch_result is not None, pilot, attempts=80)
                self.assertEqual(app.batch_result.total_files, 1)
                self.assertGreater(app.batch_result.total_prompts, 0)
                self.assertGreater(app.batch_result.budget_failures, 0)

    async def test_corpus_compare_runs_when_compare_and_corpus_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            primary = make_tokenizer_dir(workspace / "primary", ["hello world alpha beta"])
            compare = make_tokenizer_dir(workspace / "compare", ["hello world alpha beta tokenscope"])
            corpus = workspace / "corpus"
            corpus.mkdir()
            (corpus / "sample.txt").write_text("hello world\nalpha beta", encoding="utf-8")
            app = TokenscopeApp(
                tokenizer_path=str(primary),
                compare_tokenizer_path=str(compare),
                corpus_path=str(corpus),
            )
            async with app.run_test(size=(170, 55)) as pilot:
                await wait_for(lambda: app.corpus_compare_result is not None, pilot, attempts=100)
                self.assertEqual(app.corpus_compare_result.total_files, 1)
                self.assertGreater(app.corpus_compare_result.primary_total_tokens, 0)

    async def test_chat_budget_tab_renders_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_chat_tokenizer_dir(Path(tmp) / "primary")
            app = TokenscopeApp(tokenizer_path=str(root), budget_limit=32)
            async with app.run_test(size=(170, 55)) as pilot:
                await wait_for(lambda: app.primary_engine is not None, pilot)
                bottom = app.query_one("#bottom-panel", MergeTreeWidget)
                chat = bottom.current_chat_budget()
                self.assertIsNotNone(chat)
                assert chat is not None
                self.assertIsNotNone(chat.primary)
                assert chat.primary is not None
                self.assertIn("system:", chat.primary.rendered_text)
                self.assertIsNone(chat.primary.error)

    async def test_recent_tokenizer_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            recent_root = workspace / "recent"
            make_tokenizer_dir(recent_root, ["hello world"])
            add_recent_tokenizer(recent_root, workspace)
            original_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                app = TokenscopeApp()
                async with app.run_test(size=(150, 45)) as pilot:
                    await pilot.pause(0.2)
                    self.assertFalse(app.query_one("#recent-tokenizers").has_class("hidden"))
            finally:
                os.chdir(original_cwd)

    async def test_cost_profile_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            original_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                app = TokenscopeApp()
                async with app.run_test(size=(150, 45)) as pilot:
                    await pilot.pause(0.2)
                    bottom = app.query_one("#bottom-panel", MergeTreeWidget)
                    bottom.query_one("#cost-profile-name").value = "team"
                    bottom.query_one("#cost-input-price").value = "1.5"
                    bottom.query_one("#cost-output-price").value = "4"
                    bottom.query_one("#cost-output-tokens").value = "250"
                    bottom._save_cost_profile()
                    await pilot.pause(0.1)

                saved = json.loads((workspace / ".tokenscope_pricing.json").read_text(encoding="utf-8"))
                self.assertIn("team", [profile["name"] for profile in saved["profiles"]])

                app = TokenscopeApp()
                async with app.run_test(size=(150, 45)) as pilot:
                    await pilot.pause(0.2)
                    bottom = app.query_one("#bottom-panel", MergeTreeWidget)
                    bottom._apply_cost_profile("team")
                    self.assertEqual(bottom.cost_profile.name, "team")
                    self.assertEqual(bottom.cost_profile.output_per_million, 4.0)
            finally:
                os.chdir(original_cwd)

    async def test_headless_cli_analyze_exports_new_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = make_tokenizer_dir(workspace / "primary", ["hello world alpha beta"])
            output = workspace / "report.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "analyze",
                    "--tokenizer",
                    str(root),
                    "--input",
                    "hello world alpha beta",
                    "--budget",
                    "2",
                    "--rag-max-tokens",
                    "2",
                    "--export",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("packing", payload)
            self.assertIn("unicode", payload)
            self.assertIn("rag_chunking", payload)
            self.assertIn("cost", payload)
            self.assertIn("repair", payload)

    async def test_headless_cli_analyze_uses_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = make_tokenizer_dir(workspace / "primary", ["hello world alpha beta"])
            output = workspace / "report.md"
            (workspace / ".tokenscoperc").write_text(
                json.dumps(
                    {
                        "default_tokenizer": str(root),
                        "default_budget": 8,
                        "default_export_format": "md",
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "main.py"),
                    "analyze",
                    "--input",
                    "hello world",
                    "--export",
                    str(output),
                ],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            exported = output.read_text(encoding="utf-8")
            self.assertIn("## Token Table", exported)


if __name__ == "__main__":
    unittest.main()
