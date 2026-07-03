from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from config import TokenScopeConfig, load_config, save_config, _optional_positive_int, _optional_str, _export_format_choice
from main import main


class ConfigTests(unittest.TestCase):
    def test_optional_parsing(self) -> None:
        self.assertEqual(_optional_str(""), None)
        self.assertEqual(_optional_str("hello"), "hello")
        self.assertEqual(_optional_str(None), None)

        self.assertEqual(_optional_positive_int(123), 123)
        self.assertEqual(_optional_positive_int("456"), 456)
        self.assertEqual(_optional_positive_int(-5), None)
        self.assertEqual(_optional_positive_int("abc"), None)
        self.assertEqual(_optional_positive_int(None), None)

    def test_export_format_choice(self) -> None:
        self.assertEqual(_export_format_choice("md"), "md")
        self.assertEqual(_export_format_choice("MD"), "md")
        self.assertEqual(_export_format_choice("invalid"), "json")
        self.assertEqual(_export_format_choice(None), "json")

    def test_config_load_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = TokenScopeConfig(
                default_tokenizer="some/path",
                default_budget=1024,
                default_export_format="csv",
            )
            # Save config to file
            config_file = tmppath / ".tokenscoperc"
            save_config(config, config_file)

            # Load it back
            loaded = load_config(workspace=tmppath)
            self.assertEqual(loaded.default_tokenizer, "some/path")
            self.assertEqual(loaded.default_budget, 1024)
            self.assertEqual(loaded.default_export_format, "csv")

    def test_init_config_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cfg_file = tmppath / ".tokenscoperc"

            test_args = ["main.py", "init-config", "--path", str(cfg_file)]
            with patch.object(sys, "argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 0)

            self.assertTrue(cfg_file.is_file())
            loaded = load_config(workspace=tmppath)
            self.assertEqual(loaded.default_export_format, "json")


    def test_load_config_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # No config files in tmppath or home (we mock home to empty)
            with patch("pathlib.Path.home", return_value=tmppath):
                loaded = load_config(workspace=tmppath)
                # Should return default config (default_tokenizer is None)
                self.assertIsNone(loaded.default_tokenizer)
                self.assertEqual(loaded.default_export_format, "json")

    def test_load_config_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_file = tmppath / ".tokenscoperc"
            config_file.write_text("invalid json { content", encoding="utf-8")
            
            with patch("pathlib.Path.home", return_value=tmppath):
                loaded = load_config(workspace=tmppath)
                # Should fallback to default config
                self.assertIsNone(loaded.default_tokenizer)
                self.assertEqual(loaded.default_export_format, "json")


if __name__ == "__main__":
    unittest.main()
