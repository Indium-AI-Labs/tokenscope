from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from main import TokenscopeApp


class ClipboardTests(unittest.TestCase):
    def test_copy_to_clipboard_no_result(self) -> None:
        app = TokenscopeApp()
        app.primary_result = None
        
        with patch.object(app, "notify") as mock_notify:
            app.action_copy_to_clipboard()
            mock_notify.assert_called_once_with("Nothing to copy yet.", severity="warning")

    def test_copy_to_clipboard_success(self) -> None:
        app = TokenscopeApp()
        mock_result = MagicMock()
        mock_result.token_ids = [101, 2054, 999]
        app.primary_result = mock_result

        with patch("pyperclip.copy") as mock_copy, patch.object(app, "notify") as mock_notify:
            app.action_copy_to_clipboard()
            mock_copy.assert_called_once_with("101 2054 999")
            mock_notify.assert_called_once_with("Copied 3 token IDs to clipboard.")

    def test_copy_to_clipboard_import_error(self) -> None:
        app = TokenscopeApp()
        mock_result = MagicMock()
        mock_result.token_ids = [42]
        app.primary_result = mock_result

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyperclip":
                raise ImportError("Mocked ImportError")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import), patch.object(app, "notify") as mock_notify:
            app.action_copy_to_clipboard()
            mock_notify.assert_called_once()
            args, kwargs = mock_notify.call_args
            self.assertIn("pyperclip not installed", args[0])
            self.assertEqual(kwargs.get("severity"), "warning")


if __name__ == "__main__":
    unittest.main()
