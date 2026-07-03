from __future__ import annotations

import argparse
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from main import _read_headless_input


class StdinInputTests(unittest.TestCase):
    def test_read_headless_input_variants(self) -> None:
        # Inline input option
        args = argparse.Namespace(input="hello inline", input_file=None, stdin=False)
        self.assertEqual(_read_headless_input(args), "hello inline")

        # File input option
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write("hello from input file")
            tmp_name = tmp.name
        try:
            args = argparse.Namespace(input=None, input_file=tmp_name, stdin=False)
            self.assertEqual(_read_headless_input(args), "hello from input file")
        finally:
            Path(tmp_name).unlink()

        # stdin mode with mock stdin stream
        args = argparse.Namespace(input=None, input_file=None, stdin=True)
        with patch("sys.stdin.read", return_value="hello from stdin"):
            self.assertEqual(_read_headless_input(args), "hello from stdin")


if __name__ == "__main__":
    unittest.main()
