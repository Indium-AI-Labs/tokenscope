# tokenscope v0.2.0

Second public release of `tokenscope`, adding optional HuggingFace Hub tokenizer loading, speed benchmarking, budget alerts, clipboard utilities, configuration file support, and stdin pipe mode.

## Highlights

- **HuggingFace Hub Tokenizer Loading**: Added on-demand hub tokenizer downloads. Users can load tokenizers from HF Hub using the `--hub` CLI flag, or by entering a model ID (e.g. `gpt2` or `meta-llama/Llama-3.1-8B`) directly in the TUI folder browser.
- **Speed Benchmarking**: Added a "Benchmark" tab to measure and compare tokenization throughput (tokens/sec and characters/sec) between primary and compare tokenizers.
- **Budget Threshold Alerts**: Added an ambient budget badge in the TUI header and a dedicated row in the Stats Panel. Both dynamically update and color-code depending on utilization (green < 80%, yellow 80-95%, red > 95%, bold red > 100%).
- **Clipboard Integration**: Added `Ctrl+Y` shortcut to instantly copy space-separated primary token IDs to the system clipboard (integrates `pyperclip`).
- **Configuration File Support**: Added `.tokenscoperc` and `tokenscope.json` file parsing (looks for defaults in CWD first, falling back to user's home directory). Added an `init-config` subcommand to generate a default configuration file template.
- **stdin Pipe Mode**: Enhanced headless `analyze` mode to read piped text inputs directly from standard input (useful for Unix pipes and automation).

## Validation

- `python -m compileall .`
- `python -m unittest discover -v`
- Headless stdin analysis integration verification.
