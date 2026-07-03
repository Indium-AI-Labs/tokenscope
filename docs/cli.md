# Command Line Interface (CLI) Reference

`tokenscope` can be started in either interactive TUI mode or headless analysis mode.

---

## 1. TUI Interactive Mode

To open the terminal interface, launch `tokenscope` without subcommands:
```bash
tokenscope [options]
```

### Options
- **`--tokenizer`** `<path>`:
  Load a local directory containing tokenizer files or a direct path to a `tokenizer.json` file.
- **`--hub`** `<model_id>`:
  Download and load a tokenizer from the Hugging Face Hub (e.g. `meta-llama/Llama-3-8b`). Subsequent launches are cached.
- **`--compare-tokenizer`** `<path>`:
  Load a second comparison tokenizer directory or file path.
- **`--file`** `<path>`:
  Pre-populate a corpus text file or directory to compare tokenizer behaviors across larger inputs.
- **`--batch`** `<path>`:
  Pre-populate a batch JSONL file containing evaluation/instruction prompts to analyze.
- **`--budget`** `<limit>`:
  Configure a token limit to track prompt size allocations.
- **`--project`** `<project_file.json>`:
  Load a saved `tokenscope` project state (active tab, loaded models, options, inputs).

---

## 2. Subcommands

### A. `init-config`
Creates a default configuration file (`.tokenscoperc`).
```bash
tokenscope init-config [--path <custom_path>]
```
- **`--path`** `<path>`:
  Target location to write the configuration file. Defaults to `./.tokenscoperc`.

### B. `analyze` (Headless Mode)
Runs tokenizer computations and reports results immediately without loading the terminal UI.
```bash
tokenscope analyze --tokenizer <path> [options]
```

If `--tokenizer` and `--hub` are omitted, headless mode uses `default_tokenizer` or `default_hub_model` from `.tokenscoperc` when available.

#### Input Selection
- **`--hub`** `<model_id>`:
  Download and analyze with a tokenizer from the Hugging Face Hub.
- **`--input`** `<text>`:
  Passes text inline directly on the command line.
- **`--input-file`** `<path>`:
  Loads input text from a local file.
- **`--stdin`**:
  Reads input text from standard input (useful for Unix piping). Piped inputs are automatically read if standard input is not a terminal (TTY).
  ```bash
  cat article.txt | tokenscope analyze --tokenizer ./my-tokenizer --stdin --export-format md
  ```

#### Analysis Scope
- **`--compare-tokenizer`** `<path>`:
  Path to secondary tokenizer to run comparison analysis.
- **`--file`** `<path>`:
  Corpus file or folder path for batch statistics.
- **`--batch`** `<path>`:
  JSONL prompt files for multi-turn benchmark/cost predictions.
- **`--regression-suite`** `<path>`:
  Path to a JSON file containing regression test cases (inputs and expected token IDs).
- **`--rag-max-tokens`** `<int>`:
  Max token limit per chunk to run RAG partition simulation.
- **`--rag-overlap-tokens`** `<int>`:
  Token overlap between subsequent RAG chunks (default: `0`).
- **`--budget`** `<int>`:
  Numeric budget threshold limit to evaluate prompt sizing.

#### Budget & Cost Estimation
- **`--input-cost-per-million`** `<float>`:
  Price in USD per million tokens parsed.
- **`--output-cost-per-million`** `<float>`:
  Price in USD per million tokens generated.
- **`--estimated-output-tokens`** `<int>`:
  Estimated completion length to project full cost.

#### Output Control
- **`--export`** `<path>`:
  File path to write the output report. Prints directly to `stdout` if omitted.
- **`--export-format`** `{"json" | "csv" | "md" | "html"}`:
  Format of the generated report (default: `json`).
- **`--fail-on-budget`**:
  If specified, the program exits with code `2` if token usage exceeds the configured budget.
- **`--fail-on-regression`**:
  If specified, the program exits with code `2` if comparison or regression tests fail (e.g. any regression suite case failure, or token-count delta regression).
