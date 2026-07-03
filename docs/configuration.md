# Configuration Guide

`tokenscope` can be configured globally or per-project using configuration files. This document details how to initialize, modify, and utilize these configuration files.

---

## The Configuration File: `.tokenscoperc`

The application reads tokenizer defaults and preferences from a file named `.tokenscoperc`.

### Resolution Order
When starting up, `tokenscope` searches for the configuration file in the following order:
1. **Workspace Directory**: The current working directory where `tokenscope` was executed (`./.tokenscoperc`).
2. **User Home Directory**: The global user folder (`~/.tokenscoperc` or `C:\Users\<Name>\.tokenscoperc`).

If both files exist, settings in the **Workspace Directory** take precedence over the home directory. If neither file exists, default hardcoded options are applied.

### Initialization
You can generate a default JSON configuration file with default settings by running:
```bash
tokenscope init-config
```
To initialize it in a specific path:
```bash
tokenscope init-config --path /path/to/.tokenscoperc
```

### Config Properties
The `.tokenscoperc` file uses the JSON format:
```json
{
  "default_tokenizer": null,
  "default_compare_tokenizer": null,
  "default_budget": null,
  "default_export_format": "json",
  "default_hub_model": null,
  "hub_cache_dir": null,
  "encode_special_tokens": false,
  "default_corpus_path": null,
  "default_batch_path": null
}
```

- **`default_tokenizer`** (`string` or `null`): 
  Specifies the path to a local folder containing tokenizer files or a direct `tokenizer.json` path. If set, this tokenizer loads automatically at startup.
- **`default_compare_tokenizer`** (`string` or `null`):
  Specifies a local tokenizer path to load as the comparison tokenizer.
- **`default_budget`** (`integer` or `null`):
  Specifies a default limit for the prompt budget. Exceeding this limit will trigger red threshold alerts in the stats panel.
- **`default_export_format`** (`"json"` | `"csv"` | `"md"` | `"html"`):
  Determines the default export format used by TUI exports and headless analysis when `--export-format` is omitted.
- **`default_hub_model`** (`string` or `null`):
  Specifies a Hugging Face Hub repository ID to download and load when no local tokenizer path is configured.
- **`hub_cache_dir`** (`string` or `null`):
  Overrides the Hugging Face Hub cache directory used for Hub downloads.
- **`encode_special_tokens`** (`boolean`):
  Starts the TUI with special-token encoding enabled when no project overrides it.
- **`default_corpus_path`** / **`default_batch_path`** (`string` or `null`):
  Preload corpus or batch analysis paths when a primary tokenizer is available.

---

## Pricing Profiles: `.tokenscope_pricing.json`

To estimate costs for prompt processing and generation, `tokenscope` uses cost rates defined in `.tokenscope_pricing.json` located in either the workspace or home directory.

### Structure Example
```json
{
  "profiles": [
    {
      "name": "gpt-4o",
      "input_per_million": 5.0,
      "output_per_million": 15.0,
      "estimated_output_tokens": 100
    },
    {
      "name": "llama3-70b-instruct",
      "input_per_million": 0.65,
      "output_per_million": 2.75,
      "estimated_output_tokens": 150
    }
  ]
}
```

### Properties
- **`name`** (`string`): The name/label of the profile.
- **`input_per_million`** (`float`): Cost in USD per 1,000,000 input tokens.
- **`output_per_million`** (`float`): Cost in USD per 1,000,000 output tokens.
- **`estimated_output_tokens`** (`integer`): Default assumed count of generated tokens for estimating total transaction costs when inputting prompts.
