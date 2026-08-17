# tokenscope v1.0

First stable release of `tokenscope`, consolidating the offline tokenizer explorer, comparison workflow, corpus and batch analysis tools, export formats, project state support, and packaging automation into a v1.0 release.

## Highlights

- **Stable CLI and TUI versioning**: Promoted the public app version to `1.0.0`, displayed as `tokenscope v1.0.0` in the TUI and `tokenscope 1.0.0` from `--version`.
- **Release metadata**: Added dedicated v1.0 release notes for GitHub Releases and the README release pointer.
- **CI reliability**: Hardened token table selection handling so programmatic table refreshes do not emit stale highlight events that reset the selected token during Textual smoke tests.

## Validation

- `python -m compileall .`
- `python -m unittest discover -v`
