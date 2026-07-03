# Developer Guide

This document outlines how to set up the development environment, run test suites, and build binaries for release.

---

## 1. Setup

### Requirements
- Python 3.8 or newer
- pip package manager

### Environment Configuration
1. Clone the repository and navigate into the project directory:
   ```bash
   cd tokenscope
   ```
2. Install dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Running Unit Tests

`tokenscope` separates its test cases by feature area inside the `tests/` directory:
- **`tests/test_analysis_models.py`**: Asserts cost estimations, prompt budgets, RAG simulations, and serialization formats.
- **`tests/test_app_smoke.py`**: Textual TUI smoke and interaction checks.
- **`tests/test_benchmark.py`**: Timers and comparative benchmarking models.
- **`tests/test_budget_alerts.py`**: Color thresholds for stats alerts.
- **`tests/test_clipboard.py`**: Clipboard export integrations and fallback notifications.
- **`tests/test_compare_engine.py`**: Alignment alignment logic.
- **`tests/test_config.py`**: Configuration parser settings.
- **`tests/test_stdin_input.py`**: Headless input stream resolvers.

### Running all tests
Run the Python test runner from the root of the project:
```bash
python -m unittest discover -s tests -v
```

---

## 3. Packaging & Distribution

`tokenscope` uses PyInstaller to bundle the application and its dependencies into a standalone executable.

### Building
To build a standalone executable:
```bash
pyinstaller tokenscope.spec
```
Or run the build helper script:
- On Linux/macOS:
  ```bash
  ./build.sh
  ```
- On Windows (PowerShell):
  ```powershell
  python -m pyinstaller tokenscope.spec
  ```

The compiled binaries will be output to the `dist/` directory.
