TOKEN_COLORS = [
    "#49d6e8",
    "#f2cc60",
    "#d779ff",
    "#74a7ff",
    "#ffb15c",
    "#7ee787",
    "#ff8cc6",
    "#f0f6fc",
]

APP_CSS = """
Screen {
    background: #0d1117;
    color: #c9d1d9;
}

.hidden {
    display: none;
}

#header {
    dock: top;
    height: 1;
    padding: 0 1;
    background: #161b22;
    color: #f0f6fc;
}

#browser-screen {
    margin: 1 2;
    padding: 1 2;
    border: round #30363d;
    height: 1fr;
}

#browser-title {
    text-style: bold;
    color: #f0f6fc;
    margin-bottom: 1;
}

#browser-help,
#browser-root {
    height: auto;
    min-height: 1;
    color: #8b949e;
}

#folder-tree {
    height: 1fr;
    margin-top: 1;
    border: round #30363d;
}

#recent-tokenizers {
    height: 5;
    margin-top: 1;
}

#browser-status {
    height: auto;
    min-height: 1;
    margin-top: 1;
    color: #8b949e;
}

#browser-loading {
    margin-top: 1;
}

#main-layout {
    height: 1fr;
    layout: vertical;
}

#text-input {
    height: 3;
    margin: 1 1 0 1;
    border: round #30363d;
}

#content-row {
    height: 3fr;
    min-height: 10;
}

TokenView {
    width: 2fr;
    margin: 1 0 0 1;
    padding: 1;
    border: round #30363d;
    overflow-y: auto;
}

#compare-token-view {
    margin-left: 1;
}

StatsPanel {
    width: 1fr;
    min-width: 32;
    margin: 1 1 0 1;
    padding: 1;
    border: round #30363d;
}

MergeTreeWidget {
    height: 2fr;
    min-height: 12;
    margin: 1;
    padding: 0 1 1 1;
    border: round #30363d;
}

TabbedContent {
    height: 1fr;
}

TabPane {
    padding: 1 0 0 0;
}

#token-table,
#vocab-table,
#compare-table,
#special-table,
#corpus-lines,
#search-results,
#chat-messages,
#batch-table,
#corpus-compare-table,
#pipeline-table,
#tokenizer-diff-table,
#packing-table,
#regression-table,
#unicode-table,
#rag-table,
#distribution-table,
#repair-table,
#benchmark-table {
    height: 1fr;
}

#bottom-controls,
#budget-controls,
#token-search-controls,
#chat-controls,
#project-controls,
#packing-controls,
#regression-controls,
#rag-controls,
#cost-controls,
#hub-controls {
    height: 3;
}

#source-select,
#export-format-select,
#budget-select,
#token-search-mode,
#chat-role,
#packing-strategy,
#rag-mode {
    height: 3;
    margin: 0 0 1 0;
}

#merge-tree {
    height: 1fr;
    overflow-y: auto;
}

#vocab-search,
#token-search,
#budget-input,
#chat-content,
#project-path,
#regression-path,
#rag-max-tokens,
#rag-overlap-tokens,
#cost-input-price,
#cost-output-price,
#cost-output-tokens,
#hub-model-id {
    height: 3;
    margin-bottom: 1;
}

#inspector-details,
#decode-details,
#metadata-details,
#budget-summary,
#corpus-summary,
#chat-summary,
#batch-summary,
#corpus-compare-summary,
#pipeline-summary,
#project-summary,
#tokenizer-diff-summary,
#packing-summary,
#regression-summary,
#unicode-summary,
#rag-summary,
#distribution-summary,
#cost-summary,
#repair-summary,
#benchmark-summary {
    height: auto;
    min-height: 3;
}

#special-mode {
    height: 1;
}

#open-corpus,
#open-batch,
#toggle-special,
#search-prev,
#search-next,
#chat-add,
#chat-update,
#chat-delete,
#chat-up,
#chat-down,
#chat-toggle-generation,
#project-load,
#project-save,
#regression-run,
#regression-add-current,
#repair-write-preview,
#benchmark-run,
#hub-load {
    height: 3;
}

.budget-ok {
    color: #7ee787;
}

.budget-warning {
    color: #f2cc60;
}

.budget-danger {
    color: #ff8cc6;
}

.budget-exceeded {
    color: #ff4444;
    text-style: bold;
}
"""
