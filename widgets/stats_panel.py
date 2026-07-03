from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from compare_engine import CompareResult
from tokenizer_engine import TokenStats, TokenizationResult


class StatsPanel(Static):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._budget_limit: int | None = None

    def on_mount(self) -> None:
        self.update_stats(None)

    def set_budget_limit(self, limit: int | None) -> None:
        self._budget_limit = limit

    def update_stats(self, result: TokenizationResult | None) -> None:
        if result is None:
            self.update(Text("Load a tokenizer to see stats.", style="dim"))
            return
        self.update(self._table(result.stats, self._budget_limit))

    def update_comparison(self, comparison: CompareResult | None) -> None:
        if comparison is None:
            self.update(Text("Load two tokenizers to compare stats.", style="dim"))
            return

        summary = comparison.summary
        table = Table.grid(expand=True)
        table.add_column("metric", style="dim")
        table.add_column("primary", justify="right", style="bold #f0f6fc")
        table.add_column("compare", justify="right", style="bold #f2cc60")
        table.add_row(
            "Tokens",
            f"{summary.primary_token_count:,}",
            f"{summary.compare_token_count:,} ({summary.token_count_delta:+,})",
        )
        table.add_row(
            "Chars/token",
            f"{summary.primary_chars_per_token:.2f}",
            f"{summary.compare_chars_per_token:.2f} ({summary.chars_per_token_delta:+.2f})",
        )
        table.add_row(
            "Compression",
            f"{summary.primary_compression:.2f}",
            f"{summary.compare_compression:.2f} ({summary.compression_delta:+.2f})",
        )
        table.add_row("Matching spans", str(summary.matching_spans), "")
        table.add_row("Boundary mismatches", str(summary.boundary_mismatches), "")
        table.add_row("Token mismatches", str(summary.token_mismatches), "")
        table.add_row("ID mismatches", str(summary.id_mismatches), "")
        table.add_row("Missing ranges", str(summary.missing_ranges), "")

        # Add budget row for primary when a limit is set.
        if self._budget_limit and self._budget_limit > 0:
            pct = (summary.primary_token_count / self._budget_limit) * 100.0
            remaining = self._budget_limit - summary.primary_token_count
            style = _budget_color(pct)
            table.add_row(
                "Budget",
                Text(
                    f"{summary.primary_token_count:,}/{self._budget_limit:,} ({pct:.0f}%)",
                    style=style,
                ),
                Text(f"{remaining:+,} remaining", style=style),
            )

        self.update(table)

    @staticmethod
    def _table(stats: TokenStats, budget_limit: int | None = None) -> Table:
        table = Table.grid(expand=True)
        table.add_column("metric", style="dim")
        table.add_column("value", justify="right", style="bold #f0f6fc")
        table.add_row("Vocab size", f"{stats.vocab_size:,}")
        table.add_row("Tokens", f"{stats.token_count:,}")
        table.add_row("Characters", f"{stats.character_count:,}")
        table.add_row("Chars/token", f"{stats.chars_per_token:.2f}")
        table.add_row("Compression", f"{stats.compression_ratio:.2f}")
        table.add_row("Unique IDs", f"{stats.unique_token_count:,}")
        table.add_row(
            "Most frequent ID",
            "n/a" if stats.most_frequent_token_id is None else str(stats.most_frequent_token_id),
        )
        table.add_row("Avg token length", f"{stats.avg_token_length:.2f}")

        # Budget usage row.
        if budget_limit and budget_limit > 0:
            pct = (stats.token_count / budget_limit) * 100.0
            remaining = budget_limit - stats.token_count
            style = _budget_color(pct)
            table.add_row(
                "Budget",
                Text(
                    f"{stats.token_count:,}/{budget_limit:,} ({pct:.0f}%) | {remaining:+,} rem",
                    style=style,
                ),
            )

        return table


def _budget_color(pct: float) -> str:
    """Return a Rich style string based on budget usage percentage."""
    if pct > 100:
        return "bold red"
    if pct >= 95:
        return "red"
    if pct >= 80:
        return "yellow"
    return "green"
