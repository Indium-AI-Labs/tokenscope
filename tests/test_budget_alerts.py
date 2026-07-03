from __future__ import annotations

import unittest

from widgets.stats_panel import _budget_color


class BudgetAlertsTests(unittest.TestCase):
    def test_budget_color_thresholds(self) -> None:
        self.assertEqual(_budget_color(0), "green")
        self.assertEqual(_budget_color(50), "green")
        self.assertEqual(_budget_color(79.9), "green")
        self.assertEqual(_budget_color(80), "yellow")
        self.assertEqual(_budget_color(90), "yellow")
        self.assertEqual(_budget_color(94.9), "yellow")
        self.assertEqual(_budget_color(95), "red")
        self.assertEqual(_budget_color(100), "red")
        self.assertEqual(_budget_color(100.1), "bold red")
        self.assertEqual(_budget_color(120), "bold red")


if __name__ == "__main__":
    unittest.main()
