"""Checks docs/data/site.json covers every league size in draft.TEAM_SIZES.

CI runs this right after run_pipeline.py, so a refresh that skips a size never gets committed.
"""
import csv
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffdraft import draft  # noqa: E402

SITE = json.loads((ROOT / "docs" / "data" / "site.json").read_text())
SIZES = list(draft.TEAM_SIZES)
WEEKS = sorted(SITE["boards"], key=int)


class TestEveryLeagueSize(unittest.TestCase):
    def test_sizes_listed(self):
        self.assertEqual(SITE["league"]["team_sizes"], SIZES)

    def test_weeks_present(self):
        self.assertEqual([int(w) for w in WEEKS], list(range(1, SITE["week"] + 1)))

    def test_slot_picks_every_size_week_slot(self):
        self.assertEqual(sorted(SITE["slot_picks"], key=int), [str(n) for n in SIZES])
        for n in SIZES:
            for w in WEEKS:
                sheet = SITE["slot_picks"][str(n)].get(w)
                self.assertIsNotNone(sheet, f"{n} teams, week {w}: no pick sheet")
                self.assertEqual(sorted(sheet, key=int), [str(k) for k in range(1, n + 1)], f"{n} teams, week {w}")
                for k, picks in sheet.items():
                    self.assertEqual(len(picks), draft.ROUNDS, f"{n} teams, week {w}, slot {k}")

    def test_board_vorp_every_size(self):
        for w in WEEKS:
            for r in SITE["boards"][w]:
                for n in SIZES:
                    self.assertIsNotNone(r.get(f"vorp_{n}"), f"week {w} {r['name']}: vorp_{n} missing")

    def test_backtest_every_size(self):
        weeks_by_size = Counter((r["teams"], r["strategy"]) for r in SITE["draft_by_week"])
        n12 = weeks_by_size[(draft.TEAMS, "ensemble")]
        self.assertGreater(n12, 0)
        for n in SIZES:
            self.assertEqual(weeks_by_size[(n, "ensemble")], n12, f"{n} teams: backtest weeks differ from 12")
        for key in ("draft_hist", "draft_current"):
            self.assertEqual(sorted({r["teams"] for r in SITE[key] if r["strategy"] == "ensemble"}), SIZES, key)

    def test_reports_every_size(self):
        with open(ROOT / "reports" / "draft_sim_by_week.csv") as f:
            self.assertEqual(sorted({int(r["teams"]) for r in csv.DictReader(f)}), SIZES)
        summary = json.loads((ROOT / "reports" / "summary.json").read_text())
        self.assertEqual(sorted(summary["draft_hist_by_teams"], key=int), [str(n) for n in SIZES])


if __name__ == "__main__":
    unittest.main()
