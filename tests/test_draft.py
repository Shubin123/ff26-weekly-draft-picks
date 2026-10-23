"""python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffdraft import draft  # noqa: E402


def toy_board(seed=0):
    rng = np.random.default_rng(seed)
    counts = {"QB": 40, "RB": 70, "WR": 90, "TE": 40}
    rows = []
    for p, n in counts.items():
        proj = np.sort(rng.gamma(4, 3, n))[::-1]
        for i, v in enumerate(proj):
            rows.append(dict(player_id=f"{p}{i}", position=p, proj=v,
                             naive_exp=v + rng.normal(0, 2), actual=max(0, v + rng.normal(0, 6))))
    return pd.DataFrame(rows)


class TestTeamSizes(unittest.TestCase):
    def test_repl_rank_scales(self):
        self.assertEqual(draft.repl_rank(12), draft.REPL_RANK)
        for n in draft.TEAM_SIZES:
            r = draft.repl_rank(n)
            self.assertEqual((r["QB"], r["RB"], r["WR"], r["TE"]), (n, 2 * n + 5, 2 * n + 6, n + 1))

    def test_vorp_replacement_player_is_zero(self):
        b = toy_board()
        for n in draft.TEAM_SIZES:
            v = draft.vorp(b, "proj", n)
            for p, k in draft.repl_rank(n).items():
                s = b[b.position == p].proj.sort_values(ascending=False)
                self.assertAlmostEqual(v[s.index[k]], 0.0)

    def test_smaller_league_lowers_replacement(self):
        b = toy_board()
        v8, v16 = draft.vorp(b, "proj", 8), draft.vorp(b, "proj", 16)
        self.assertTrue((v8 <= v16 + 1e-9).all())

    def test_slot_sheet_per_size(self):
        b = toy_board()
        b["vorp"] = draft.vorp(b, "proj")
        for n in draft.TEAM_SIZES:
            sheet = draft.recommend_by_slot(b, teams=n)
            self.assertEqual(sorted(sheet), list(range(1, n + 1)))
            for picks in sheet.values():
                self.assertEqual(len(picks), draft.ROUNDS)
                self.assertEqual(len(set(picks)), draft.ROUNDS)
                pos = b.set_index("player_id").loc[picks, "position"]
                self.assertEqual((pos == "QB").sum(), 1)
                self.assertGreaterEqual((pos == "TE").sum(), 1)

    def test_draft_eval_per_size(self):
        b = toy_board()
        for n in draft.TEAM_SIZES:
            res = draft.draft_eval(b, ["proj", "naive_exp"], teams=n)
            self.assertAlmostEqual(res["naive_exp"]["win_rate"], 0.5)  # field vs itself
            self.assertTrue(0 <= res["proj"]["win_rate"] <= 1)

    def test_snake_rosters_full(self):
        b = toy_board()
        pos = b.position.to_numpy()
        prefs = list(np.argsort(-b.proj.to_numpy()))
        for n in draft.TEAM_SIZES:
            rosters = draft.snake_draft(pos, [prefs] * n, n)
            self.assertEqual(len(rosters), n)
            self.assertTrue(all(len(r) == draft.ROUNDS for r in rosters))
            flat = [i for r in rosters for i in r]
            self.assertEqual(len(flat), len(set(flat)))


if __name__ == "__main__":
    unittest.main()
