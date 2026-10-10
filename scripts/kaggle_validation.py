"""Cross-checks using the remaining Kaggle datasets.

1. mexwell play-by-play 2013-2021: defensive yards/play allowed (pass, rush) from raw
   plays vs. the fantasy-points-allowed ratio we build from box scores. If they agree,
   the matchup feature captures real defensive quality, not noise.
2. heefjones 1970-2024 season data: year-over-year stability of points-per-game by
   position. That is the empirical basis for how hard the empirical-Bayes model
   shrinks a player's prior season toward the position mean.

Writes reports/kaggle_validation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffdraft import data, features  # noqa: E402

MEX = data.KAG / "mexwell__ultimate-nfl-database"


def pbp_defense():
    frames = []
    for y in range(2013, 2022):
        p = pd.read_csv(MEX / f"pbp-{y}.csv", usecols=[
            "SeasonYear", "DefenseTeam", "Yards", "IsRush", "IsPass", "IsTouchdown",
            "PlayType", "IsSack", "IsInterception"], low_memory=False)
        p["season"] = y  # file year == season (SeasonYear is blank in some files)
        frames.append(p)
    p = pd.concat(frames)
    p = p[p.DefenseTeam.notna() & (p.IsRush.eq(1) | p.IsPass.eq(1))]
    p["team"] = data.canon_team(p.DefenseTeam)
    agg = p.groupby(["season", "team"]).apply(lambda d: pd.Series({
        "pass_ypp": d.loc[d.IsPass == 1, "Yards"].mean(),
        "rush_ypc": d.loc[d.IsRush == 1, "Yards"].mean(),
        "td_rate": d.IsTouchdown.mean(),
        "plays": len(d)}), include_groups=False).reset_index()
    return agg


def fpa_by_defense(weekly):
    g = features.add_shares(weekly)
    d = g.groupby(["opp", "season", "week", "position"]).ppr.sum().reset_index()
    d = d.groupby(["opp", "season", "position"]).ppr.mean().unstack("position").reset_index()
    return d.rename(columns={"opp": "team"})


def yoy_stability():
    h = pd.read_csv(data.KAG / "heefjones__nfl-fantasy-data-1970-2024" / "fantasy_data.csv")
    h = h[h.Pos.isin(data.POSITIONS) & (h.G >= 8)]
    h = h[["Player", "Pos", "Year", "PPG_half-ppr"]].rename(columns={"PPG_half-ppr": "ppg"})
    nxt = h.assign(Year=h.Year - 1).rename(columns={"ppg": "ppg_next"})
    m = h.merge(nxt, on=["Player", "Pos", "Year"])
    out = {}
    for era, lo, hi in [("1970-1999", 1970, 1999), ("2000-2024", 2000, 2024)]:
        e = m[(m.Year >= lo) & (m.Year <= hi)]
        out[era] = {p: {"r": round(e[e.Pos == p][["ppg", "ppg_next"]].corr().iloc[0, 1], 3),
                        "n": int((e.Pos == p).sum())} for p in data.POSITIONS}
    return out


def main():
    weekly = data.build_weekly(2026)
    pbp = pbp_defense()
    fpa = fpa_by_defense(weekly[weekly.season <= 2021])
    j = pbp.merge(fpa, on=["season", "team"])
    pairs = {"pass_ypp~WR": ("pass_ypp", "WR"), "pass_ypp~TE": ("pass_ypp", "TE"),
             "pass_ypp~QB": ("pass_ypp", "QB"), "rush_ypc~RB": ("rush_ypc", "RB"),
             "td_rate~QB": ("td_rate", "QB")}
    same = {k: round(j[a].corr(j[b]), 3) for k, (a, b) in pairs.items()}
    # does last season's pbp defense predict this season's points allowed?
    lag = pbp.assign(season=pbp.season + 1).merge(fpa, on=["season", "team"])
    nxt = {k: round(lag[a].corr(lag[b]), 3) for k, (a, b) in pairs.items()}
    res = {"pbp_defense_vs_fpa_same_season": same,
           "pbp_defense_prev_season_vs_fpa": nxt,
           "defense_team_seasons": int(len(j)),
           "ppg_year_over_year_correlation": yoy_stability()}
    out = ROOT / "reports" / "kaggle_validation.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
