"""Load the raw Kaggle + nflverse files and harmonise them into one weekly table.

One row per (player, season, week) for QB/RB/WR/TE regular-season games, with
identical column names whether the row came from the historical Kaggle weekly
file (2012-2025) or the live nflverse 2026 box scores.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
KAG = RAW / "kaggle"
NFLV = RAW / "nflverse"

POSITIONS = ["QB", "RB", "WR", "TE"]
STAT_COLS = [
    "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int",
    "carries", "rush_yds", "rush_td",
    "targets", "receptions", "rec_yds", "rec_td", "air_yards",
    "snap_pct", "ppr",
]
# Team abbreviation drift across sources -> one canonical code.
TEAM_FIX = {"LA": "LAR", "LVR": "LV", "STL": "LAR", "SD": "LAC", "OAK": "LV", "JAC": "JAX",
            "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


def canon_team(s):
    return s.replace(TEAM_FIX)


def ppr_points(df):
    """Standard PPR scoring from box-score columns (used to verify sources agree)."""
    return (0.04 * df.pass_yds + 4 * df.pass_td - 2 * df.pass_int
            + 0.1 * (df.rush_yds + df.rec_yds) + 6 * (df.rush_td + df.rec_td)
            + df.receptions)


def load_history():
    """Kaggle philiphyde1 weekly offense, 2012-2025 regular season."""
    f = KAG / "philiphyde1__nfl-stats-1999-2022" / "weekly_player_stats_offense.csv"
    use = ["player_id", "player_name", "position", "team", "season", "week",
           "season_type", "game_id", "pass_attempts", "complete_pass",
           "passing_yards", "pass_touchdown", "interception", "rush_attempts",
           "rushing_yards", "rush_touchdown", "targets", "receptions",
           "receiving_yards", "receiving_touchdown", "receiving_air_yards",
           "offense_pct", "fantasy_points_ppr"]
    d = pd.read_csv(f, usecols=use, low_memory=False)
    d = d[(d.season_type == "REG") & d.position.isin(POSITIONS)]
    d = d.rename(columns={
        "player_name": "name", "pass_attempts": "pass_att", "complete_pass": "pass_cmp",
        "passing_yards": "pass_yds", "pass_touchdown": "pass_td", "interception": "pass_int",
        "rush_attempts": "carries", "rushing_yards": "rush_yds", "rush_touchdown": "rush_td",
        "receiving_yards": "rec_yds", "receiving_touchdown": "rec_td",
        "receiving_air_yards": "air_yards", "offense_pct": "snap_pct",
        "fantasy_points_ppr": "ppr"})
    d["source"] = "kaggle:philiphyde1"
    return d.drop(columns="season_type")


def load_live(season):
    """nflverse weekly box scores + snap counts for the current season."""
    f = NFLV / f"stats_player_week_{season}.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, low_memory=False)
    d = d[(d.season_type == "REG") & d.position.isin(POSITIONS)]
    d = d.rename(columns={
        "player_display_name": "name", "attempts": "pass_att", "completions": "pass_cmp",
        "passing_yards": "pass_yds", "passing_tds": "pass_td",
        "passing_interceptions": "pass_int", "rushing_yards": "rush_yds",
        "rushing_tds": "rush_td", "receiving_yards": "rec_yds", "receiving_tds": "rec_td",
        "receiving_air_yards": "air_yards", "fantasy_points_ppr": "ppr"})
    # snap share: pfr id -> gsis id through the nflverse players crosswalk
    snaps = pd.read_csv(NFLV / f"snap_counts_{season}.csv")
    snaps = snaps[snaps.game_type == "REG"]
    xw = players()[["gsis_id", "pfr_id"]].dropna()
    snaps = snaps.merge(xw, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    snaps = snaps.groupby(["gsis_id", "week"], as_index=False).offense_pct.max()
    d = d.merge(snaps, left_on=["player_id", "week"], right_on=["gsis_id", "week"], how="left").copy()
    d["snap_pct"] = d.offense_pct
    d["source"] = "nflverse:live"
    keep = ["player_id", "name", "position", "team", "season", "week", "game_id",
            *[c for c in STAT_COLS], "source"]
    return d[keep]


_players = None


def players():
    global _players
    if _players is None:
        _players = pd.read_csv(NFLV / "players.csv", low_memory=False)
    return _players


def load_combine():
    """mexwell combine -> athletic profile per (name, position)."""
    f = KAG / "mexwell__ultimate-nfl-database" / "combine.csv"
    c = pd.read_csv(f, on_bad_lines="skip", engine="python")
    c = c[["name", "position", "weight", "fortyyd", "vertical", "broad"]].copy()
    for k in ["weight", "fortyyd", "vertical", "broad"]:
        c[k] = pd.to_numeric(c[k], errors="coerce").replace(0, np.nan)
    # Barnwell speed score: weight * 200 / forty^4
    c["speed_score"] = c.weight * 200 / c.fortyyd ** 4
    c["key"] = c.name.str.lower().str.replace(r"[^a-z]", "", regex=True)
    return c.groupby("key", as_index=False)[["speed_score", "fortyyd", "vertical", "broad"]].mean()


def load_lines(max_hist_season=2025):
    """Closing spread/total per team-game.

    Kaggle tobycrabtree spreadspoke for history (<= max_hist_season); nflverse games.csv
    fills the live season. Returns team, season, week, home, spread (team-perspective,
    negative = favourite), total, implied team total.
    """
    k = pd.read_csv(KAG / "tobycrabtree__nfl-scores-and-betting-data" / "spreadspoke_scores.csv")
    teams = pd.read_csv(KAG / "tobycrabtree__nfl-scores-and-betting-data" / "nfl_teams.csv")
    name2id = dict(zip(teams.team_name, teams.team_id))
    k = k[(~k.schedule_playoff) & (k.schedule_season >= 2012)].copy()
    k = k[pd.to_numeric(k.schedule_week, errors="coerce").notna()]
    k["week"] = k.schedule_week.astype(int)
    k["season"] = k.schedule_season
    k["home"] = canon_team(k.team_home.map(name2id))
    k["away"] = canon_team(k.team_away.map(name2id))
    k["fav"] = canon_team(k.team_favorite_id)
    k["total"] = pd.to_numeric(k.over_under_line, errors="coerce")
    # home spread in "home team minus points" convention (negative = home favoured)
    k["home_spread"] = np.where(k.fav == k.home, k.spread_favorite, -k.spread_favorite)
    k.loc[k.fav == "PICK", "home_spread"] = 0.0
    k = k[["season", "week", "home", "away", "home_spread", "total"]]

    g = pd.read_csv(NFLV / "games.csv")
    g = g[(g.game_type == "REG") & (g.season > max_hist_season)].copy()
    g["home"] = canon_team(g.home_team)
    g["away"] = canon_team(g.away_team)
    g["home_spread"] = -g.spread_line  # nflverse spread_line is home margin
    g["total"] = g.total_line
    g = g[["season", "week", "home", "away", "home_spread", "total"]]
    games = pd.concat([k, g], ignore_index=True)

    h = games.rename(columns={"home": "team", "away": "opp"}).assign(is_home=1)
    h["spread"] = h.home_spread
    a = games.rename(columns={"away": "team", "home": "opp"}).assign(is_home=0)
    a["spread"] = -a.home_spread
    t = pd.concat([h, a], ignore_index=True)
    t["implied_total"] = t.total / 2 - t.spread / 2
    return t[["season", "week", "team", "opp", "is_home", "spread", "total", "implied_total"]]


def load_injuries(seasons):
    """Final game-status designations (Out / Doubtful / Questionable)."""
    out = []
    for y in seasons:
        f = NFLV / f"injuries_{y}.csv"
        if f.exists():
            i = pd.read_csv(f, low_memory=False)
            i = i[i.game_type == "REG"][["season", "week", "gsis_id", "report_status"]]
            out.append(i)
    i = pd.concat(out, ignore_index=True).dropna(subset=["gsis_id"])
    return i.drop_duplicates(["season", "week", "gsis_id"], keep="last").rename(
        columns={"gsis_id": "player_id", "report_status": "injury"})


def opp_from_game_id(gid, team):
    """2023_11_NYJ_BUF -> the team that is not `team`."""
    parts = gid.str.split("_")
    away, home = parts.str[2], parts.str[3]
    return np.where(canon_team(away) == team, canon_team(home), canon_team(away))


def build_weekly(season=2026):
    hist = load_history()
    hist = hist[hist.season < season]
    live = load_live(season)
    d = pd.concat([hist, live], ignore_index=True)
    d["team"] = canon_team(d.team)
    d["opp"] = opp_from_game_id(d.game_id, d.team)
    for c in STAT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    fill0 = [c for c in STAT_COLS if c not in ("snap_pct",)]
    d[fill0] = d[fill0].fillna(0.0)
    # snap_pct is 0-1 in nflverse, sometimes 0-100 in older exports
    d.loc[d.snap_pct > 1.5, "snap_pct"] /= 100
    # The Kaggle file's own PPR column has fumble/2-pt quirks; rescore every row
    # from box-score fields so old and new sources share one scoring formula.
    d["ppr_source"] = d.ppr
    d["ppr"] = ppr_points(d)
    d = d.sort_values(["season", "week", "player_id"]).drop_duplicates(
        ["player_id", "season", "week"], keep="last")

    p = players()[["gsis_id", "birth_date", "draft_pick", "draft_round", "rookie_season"]]
    d = d.merge(p, left_on="player_id", right_on="gsis_id", how="left").drop(columns="gsis_id")
    d["age"] = (pd.to_datetime(d.season.astype(str) + "-09-01")
                - pd.to_datetime(d.birth_date, errors="coerce")).dt.days / 365.25
    d["exp"] = d.season - d.rookie_season
    d["draft_pick"] = d.draft_pick.fillna(300)

    comb = load_combine()
    d["key"] = d.name.str.lower().str.replace(r"[^a-z]", "", regex=True)
    d = d.merge(comb, on="key", how="left").drop(columns="key")
    return d.reset_index(drop=True)
