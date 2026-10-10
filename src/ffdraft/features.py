"""Point-in-time feature engineering.

Every candidate row (player, season, week) only sees games strictly before that
week. History is summarised as a "state after each game" table and attached to
candidate rows with merge_asof(allow_exact_matches=False), so there is no leakage.
"""
import numpy as np
import pandas as pd

from .data import POSITIONS

BASE = ["ppr", "targets", "carries", "receptions", "rec_yds", "rush_yds", "pass_att",
        "pass_yds", "pass_td", "pass_int", "tds", "air_yards", "snap_pct",
        "tgt_share", "rush_share", "ay_share"]
EWM_FAST = 3   # half-life in games
EWM_SLOW = 10


def tkey(season, week):
    return season * 100 + week


def add_shares(g):
    g = g.copy()
    g["tds"] = g.rush_td + g.rec_td
    team = g.groupby(["season", "week", "team"])[["targets", "carries", "air_yards"]].transform("sum")
    g["tgt_share"] = g.targets / team.targets.replace(0, np.nan)
    g["rush_share"] = g.carries / team.carries.replace(0, np.nan)
    g["ay_share"] = g.air_yards / team.air_yards.replace(0, np.nan)
    g[["tgt_share", "rush_share", "ay_share"]] = g[["tgt_share", "rush_share", "ay_share"]].fillna(0)
    g["t"] = tkey(g.season, g.week)
    return g


def player_state(g):
    """Rolling player summaries *including* each game (consumed strictly after it)."""
    g = g.sort_values(["player_id", "t"]).copy()
    by = g.groupby("player_id")
    st = g[["player_id", "t", "season", "team", "position", "name"]].copy()
    for c in BASE:
        st[f"ewf_{c}"] = by[c].transform(lambda s: s.ewm(halflife=EWM_FAST).mean())
    for c in ["ppr", "targets", "carries", "snap_pct", "tgt_share", "tds"]:
        st[f"ews_{c}"] = by[c].transform(lambda s: s.ewm(halflife=EWM_SLOW).mean())
    bys = g.groupby(["player_id", "season"])
    for c in BASE:
        st[f"std_{c}"] = bys[c].transform(lambda s: s.expanding().mean())
    st["std_games"] = bys.cumcount() + 1
    st["career_games"] = by.cumcount() + 1
    st["last_ppr"] = g.ppr
    st["sd8_ppr"] = by.ppr.transform(lambda s: s.rolling(8, min_periods=3).std())
    st["max8_ppr"] = by.ppr.transform(lambda s: s.rolling(8, min_periods=1).max())
    st["last_t"] = g.t
    return st


def prev_season(g):
    a = g.groupby(["player_id", "season"]).agg(
        ps_games=("ppr", "size"), ps_ppg=("ppr", "mean"), ps_tgt=("targets", "mean"),
        ps_car=("carries", "mean"), ps_snap=("snap_pct", "mean"),
        ps_tgt_share=("tgt_share", "mean"), ps_sd=("ppr", "std")).reset_index()
    a["season"] += 1
    return a


def team_state(g):
    """Offense environment: pass volume, rush volume, skill-position PPR per game."""
    t = g.groupby(["team", "season", "week", "t"], as_index=False).agg(
        tm_pass=("pass_att", "sum"), tm_car=("carries", "sum"), tm_ppr=("ppr", "sum"))
    t = t.sort_values(["team", "t"])
    by = t.groupby("team")
    out = t[["team", "t", "season"]].copy()
    for c in ["tm_pass", "tm_car", "tm_ppr"]:
        out[f"ew_{c}"] = by[c].transform(lambda s: s.ewm(halflife=4).mean())
    return out


def defense_state(g):
    """Fantasy points allowed to each position, smoothed, as a ratio to league mean."""
    d = g.groupby(["opp", "season", "week", "t", "position"], as_index=False).ppr.sum()
    lg = d.groupby(["season", "position"]).ppr.mean().rename("lg").reset_index()
    lg["season"] += 1  # previous-season league mean: known before the season starts
    d = d.merge(lg, on=["season", "position"], how="left")
    d["lg"] = d.lg.fillna(d.groupby("position").ppr.transform("mean"))
    d["ratio"] = d.ppr / d.lg
    d = d.sort_values(["opp", "position", "t"])
    d["def_ratio"] = d.groupby(["opp", "position"]).ratio.transform(
        lambda s: s.ewm(halflife=6).mean())
    d["def_n"] = d.groupby(["opp", "position"]).cumcount() + 1
    return d[["opp", "position", "t", "def_ratio", "def_n"]]


def build_pool(g, lines, seasons, injuries):
    """Candidate players for each (season, week) as they'd look before kickoff."""
    last = g[["player_id", "season", "week", "t", "team", "position", "name"]]
    rows = []
    for s in seasons:
        weeks = sorted(lines.loc[lines.season == s, "week"].unique())
        prev_n = g[g.season == s - 1].groupby("player_id").size()
        vets = set(prev_n[prev_n >= 6].index)
        for w in weeks:
            T = tkey(s, w)
            recent = last[(last.t < T) & (last.t >= tkey(s, max(1, w - 3))) & (last.season == s)]
            ids = set(recent.player_id)
            if w <= 4:
                ids |= vets
            rows.append(pd.DataFrame({"player_id": list(ids), "season": s, "week": w}))
    pool = pd.concat(rows, ignore_index=True)
    pool["player_id"] = pool.player_id.astype(g.player_id.dtype)
    pool["t"] = tkey(pool.season, pool.week)
    # last known team/position before this week
    pool = pool.sort_values("t")
    pool = pd.merge_asof(pool, last.sort_values("t")[["player_id", "t", "team", "position", "name"]],
                         on="t", by="player_id", allow_exact_matches=False)
    pool = pool.dropna(subset=["team"])
    pool = pool.merge(lines, on=["season", "week", "team"], how="inner")  # drops byes
    # outcome: did he play, how many points (0 if not)
    out = g[["player_id", "season", "week", "ppr", "team"]].rename(columns={"team": "team_actual"})
    pool = pool.merge(out, on=["player_id", "season", "week"], how="left")
    pool["played"] = pool.ppr.notna().astype(int)
    pool["actual"] = pool.ppr.fillna(0.0)
    pool = pool.drop(columns="ppr")
    pool = pool.merge(injuries, on=["player_id", "season", "week"], how="left")
    pool["injury"] = pool.injury.fillna("None")
    return pool


def attach(pool, g):
    """Join all point-in-time state onto the candidate pool."""
    st = player_state(g).drop(columns=["season", "team", "position", "name"])
    pool = pool.sort_values("t")
    f = pd.merge_asof(pool, st.sort_values("t"), on="t", by="player_id",
                      allow_exact_matches=False)
    same = f.season == (f.last_t // 100)
    for c in [c for c in f.columns if c.startswith("std_")]:
        f.loc[~same, c] = np.nan
    f.loc[~same, "std_games"] = 0
    f["gap_weeks"] = np.where(same, f.week - f.last_t % 100, f.week + 17)
    f = f.merge(prev_season(g), on=["player_id", "season"], how="left")

    ts = team_state(g).drop(columns="season").sort_values("t")
    f = pd.merge_asof(f.sort_values("t"), ts, on="t", by="team", allow_exact_matches=False)
    ds = defense_state(g).sort_values("t")
    f = pd.merge_asof(f.sort_values("t"), ds, on="t", by=["opp", "position"],
                      allow_exact_matches=False)
    f["def_ratio"] = f.def_ratio.fillna(1.0)

    bio = g.drop_duplicates("player_id", keep="last").set_index("player_id")[
        ["draft_pick", "age", "exp", "speed_score", "rookie_season", "birth_date"]]
    f = f.join(bio[["draft_pick", "speed_score", "rookie_season", "birth_date"]], on="player_id")
    f["exp"] = f.season - f.rookie_season
    f["age"] = (pd.to_datetime(f.season.astype(str) + "-09-01")
                - pd.to_datetime(f.birth_date, errors="coerce")).dt.days / 365.25
    f["log_pick"] = np.log1p(f.draft_pick.fillna(300))
    for p in POSITIONS:
        f[f"pos_{p}"] = (f.position == p).astype(int)
    f["inj_Q"] = (f.injury == "Questionable").astype(int)
    f = f.drop(columns=["rookie_season", "birth_date"])
    return f.sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def feature_columns(f):
    skip = {"player_id", "season", "week", "t", "team", "position", "name", "opp",
            "team_actual", "played", "actual", "injury", "last_t", "draft_pick"}
    return [c for c in f.columns if c not in skip and f[c].dtype != object]


def build(weekly, lines, injuries, seasons):
    g = add_shares(weekly.dropna(subset=["player_id"]))
    pool = build_pool(g, lines, seasons, injuries)
    return attach(pool, g)
