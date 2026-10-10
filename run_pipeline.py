"""End-to-end: features -> rolling backtest (2015-2026) -> weekly draft board.

python run_pipeline.py --season 2026 --week 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ffdraft import data, draft, features, models  # noqa: E402

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
SITE = ROOT / "docs" / "data"
REPORTS = ROOT / "reports"
FIRST_FEATURE_SEASON = 2013
FIRST_TEST_SEASON = 2015
STRATS = models.BASE_MODELS + ["ensemble"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_features(season, week):
    weekly = data.build_weekly(season)
    lines = data.load_lines(season - 1)
    inj = data.load_injuries(range(FIRST_FEATURE_SEASON - 1, season + 1))
    f = features.build(weekly, lines, inj, range(FIRST_FEATURE_SEASON, season + 1))
    f = f[(f.season < season) | (f.week <= week)].reset_index(drop=True)
    return weekly, f


def rolling_backtest(f, season):
    feats = features.feature_columns(f)
    preds = []
    for s in range(FIRST_TEST_SEASON, season + 1):
        tr = f[f.season < s]
        te = f[f.season == s]
        log(f"fit <{s}  ({tr.played.sum():,} played rows)  -> predict {s} ({len(te):,} candidates)")
        m = models.Projector(feats).fit(tr)
        p = m.predict(te)
        preds.append(pd.concat([te[["player_id", "name", "position", "team", "opp", "season",
                                    "week", "injury", "played", "actual", "is_home",
                                    "implied_total", "spread", "def_ratio", "std_ppr",
                                    "ewf_snap_pct", "ewf_tgt_share", "ewf_carries"]], p], axis=1))
        if s == season:
            final = m
    oof = pd.concat(preds, ignore_index=True)
    # stack weights for season s use only seasons < s
    oof["ensemble"] = oof[models.BASE_MODELS].mean(axis=1)
    weights = {}
    for s in range(FIRST_TEST_SEASON + 1, season + 1):
        w = models.fit_stack(oof[oof.season < s])
        weights[s] = {p: dict(zip(models.BASE_MODELS, np.round(v, 3))) for p, v in w.items()}
        m = oof.season == s
        oof.loc[m, "ensemble"] = models.apply_stack(oof[m], oof.loc[m, "position"], w)
    for c in STRATS:
        oof[f"{c}_exp"] = oof[c] * oof.p_play
    oof["proj"] = oof.ensemble_exp
    imp = pd.Series(final.lgbm.feature_importances_, index=feats).sort_values(ascending=False)
    return oof, weights, imp


def accuracy(oof):
    rows = []
    played = oof[oof.played == 1]
    for (s, c) in [(s, c) for s in sorted(oof.season.unique()) for c in STRATS]:
        d = played[played.season == s]
        if d.empty:
            continue
        e = d[c] - d.actual
        rho, hit = [], []
        for (_, _), g in d.groupby(["week", "position"]):
            if len(g) >= 8:
                rho.append(spearmanr(g[c], g.actual).statistic)
                k = 12 if g.position.iloc[0] in ("QB", "TE") else 24
                k = min(k, len(g) // 2)
                hit.append(len(set(g.nlargest(k, c).index) & set(g.nlargest(k, "actual").index)) / k)
        rows.append(dict(season=int(s), model=c, n=len(d), mae=e.abs().mean(),
                         rmse=np.sqrt((e ** 2).mean()),
                         r2=1 - (e ** 2).sum() / ((d.actual - d.actual.mean()) ** 2).sum(),
                         spearman=np.nanmean(rho) if rho else np.nan,
                         topk_hit=np.mean(hit) if hit else np.nan))
    return pd.DataFrame(rows)


def draft_backtest(oof, season, week):
    rows = []
    strats = [f"{c}_exp" for c in STRATS] + ["oracle"]
    for (s, w), g in oof.groupby(["season", "week"]):
        if s == season and w >= week:
            continue  # target week not finished
        g = g[~g.injury.isin(["Out", "Doubtful"])].copy()
        g["oracle"] = g.actual
        res = draft.draft_eval(g, strats)
        for k, v in res.items():
            rows.append(dict(season=int(s), week=int(w), strategy=k.replace("_exp", ""), **v))
    return pd.DataFrame(rows)


def board(oof, season, week, final_cols):
    b = oof[(oof.season == season) & (oof.week == week)].copy()
    b = b[~b.injury.isin(["Out"])]
    b["vorp"] = draft.vorp(b, "proj")
    b["tier"] = draft.tiers(b, "proj")
    b = b.sort_values("vorp", ascending=False)
    b["overall_rank"] = np.arange(1, len(b) + 1)
    b["pos_rank"] = b.groupby("position").proj.rank(ascending=False, method="first").astype(int)
    slots = draft.recommend_by_slot(b)
    b = b[b.pos_rank <= b.position.map({"QB": 32, "RB": 60, "WR": 80, "TE": 32})]
    return b[final_cols], slots


def to_records(df):
    return json.loads(df.to_json(orient="records", double_precision=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=5, help="week to draft for (next to play)")
    a = ap.parse_args()
    for d in (PROC, SITE, REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    log("building features")
    weekly, f = build_features(a.season, a.week)
    log(f"features: {f.shape}")
    oof, weights, imp = rolling_backtest(f, a.season)
    oof.to_parquet(PROC / "predictions_all.parquet")

    log("accuracy metrics")
    acc = accuracy(oof)
    acc.to_csv(REPORTS / "accuracy_by_season.csv", index=False)
    log("draft simulations")
    dr = draft_backtest(oof, a.season, a.week)
    dr.to_csv(REPORTS / "draft_sim_by_week.csv", index=False)

    cols = ["overall_rank", "pos_rank", "tier", "player_id", "name", "position", "team", "opp",
            "is_home", "implied_total", "spread", "def_ratio", "injury", "p_play", "proj",
            "floor", "ceiling", "vorp", "naive_exp", "ewma", "eb", "ridge", "lgbm", "xgb",
            "ensemble", "std_ppr", "ewf_snap_pct", "ewf_tgt_share", "ewf_carries",
            "played", "actual"]
    boards, slot_sheets = {}, {}
    for w in range(1, a.week + 1):
        b, slots = board(oof, a.season, w, cols)
        boards[w] = b
        slot_sheets[w] = slots
        b.to_csv(PROC / f"draft_board_{a.season}_w{w:02d}.csv", index=False)

    # ---------- site payload ----------
    summ = acc.groupby("model")[["mae", "rmse", "r2", "spearman", "topk_hit"]].mean()
    hist = acc[acc.season < a.season]
    summ_hist = hist.groupby("model")[["mae", "rmse", "r2", "spearman", "topk_hit"]].mean()
    cur = acc[acc.season == a.season].set_index("model")
    dsum = dr.groupby("strategy")[["points", "margin", "win_rate"]].mean()
    dsum_hist = dr[dr.season < a.season].groupby("strategy")[["points", "margin", "win_rate"]].mean()
    dsum_cur = dr[dr.season == a.season].groupby("strategy")[["points", "margin", "win_rate"]].mean()
    cur_weeks = dr[dr.season == a.season].pivot_table(index="week", columns="strategy",
                                                      values=["win_rate", "points"])
    payload = {
        "season": a.season, "week": a.week,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "league": {"teams": draft.TEAMS, "slots": draft.SLOTS, "flex": sorted(draft.FLEX),
                   "scoring": "PPR"},
        "accuracy_by_season": to_records(acc),
        "accuracy_hist": to_records(summ_hist.reset_index()),
        "accuracy_current": to_records(cur.reset_index()),
        "draft_by_week": to_records(dr),
        "draft_hist": to_records(dsum_hist.reset_index()),
        "draft_current": to_records(dsum_cur.reset_index()),
        "stack_weights": {str(k): v for k, v in weights.items()},
        "feature_importance": to_records(imp.head(25).rename("gain").reset_index()
                                         .rename(columns={"index": "feature"})),
        "boards": {str(w): to_records(b) for w, b in boards.items()},
        "slot_picks": {str(w): {str(k): v for k, v in s.items()} for w, s in slot_sheets.items()},
    }
    (SITE / "site.json").write_text(json.dumps(payload))
    summary = {"accuracy_all": summ.round(3).to_dict(), "accuracy_hist": summ_hist.round(3).to_dict(),
               "draft_all": dsum.round(3).to_dict(), "draft_hist": dsum_hist.round(3).to_dict(),
               "draft_current": dsum_cur.round(3).to_dict()}
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=1))
    log("done")
    print(summ_hist.round(3).to_string())
    print(cur.round(3).to_string())
    print(dsum_hist.round(3).to_string())
    print(dsum_cur.round(3).to_string())
    print(cur_weeks.round(3).to_string())


if __name__ == "__main__":
    main()
