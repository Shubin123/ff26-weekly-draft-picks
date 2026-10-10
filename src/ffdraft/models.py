"""Projection models.

Two stages, combined as  E[points] = P(plays) * E[points | plays]:

Points-if-active models (fit on candidates who played):
  naive    - season-to-date average, falling back to last season, then position mean
  ewma     - exponentially weighted career average (half-life 3 games)
  eb       - empirical-Bayes (normal-normal) shrinkage of this season's average toward a
             prior built from last season, itself shrunk to the position mean. Shrinkage
             constants are method-of-moments estimates (sigma^2 / tau^2) per position.
  ridge    - per-position RidgeCV on standardised usage / matchup / Vegas features
  lgbm     - LightGBM gradient boosting (L2), plus q10 / q90 quantile models
  xgb      - XGBoost with a Tweedie objective (right-skewed, non-negative target)
  ensemble - per-position non-negative least squares stack of the above, fit only on
             earlier seasons' out-of-sample predictions

Availability model:
  p_play   - logistic regression on injury designation + recent usage / inactivity
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import POSITIONS

warnings.filterwarnings("ignore", category=UserWarning)

BASE_MODELS = ["naive", "ewma", "eb", "ridge", "lgbm", "xgb"]


class EmpiricalBayes:
    def fit(self, tr):
        self.pos_mean, self.k_season, self.k_prev = {}, {}, {}
        for p in POSITIONS:
            d = tr[tr.position == p]
            self.pos_mean[p] = d.actual.mean()
            # player-season means of realised points -> within / between variance
            ps = d.groupby(["player_id", "season"]).actual.agg(["mean", "var", "size"])
            ps = ps[ps["size"] >= 4]
            sigma2 = ps["var"].mean()
            tau2 = max(ps["mean"].var() - sigma2 / ps["size"].mean(), 1e-3)
            self.k_season[p] = sigma2 / tau2
            # year-to-year: how much a full prior season predicts this one
            self.k_prev[p] = self.k_season[p] * 1.5
        return self

    def predict(self, f):
        m = f.position.map(self.pos_mean)
        kp = f.position.map(self.k_prev)
        ks = f.position.map(self.k_season)
        ng = f.ps_games.fillna(0)
        prior = (ng * f.ps_ppg.fillna(0) + kp * m) / (ng + kp)
        # no prior season: lean on fast EWMA of whatever history exists
        prior = np.where(ng > 0, prior, np.where(f.ewf_ppr.notna(), 0.5 * f.ewf_ppr + 0.5 * m, m))
        n = f.std_games.fillna(0)
        x = f.std_ppr.fillna(0)
        return ((n * x + ks * prior) / (n + ks)).to_numpy()


def naive(f):
    pos_mean = f.groupby("position").ewf_ppr.transform("median")
    return f.std_ppr.fillna(f.ps_ppg).fillna(f.ewf_ppr).fillna(pos_mean).to_numpy()


def ewma(f):
    return f.ewf_ppr.fillna(f.ps_ppg).fillna(f.groupby("position").ewf_ppr.transform("median")).to_numpy()


class Projector:
    def __init__(self, feats, seed=7):
        self.feats = feats
        self.seed = seed

    def fit(self, tr_all):
        tr = tr_all[tr_all.played == 1]
        X, y = tr[self.feats], tr.actual.to_numpy()
        self.eb = EmpiricalBayes().fit(tr)
        self.ridge = {}
        for p in POSITIONS:
            m = tr.position == p
            self.ridge[p] = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                RidgeCV(alphas=np.logspace(-1, 3, 9))).fit(X[m], y[m])
        lp = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=self.seed, verbose=-1)
        self.lgbm = lgb.LGBMRegressor(**lp).fit(X, y)
        self.q10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **lp).fit(X, y)
        self.q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **lp).fit(X, y)
        self.xgb = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3,
                                    n_estimators=500, learning_rate=0.04, max_depth=5,
                                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                                    random_state=self.seed).fit(X, np.clip(y, 0, None))
        # availability: everyone in the pool who wasn't ruled Out/Doubtful pre-game
        av = tr_all[~tr_all.injury.isin(["Out", "Doubtful"])]
        self.pfeats = ["inj_Q", "gap_weeks", "std_games", "ewf_snap_pct", "ews_snap_pct",
                       "career_games", "ewf_ppr", "log_pick", "week"]
        self.pplay = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=500)).fit(
            av[self.pfeats], av.played)
        return self

    def predict(self, f):
        X = f[self.feats]
        out = pd.DataFrame(index=f.index)
        out["naive"] = naive(f)
        out["ewma"] = ewma(f)
        out["eb"] = self.eb.predict(f)
        out["ridge"] = np.nan
        for p in POSITIONS:
            m = (f.position == p).to_numpy()
            if m.any():
                out.loc[m, "ridge"] = self.ridge[p].predict(X[m])
        out["lgbm"] = self.lgbm.predict(X)
        out["xgb"] = self.xgb.predict(X)
        out["floor"] = self.q10.predict(X)
        out["ceiling"] = self.q90.predict(X)
        out["p_play"] = self.pplay.predict_proba(f[self.pfeats])[:, 1]
        out.loc[f.injury.isin(["Out", "Doubtful"]).to_numpy(), "p_play"] = 0.0
        return out


def fit_stack(oof):
    """Per-position NNLS weights on earlier seasons' out-of-sample predictions."""
    w = {}
    for p in POSITIONS:
        d = oof[(oof.position == p) & (oof.played == 1)]
        A = d[BASE_MODELS].to_numpy()
        coef, _ = nnls(A, d.actual.to_numpy())
        w[p] = coef / coef.sum() if coef.sum() > 0 else np.full(len(BASE_MODELS), 1 / len(BASE_MODELS))
    return w


def apply_stack(pred, positions, w):
    e = np.zeros(len(pred))
    for p in POSITIONS:
        m = (positions == p).to_numpy()
        e[m] = pred.loc[m, BASE_MODELS].to_numpy() @ w[p]
    return e
