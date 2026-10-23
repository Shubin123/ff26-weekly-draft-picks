# FF26: Weekly Fantasy Football Draft Picks, 2026

A weekly **12-team PPR snake-draft board** for the 2026 NFL season. It's built from a mix of older Kaggle datasets and live 2026 nflverse data. A stacked ensemble of statistical models produces the projections. Everything is backtested with a rolling origin over every season from 2015 to 2026.

**Live board:** https://shubin123.github.io/ff26-weekly-draft-picks/ · **Week 5 gist:** https://gist.github.com/Shubin123/d52acafd601c738d2d7fc46fbb11005b

Each week you get:

- **A draft board** ranked by value over replacement (VORP), with positional tiers, a 10th–90th percentile floor/ceiling, the chance the player suits up, the Vegas team total and the matchup.
- **A pick sheet by draft slot**: who to take in each of the 7 rounds from pick 1 to 12, assuming the rest of the league drafts off season-average rankings.
- **A league-size slider** (8, 10, 12, 14 or 16 teams) on the live board. Replacement levels scale with league size (QB n, RB 2n+5, WR 2n+6, TE n+1), so the board re-ranks. The pick sheet and the backtest charts switch to drafts simulated and backtested for that many teams. The tables below are the 12-team default.
- **A scorecard**: how the picks did in every completed 2026 week and across 11 earlier seasons.

## Results

### Simulated drafts

These are the goal metric. Each week a 12-team draft is run once from every slot. One manager drafts with the strategy being tested and the other 11 draft by season average, which stands in for consensus. Players who don't play score 0, as in a real league.

| Strategy | Win rate vs field 2015–25 | Pts over field 2015–25 | Win rate 2026 wk 1–4 |
|---|---|---|---|
| **Stacked ensemble** | **72.6%** | **+20.3** | **81.8%** |
| LightGBM | 69.2% | +16.7 | 72.9% |
| Ridge (per position) | 69.7% | +17.0 | 71.8% |
| XGBoost (Tweedie) | 69.0% | +17.7 | 74.4% |
| Empirical Bayes | 61.8% | +9.9 | 62.9% |
| EWMA | 58.9% | +7.6 | 71.2% |
| Season average (field) | 50.0% | 0 | 50.0% |
| Oracle (hindsight ceiling) | 100% | +123 | 100% |

**By league size** (ensemble vs. a season-average field, full run for week 7; the projections are the same, and the drafts, VORP and backtest are run per size):

| Teams | Win rate 2015–25 | Pts over field 2015–25 | Win rate 2026 |
|---|---|---|---|
| 8 | 68.8% | +17.1 | 71.7% |
| 10 | 70.8% | +19.0 | 76.1% |
| 12 | 72.0% | +19.7 | 84.0% |
| 14 | 72.4% | +19.2 | 78.5% |
| 16 | 72.9% | +19.6 | 76.5% |

The strategy table above is from the week-5 run; the 12-team row here is the week-7 run, after more 2026 games went into the data.

The ensemble beats the consensus field in every individual season: 2015 0.71 · 2016 0.70 · 2017 0.75 · 2018 0.80 · 2019 0.78 · 2020 0.63 · 2021 0.73 · 2022 0.76 · 2023 0.73 · 2024 0.68 · 2025 0.72 · 2026 0.82.

### Projection accuracy

These numbers cover players who played (PPR points per game).

| Model | MAE 2015–25 | Spearman 2015–25 | MAE 2026 | Spearman 2026 |
|---|---|---|---|---|
| **Stacked ensemble** | **5.07** | **0.515** | **4.68** | **0.574** |
| Ridge | 5.12 | 0.511 | 4.73 | 0.570 |
| XGBoost Tweedie | 5.10 | 0.505 | 4.68 | 0.574 |
| LightGBM | 5.13 | 0.504 | 4.74 | 0.564 |
| Empirical Bayes | 5.22 | 0.479 | 4.93 | 0.521 |
| EWMA | 5.22 | 0.481 | 4.77 | 0.529 |
| Season average | 5.38 | 0.462 | 5.20 | 0.507 |

The 2026 rows cover weeks 1–4 plus the one finished week-5 game (TB 24 @ DAL 16, Thursday night). The models were trained on 2012–2025 only and never refit on 2026 outcomes.

## Data: old and new

| Source | Era | Used for |
|---|---|---|
| [Kaggle: mexwell/ultimate-nfl-database](https://www.kaggle.com/datasets/mexwell/ultimate-nfl-database) | 2013–2021 | Combine speed scores as a feature; play-by-play to validate the matchup feature |
| [Kaggle: philiphyde1/nfl-stats-1999-2022](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022) | 2012–2025 | Weekly player box scores and snap share (the training set) |
| [Kaggle: tobycrabtree/nfl-scores-and-betting-data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data) | 2012–2025 | Closing spreads and totals, turned into implied team totals |
| [Kaggle: heefjones/nfl-fantasy-data-1970-2024](https://www.kaggle.com/datasets/heefjones/nfl-fantasy-data-1970-2024) | 1970–2024 | Year-over-year PPG stability, a check on how strongly to shrink last season |
| [nflverse](https://github.com/nflverse/nflverse-data) | 2026 (live) | Weekly box scores, snap counts, injury reports, 2026 lines, ID crosswalk |

### Data checks

These are written to `reports/kaggle_validation.json`.

- **Defensive quality.** Play-by-play defensive yards per play agrees with the box-score "points allowed" feature within a season: r = 0.43 for pass yards/play vs WR points allowed and 0.51 for rush yards/carry vs RB points allowed. It barely carries over into the next season (r ≈ 0.2), so the matchup feature is an in-season EWMA rather than last year's number.
- **Kaggle PPR column.** The Kaggle weekly file's own `fantasy_points_ppr` disagrees with nflverse on about 20% of 2025 rows (fumble and 2-point quirks). To keep old and new data identical, every row is rescored from box-score fields with one formula.

## Method

1. **Candidate pool, point-in-time.** For each season and week the pool holds everyone active in the previous 3 weeks, plus last year's regulars in weeks 1–4. Each player is placed on his last known team. Every feature comes from an as-of join (`merge_asof`, strictly before the week), so nothing leaks from the future. Players ruled **Out** are removed.
2. **Features (69).** These include:
   - Fast and slow EWMAs and season-to-date averages of points, targets, carries, air yards, snap share, and target/rush/air-yard shares.
   - Last season's per-game line and the player's 8-game volatility.
   - Team pass and rush volume.
   - An opponent points-allowed ratio by position.
   - The Vegas implied total and spread, plus home/away.
   - Age, experience, draft capital, combine speed score and the injury designation.
3. **Models:**
   - `naive`: season-to-date average.
   - `ewma`: exponentially weighted career average.
   - `eb`: empirical-Bayes normal–normal shrinkage, with σ²/τ² estimated per position by method of moments.
   - `ridge`: RidgeCV per position.
   - `lgbm`: LightGBM, plus q10/q90 quantile models for floor and ceiling.
   - `xgb`: XGBoost with a Tweedie objective.
   - `ensemble`: a per-position NNLS stack fitted **only on earlier seasons' out-of-sample predictions**.
   - `p_play`: a logistic model of availability. The final projection is `P(plays) × points-if-active`.
4. **Draft logic:**
   - VORP against 12-team replacement levels: QB12, RB29, WR30, TE13.
   - Tiers break wherever the projection drops by more than 1.25 points.
   - A snake-draft simulator fills QB/RB/RB/WR/WR/TE/FLEX.

The 2026 stack weights lean on ridge and LightGBM. For WRs they are ridge 0.54, LightGBM 0.32, EB 0.11. For QBs they are LightGBM 0.31, XGBoost 0.26, ridge 0.23, EWMA 0.15.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_data.py --season 2026      # Kaggle + nflverse, no API key needed
.venv/bin/python scripts/kaggle_validation.py
.venv/bin/python run_pipeline.py --season 2026 --week 7  # about 8 minutes (5 league sizes)
cp reports/kaggle_validation.json docs/data/
```

The outputs are:

- `data/processed/draft_board_2026_wNN.csv`: one board per week.
- `reports/accuracy_by_season.csv` and `reports/draft_sim_by_week.csv`: the backtests.
- `docs/data/site.json`: everything the site reads.

Tests (CI runs both on every weekly refresh, before committing):

```bash
.venv/bin/python -m unittest discover tests      # draft logic per league size
cd tests && npm install && npm test              # drives the slider in headless Chrome
```

On macOS LightGBM needs OpenMP: `brew install libomp`.

A GitHub Action (`.github/workflows/weekly.yml`) rebuilds the board every Tuesday after Monday Night Football and commits it. GitHub Pages serves `docs/`.

## Limitations

- **Rookies.** A rookie enters the pool after his first NFL game, so week-1 rookies are missing.
- **Week-1 team assignments.** These use the player's last known team, so offseason moves can be wrong for week 1.
- **No in-season refit.** 2026 projections update through features as new games arrive, but the models themselves are not retrained mid-season. That keeps 2026 a true out-of-sample test.
- **Scoring.** Fumbles lost and 2-point conversions are left out so every source is scored the same way.
- **Reproducibility.** The numbers above come from the GitHub Actions run (Python 3.12). A local run on Python 3.14 with newer LightGBM/XGBoost gave the same model ranking but slightly different figures, for example a 2026 ensemble win rate of 87.7% instead of 81.8%. Tree models fitted with multiple threads are not bit-for-bit reproducible across library versions.
- **Week 5 sample.** The week-5 board was produced before Sunday's games. Only TB @ DAL can be scored so far.
