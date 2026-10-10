"""Download every raw input into data/raw/.

Old / historical sources (Kaggle, public downloads, no API key needed):
  * mexwell/ultimate-nfl-database        - play-by-play 2013-2021, combine, weather
  * philiphyde1/nfl-stats-1999-2022      - weekly player + team offense 2012-2025
  * tobycrabtree/nfl-scores-and-betting-data - scores, spreads, totals 1966-2025
  * heefjones/nfl-fantasy-data-1970-2024 - season-level fantasy results 1970-2024

New / live sources (nflverse GitHub releases, refreshed nightly in season):
  * stats_player_week_<season>.csv   - 2026 weekly player box scores
  * snap_counts_<season>.csv         - 2026 snap shares
  * injuries_<season>.csv            - final injury designations 2012-2026
  * players.csv                      - id crosswalk (gsis <-> pfr), draft info
  * games.csv                        - schedule + closing spread/total (2026 lines)

Usage: python scripts/fetch_data.py [--season 2026] [--force]
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KAGGLE = [
    "mexwell/ultimate-nfl-database",
    "philiphyde1/nfl-stats-1999-2022",
    "tobycrabtree/nfl-scores-and-betting-data",
    "heefjones/nfl-fantasy-data-1970-2024",
]
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def get(url):
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def fetch_kaggle(force):
    out = RAW / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    for ref in KAGGLE:
        dest = out / ref.replace("/", "__")
        if dest.exists() and not force:
            print(f"skip  {ref}")
            continue
        print(f"get   kaggle {ref}")
        blob = get(f"https://www.kaggle.com/api/v1/datasets/download/{ref}")
        dest.mkdir(exist_ok=True)
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)


def fetch_nflverse(season, force):
    out = RAW / "nflverse"
    out.mkdir(parents=True, exist_ok=True)
    live = [
        (f"stats_player/stats_player_week_{season}.csv", True),
        (f"snap_counts/snap_counts_{season}.csv", True),
        ("players/players.csv", True),
    ]
    live += [(f"injuries/injuries_{y}.csv", y == season) for y in range(2012, season + 1)]
    for rel, always in live:
        dest = out / Path(rel).name
        if dest.exists() and not (force or always):
            continue
        print(f"get   nflverse {rel}")
        try:
            dest.write_bytes(get(f"{NFLVERSE}/{rel}"))
        except requests.HTTPError as e:
            print(f"warn  {rel}: {e}", file=sys.stderr)
    print("get   nflverse games.csv")
    (out / "games.csv").write_bytes(get(GAMES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fetch_kaggle(a.force)
    fetch_nflverse(a.season, a.force)
