"""Weekly draft logic: value over replacement, tiers, and snake-draft simulation.

League default: 12 teams, PPR, lineup QB / RB / RB / WR / WR / TE / FLEX(RB-WR-TE),
7-round snake draft held fresh every week.
"""
import numpy as np
import pandas as pd

TEAMS = 12
TEAM_SIZES = (10, 11, 12)  # league sizes the site's slider offers
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX = {"RB", "WR", "TE"}
ROUNDS = sum(SLOTS.values()) + 1
# replacement = first player outside the starters league-wide (flex split ~ RB 40 / WR 50 / TE 10)
def repl_rank(teams=TEAMS):
    return {"QB": teams, "RB": teams * 2 + 5, "WR": teams * 2 + 6, "TE": teams + 1}


REPL_RANK = repl_rank()


def vorp(df, col="proj", teams=TEAMS):
    out = pd.Series(0.0, index=df.index)
    for p, r in repl_rank(teams).items():
        m = df.position == p
        s = df.loc[m, col].sort_values(ascending=False)
        repl = s.iloc[min(r, len(s) - 1)] if len(s) else 0.0
        out[m] = df.loc[m, col] - repl
    return out


def tiers(df, col="proj", gap=1.25):
    """Break a positional list into tiers wherever the projection drops by > `gap` points."""
    t = pd.Series(0, index=df.index)
    for p in SLOTS:
        s = df.loc[df.position == p, col].sort_values(ascending=False)
        tier, prev = 1, None
        for i, v in s.items():
            if prev is not None and prev - v > gap:
                tier += 1
            t[i] = tier
            prev = v
    return t


def _can_take(need, flex_open, pos):
    return need.get(pos, 0) > 0 or (flex_open and pos in FLEX)


def snake_draft(pos, rank_by_team, teams=TEAMS, rounds=ROUNDS):
    """Run one snake draft.

    pos: array of positions per player. rank_by_team[k]: player indices in that team's
    preference order. Each team takes its highest-ranked available player that fills an
    open lineup slot. Returns list of picked index arrays per team.
    """
    taken = np.zeros(len(pos), dtype=bool)
    need = [dict(SLOTS) for _ in range(teams)]
    flex = [True] * teams
    ptr = [0] * teams
    rosters = [[] for _ in range(teams)]
    for r in range(rounds):
        order = range(teams) if r % 2 == 0 else range(teams - 1, -1, -1)
        for k in order:
            prefs = rank_by_team[k]
            j = ptr[k]
            while j < len(prefs):
                i = prefs[j]
                if not taken[i] and _can_take(need[k], flex[k], pos[i]):
                    break
                j += 1
            if j >= len(prefs):
                continue
            i = prefs[j]
            taken[i] = True
            rosters[k].append(i)
            p = pos[i]
            if need[k].get(p, 0) > 0:
                need[k][p] -= 1
            else:
                flex[k] = False
            # advance pointer past already-taken prefixes only
            while ptr[k] < len(prefs) and taken[prefs[ptr[k]]]:
                ptr[k] += 1
    return rosters


def draft_eval(week_df, strategy_cols, field_col="naive_exp"):
    """Score each strategy by drafting from every slot against a field that drafts by
    `field_col` (season-average consensus). Returns per-strategy mean lineup points,
    margin over the field average, and head-to-head win rate."""
    d = week_df.reset_index(drop=True)
    pos = d.position.to_numpy()
    actual = d.actual.to_numpy()

    def order(col):
        v = vorp(d, col).to_numpy()
        return list(np.argsort(-v, kind="stable"))

    field = order(field_col)
    res = {}
    for col in strategy_cols:
        mine = order(col)
        pts, margin, wins = [], [], []
        for slot in range(TEAMS):
            prefs = [field] * TEAMS
            prefs[slot] = mine
            rosters = snake_draft(pos, prefs)
            score = np.array([actual[r].sum() for r in rosters])
            others = np.delete(score, slot)
            pts.append(score[slot])
            margin.append(score[slot] - others.mean())
            wins.append((score[slot] > others).mean() + 0.5 * (score[slot] == others).mean())
        res[col] = dict(points=np.mean(pts), margin=np.mean(margin), win_rate=np.mean(wins))
    return res


def recommend_by_slot(board, teams=TEAMS, rounds=ROUNDS):
    """For each draft slot, the picks our projections would make if the rest of the league
    drafts by consensus (season average) - a 'who to take at pick N' sheet."""
    d = board.reset_index(drop=True)
    pos = d.position.to_numpy()
    mine = list(np.argsort(-vorp(d, "proj", teams).to_numpy(), kind="stable"))
    field = list(np.argsort(-vorp(d, "naive_exp", teams).to_numpy(), kind="stable"))
    out = {}
    for slot in range(teams):
        prefs = [field] * teams
        prefs[slot] = mine
        r = snake_draft(pos, prefs, teams, rounds)[slot]
        out[slot + 1] = [d.loc[i, "player_id"] for i in r]
    return out
