# EPL Match Outcome Predictor

A machine learning pipeline that predicts English Premier League match outcomes (Win / Draw / Loss) using historical match data from 2019/20 to 2022/23, tested on the 2023/24 season.

---

## Overview

The pipeline builds a custom ELO rating system per team and combines it with rolling performance features (goals, xG, shots, possession) to train four ML classifiers. The best model — Random Forest — achieves **~65% accuracy** on the held-out 2023/24 season, compared to a naive baseline of **~39%** (always guessing the most common outcome).

---

## Project Structure

```
Football.ipynb
matches.csv                  # Input data (EPL matches 2019/20 – 2023/24)
model_comparison.png         # Output: F1 score bar chart + confusion matrix
feature_importance.png       # Output: Random Forest feature importances
```

---

## Dataset

The `matches.csv` file contains one row per team per match (so each fixture appears twice). Key columns used:

| Column | Description |
|---|---|
| `date` | Match date |
| `team` | Team name |
| `opponent` | Opponent name |
| `venue` | Home or Away |
| `result` | W / D / L (from the team's perspective) |
| `gf` / `ga` | Goals for / against |
| `xg` / `xga` | Expected goals for / against |
| `poss` | Possession % |
| `sot` | Shots on target |
| `round` | Match week (e.g. "Matchweek 12") |

> **Note:** The `season` column in the raw CSV is unreliable (a scraping artifact). The pipeline derives the true EPL season from the match date — seasons run August to May, so August 2021 → May 2022 = season 2021.

---

## How It Works

### 1. ELO Rating System

Each team starts at 1500 ELO. After every match, ELO updates based on the result and the pre-match rating difference between the two teams:

```
E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
R_A_new = R_A + K × (score - E(A))
```

Where `score` = 1 (win), 0.5 (draw), 0 (loss) and `K = 32`.

ELO is computed strictly in chronological order to prevent data leakage — only past matches inform each prediction.

### 2. Rolling Performance Features

For each team, rolling 5-match averages are computed using `.shift(1).rolling(5)` — the shift ensures only matches *before* the current one contribute:

- Goals scored / conceded average
- xG / xGA average
- Shots on target average
- Possession average
- Form points (3 for W, 1 for D, 0 for L)

### 3. Train / Test Split

| Split | Seasons | Rows |
|---|---|---|
| Train | 2019/20, 2020/21, 2021/22, 2022/23 | ~4,000 |
| Test | 2023/24 | 760 |

A strict time-based split is used — no random shuffling, no data from the future leaking into training.

### 4. Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Best overall performer |
| Gradient Boosting | Good but slower to train |
| XGBoost | Competitive; better on draws |

---

## Results (2023/24 Season)

| Model | Macro F1 | Accuracy |
|---|---|---|
| **Random Forest** | **0.61** | **65%** |
| XGBoost | 0.60 | 63% |
| Gradient Boosting | ~0.55 | ~59% |
| Logistic Regression | ~0.38 | ~51% |

Naive baseline (always predict the majority class): **39.2%**

Draws are the hardest class to predict — even the best model achieves only ~0.47 F1 on draws, vs ~0.65–0.70 on wins and losses.

---


## Why the Model Struggles

### 1. Draws are fundamentally unpredictable from stats alone

Draws cluster around matches where both teams are closely rated. The ELO difference for draws averages near 0, but so does the ELO difference for many wins and losses. There is no statistical signature that reliably separates "closely contested match that ended in a goal" from "closely contested match that ended level." Even professional betting markets mis-price draws more than any other outcome.

### 2. Football is a low-scoring, high-variance sport

In a 90-minute match, the average number of goals is around 2.7. A single deflection, red card, or VAR decision can completely change the result. This randomness means the relationship between underlying performance (xG, possession) and actual outcome is weak on a per-match basis — it only becomes reliable over many matches, which is exactly what the rolling averages try to capture.

### 3. No player-level information

The model treats every match as team vs. team with no knowledge of who is actually playing. A team missing its first-choice goalkeeper and two centre-backs is rated the same as a fully fit squad. Injuries, suspensions, and rotation for European fixtures are all invisible to this pipeline.

### 4. Rolling windows ignore context

A 5-game rolling average treats a run of 5 wins against bottom-half teams the same as 5 wins against top-six opposition. Strength of schedule is not accounted for.

---

## Future Improvements

### ELO Enhancements (implemented in v2)
- **xG-based ELO** — update ratings using expected goals instead of actual scorelines, removing luck from individual shot outcomes
- **Margin-of-victory multiplier** — larger xG margins produce bigger ELO updates (with log-dampening to prevent outliers dominating)
- **Dynamic home advantage** — apply a per-match home boost directly inside the ELO calculation rather than as a flat model feature
- **Season decay** — regress ELOs toward 1500 at the start of each season to account for transfers and squad changes
- **Variable K-factor** — reduce K as the season progresses, reflecting decreasing uncertainty about team quality
