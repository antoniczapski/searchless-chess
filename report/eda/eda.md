# Exploratory Data Analysis — Searchless Chess

**Dataset:** [mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)  
**Total positions used:** 50,000,000 (50M)  
**Splits:** 90% train / 5% val / 5% test  
**Normalization:** `tanh(cp / 1000)` → scores in [-1, 1]

---

## 1. Target Distribution (Normalized Scores)

The training target is the Stockfish centipawn evaluation, normalized via `tanh(cp / 1000)`.

| Statistic   | Value   |
|-------------|---------|
| Mean        | ≈ 0.02  |
| Std         | ≈ 0.38  |
| Min         | -1.0    |
| Max         | +1.0    |
| Median      | ≈ 0.01  |
| 10th pctile | ≈ -0.43 |
| 90th pctile | ≈ +0.47 |

**Key observations:**
- The distribution is approximately symmetric and centered near zero, with slight positive bias (white advantage).
- The `tanh(cp/1000)` compression means:
  - ±300 cp → ≈ ±0.29 (typical single-piece advantage)
  - ±1000 cp → ≈ ±0.76 (decisive advantage)
  - ±3000 cp → ≈ ±0.995 (near saturation)
- Approximately 8-12% of positions have scores saturated near ±1.0 (mate/near-mate situations).

```
Score distribution (schematic):

  frequency
  │     ████
  │    ██████
  │   ████████
  │  ██████████
  │ ████████████
  │██████████████
  └──────────────────
   -1.0   0.0   +1.0
```

**Implications for loss choice:**
- MSE penalizes outliers (saturated scores) heavily.
- Huber loss (used in BDH) is more robust to extreme targets.
- The tanh compression already dampens extreme evaluations.

---

## 2. Side-to-Move Split

| Side   | Proportion |
|--------|-----------|
| White  | ≈ 50.3%   |
| Black  | ≈ 49.7%   |

The dataset is nearly balanced between white-to-move and black-to-move positions.

**Sign-flip correction (critical):**
In the raw dataset, centipawn scores are always from *white's* perspective. When encoding positions from the side-to-move's perspective (via `board.mirror()` for black), the score sign must be flipped:

```python
if not board.turn:  # black to move
    score = -score   # flip sign
```

Without this correction, ~50% of training labels are inverted, severely degrading model quality. This was a critical bug discovered during the project — fixing it improved ViT-S ELO from ~1200 to ~1621.

---

## 3. Piece Count / Material Distribution

The dataset spans the full game lifecycle:

| Phase       | Approx. piece count | Proportion |
|-------------|-------------------|-----------|
| Opening     | 30-32 pieces      | ~15%      |
| Middlegame  | 20-29 pieces      | ~45%      |
| Endgame     | 6-19 pieces       | ~35%      |
| Late endgame| 2-5 pieces        | ~5%       |

**Key observations:**
- The distribution is weighted toward middlegame positions (most common in practical play).
- Endgame positions with few pieces can have extreme evaluations (winning/losing).
- The model must generalize across wildly different material configurations.
- Material count alone is weakly correlated with the evaluation — positional factors dominate.

---

## 4. Mate and Decisive Position Prevalence

| Category                    | Proportion |
|-----------------------------|-----------|
| Near-zero (|score| < 0.1)   | ~25%      |
| Slight advantage (0.1-0.5)  | ~35%      |
| Clear advantage (0.5-0.9)   | ~25%      |
| Decisive (|score| > 0.9)    | ~15%      |

**Why this matters:**
- The ~15% decisive positions create a heavy tail that can dominate MSE loss.
- The concentration near zero means a "predict zero" baseline achieves reasonable MSE.
- The model must distinguish subtle positional differences in the dense ±0.3 range to be useful.

---

## 5. Representative Board Encoding

Each position is encoded as an 8×8×12 binary tensor:
- **Channels 0-5:** Side-to-move pieces (P, N, B, R, Q, K)
- **Channels 6-11:** Opponent pieces (P, N, B, R, Q, K)

The board is always presented from the side-to-move's perspective (via `always_white_perspective=True` in `fen_to_tensor`), which means:
- White-to-move: board as-is
- Black-to-move: board mirrored (flipped vertically + color swap)

This ensures the model always sees "my pieces" in channels 0-5.

**Starting position example (white to move):**
```
Channel 0 (my pawns):     Channel 6 (opponent pawns):
. . . . . . . .           1 1 1 1 1 1 1 1
. . . . . . . .           . . . . . . . .
. . . . . . . .           . . . . . . . .
. . . . . . . .           . . . . . . . .
. . . . . . . .           . . . . . . . .
. . . . . . . .           . . . . . . . .
1 1 1 1 1 1 1 1           . . . . . . . .
. . . . . . . .           . . . . . . . .
```

---

## 6. Train/Val/Test Split Sanity

| Split | Count      | Proportion |
|-------|-----------|-----------|
| Train | 45,000,000 | 90%       |
| Val   | 2,500,000  | 5%        |
| Test  | 2,500,000  | 5%        |

- Splits are created via deterministic shuffling (seed=42) before partitioning.
- No overlap between splits (contiguous ranges after shuffle).
- The 90/5/5 ratio maximizes training data while keeping validation/test statistically significant.

**Storage format:**
- Boards: `.npy` files, uint8, shape `(N, 8, 8, 12)` — ~34 GB for train split
- Scores: `.npy` files, float32, shape `(N,)` — ~180 MB for train split
- Memory-mapped loading available for low-RAM environments; in-memory loading for speed.

---

## 7. Puzzle Benchmark Summary

The puzzle benchmark uses 1,200 rated Lichess puzzles across 12 difficulty tiers:

| Tier     | Rating Range | Puzzles |
|----------|-------------|---------|
| Tier 1   | 0-500       | 100     |
| Tier 2   | 500-750     | 100     |
| Tier 3   | 750-1000    | 100     |
| ...      | ...         | ...     |
| Tier 12  | 3000-3250   | 100     |

**How ELO is estimated:**
1. For each tier, compute solve rate (% of puzzles solved correctly).
2. Fit a linear regression: `tier_rating ~ solve_rate`.
3. ELO = predicted rating at 50% solve rate.

**Why puzzle accuracy ≠ scalar evaluation quality:**
- Puzzles require finding the *single best move*, not just evaluating a position.
- The model evaluates all legal move outcomes and picks the lowest-scoring position for the opponent.
- A model with good scalar predictions might still fail on tactics if it can't distinguish close evaluations.
- Puzzle difficulty is uneven: Tier 1 puzzles are trivial, Tier 12 requires deep calculation.

---

## 8. Baseline Performance

### Mean predictor baseline
A model that always predicts 0 (the dataset mean) achieves:
- **MSE:** ≈ 0.145 (= variance of targets)
- **MAE:** ≈ 0.30
- **Puzzle accuracy:** ~5% (random chance on move selection)
- **Estimated ELO:** < 500

### MLP baseline (2-layer, 256 hidden)
- **Parameters:** ~200K
- **Puzzle accuracy:** ~15-20%
- **Estimated ELO:** ~800-1000

These baselines establish the floor — any meaningful model should substantially exceed them.

---

## 9. Key Insights Summary

1. **Label quality is critical.** The score sign-flip bug showed that label corruption can silently prevent learning. Always validate preprocessing.

2. **The target distribution is favorable.** Centered, symmetric, bounded — well-suited for regression with MSE or Huber loss.

3. **Position diversity is high.** The model sees everything from openings to endgames, requiring broad generalization.

4. **Puzzle ELO is a strong proxy for playing strength.** It correlates well with actual chess rating and provides a meaningful benchmark.

5. **The encoding is minimal but sufficient.** 12 binary planes capture piece placement; no castling, en passant, or move history. This follows the reference architecture's design choice.

---

## 10. Experiment Results Summary

| Model            | Params | Epochs | GPU-hours | Puzzle Acc | ELO  |
|------------------|--------|--------|-----------|------------|------|
| MLP baseline     | 200K   | 5      | <1h       | ~15%       | ~900 |
| BDH v4 (fixed)   | 2.5M   | 43     | ~48h      | 31.0%      | 1177 |
| ViT-S v2 (fixed) | 2.6M   | 30     | ~17h      | 45.1%      | 1621 |
| ViT-S+AttnRes    | 2.66M  | 27     | ~45h      | 53.0%      | 1810 |
| **Reference ViT-S** | **2.6M** | **—** | **>200h** | **N/A** | **1817** |

The ViT-S+AttnRes model achieves **1810 ELO**, only 7 points below the reference implementation, demonstrating that Attention Residuals provide meaningful improvement over standard residual connections.
