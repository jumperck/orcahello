"""Bias-sampling logic: hard + uniform sampling of detections."""

import pandas as pd


def _sample_label(pool: pd.DataFrame, hard_pool: pd.DataFrame, target: int,
                   hard_label: str, uniform_label: str, seed: int) -> pd.DataFrame:
    """Sample up to `target` examples, split 1:1 between hard and uniform with backfill."""
    target_hard = target // 2
    target_uniform = target - target_hard

    # Take hard examples (already sorted by priority)
    hard = hard_pool.head(target_hard).assign(example_type=hard_label)

    # Uniform from remainder
    remaining = pool[~pool.index.isin(hard.index)]
    n_uniform = min(target_uniform, len(remaining))
    uniform = remaining.sample(n=n_uniform, random_state=seed).assign(example_type=uniform_label)

    # Backfill: if hard pool was too small, fill from uniform pool
    total = len(hard) + len(uniform)
    if total < target:
        backfill_pool = pool[~pool.index.isin(hard.index) & ~pool.index.isin(uniform.index)]
        n_backfill = min(target - total, len(backfill_pool))
        if n_backfill > 0:
            backfill = backfill_pool.sample(n=n_backfill, random_state=seed).assign(example_type=uniform_label)
            uniform = pd.concat([uniform, backfill])

    return pd.concat([hard, uniform])


def bias_sample(
    df: pd.DataFrame,
    target_positives: int,
    negative_ratio: float,
    hard_pos_max_conf: float,
    hard_pos_min_conf: float,
    hard_neg_min_conf: float,
    seed: int,
    location: str = "",
) -> pd.DataFrame:
    """Bias-sample hard + uniform examples from detections with inference results."""
    prefix = f"  [{location}] " if location else "  "
    has_inf = df.dropna(subset=["global_confidence"]).copy()
    n_without = len(df) - len(has_inf)

    positives = has_inf[has_inf["binary_label"] == 1]
    negatives = has_inf[has_inf["binary_label"] == 0]

    print(f"{prefix}{len(df)} detections ({len(positives)} pos, {len(negatives)} neg"
          f"{f', {n_without} no inference' if n_without else ''})")

    if len(positives) < target_positives:
        print(f"{prefix}Warning: {len(positives)} positives available, below target of {target_positives}")

    # --- Positive sampling (cap at target_positives) ---
    hard_pos_pool = positives[
        (positives["global_confidence"] < hard_pos_max_conf)
        & (positives["global_confidence"] > hard_pos_min_conf)
    ].sort_values("global_confidence")

    target_pos = min(target_positives, len(positives))
    sampled_pos = _sample_label(positives, hard_pos_pool, target_pos,
                                "positive_hard", "positive_uniform", seed)

    # --- Negative sampling (cap at ratio * actual positives sampled) ---
    hard_neg_pool = negatives[
        negatives["global_confidence"] > hard_neg_min_conf
    ].sort_values("global_confidence", ascending=False)

    # If positives are scarce (< half target), use target-based negative count
    # so we still get good negative coverage for evaluation
    if len(sampled_pos) < target_positives // 2:
        target_neg = min(int(target_positives * negative_ratio), len(negatives))
    else:
        target_neg = min(int(len(sampled_pos) * negative_ratio), len(negatives))
    sampled_neg = _sample_label(negatives, hard_neg_pool, target_neg,
                                "negative_hard", "negative_uniform", seed)

    return pd.concat([sampled_pos, sampled_neg])


def print_summary(sampled_df: pd.DataFrame):
    """Print summary statistics."""
    print(f"\nSampled dataset: {len(sampled_df)} rows")
    ct = sampled_df.groupby(["binary_label", "example_type"]).size().reset_index(name="count")
    for _, row in ct.iterrows():
        print(f"  label={row['binary_label']} type={row['example_type']}: {row['count']}")
