"""Bias-sample from a complete CSV to produce a sampled dataset."""

import argparse
from pathlib import Path

import pandas as pd

from src.models import SampledDetectionRecord, format_df
from src.sampling import bias_sample, print_summary


def main():
    parser = argparse.ArgumentParser(description="Bias-sample hard + uniform examples from a complete CSV")
    parser.add_argument("input_csv", type=Path, help="Path to a *--complete.csv from create_dataset.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: replace '--complete.csv' with '--sampled.csv')")
    parser.add_argument("--target-positives", type=int, default=50, help="Target number of positive examples (cap)")
    parser.add_argument("--negative-ratio", type=float, default=2.0, help="Negative:positive ratio")
    parser.add_argument("--hard-pos-max-conf", type=float, default=0.7,
                        help="Max confidence for hard positives")
    parser.add_argument("--hard-pos-min-conf", type=float, default=0.1,
                        help="Min confidence for hard positives (exclude bad audio)")
    parser.add_argument("--hard-neg-min-conf", type=float, default=0.5,
                        help="Min confidence for hard negatives")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    csv_path = args.input_csv.resolve()
    print(f"Loading complete dataset from {csv_path}")
    det_df = pd.read_csv(csv_path, dtype=str)

    # Restore numeric types needed for sampling
    det_df["binary_label"] = pd.to_numeric(det_df["binary_label"])
    det_df["global_confidence"] = pd.to_numeric(det_df["global_confidence"], errors="coerce")
    # Alias so sampling fallback works
    det_df["meta_orcahello_confidence"] = det_df["global_confidence"]
    print(f"  {len(det_df)} detections loaded")

    # Bias-sample per location
    print("Sampling...")
    sampled_parts = []
    for loc, loc_df in det_df.groupby("location_slug"):
        loc_sampled = bias_sample(
            loc_df,
            target_positives=args.target_positives,
            negative_ratio=args.negative_ratio,
            hard_pos_max_conf=args.hard_pos_max_conf,
            hard_pos_min_conf=args.hard_pos_min_conf,
            hard_neg_min_conf=args.hard_neg_min_conf,
            seed=args.seed,
            location=loc,
        )
        sampled_parts.append(loc_sampled)
    sampled = pd.concat(sampled_parts)

    # Format and write
    output = format_df(sampled, SampledDetectionRecord)
    if args.output:
        sampled_path = args.output
    else:
        sampled_path = csv_path.with_name(csv_path.name.replace("--complete.csv", "--sampled.csv"))
    output.to_csv(sampled_path, index=False)
    print(f"\nWrote {len(output)} rows to {sampled_path}")

    print_summary(sampled)


if __name__ == "__main__":
    main()
