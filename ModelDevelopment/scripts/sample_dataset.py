"""Bias-sample from a recording-level HF Dataset to produce a sampled subset."""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from src.sampling import bias_sample, print_summary


def _dataset_to_sampling_df(dataset) -> pd.DataFrame:
    """Extract columns needed for bias_sample from an HF Dataset."""
    rows = []
    for i, ex in enumerate(dataset):
        tags = ex.get("tags", [])
        tag = tags[0]["tag"] if tags else "srkw_negative"
        score = tags[0]["score"] if tags else 0.0
        meta = ex.get("metadata", {})
        rows.append({
            "_index": i,
            "location_slug": meta.get("location_slug", ""),
            "binary_label": 1 if tag == "srkw_positive" else 0,
            "global_confidence": score,
            "meta_orcahello_confidence": score,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Bias-sample hard + uniform examples from a recording-level HF Dataset"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Path to dataset directory (must contain recording_dataset/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for sampled dataset (default: sampled_dataset/ in dataset-dir)")
    parser.add_argument("--target-positives", type=int, default=50,
                        help="Target number of positive examples (cap)")
    parser.add_argument("--negative-ratio", type=float, default=2.0,
                        help="Negative:positive ratio")
    parser.add_argument("--hard-pos-max-conf", type=float, default=0.7,
                        help="Max confidence for hard positives")
    parser.add_argument("--hard-pos-min-conf", type=float, default=0.1,
                        help="Min confidence for hard positives (exclude bad audio)")
    parser.add_argument("--hard-neg-min-conf", type=float, default=0.5,
                        help="Min confidence for hard negatives")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    recording_path = dataset_dir / "recording_dataset"

    if not recording_path.exists():
        print(f"Error: no recording_dataset/ found in {dataset_dir}")
        raise SystemExit(1)

    print(f"Loading recording dataset from {recording_path}")
    dataset = load_from_disk(str(recording_path))
    print(f"  {len(dataset)} recordings loaded")

    # Convert to DataFrame for sampling logic
    df = _dataset_to_sampling_df(dataset)

    # Bias-sample per location
    print("Sampling...")
    sampled_parts = []
    for loc, loc_df in df.groupby("location_slug"):
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

    sampled_df = pd.concat(sampled_parts)
    print_summary(sampled_df)

    # Filter HF Dataset to sampled indices
    sampled_indices = sorted(sampled_df["_index"].tolist())
    sampled_dataset = dataset.select(sampled_indices)

    output_path = args.output or (dataset_dir / "sampled_dataset")
    sampled_dataset.save_to_disk(str(output_path))
    print(f"\nSaved sampled dataset: {output_path} ({len(sampled_dataset)} recordings)")


if __name__ == "__main__":
    main()
