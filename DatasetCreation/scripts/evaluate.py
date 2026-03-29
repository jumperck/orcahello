#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "scikit-learn",
#   "matplotlib",
# ]
# ///
"""
Evaluate model predictions against ground truth labels.

Uses ground truth from a CSV with file_path, binary_label (1=positive, 0=false_positive)
and predictions from a summary CSV with file_path, global_confidence. Writes
results.txt and roc_curve.png to the output directory (default: same folder as summary).

Column indexes (0-based) can override the default column names:
    --label-col N   column index for ground truth binary label (default: auto-detect)
    --score-col N   column index for prediction confidence score (default: auto-detect)
    --id-col N      column index for row identifier in FN/FP listings (default: auto-detect)

Usage:
    uv run scripts/evaluate.py --ground-truth path/to/ground_truth_labels.csv --predictions path/to/summary.csv
    uv run scripts/evaluate.py --predictions 2026-01--v1.2/summary.csv   # ground truth and output use defaults
    # Single file with column indexes:
    uv run scripts/evaluate.py --ground-truth data.csv --predictions data.csv --label-col 5 --score-col 8 --id-col 4
    python scripts/evaluate.py ...   # with pandas, scikit-learn, matplotlib installed
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def resolve_col(df: pd.DataFrame, index: int | None, default_name: str) -> str:
    """Return column name from integer index or fall back to default_name."""
    if index is not None:
        if index < 0 or index >= len(df.columns):
            print(f"Error: column index {index} out of range (0..{len(df.columns)-1})", file=sys.stderr)
            sys.exit(1)
        return df.columns[index]
    if default_name in df.columns:
        return default_name
    return None


def load_and_merge(
    ground_truth_path: str,
    predictions_path: str,
    label_col: int | None = None,
    score_col: int | None = None,
    id_col: int | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Load and merge gt/predictions CSVs. Returns (df, label_col_name, score_col_name, id_col_name)."""
    gt = pd.read_csv(ground_truth_path)
    same_file = Path(ground_truth_path).resolve() == Path(predictions_path).resolve()

    if same_file:
        merged = gt
    else:
        preds = pd.read_csv(predictions_path)

        # Find a common merge key between gt and preds
        merge_key = None
        for candidate in ("detection_id", "file_path", "id"):
            if candidate in gt.columns and candidate in preds.columns:
                merge_key = candidate
                break

        if merge_key is None:
            # Derive detection_id from file_path stem in whichever df has file_path
            if "file_path" in preds.columns and "detection_id" in gt.columns:
                preds["detection_id"] = preds["file_path"].apply(lambda p: Path(p).stem)
                merge_key = "detection_id"
            elif "file_path" in gt.columns and "detection_id" in preds.columns:
                gt["detection_id"] = gt["file_path"].apply(lambda p: Path(p).stem)
                merge_key = "detection_id"
            else:
                print(
                    f"Error: no common merge key found.\n"
                    f"  GT columns: {list(gt.columns)}\n"
                    f"  Pred columns: {list(preds.columns)}",
                    file=sys.stderr,
                )
                sys.exit(1)

        merged = gt.merge(preds, on=merge_key, how="inner", suffixes=("_gt", "_pred"))
        n_gt, n_pred, n_merged = len(gt), len(preds), len(merged)
        print(f"Merged on {merge_key!r}: {n_merged} rows ({n_gt} gt, {n_pred} pred)")
        if n_merged < max(n_gt, n_pred):
            print(
                f"Warning: {n_gt} ground-truth rows, {n_pred} prediction rows, "
                f"but only {n_merged} matched by {merge_key}.",
                file=sys.stderr,
            )

    label_name = (
        resolve_col(merged, label_col, "binary_label")
        or resolve_col(merged, label_col, "binary_label_gt")
        or resolve_col(merged, label_col, "label_binary")
    )
    score_name = (
        resolve_col(merged, score_col, "global_confidence")
        or resolve_col(merged, score_col, "global_confidence_pred")
    )
    id_name = (
        resolve_col(merged, id_col, "file_path")
        or resolve_col(merged, id_col, "file_path_pred")
        or resolve_col(merged, id_col, "detection_id")
    )

    if not label_name or label_name not in merged.columns:
        print(f"Error: label column not found. Use --label-col to specify. Columns: {list(merged.columns)}", file=sys.stderr)
        sys.exit(1)
    if not score_name or score_name not in merged.columns:
        print(f"Error: score column not found. Use --score-col to specify. Columns: {list(merged.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Using columns: label={label_name!r}, score={score_name!r}, id={id_name!r}")
    return merged, label_name, score_name, id_name


def predictions_summary_table(predictions_path: str) -> str:
    """Build summary table (folder, pred=1, total, pct) from predictions CSV."""
    preds = pd.read_csv(predictions_path)
    preds["folder"] = preds["file_path"].str.split("/").str[0]
    grp = preds.groupby("folder", sort=True).agg(
        pred_1=("global_prediction", lambda x: (x == 1).sum()),
        total=("global_prediction", "count"),
    ).reset_index()
    grp["pct"] = (100 * grp["pred_1"] / grp["total"]).round(1).astype(str) + "%"
    t_pred, t_total = grp["pred_1"].sum(), grp["total"].sum()
    pct_total = f"{100 * t_pred / t_total:.1f}%" if t_total else "N/A"
    total_row = pd.DataFrame({
        "folder": ["TOTAL"],
        "pred_1": [t_pred],
        "total": [t_total],
        "pct": [pct_total],
    })
    table_df = pd.concat([grp, total_row], ignore_index=True)
    return "\n".join([
        "",
        "Predictions by folder (global_prediction=1 vs total):",
        "",
        table_df.to_string(index=False),
    ])


def fpr_at_recall(
    fpr_arr, tpr_arr, thresholds, target_recall: float
) -> tuple[float, float, float, bool]:
    """Return (FPR, actual_recall, threshold, achieved) at the lowest-FPR point where TPR >= target_recall."""
    n_real = len(thresholds)
    for i in range(n_real):
        if tpr_arr[i] >= target_recall:
            return float(fpr_arr[i]), float(tpr_arr[i]), float(thresholds[i]), True
    max_idx = len(tpr_arr) - 2
    thresh = thresholds[max_idx] if max_idx < len(thresholds) else 0.0
    return float(fpr_arr[max_idx]), float(tpr_arr[max_idx]), float(thresh), False


def print_false_negatives(
    op_index: int,
    op_results: list,
    df: pd.DataFrame,
    y_true,
    y_score,
    id_col: str,
    score_col: str,
    top_n: int = 5,
) -> list[str]:
    """Generate text lines showing false negatives at the specified operating point."""
    op_name, op_target, op_fpr, op_rec, op_thr, op_achieved = op_results[op_index]
    fn_mask = (y_true == 1) & (y_score < op_thr)
    fn_df = df.loc[fn_mask, [id_col, score_col]].copy()
    total_pos = int(y_true.sum())

    lines = []
    lines.append(f"\nFalse negatives at {op_name} threshold ({op_thr:.2f}):")
    lines.append(f"  {len(fn_df)} of {total_pos} true positives missed")

    sorted_fn = fn_df.sort_values(score_col)
    for i, (_, row) in enumerate(sorted_fn.iterrows()):
        if i >= top_n:
            lines.append(f"  ... and {len(fn_df) - top_n} more")
            break
        lines.append(f"  {row[id_col]}  (score={row[score_col]:.2f})")

    return lines


def print_false_positives(
    op_index: int,
    op_results: list,
    df: pd.DataFrame,
    y_true,
    y_score,
    id_col: str,
    score_col: str,
    top_n: int = 5,
) -> list[str]:
    """Generate text lines showing hardest false positives at the specified operating point."""
    op_name, op_target, op_fpr, op_rec, op_thr, op_achieved = op_results[op_index]
    fp_mask = (y_true == 0) & (y_score >= op_thr)
    fp_df = df.loc[fp_mask, [id_col, score_col]].copy()
    total_neg = int((y_true == 0).sum())

    lines = []
    lines.append(f"\nHardest false positives at {op_name} threshold ({op_thr:.2f}):")
    lines.append(f"  {len(fp_df)} of {total_neg} true negatives incorrectly flagged")

    sorted_fp = fp_df.sort_values(score_col, ascending=False)
    for i, (_, row) in enumerate(sorted_fp.iterrows()):
        if i >= top_n:
            lines.append(f"  ... and {len(fp_df) - top_n} more")
            break
        lines.append(f"  {row[id_col]}  (score={row[score_col]:.2f})")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inference predictions against ground truth (e.g. OrcaHello labels)."
    )
    parser.add_argument(
        "--ground-truth", "--gt",
        default="../detection_downloads/2026-01/ground_truth_labels.csv",
        help="Path to ground truth CSV with file_path, binary_label, label",
    )
    parser.add_argument(
        "--predictions", "--pred",
        default="2026-01--v1.2/summary.csv",
        help="Path to predictions summary CSV with file_path, global_confidence (default: 2026-01--v1.2/summary.csv)",
    )
    parser.add_argument(
        "--output-dir", "--o",
        default=None,
        help="Directory for results.txt and roc_curve.png (default: same folder as predictions file)",
    )
    parser.add_argument(
        "--label-col", type=int, default=None,
        help="0-based column index for ground truth binary label (default: auto-detect 'binary_label|label_binary')",
    )
    parser.add_argument(
        "--score-col", type=int, default=None,
        help="0-based column index for prediction confidence score (default: auto-detect 'global_confidence')",
    )
    parser.add_argument(
        "--id-col", type=int, default=None,
        help="0-based column index for row identifier in FN/FP listings (default: auto-detect 'file_path' or 'detection_id')",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df, label_name, score_name, id_name = load_and_merge(
        args.ground_truth, args.predictions,
        label_col=args.label_col, score_col=args.score_col, id_col=args.id_col,
    )

    # Drop rows with missing label or score
    df[label_name] = pd.to_numeric(df[label_name], errors="coerce")
    df[score_name] = pd.to_numeric(df[score_name], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[label_name, score_name]).copy()
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with missing label or score ({len(df)} remaining)")

    y_true = df[label_name].astype(int).values
    y_score = df[score_name].astype(float).values

    auroc = roc_auc_score(y_true, y_score)
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)

    max_recall = float(tpr_arr[-2])
    max_recall_fpr = float(fpr_arr[-2])
    max_recall_thr = float(thresholds[-2])

    operating_points = [
        ("OP1", 0.95),
        ("OP2", 0.90),
    ]
    op_results = [
        (name, target, *fpr_at_recall(fpr_arr, tpr_arr, thresholds, target))
        for name, target in operating_points
    ]

    # --- Build results text ---
    lines = []
    # Only show predictions-by-folder table when the predictions file has the expected columns
    preds_check = pd.read_csv(args.predictions, nrows=0)
    if "file_path" in preds_check.columns and "global_prediction" in preds_check.columns:
        lines.append(predictions_summary_table(args.predictions))
        lines.append("")
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    lines.append(f"Rows:                 {len(df)} ({n_pos} positive, {n_neg} negative)")
    lines.append(f"AUROC:                {auroc:.4f}")
    lines.append(
        f"Max recall:           {max_recall:.4f}  (FPR={max_recall_fpr:.4f}, threshold={max_recall_thr:.2f})"
    )
    for name, target, fpr, rec, thr, achieved in op_results:
        if achieved and fpr < 1.0:
            lines.append(
                f"{name} (recall≥{target}):   FPR={fpr:.4f}, recall={rec:.4f}, threshold={thr:.2f}"
            )
        else:
            lines.append(
                f"{name} (recall≥{target}):   not achievable without FPR=1 (max recall={max_recall:.4f})"
            )

    # --- False negatives at chosen operating point ---
    lines.extend(print_false_negatives(1, op_results, df, y_true, y_score, id_name, score_name, top_n=5))

    # --- False positives at chosen operating point ---
    lines.extend(print_false_positives(1, op_results, df, y_true, y_score, id_name, score_name, top_n=5))

    results_text = "\n".join(lines)
    print(results_text)

    results_path = out_dir / "results.txt"
    results_path.write_text(results_text + "\n")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"ROC (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]
    for (name, target, fpr, rec, thr, achieved), color in zip(op_results, colors):
        if achieved and fpr < 1.0:
            ax.scatter(
                [fpr],
                [rec],
                zorder=5,
                color=color,
                label=f"{name}: recall={rec:.3f}, FPR={fpr:.3f}, thr={thr:.1f}",
            )
    ax.scatter(
        [max_recall_fpr],
        [max_recall],
        zorder=5,
        color="tab:red",
        marker="x",
        s=80,
        label=f"Max recall={max_recall:.3f}, FPR={max_recall_fpr:.3f}, thr={max_recall_thr:.1f}",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Inference vs ground truth")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()

    plot_path = out_dir / "roc_curve.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nOutput written to: {out_dir}/")
    print(f"  {plot_path.name}")
    print(f"  {results_path.name}")


if __name__ == "__main__":
    main()
