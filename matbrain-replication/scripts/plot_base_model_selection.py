"""Fig. 2b-style plot: initial LM loss vs parameter count, annotated with GPQA / HLE.

    python scripts/plot_base_model_selection.py --loss results/base_model_loss.csv \
        --scores configs/base_model_scores.csv --out results/base_model_selection.png

``--scores`` is a CSV with columns model,gpqa,hle taken from the Artificial
Analysis leaderboard at the time of your run.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True)
    ap.add_argument("--scores")
    ap.add_argument("--out", default="results/base_model_selection.png")
    args = ap.parse_args()
    df = pd.read_csv(args.loss)
    if args.scores:
        df = df.merge(pd.read_csv(args.scores), on="model", how="left")
    fig, ax = plt.subplots(figsize=(6, 4))
    size = 40 + 6 * df["gpqa"].fillna(0) if "gpqa" in df else 60
    sc = ax.scatter(df["params_b"], df["initial_lm_loss"], s=size, c=df["hle"] if "hle" in df else None, cmap="viridis")
    for _, r in df.iterrows():
        ax.annotate(r["model"].split("/")[-1], (r["params_b"], r["initial_lm_loss"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("Initial LM loss (lower is better)")
    if "hle" in df:
        fig.colorbar(sc, label="HLE (%)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
