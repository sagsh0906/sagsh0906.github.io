"""Build the Mat-R1 SFT mixture (ms-swift ``messages`` jsonl).

    python training/sft/prepare_sft_data.py \
        --mat-sft data/Mat-252K-SFT \
        --mot-science data/Mixture-of-Thoughts/science \
        --out-dir data/sft

Mixture (paper): Mat-252K-SFT (~252k) + open-r1/Mixture-of-Thoughts science
subset (173k reasoning traces) + self-identity data, with the customised
Mat-R1 system prompt. Download the sources beforehand, e.g.
    huggingface-cli download open-r1/Mixture-of-Thoughts --repo-type dataset --include "science/*"
    huggingface-cli download <Mat-252K-SFT repo, doi:10.57967/hf/9652> --repo-type dataset
Records whose MP ids belong to the validation/test split can be removed with
--exclude-ids (one id per line) to keep the benchmark leakage-free.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from matbrain.data.formats import iter_records, to_messages, with_system, write_jsonl  # noqa: E402
from matbrain.data.identity import identity_samples  # noqa: E402
from matbrain.prompts import MAT_R1_SYSTEM  # noqa: E402


def load(path: str, source: str, system: str | None, limit: int | None, exclude: set[str]) -> list[dict]:
    rows = []
    for rec in iter_records(path):
        mp_id = rec.get("material_id") or rec.get("mp_id")
        if mp_id and mp_id in exclude:
            continue
        msgs = to_messages(rec)
        if not msgs or msgs[-1]["role"] != "assistant" or not msgs[-1]["content"]:
            continue
        rows.append({"messages": with_system(msgs, system), "source": source})
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-sft", required=True, help="Mat-252K-SFT file or directory")
    ap.add_argument("--mot-science", help="Mixture-of-Thoughts science subset (parquet dir)")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--identity-samples", type=int, default=500)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None, help="cap per source (debugging / small-scale runs)")
    ap.add_argument("--exclude-ids", help="file with MP ids to drop (validation/test split)")
    ap.add_argument("--no-system-for-mot", action="store_true", help="keep MoT traces without the Mat-R1 system prompt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exclude = set(open(args.exclude_ids).read().split()) if args.exclude_ids else set()
    rows = load(args.mat_sft, "mat_252k_sft", MAT_R1_SYSTEM, args.limit, exclude)
    print(f"Mat-252K-SFT: {len(rows)}")
    if args.mot_science:
        mot = load(args.mot_science, "mot_science", None if args.no_system_for_mot else MAT_R1_SYSTEM, args.limit, set())
        print(f"Mixture-of-Thoughts/science: {len(mot)}")
        rows += mot
    ident = [{"messages": with_system(r["messages"], MAT_R1_SYSTEM), "source": r["source"]} for r in identity_samples(args.identity_samples, args.seed)]
    rows += ident
    print(f"self-identity: {len(ident)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    write_jsonl(os.path.join(args.out_dir, "mat_r1_val.jsonl"), rows[:n_val])
    write_jsonl(os.path.join(args.out_dir, "mat_r1_train.jsonl"), rows[n_val:])
    print(f"train={len(rows) - n_val} val={n_val} -> {args.out_dir}")


if __name__ == "__main__":
    main()
