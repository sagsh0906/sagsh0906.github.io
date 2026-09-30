"""Base-model selection probe (paper Fig. 2b).

Computes the *initial* language-modelling loss of candidate base models on the
same slice of the training corpus under a unified configuration: seed 42,
packing to 10,240-token sequences, one global batch of 16 packed sequences,
loss over assistant tokens only (as in SFT). Samples are packed without
cross-sample attention (each sample is scored independently and token losses
are pooled over the global batch), which is what packed SFT with per-sample
attention masks computes.

    python training/sft/base_model_probe.py --data data/sft/mat_r1_train.jsonl \
        --models Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-30B-A3B --out results/base_model_loss.csv

Then combine with GPQA / HLE scores (Artificial Analysis leaderboard) using
scripts/plot_base_model_selection.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from matbrain.data.formats import iter_records, to_messages  # noqa: E402


def tokenize_with_labels(tokenizer, messages):
    """Token ids + labels masking everything except assistant turns."""
    ids, labels = [], []
    for i in range(len(messages)):
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False) if i else ""
        full = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        piece = tokenizer(full[len(prefix):], add_special_tokens=False)["input_ids"]
        ids += piece
        labels += piece if messages[i]["role"] == "assistant" else [-100] * len(piece)
    return ids, labels


def select_global_batch(tokenizer, records, seq_len: int, batch: int, seed: int):
    """Greedy packing (after a seeded shuffle) until `batch` sequences of `seq_len` are filled."""
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    bins: list[list[tuple[list[int], list[int]]]] = [[]]
    fill = 0
    for rec in records:
        msgs = to_messages(rec)
        if not msgs:
            continue
        ids, labels = tokenize_with_labels(tokenizer, msgs)
        ids, labels = ids[:seq_len], labels[:seq_len]
        if fill + len(ids) > seq_len:
            if len(bins) == batch:
                break
            bins.append([])
            fill = 0
        bins[-1].append((ids, labels))
        fill += len(ids)
    return [s for b in bins for s in b]


def probe(model_name: str, samples_by_tok, args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    samples = select_global_batch(tok, samples_by_tok, args.seq_len, args.global_batch, args.seed)
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for ids, labels in samples:
            x = torch.tensor([ids], device=model.device)
            y = torch.tensor([labels], device=model.device)
            n = int((y[:, 1:] != -100).sum())
            if n == 0:
                continue
            out = model(input_ids=x, labels=y)
            total_loss += float(out.loss) * n
            total_tok += n
    n_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {"model": model_name, "params_b": round(n_params / 1e9, 2), "initial_lm_loss": total_loss / max(total_tok, 1), "scored_tokens": total_tok, "n_samples": len(samples)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=10240)
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-records", type=int, default=20000)
    ap.add_argument("--out", default="results/base_model_loss.csv")
    args = ap.parse_args()

    records = []
    for i, rec in enumerate(iter_records(args.data)):
        if i >= args.max_records:
            break
        records.append(rec)
    rows = []
    for name in args.models:
        row = probe(name, records, args)
        print(json.dumps(row))
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["initial_lm_loss"])
    print(f"lowest initial loss: {best['model']} ({best['initial_lm_loss']:.4f}); perplexity {math.exp(best['initial_lm_loss']):.2f}")


if __name__ == "__main__":
    main()
