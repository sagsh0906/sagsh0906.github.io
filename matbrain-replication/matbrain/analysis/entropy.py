"""Token-level Shannon entropy diagnostics (paper Fig. 4, Supplementary Note 1).

H_t = - sum_v p_t(v) log2 p_t(v)   [bits], from the model's *raw* next-token
distribution (temperature 1, before sampling warpers).

* trajectories for single matched inputs (Fig. 4a-e): raw entropy + 15-point
  adjacent averaging (Origin "Adjacent-Averaging") + mean line;
* segment-level entropy for <think> / <tool_call> / <answer> spans (Fig. 4e);
* kernel density estimates over 200 held-out prompts (Fig. 4f): Gaussian KDE,
  evaluation range extended by 500% of the bandwidth at both ends, each curve
  scaled to the maximum count of its own histogram.

    python -m matbrain.analysis.entropy generate --model Qwen/Qwen3-14B --prompts prompts.jsonl --out results/entropy/t1_base.jsonl
    python -m matbrain.analysis.entropy plot --inputs results/entropy/*.jsonl --out-dir results/entropy/figs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Sequence

import numpy as np

SEGMENT_RE = {
    "think": re.compile(r"<think>.*?</think>", re.S),
    "tool_call": re.compile(r"<tool_call>.*?</tool_call>", re.S),
    "answer": re.compile(r"<answer>.*?</answer>", re.S),
}


# --------------------------------------------------------------------------- #
# Entropy computation
# --------------------------------------------------------------------------- #
def entropy_from_logits(logits):
    """Shannon entropy in bits for each row of a [..., vocab] torch tensor."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1) / math.log(2)


def entropy_from_top_logprobs(top_logprobs: Sequence[dict[str, float]]) -> list[float]:
    """Approximate entropy from top-k logprobs (e.g. vLLM ``top_logprobs=20``).

    The residual probability mass is treated as one extra outcome, giving a
    lower bound of the true entropy. Prefer ``entropy_from_logits``.
    """
    out = []
    for d in top_logprobs:
        p = np.exp(np.array(list(d.values()), dtype=float))
        rest = max(0.0, 1.0 - p.sum())
        probs = np.append(p, rest) if rest > 1e-12 else p
        probs = probs[probs > 0]
        out.append(float(-(probs * np.log2(probs)).sum()))
    return out


def generate_with_entropy(model, tokenizer, messages: list[dict], max_new_tokens: int = 4096, tools: list[dict] | None = None, do_sample: bool = False, **gen_kwargs) -> dict[str, Any]:
    """Generate a response and record the entropy of every generated token."""
    import torch

    prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample, output_logits=True, return_dict_in_generate=True, **gen_kwargs)
    new_ids = out.sequences[0, enc["input_ids"].shape[1] :].tolist()
    ent = [float(entropy_from_logits(l[0])) for l in out.logits][: len(new_ids)]
    return {"token_ids": new_ids, "tokens": tokenizer.convert_ids_to_tokens(new_ids), "entropy": ent, "text": tokenizer.decode(new_ids, skip_special_tokens=False)}


def teacher_forced_entropy(model, tokenizer, messages: list[dict], response: str, tools: list[dict] | None = None) -> dict[str, Any]:
    """Entropy along a fixed response (identical text for every model = matched inputs)."""
    import torch

    prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p_ids + r_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    # logits at position i predict token i+1
    ent = entropy_from_logits(logits[len(p_ids) - 1 : len(p_ids) - 1 + len(r_ids)]).tolist()
    return {"token_ids": r_ids, "tokens": tokenizer.convert_ids_to_tokens(r_ids), "entropy": ent, "text": response}


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def adjacent_average(x: Sequence[float], window: int = 15) -> np.ndarray:
    """Centered moving average; near the edges the available points are averaged."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    half = window // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    lo = np.clip(idx - half, 0, len(x))
    hi = np.clip(idx + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def token_char_offsets(token_texts: Sequence[str]) -> list[int]:
    offs, pos = [], 0
    for t in token_texts:
        offs.append(pos)
        pos += len(t)
    return offs


def segment_entropy(token_texts: Sequence[str], entropy: Sequence[float]) -> dict[str, dict[str, float]]:
    """Mean entropy per segment type; ``token_texts`` are decoded token strings."""
    text = "".join(token_texts)
    offs = token_char_offsets(token_texts)
    labels = ["other"] * len(offs)
    for name, rx in SEGMENT_RE.items():
        for m in rx.finditer(text):
            for i, o in enumerate(offs):
                if m.start() <= o < m.end():
                    labels[i] = name
    out = {}
    for name in list(SEGMENT_RE) + ["other", "all"]:
        vals = [e for e, l in zip(entropy, labels) if name == "all" or l == name]
        if vals:
            out[name] = {"mean": float(np.mean(vals)), "n_tokens": len(vals)}
    return out


def kde_curve(values: Sequence[float], bins: int = 40, extend: float = 5.0, grid: int = 512) -> dict[str, np.ndarray]:
    """Gaussian KDE scaled to the histogram's max count; range extended by `extend` x bandwidth."""
    from scipy.stats import gaussian_kde

    v = np.asarray(values, dtype=float)
    kde = gaussian_kde(v)
    bw = float(np.sqrt(kde.covariance[0, 0]))  # bandwidth in data units
    xs = np.linspace(v.min() - extend * bw, v.max() + extend * bw, grid)
    ys = kde(xs)
    counts, edges = np.histogram(v, bins=bins)
    ys = ys / ys.max() * counts.max()
    return {"x": xs, "y": ys, "hist_counts": counts, "hist_edges": edges, "bandwidth": bw}


def kde_peaks(curve: dict[str, np.ndarray], min_rel_height: float = 0.05) -> list[float]:
    y, x = curve["y"], curve["x"]
    peaks = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > min_rel_height * y.max()]
    return [float(x[i]) for i in peaks]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_trajectory(entropy: Sequence[float], title: str, out_path: str, window: int = 15, second: Sequence[float] | None = None, labels=("entropy", "second")) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(entropy))
    ax.plot(x, entropy, lw=0.5, alpha=0.35, color="#4C72B0", label=f"Origin {labels[0]}")
    ax.plot(x, adjacent_average(entropy, window), lw=1.4, color="#4C72B0", label=f"{window} pts AAv smooth")
    if second is not None:
        x2 = np.arange(len(entropy), len(entropy) + len(second))
        ax.plot(x2, second, lw=0.5, alpha=0.35, color="#DD8452", label=f"Origin {labels[1]}")
        ax.plot(x2, adjacent_average(second, window), lw=1.4, color="#DD8452", label=f"{window} pts AAv smooth ({labels[1]})")
    else:
        ax.axhline(float(np.mean(entropy)), ls="--", color="k", lw=0.8, label=f"Average ({np.mean(entropy):.3f})")
    ax.set_xlabel("Token")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_kde(series: dict[str, Sequence[float]], out_path: str) -> dict[str, list[float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    peaks = {}
    for name, vals in series.items():
        c = kde_curve(vals)
        ax.fill_between(c["x"], c["y"], alpha=0.25)
        ax.plot(c["x"], c["y"], lw=1.2, label=name)
        peaks[name] = kde_peaks(c)
    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return peaks


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate responses and record token entropies with a HF model")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True, help="jsonl with 'messages' (and optional 'tools')")
    g.add_argument("--out", required=True)
    g.add_argument("--max-new-tokens", type=int, default=4096)
    g.add_argument("--limit", type=int, default=200)
    p = sub.add_parser("plot")
    p.add_argument("--inputs", nargs="+", required=True, help="jsonl files from `generate` (one per model)")
    p.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.cmd == "generate":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from matbrain.data.formats import iter_records

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            for i, rec in enumerate(iter_records(args.prompts)):
                if i >= args.limit:
                    break
                res = generate_with_entropy(model, tok, rec["messages"], args.max_new_tokens, tools=rec.get("tools"))
                res["segments"] = segment_entropy([tok.convert_tokens_to_string([t]) for t in res["tokens"]], res["entropy"])
                fh.write(json.dumps({"id": rec.get("id", i), **res}) + "\n")
    else:
        from matbrain.data.formats import iter_records

        os.makedirs(args.out_dir, exist_ok=True)
        series, summary = {}, {}
        for path in args.inputs:
            name = os.path.splitext(os.path.basename(path))[0]
            rows = list(iter_records(path))
            plot_trajectory(rows[0]["entropy"], name, os.path.join(args.out_dir, f"trajectory_{name}.png"))
            series[name] = [e for r in rows for e in r["entropy"]]
            summary[name] = {"mean_entropy": float(np.mean(series[name])), "n_tokens": len(series[name]), "n_samples": len(rows)}
        summary["kde_peaks"] = plot_kde(series, os.path.join(args.out_dir, "entropy_kde.png"))
        with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
