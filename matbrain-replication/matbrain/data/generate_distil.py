"""Generate-distil construction of Mat-252K-SFT (paper Methods).

Phase 1  context-aware question generation   (DeepSeek-V3 on corpus chunks)
Phase 2  reasoning-driven knowledge distillation (DeepSeek-R1 answers with source text)
Phase 3  automated self-evaluation            (DeepSeek-R1 scores 0-5 for factuality and
                                                consistency; keep score > 4 by default)
(The expert random-sampling review of phase 3 is a manual step.)

MP-grounded structure-generation / property-prediction samples are produced
from training-split records (never validation/test) with templated questions
whose reference answers are the database values; R1 writes the reasoning
trace towards the known answer and samples whose final answer disagrees with
the database value are discarded.

    python -m matbrain.data.generate_distil literature --chunks data/chunks.jsonl --out data/sft_raw/lit.jsonl
    python -m matbrain.data.generate_distil mp --records data/splits/train.jsonl --out data/sft_raw/mp.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from typing import Any

import yaml

from matbrain.agent.llm import ChatBackend, backend_from_config
from matbrain.agent.parsing import extract_answer
from matbrain.data.formats import iter_records

QUESTION_PROMPT = (
    "You are a materials-science professor writing exam questions. Based ONLY on the passage below "
    "(category: {category}), write {n} distinct, self-contained questions that test understanding of key "
    "material concepts, structural characteristics, properties or synthesis protocols described in it. "
    "Each question must be answerable from the passage plus standard domain knowledge, and must name the "
    "material(s) explicitly. Return a JSON list of strings only.\n\nPassage:\n{context}"
)
ANSWER_PROMPT = (
    "Answer the question as an expert materials scientist. Ground your answer in the reference passage, "
    "reason step by step, and state the final answer clearly.\n\nReference passage:\n{context}\n\nQuestion: {question}"
)
JUDGE_PROMPT = (
    "Evaluate the instruction-response pair below for factual accuracy and consistency with the context. "
    "Score from 0 (wrong / unsupported) to 5 (fully correct, consistent and well reasoned). "
    "Reply with the integer score inside <score></score>.\n\nContext:\n{context}\n\nInstruction:\n{question}\n\nResponse:\n{answer}"
)
MP_RATIONALE_PROMPT = (
    "{question}\n\n(For the annotator only: the database reference value is {reference}. Write a rigorous "
    "expert reasoning that analyses the structure and arrives at this value, without mentioning that a "
    "reference was provided. End with <answer>{answer_format}</answer>.)"
)


def parse_json_list(text: str) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in items if isinstance(q, str) and len(q.strip()) > 15]


def parse_score(text: str) -> float | None:
    m = re.search(r"<score>\s*([0-5](?:\.\d+)?)\s*</score>", text or "")
    if not m:
        m = re.search(r"\b([0-5](?:\.\d+)?)\s*/\s*5\b", text or "")
    return float(m.group(1)) if m else None


def passes(score: float | None, threshold: float = 4.0, strict: bool = True) -> bool:
    """Paper: samples 'exceeding a threshold of 4' are retained."""
    if score is None:
        return False
    return score > threshold if strict else score >= threshold


def _assistant(reply) -> str:
    return f"<think>{reply.reasoning}</think>\n{reply.content}" if reply.reasoning else reply.content


async def distil_chunk(chunk: dict, v3: ChatBackend, r1: ChatBackend, judge: ChatBackend, n_questions: int, threshold: float, strict: bool) -> list[dict]:
    q_reply = await v3.chat([{"role": "user", "content": QUESTION_PROMPT.format(category=chunk.get("category", "general"), n=n_questions, context=chunk["text"])}], temperature=0.7)
    out = []
    for q in parse_json_list(q_reply.content)[:n_questions]:
        a = await r1.chat([{"role": "user", "content": ANSWER_PROMPT.format(context=chunk["text"], question=q)}])
        answer = _assistant(a)
        j = await judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(context=chunk["text"], question=q, answer=a.content)}])
        score = parse_score(j.content)
        if passes(score, threshold, strict):
            out.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}], "score": score, "source": "literature", "chunk_id": chunk.get("chunk_id"), "category": chunk.get("category")})
    return out


# --------------------------------------------------------------------------- #
# MP-grounded samples
# --------------------------------------------------------------------------- #
PROPERTY_QUESTIONS = {
    "formation_energy_per_atom": ("What is the formation energy per atom (eV/atom) of the material with the following crystal structure?", "{:.3f}"),
    "energy_above_hull": ("Estimate the energy above the convex hull (eV/atom) of the following crystal structure and comment on its thermodynamic stability.", "{:.3f}"),
    "band_gap": ("Estimate the DFT-PBE band gap (eV) of the following crystal structure. Is it a metal, semiconductor or insulator?", "{:.3f}"),
    "efermi": ("Estimate the Fermi energy (eV) of the following crystal structure.", "{:.3f}"),
    "is_metal": ("Is the material with the following crystal structure metallic?", "{}"),
    "is_magnetic": ("Is the material with the following crystal structure magnetic?", "{}"),
    "ordering": ("What is the collinear magnetic ordering (NM, FM, FiM or AFM) of the following crystal structure?", "{}"),
}
GEN_QUESTION = (
    "Generate the crystal structure of {formula} in space group {sg} (No. {num}, {system} crystal system) "
    "with {n} atoms in the unit cell. Output a complete CIF."
)


def mp_questions(rec: dict) -> list[dict[str, Any]]:
    items = []
    for prop, (q, fmt) in PROPERTY_QUESTIONS.items():
        v = rec.get(prop)
        if v is None or (prop == "ordering" and str(v).lower() == "unknown"):
            continue
        items.append({"task": f"property:{prop}", "question": f"{q}\n\n{rec['cif'].strip()}", "reference": fmt.format(v) if not isinstance(v, bool) else str(v).lower(), "answer_format": '{"value": ...}'})
    if rec.get("spacegroup_symbol"):
        items.append({
            "task": "structure_generation",
            "question": GEN_QUESTION.format(formula=rec["formula"], sg=rec["spacegroup_symbol"], num=rec.get("spacegroup_number"), system=rec.get("crystal_system"), n=rec.get("nsites")),
            "reference": rec["cif"].strip(),
            "answer_format": "CIF",
        })
    return items


def consistent(task: str, answer: str | None, reference: str, rel_tol: float = 0.15, abs_tol: float = 0.1) -> bool:
    if answer is None:
        return False
    if task == "structure_generation":
        from matbrain.chem import StructureParseError, parse_structure, reduced_formula

        try:
            return reduced_formula(parse_structure(answer)) == reduced_formula(parse_structure(reference))
        except StructureParseError:
            return False
    m = re.search(r'"value"\s*:\s*"?([^",}]+)', answer)
    val = (m.group(1) if m else answer).strip().strip('"').lower()
    try:
        a, b = float(val), float(reference)
        return abs(a - b) <= max(abs_tol, rel_tol * abs(b))
    except ValueError:
        return val == reference.lower()


async def distil_mp(rec: dict, r1: ChatBackend | None, judge: ChatBackend | None, threshold: float, strict: bool) -> list[dict]:
    out = []
    for item in mp_questions(rec):
        if r1 is None:  # template-only mode: reference answer without reasoning trace
            answer = f"<answer>{item['reference']}</answer>" if item["task"] == "structure_generation" else f'<answer>{{"value": {json.dumps(item["reference"])}}}</answer>'
            out.append({"messages": [{"role": "user", "content": item["question"]}, {"role": "assistant", "content": answer}], "source": "mp_template", "task": item["task"], "material_id": rec["material_id"]})
            continue
        prompt = MP_RATIONALE_PROMPT.format(question=item["question"], reference=item["reference"][:4000], answer_format=item["answer_format"])
        a = await r1.chat([{"role": "user", "content": prompt}])
        if not consistent(item["task"], extract_answer(a.content), item["reference"]):
            continue
        score = None
        if judge is not None:
            j = await judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(context=f"reference value: {item['reference'][:2000]}", question=item["question"], answer=a.content)}])
            score = parse_score(j.content)
            if not passes(score, threshold, strict):
                continue
        out.append({"messages": [{"role": "user", "content": item["question"]}, {"role": "assistant", "content": _assistant(a)}], "score": score, "source": "mp_distil", "task": item["task"], "material_id": rec["material_id"]})
    return out


# --------------------------------------------------------------------------- #
async def run(items: list[dict], worker, out_path: str, concurrency: int) -> int:
    done = set()
    if os.path.exists(out_path):
        for r in iter_records(out_path):
            done.add(r.get("chunk_id") or r.get("material_id"))
    todo = [it for it in items if (it.get("chunk_id") or it.get("material_id")) not in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:

        async def one(it):
            nonlocal written
            async with sem:
                try:
                    rows = await worker(it)
                except Exception as exc:
                    print(f"error on {it.get('chunk_id') or it.get('material_id')}: {exc}")
                    return
            async with lock:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += 1
                fh.flush()

        await asyncio.gather(*(one(it) for it in todo))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["literature", "mp"])
    ap.add_argument("--chunks")
    ap.add_argument("--records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/data_llms.yaml")
    ap.add_argument("--questions-per-chunk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--non-strict", action="store_true", help="keep score >= threshold instead of > threshold")
    ap.add_argument("--template-only", action="store_true", help="MP mode without teacher LLM calls")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    strict = not args.non_strict

    cfg = yaml.safe_load(open(args.config)) if not args.template_only else {}
    teachers = {k: backend_from_config(v) for k, v in cfg.get("teachers", {}).items()}
    if args.mode == "literature":
        chunks = [c for c in iter_records(args.chunks) if c.get("category")]
        random.Random(args.seed).shuffle(chunks)
        chunks = chunks[: args.limit] if args.limit else chunks
        n = asyncio.run(run(chunks, lambda c: distil_chunk(c, teachers["question"], teachers["answer"], teachers["judge"], args.questions_per_chunk, args.threshold, strict), args.out, args.concurrency))
    else:
        recs = list(iter_records(args.records))
        recs = recs[: args.limit] if args.limit else recs
        r1 = None if args.template_only else teachers["answer"]
        judge = None if args.template_only else teachers.get("judge")
        n = asyncio.run(run(recs, lambda r: distil_mp(r, r1, judge, args.threshold, strict), args.out, args.concurrency))
    print(f"wrote {n} instruction pairs to {args.out}")


if __name__ == "__main__":
    main()
