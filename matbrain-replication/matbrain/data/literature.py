"""Literature preprocessing for the Mat-SFT corpus.

1. PDF -> Markdown with MinerU v1.2.0 (``magic-pdf`` CLI).
2. Markdown -> chunks on structural delimiters (headings / paragraphs).
3. Chunk classification into Structure / Properties / Synthesis / Applications
   by three LLMs (DeepSeek-V3, GPT-4o, Claude-3.5 in the paper) with majority vote.
4. Extraction of synthesis-pathway and application passages with LangExtract v1.0.3.

    python -m matbrain.data.literature convert --pdf-dir papers/ --out-dir data/markdown
    python -m matbrain.data.literature classify --md-dir data/markdown --config configs/data_llms.yaml --out data/chunks.jsonl
    python -m matbrain.data.literature extract --chunks data/chunks.jsonl --out data/extractions.jsonl --model-id gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from matbrain.agent.llm import ChatBackend, backend_from_config
from matbrain.data.formats import iter_records, write_jsonl
from matbrain.mcp.tools.search import chunk_markdown

CATEGORIES = ("Structure", "Properties", "Synthesis", "Applications")
CLASSIFY_PROMPT = (
    "You are annotating text chunks from materials-science papers. Classify the chunk into exactly one of: "
    "Structure (crystal structure, lattice, symmetry, characterisation of structure), Properties (measured or "
    "computed physical/chemical properties), Synthesis (preparation methods, precursors, conditions), "
    "Applications (devices, performance in applications). Reply with the category name only.\n\nChunk:\n{chunk}"
)


# --------------------------------------------------------------------------- #
def convert_pdfs(pdf_dir: str, out_dir: str, method: str = "auto") -> list[str]:
    """Run MinerU (v1.x CLI ``magic-pdf``) on every PDF; corrupted PDFs are skipped."""
    done = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            with open(pdf, "rb") as fh:
                if fh.read(5) != b"%PDF-":
                    raise ValueError("not a PDF")
            subprocess.run(["magic-pdf", "-p", str(pdf), "-o", out_dir, "-m", method], check=True, capture_output=True, timeout=1800)
            done.append(pdf.stem)
        except Exception as exc:  # integrity check failed / MinerU error -> excluded
            print(f"skip {pdf.name}: {exc}")
    return done


def iter_chunks(md_dir: str, max_chars: int = 2000):
    for md in sorted(Path(md_dir).rglob("*.md")):
        doc_id = md.stem
        for i, chunk in enumerate(chunk_markdown(md.read_text(encoding="utf-8", errors="ignore"), max_chars)):
            if len(chunk) > 200:  # skip captions / fragments
                yield {"doc_id": doc_id, "chunk_id": f"{doc_id}#{i}", "text": chunk}


# --------------------------------------------------------------------------- #
def parse_category(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for cat in CATEGORIES:
        if re.search(rf"\b{cat}\b", text, re.I):
            return cat
    if re.search(r"\bpropert", text, re.I):
        return "Properties"
    return None


def majority_vote(labels: list[str | None], tie_break: str = "drop") -> str | None:
    votes = Counter(label for label in labels if label)
    if not votes:
        return None
    (top, n), *rest = votes.most_common()
    if n >= 2 or (not rest and n == 1 and len(labels) == 1):
        return top
    if tie_break == "first":
        return next((label for label in labels if label), None)
    return None


async def classify_chunk(chunk: dict, backends: list[ChatBackend], tie_break: str = "drop") -> dict:
    prompt = CLASSIFY_PROMPT.format(chunk=chunk["text"][:6000])
    replies = await asyncio.gather(*(b.chat([{"role": "user", "content": prompt}], temperature=0.0) for b in backends), return_exceptions=True)
    labels = [parse_category(r.content) if not isinstance(r, Exception) else None for r in replies]
    return {**chunk, "votes": dict(zip([b.name for b in backends], labels)), "category": majority_vote(labels, tie_break)}


async def classify_all(chunks: list[dict], backends: list[ChatBackend], concurrency: int = 16, tie_break: str = "drop") -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(c):
        async with sem:
            return await classify_chunk(c, backends, tie_break)

    return await asyncio.gather(*(one(c) for c in chunks))


# --------------------------------------------------------------------------- #
SYNTHESIS_PROMPT = (
    "Extract synthesis procedures and application scenarios of inorganic crystalline materials. "
    "Use exact text spans. For synthesis steps record target material, precursors, method, temperature, "
    "time and atmosphere as attributes when present."
)


def langextract_passages(text: str, model_id: str, **kwargs) -> list[dict[str, Any]]:
    import langextract as lx

    examples = [
        lx.data.ExampleData(
            text="LiFePO4 was synthesized by a solid-state reaction of Li2CO3, FeC2O4 and NH4H2PO4 at 700 C for 10 h under Ar. The cathode delivered 160 mAh/g.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="synthesis",
                    extraction_text="synthesized by a solid-state reaction of Li2CO3, FeC2O4 and NH4H2PO4 at 700 C for 10 h under Ar",
                    attributes={"target": "LiFePO4", "method": "solid-state", "precursors": "Li2CO3, FeC2O4, NH4H2PO4", "temperature": "700 C", "time": "10 h", "atmosphere": "Ar"},
                ),
                lx.data.Extraction(extraction_class="application", extraction_text="The cathode delivered 160 mAh/g", attributes={"application": "Li-ion battery cathode"}),
            ],
        )
    ]
    result = lx.extract(text_or_documents=text, prompt_description=SYNTHESIS_PROMPT, examples=examples, model_id=model_id, **kwargs)
    return [{"class": e.extraction_class, "text": e.extraction_text, "attributes": e.attributes or {}} for e in result.extractions]


# --------------------------------------------------------------------------- #
def load_backends(config_path: str, key: str) -> list[ChatBackend]:
    cfg = yaml.safe_load(open(config_path))
    return [backend_from_config(c) for c in cfg[key]]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--pdf-dir", required=True)
    c.add_argument("--out-dir", required=True)
    k = sub.add_parser("classify")
    k.add_argument("--md-dir", required=True)
    k.add_argument("--config", default="configs/data_llms.yaml")
    k.add_argument("--out", required=True)
    k.add_argument("--concurrency", type=int, default=16)
    k.add_argument("--tie-break", choices=["drop", "first"], default="drop")
    e = sub.add_parser("extract")
    e.add_argument("--chunks", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--model-id", default=os.environ.get("LANGEXTRACT_MODEL", "gpt-4o"))
    args = ap.parse_args()

    if args.cmd == "convert":
        print(f"converted {len(convert_pdfs(args.pdf_dir, args.out_dir))} PDFs")
    elif args.cmd == "classify":
        chunks = list(iter_chunks(args.md_dir))
        rows = asyncio.run(classify_all(chunks, load_backends(args.config, "classifiers"), args.concurrency, args.tie_break))
        write_jsonl(args.out, rows)
        print(Counter(r["category"] for r in rows))
    else:
        out = []
        for ch in iter_records(args.chunks):
            if ch.get("category") in ("Synthesis", "Applications"):
                try:
                    out.append({**ch, "extractions": langextract_passages(ch["text"], args.model_id)})
                except Exception as exc:
                    print(f"{ch['chunk_id']}: {exc}")
        write_jsonl(args.out, out)
        print(json.dumps({"extracted_chunks": len(out)}))


if __name__ == "__main__":
    main()
