"""Generative-design tools: CrystaLLM (text-to-CIF) and MatterGen (diffusion).

Both wrap the upstream command-line tools, which run inside their own GPU
containers in the paper's Kubernetes deployment. Configure with:

* ``CRYSTALLM_HOME``      checkout of https://github.com/lantunes/CrystaLLM
* ``CRYSTALLM_MODEL_DIR`` model directory (e.g. crystallm_v1_small)
* ``MATTERGEN_CMD``       mattergen-generate executable (default: on PATH)
* ``MATBRAIN_GPUS``       comma-separated GPU ids used for multi-worker generation
"""

from __future__ import annotations

import glob
import io
import os
import shlex
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import Field
from pymatgen.core import Composition

from matbrain import chem
from matbrain.mcp.artifacts import get_store
from matbrain.mcp.registry import GENERATION, REGISTRY, ChemSys, Formula, ToolArgs, ToolError, ToolOutput


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise ToolError(f"command failed ({proc.returncode}): {shlex.join(cmd[:4])} ...", stderr=proc.stderr[-4000:])
    return proc


def _store_cifs(cif_texts: list[str], source: str, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out = []
    for text in cif_texts:
        row: dict[str, Any] = {}
        try:
            s = chem.parse_structure(text)
            row["handle"] = get_store().put_structure(s, {"source": source, **(extra or {})})
            row["formula"] = s.composition.reduced_formula
            row["num_sites"] = len(s)
            row.update({k: v for k, v in chem.structural_validity(s).items() if k in ("valid", "min_distance")})
            try:
                row["space_group"] = chem.symmetry_info(s)["space_group_symbol"]
            except Exception:
                row["space_group"] = None
        except chem.StructureParseError as exc:
            row = {"handle": None, "parse_error": str(exc)[:200]}
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
class CrystaLLMArgs(ToolArgs):
    formula: Formula = Field(..., description="Cell composition with integer counts, e.g. Na2Cl2 (Z included)")
    spacegroup: str | None = Field(None, description="Optional Hermann-Mauguin symbol, e.g. Fm-3m")
    num_samples: int = Field(3, ge=1, le=64)
    top_k: int = Field(10, ge=1, le=100)
    max_new_tokens: int = Field(3000, ge=256, le=8000)


@REGISTRY.tool(category=GENERATION, timeout=3600)
def crystallm_generate(args: CrystaLLMArgs) -> ToolOutput:
    """Generate candidate crystal structures (CIF) for a composition with CrystaLLM,
    optionally conditioned on a space group. Returns artifact handles."""
    comp = Composition(args.formula)
    if any(abs(v - round(v)) > 1e-6 for v in comp.values()):
        raise ToolError("Non-stoichiometric compositions are not supported by CrystaLLM; use an integer approximant")
    home = os.environ.get("CRYSTALLM_HOME")
    model_dir = os.environ.get("CRYSTALLM_MODEL_DIR", "crystallm_v1_small")
    if not home:
        raise ToolError("CRYSTALLM_HOME is not configured in the Mat-MCP environment")
    device = os.environ.get("MATBRAIN_DEVICE", "cuda")
    formula = comp.formula.replace(" ", "")
    with tempfile.TemporaryDirectory() as tmp:
        prompt = os.path.join(tmp, "prompt.txt")
        cmd = [sys.executable, os.path.join(home, "bin", "make_prompt_file.py"), formula, prompt]
        if args.spacegroup:
            cmd += ["--spacegroup", args.spacegroup]
        _run(cmd, cwd=tmp)
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        sample = _run(
            [
                sys.executable, os.path.join(home, "bin", "sample.py"),
                f"out_dir={os.path.join(home, model_dir) if not os.path.isabs(model_dir) else model_dir}",
                f"start=FILE:{prompt}", f"num_samples={args.num_samples}", f"top_k={args.top_k}",
                f"max_new_tokens={args.max_new_tokens}", f"device={device}", "target=file",
            ],
            cwd=raw,
        )
        processed = os.path.join(tmp, "processed")
        _run([sys.executable, os.path.join(home, "bin", "postprocess.py"), raw, processed], cwd=tmp)
        texts = [open(p, encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(processed, "*.cif")))]
    results = _store_cifs(texts, "crystallm", {"prompt_formula": formula, "spacegroup": args.spacegroup})
    return ToolOutput(
        data={"n_generated": len(results), "n_parseable": sum(r.get("handle") is not None for r in results), "results": results},
        stdout=sample.stdout[-2000:],
        stderr=sample.stderr[-2000:],
    )


# --------------------------------------------------------------------------- #
def _mattergen_job(chemsys: str, n: int, batch_size: int, guidance: float, out_dir: str, gpu: str | None) -> tuple[list[str], str]:
    exe = os.environ.get("MATTERGEN_CMD", "mattergen-generate")
    num_batches = max(1, -(-n // batch_size))
    cmd = [
        exe, out_dir, "--pretrained-name=chemical_system", f"--batch_size={batch_size}", f"--num_batches={num_batches}",
        f"--properties_to_condition_on={{'chemical_system': '{chemsys}'}}", "--record_trajectories=False",
        f"--diffusion_guidance_factor={guidance}",
    ]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = _run(cmd, env=env, timeout=24 * 3600)
    zpath = os.path.join(out_dir, "generated_crystals_cif.zip")
    if not os.path.exists(zpath):
        raise ToolError("MatterGen finished without generated_crystals_cif.zip", stderr=proc.stderr[-2000:])
    with zipfile.ZipFile(zpath) as zf:
        texts = [io.TextIOWrapper(zf.open(n), encoding="utf-8").read() for n in zf.namelist() if n.endswith(".cif")]
    return texts[:n], proc.stderr[-1000:]


class MatterGenArgs(ToolArgs):
    chemical_system: ChemSys = Field(..., description="e.g. Co-V-S")
    num_samples: int = Field(64, ge=1, le=50000)
    batch_size: int = Field(64, ge=1, le=1024)
    guidance_factor: float = Field(2.0, ge=0, le=10)
    num_workers: int = Field(1, ge=1, le=64, description="Parallel generation workers (one GPU each)")
    output_dir: str | None = None


@REGISTRY.tool(category=GENERATION, timeout=48 * 3600)
def mattergen_generate(args: MatterGenArgs) -> ToolOutput:
    """Generate crystal structures in a chemical system with MatterGen (chemical_system
    conditioned diffusion). With num_workers > 1 generation is split across GPUs in
    parallel (the paper's multithreaded generation of 30,000 M-V-S structures)."""
    gpus = [g for g in os.environ.get("MATBRAIN_GPUS", "").split(",") if g] or [None]
    workers = max(1, min(args.num_workers, args.num_samples))
    share = [args.num_samples // workers + (i < args.num_samples % workers) for i in range(workers)]
    base = args.output_dir or tempfile.mkdtemp(prefix="mattergen_")
    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, n in enumerate(share):
            out = os.path.join(base, f"worker_{i}")
            os.makedirs(out, exist_ok=True)
            jobs.append(pool.submit(_mattergen_job, args.chemical_system, n, min(args.batch_size, n), args.guidance_factor, out, gpus[i % len(gpus)]))
        texts, logs = [], []
        for job in jobs:
            t, log = job.result()
            texts.extend(t)
            logs.append(log)
    results = _store_cifs(texts, "mattergen", {"chemical_system": args.chemical_system})
    handles = [r["handle"] for r in results if r.get("handle")]
    summary = {
        "n_generated": len(results),
        "n_parseable": len(handles),
        "output_dir": base,
        # Only a preview goes into the observation; the full list is on disk / in the store.
        "handles_preview": handles[:20],
    }
    manifest = os.path.join(base, "handles.txt")
    with open(manifest, "w") as fh:
        fh.write("\n".join(handles))
    summary["handles_file"] = manifest
    return ToolOutput(data=summary, stderr="\n".join(logs)[-2000:])
