"""Project-local Llama 3.1 8B GGUF scientific mutation hypothesis generator."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .agents import Hypothesis, parse_hypothesis
from .config import AMINO_ACIDS, POSITIONS, ROOT, WT
from .knowledge import AA_GROUP

MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_PATH = ROOT / "models" / "llm" / MODEL_FILENAME
MODEL_ID = "Meta-Llama-3.1-8B-Instruct-Q4_K_M"
EXPECTED_SIZE = 4_920_739_232
EXPECTED_SHA256 = "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
MODEL_URL = (
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/"
    + MODEL_FILENAME
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ground_hypothesis_reasons(summary: dict, hypotheses: list[Hypothesis]) -> tuple[list[Hypothesis], list[dict]]:
    """Replace unverified numerical prose with observed-only evidence.

    Does not change the proposed positions or residues. Association is not
    causal evidence, especially for four-site epistatic GB1 variants.
    """
    by_key = {
        (int(a["position"]), str(a["amino_acid"])): a
        for a in summary.get("site_associations", [])
    }
    grounded, issues = [], []
    for h in hypotheses:
        evidence = by_key.get((h.position, h.amino_acid))
        claimed = re.search(r"\bobserved\s+in\s+(\d+)\s+variants\b", h.reason, re.I)
        if claimed and evidence is not None and int(claimed.group(1)) != int(evidence["count"]):
            issues.append({"position": h.position, "amino_acid": h.amino_acid,
                           "issue": "observed_count_mismatch", "claimed": int(claimed.group(1)),
                           "observed": int(evidence["count"]), "original_reason": h.reason})
        if evidence is None:
            reason = "No matching association in the supplied observed-data summary; unverified hypothesis"
            issues.append({"position": h.position, "amino_acid": h.amino_acid,
                           "issue": "no_observed_summary_support", "original_reason": h.reason})
        else:
            reason = (
                f"Observed count={int(evidence['count'])}; top fraction="
                f"{float(evidence['top_fraction']):.3f}; marginal association only, epistasis unverified"
            )
        grounded.append(Hypothesis(h.position, h.amino_acid, reason, h.source))
    return grounded, issues


@lru_cache(maxsize=1)
def _load(path: str, n_ctx: int, threads: int, seed: int, gpu_layers: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("Install llama-cpp-python for local GGUF inference") from exc
    return Llama(
        model_path=path, n_ctx=n_ctx, n_batch=256, n_threads=threads,
        n_gpu_layers=gpu_layers, seed=seed, use_mmap=True,
        verbose=os.getenv("PROTEINEVO_LOCAL_VERBOSE", "0").lower() in {"1", "true", "yes"},
    )


class LocalGGUFHypothesisGenerator:
    """Structured scientific hypotheses; never receives hidden fitness labels."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, seed: int = 20261027,
                 knowledge_enhanced: bool = False, exact_hypotheses: int | None = None,
                 trace_directory: Path | None = None):
        self.model_path = Path(model_path).resolve()
        self.seed = seed
        self.knowledge_enhanced = knowledge_enhanced
        self.model = MODEL_ID
        self.backend = "local_gguf"
        self.n_ctx = int(os.getenv("PROTEINEVO_LOCAL_CTX", "4096"))
        self.threads = int(os.getenv("PROTEINEVO_LOCAL_THREADS", "8"))
        self.gpu_layers = int(os.getenv("PROTEINEVO_LOCAL_GPU_LAYERS", "0"))
        self.max_hypotheses = int(os.getenv("PROTEINEVO_LOCAL_MAX_HYPOTHESES", "4"))
        self.max_tokens = int(os.getenv("PROTEINEVO_LOCAL_MAX_TOKENS", "384"))
        self.n_top = int(os.getenv("PROTEINEVO_LOCAL_TOP_VARIANTS", "8"))
        self.n_sites = int(os.getenv("PROTEINEVO_LOCAL_SITE_ASSOCIATIONS", "12"))
        self.ground_reasons = os.getenv("PROTEINEVO_LOCAL_GROUND_REASONS", "1").strip().lower() not in ("0", "false", "off")
        self.exact_hypotheses = exact_hypotheses
        self.trace_directory = Path(trace_directory) if trace_directory is not None else None
        if not (1 <= self.max_hypotheses <= 12 and 128 <= self.max_tokens <= 1600):
            raise ValueError("Unsupported local LLM hypothesis/token budget")
        if exact_hypotheses is not None and (type(exact_hypotheses) is not int or not 1 <= exact_hypotheses <= self.max_hypotheses):
            raise ValueError("Exact hypothesis count must fit the configured budget")
        self.last_trace: dict = {}

    def _save_trace(self):
        if self.trace_directory is not None:
            self.trace_directory.mkdir(parents=True, exist_ok=True)
            (self.trace_directory / (self.last_trace["call_id"] + ".json")).write_text(
                json.dumps(self.last_trace, indent=2, ensure_ascii=False), encoding="utf-8")

    def check_model(self) -> dict:
        if not self.model_path.is_file():
            raise RuntimeError(f"Missing GGUF: {self.model_path}; run scripts/download_local_llm.py")
        size = self.model_path.stat().st_size
        if self.model_path.name == MODEL_FILENAME and size != EXPECTED_SIZE:
            raise RuntimeError(f"Incomplete GGUF: {size}/{EXPECTED_SIZE} bytes")
        with self.model_path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise RuntimeError("Invalid GGUF header")
        return {
            "backend": self.backend,
            "model": self.model,
            "model_path": str(self.model_path),
            "size_bytes": size,
            "expected_sha256": EXPECTED_SHA256,
            "n_ctx": self.n_ctx,
            "threads": self.threads,
            "seed": self.seed,
            "gpu_layers": self.gpu_layers,
            "max_hypotheses": self.max_hypotheses,
            "max_tokens": self.max_tokens,
            "max_top_variants": self.n_top,
            "max_site_associations": self.n_sites,
            "exact_hypotheses": self.exact_hypotheses,
            "ground_reasons_in_observed_evidence": self.ground_reasons,
        }

    def propose(self, summary: dict, n: int = 12) -> list[Hypothesis]:
        self.check_model()
        concise = {
            "n_observed": summary["n_observed"],
            "best_seen_fitness": summary["best_seen_fitness"],
            "top_variants": summary["top_variants"][:self.n_top],
            "site_associations": summary["site_associations"][:self.n_sites],
            "caution": summary["caution"],
        }
        system = (
            "You are a scientific Agent proposing mutations for GB1 IgG-Fc binding. "
            "The ONLY mutable protein sites are 39,40,41,54. Wild type is VDGV. "
            "Use ONLY the provided measured observations and correlations; NEVER "
            "invent experimental fitness values or imply that marginal enrichment "
            "proves causality. Interactions between mutation sites (epistasis) matter. "
            "Output ONLY JSON with this exact shape: "
            '{"hypotheses":[{"position":39,"amino_acid":"A",'
            '"reason":"Supported by observed fitness associations"}]}. '
            f"Provide 1 to {min(n, self.max_hypotheses)} DISTINCT hypotheses. "
            "Every object needs position (integer), amino_acid (one uppercase "
            "standard residue), reason (one brief evidence-grounded sentence). "
            "Never suggest unchanged WT V39,D40,G41,V54. Valid amino acids: "
            "ACDEFGHIKLMNPQRSTVWY. Prefer several sites over repeated alleles. "
            "All reasons must come from the measured statistics."
        )
        if self.exact_hypotheses is not None:
            system += f" For this budget-matched experiment provide EXACTLY {self.exact_hypotheses} distinct valid hypotheses."
        if self.knowledge_enhanced:
            system += (
                " Physicochemical guidance: changes in charge or sidechain bulk "
                "may destabilize binding; glycine/proline can alter backbone geometry. "
                "Conservative substitutions may be safer but do not override data. "
                "Do not assume beneficial single mutations combine additively. "
                f"AA classes: {json.dumps(AA_GROUP, sort_keys=True)}."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(concise, sort_keys=True)},
        ]
        start = time.perf_counter()
        self.last_trace = {
            "call_id": uuid.uuid4().hex,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "started", "new_llm_call_performed": True,
            "backend": self.backend, "model": self.model,
            "knowledge_enhanced": self.knowledge_enhanced,
            "prompt": messages, "raw_response": None,
            "hypotheses": [], "rejected": [], "fact_check_issues": [],
            "model_metadata": self.check_model(),
            "ground_reasons_in_observed_evidence": self.ground_reasons,
            "uses_oracle_labels": False,
            "observed_summary_sha256": hashlib.sha256(json.dumps(concise, sort_keys=True).encode()).hexdigest(),
        }
        self._save_trace()
        try:
            client = _load(str(self.model_path), self.n_ctx, self.threads, self.seed, self.gpu_layers)
            response_format = {"type": "json_object"}
            if self.exact_hypotheses is not None:
                response_format["schema"] = {
                    "type": "object", "properties": {"hypotheses": {
                        "type": "array", "minItems": self.exact_hypotheses, "maxItems": self.exact_hypotheses,
                        "items": {"type": "object", "properties": {
                            "position": {"type": "integer", "enum": list(POSITIONS)},
                            "amino_acid": {"type": "string", "enum": list(AMINO_ACIDS)},
                            "reason": {"type": "string", "minLength": 1}},
                            "required": ["position", "amino_acid", "reason"], "additionalProperties": False}}},
                    "required": ["hypotheses"], "additionalProperties": False,
                }
            response = client.create_chat_completion(
                messages=messages, temperature=0, max_tokens=self.max_tokens,
                response_format=response_format,
            )
            raw = response["choices"][0]["message"]["content"]
            self.last_trace.update({"raw_response": raw, "usage": response.get("usage", {}),
                                    "finish_reason": response["choices"][0].get("finish_reason")})
            self._save_trace()
            parsed = json.loads(raw)
            entries = parsed["hypotheses"]
            if not isinstance(entries, list):
                raise TypeError("Expected hypotheses list")
        except Exception as exc:
            self.last_trace.update({"status": "failed", "error": str(exc),
                                    "llm_wall_seconds": round(time.perf_counter() - start, 3)})
            self._save_trace()
            raise RuntimeError(
                f"Local LLM failed to produce usable hypothesis JSON: "
                f"{str(locals().get('raw', ''))[:300]}"
            ) from exc

        accepted: list[Hypothesis] = []
        rejected: list[dict] = []
        seen: set[tuple[int, str]] = set()
        for item in entries:
            try:
                hypothesis = parse_hypothesis(item, "local_gguf:" + self.model)
                key = (hypothesis.position, hypothesis.amino_acid)
                if key in seen:
                    raise ValueError("Duplicate mutation hypothesis")
                seen.add(key)
                accepted.append(hypothesis)
            except (ValueError, TypeError, KeyError):
                rejected.append(item)
            if len(accepted) >= min(n, self.max_hypotheses):
                break
        raw_hypotheses = [asdict(h) for h in accepted]
        evidence_issues: list[dict] = []
        if self.ground_reasons:
            # Scientific claims are checked against the same observed statistics
            # sent to the model; no hidden/held-out fitness values are consulted.
            accepted, evidence_issues = ground_hypothesis_reasons(concise, accepted)
        self.last_trace.update({
            "backend": self.backend,
            "model": self.model,
            "knowledge_enhanced": self.knowledge_enhanced,
            "prompt": messages,
            "raw_response": raw,
            "hypotheses": [asdict(h) for h in accepted],
            "raw_hypotheses_before_grounding": raw_hypotheses,
            "fact_check_issues": evidence_issues,
            "ground_reasons_in_observed_evidence": self.ground_reasons,
            "rejected": rejected,
            "model_metadata": self.check_model(),
            "llm_wall_seconds": round(time.perf_counter() - start, 3),
            "usage": response.get("usage", {}),
            "uses_oracle_labels": False,
        })
        if not accepted or (self.exact_hypotheses is not None and len(accepted) != self.exact_hypotheses):
            self.last_trace.update({"status": "failed", "error": "No valid hypotheses or exact count mismatch"})
            self._save_trace()
            raise RuntimeError(f"Local LLM hypothesis validation/count failed; {raw[:400]}")
        self.last_trace["status"] = "complete"
        self._save_trace()
        return accepted
