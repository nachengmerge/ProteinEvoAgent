"""Reproduce the task workflow on supplied known GB1 observations only.

This is an engineering demonstration, separate from the frozen efficacy studies.
No project validation, test or unknown-fitness table is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from proteinevo.config import SPLITS, full_sequence, mutation_count, mutation_names
from proteinevo.delivery import workflow as w
from proteinevo.delivery.cli import doctor


def run(observations_path, output, include_llm=False, rounds=3):
    output = w.fresh_directory(output)
    state_file = output / "execution_state.json"

    def state(value):
        state_file.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    state({"status": "running", "scope": "small_known_observations_engineering_reproduction"})
    try:
        source = Path(observations_path)
        known = w.observations(pd.read_csv(source))
        if len(known) < 160 or len(known) > 2048:
            raise ValueError("Small reproduction requires 160–2048 supplied known observations")
        # Convert the actual supplied records, not a synthetic mutation/fitness table.
        prepared = known.assign(full_sequence=known.variant.map(full_sequence),
                                mutation_count=known.variant.map(mutation_count),
                                mutations=known.variant.map(lambda v: ";".join(mutation_names(v))))
        (output / "data").mkdir()
        prepared.to_csv(output / "data/observations.csv", index=False)
        source_info = {"source_filename": source.name,
                       "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "observations": len(known), "unique_variants": known.variant.nunique(),
                       "sequence_length": 56, "valid_residue_alphabet": "ACDEFGHIKLMNPQRSTVWY",
                       "mutation_count_distribution": known.variant.map(mutation_count).value_counts().sort_index().to_dict(),
                       "zero_fitness_records_retained": int((known.fitness == 0).sum()),
                       "project_heldout_or_unknown_fitness_read": False}
        (output / "data/processing.json").write_text(json.dumps(source_info, indent=2), encoding="utf-8")

        llm_check = None
        methods = ("random", "full_greedy", "tool_rule")
        if include_llm:
            llm_check = doctor(verify_model=True)
            if not llm_check.get("llama_model_verified"):
                raise RuntimeError("Llama weights are missing or invalid; no rule fallback is substituted")
            if not llm_check["packages"].get("llama-cpp-python"):
                raise RuntimeError("Install a working llama-cpp-python backend before --include-llm")
            methods = ("random", "full_greedy", "tool_llm", "knowledge_tool_llm")

        metrics = w.train(known, output / "predictor", predictor="ridge")
        rows = w.demo(known, output / "evolution", methods=methods, rounds=rounds)
        decisions = [json.loads(p.read_text(encoding="utf-8"))
                     for p in (output / "evolution").glob("*/round_*/decision.json")]
        calls = sum(r["real_LLM_calls"] for r in rows)
        if include_llm:
            for method in ("tool_llm", "knowledge_tool_llm"):
                if sum(r["real_LLM_calls"] for r in rows if r["method"] == method) < 1:
                    raise RuntimeError(f"{method} performed no real Llama inference; not a valid LLM demonstration")
        rec = pd.read_csv(output / "evolution/recommendations.csv")
        if len(rec) != len(methods) * rounds * 8 or not set(rec.variant) <= set(known.variant):
            raise RuntimeError("Unexpected recommendation budget or provenance")
        for _, group in rec.groupby("method"):
            if group.variant.duplicated().any():
                raise RuntimeError("Repeated measurement within a trajectory")
        packages = {}
        for name in ("numpy", "pandas", "scikit-learn", "scipy", "llama-cpp-python"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        summary = {"status": "complete", "scope": "engineering_reusing_previously_known_labels",
                   "source_observations": len(known), "methods": list(methods), "rounds": rounds,
                   "candidate_budget_per_round": 128, "assay_budget_per_round": 8,
                   "engineering_reveals": len(rec), "real_LLM_calls": calls,
                   "decision_files": len(decisions), "predictor": "three_member_Ridge",
                   "predictor_internal_metrics": metrics,
                   "new_project_measurements": 0, "new_P2_training": False,
                   "project_heldout_fitness_read": False, "formal_efficacy_result": False,
                   "llama_weights_hash_verified": bool(llm_check and llm_check.get("llama_model_verified")),
                   "environment": {"python": platform.python_version(), "system": platform.system(), "packages": packages},
                   "outputs": {"prepared_data": "data/observations.csv", "processing": "data/processing.json",
                               "predictor_metrics": "predictor/metrics.json", "model": "predictor/model.joblib",
                               "per_round_metrics": "evolution/round_metrics.csv",
                               "recommendations": "evolution/recommendations.csv",
                               "decisions_feedback_and_raw_LLM": "evolution/<method>/round_<n>/"}}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        state({k: v for k, v in summary.items() if k not in ("environment", "outputs", "predictor_internal_metrics")})
        return summary
    except Exception as exc:
        state({"status": "failed", "error": repr(exc), "scope": "engineering_only", "resume_supported": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=SPLITS / "gb1_initial_labeled.csv")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are refused")
    parser.add_argument("--rounds", type=int, choices=(2, 3), default=3)
    parser.add_argument("--include-llm", action="store_true", help="Use actual local Llama for ordinary and knowledge Agents")
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.output, args.include_llm, args.rounds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
