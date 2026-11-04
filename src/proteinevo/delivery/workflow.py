"""Observed-only training and recommendation adapters for the GB1 task.

These adapters are demonstrations, not replacements for preregistered studies.
The predictor, scientific tools, Llama planner and feedback code are reused.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import ROOT, SPLITS, WT, full_sequence, mutation_count, mutation_names, validate_variant
from ..models import evaluate
from ..study_support import write_json

METHODS = ("random", "full_greedy", "tool_rule", "tool_llm", "knowledge_tool_llm")


def fresh_directory(path):
    """Never overwrite a completed, failed or partially written output."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def observations(frame):
    if not {"variant", "fitness"}.issubset(frame.columns):
        raise ValueError("Observations require variant,fitness columns")
    result = frame[["variant", "fitness"]].copy()
    if result.empty or result.variant.duplicated().any():
        raise ValueError("Observations must be nonempty and unique")
    for variant in result.variant:
        validate_variant(variant)
    result["fitness"] = pd.to_numeric(result.fitness, errors="raise")
    if not np.isfinite(result.fitness).all() or (result.fitness < 0).any():
        raise ValueError("Observed fitness must be finite and nonnegative")
    return result.sort_values("variant").reset_index(drop=True)


def state_hash(frame):
    return hashlib.sha256(observations(frame).to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def candidate_ids(frame, known):
    if set(frame.columns) != {"variant"}:
        raise ValueError("Candidate CSV must contain only variant; candidate fitness is forbidden")
    if frame.variant.duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    for variant in frame.variant:
        validate_variant(variant)
    return set(frame.variant) - set(known)


def fit_model(known, predictor="ridge", seed=20261031, device="cpu"):
    from ..active_science.deep_ensemble import LoggedEnsemble
    known = observations(known)
    os.environ["PROTEINEVO_TORCH_DEVICE"] = device
    model = LoggedEnsemble(predictor, seed, n_members=3, mode="full").fit(known.variant, known.fitness)
    return {"format": "proteinevo-delivery-v1", "model": model, "observed_sha256": state_hash(known),
            "observed_ids": known.variant.tolist(), "predictor": predictor, "seed": seed,
            "training_n": len(known), "disagreement_is_calibrated": False}


def train(known, output, predictor="ridge", seed=20261031, device="cpu"):
    known = observations(known)
    if len(known) < 30:
        raise ValueError("At least 30 known variants are required for the internal evaluation")
    output = fresh_directory(output)
    write_json(output / "execution_state.json", {"status": "running", "scope": "observed_only_training"})
    try:
        # No project validation/test file is read, and no hyperparameter is chosen here.
        train_rows, valid_rows = train_test_split(known, test_size=.2, random_state=seed)
        diagnostic = fit_model(train_rows, predictor, seed, device)
        predictions = diagnostic["model"].predict_members(valid_rows.variant.tolist()).mean(axis=0)
        metrics = evaluate(valid_rows.fitness, predictions)
        valid_rows.assign(predicted_fitness=predictions).to_csv(output / "internal_validation_predictions.csv", index=False)
        write_json(output / "metrics.json", {"scope": "fixed_20_percent_holdout_within_supplied_observations",
                   "train_n": len(train_rows), "validation_n": len(valid_rows), **metrics})
        artifact = fit_model(known, predictor, seed, device)
        joblib.dump(artifact, output / "model.joblib")
        write_json(output / "training.json", {k: v for k, v in artifact.items() if k != "model"})
        write_json(output / "execution_state.json", {"status": "complete", "new_project_oracle_queries": 0,
                   "real_LLM_calls": 0, "model_member_fits": 6})
        return metrics
    except Exception as exc:
        write_json(output / "execution_state.json", {"status": "failed", "error": repr(exc)})
        raise


def evaluate_model(artifact, labeled, output):
    labeled = observations(labeled)
    if set(labeled.variant) & set(artifact["observed_ids"]):
        raise ValueError("Evaluation variants overlap model training observations")
    predictions = artifact["model"].predict_members(labeled.variant.tolist()).mean(axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions")
    output = fresh_directory(output)
    metrics = evaluate(labeled.fitness, predictions)
    labeled.assign(predicted_fitness=predictions).to_csv(output / "predictions.csv", index=False)
    write_json(output / "metrics.json", {"scope": "explicit_user_supplied_labels; not automatically an external blind test", **metrics})
    return metrics


def recommend(known, artifact, output, method="full_greedy", available=None,
              candidate_budget=800, batch_size=16, seed=20261031, registry=None):
    from ..agents import DataAnalyst
    from ..candidates import generate_proposals, sample_uniform
    from ..active_science.planner import Planner, add_tools
    from ..active_science.tools import attach_forecast
    from ..workbench_v2.science import Scoring, SharedSelector, build_bank

    known = observations(known)
    if artifact.get("format") != "proteinevo-delivery-v1" or state_hash(known) != artifact["observed_sha256"]:
        raise ValueError("Model must have been trained on exactly these current observations")
    if method not in METHODS or not 4 <= batch_size <= candidate_budget:
        raise ValueError("Unknown method or invalid budgets (4 <= batch <= candidates)")
    if available is None:
        partition = pd.read_csv(SPLITS / "gb1_partition.csv.gz", usecols=["variant", "split"])
        available = set(partition.loc[partition.split == "discovery", "variant"])
    available = set(available) - set(known.variant)
    for variant in available:
        validate_variant(variant)
    if len(available) < batch_size:
        raise ValueError("Insufficient unseen candidates")
    output = fresh_directory(output)
    write_json(output / "execution_state.json", {"status": "running", "scope": "recommendation_only"})
    try:
        c = json.loads((ROOT / "configs/llama_baseline_v1.json").read_text(encoding="utf-8"))
        c.update(batch_size=batch_size, candidate_budget=candidate_budget)
        rng = np.random.default_rng(seed)
        provenance = {}
        ids = (sample_uniform(available, candidate_budget, rng) if method == "random" else
               generate_proposals(available, known, [], candidate_budget, rng, provenance=provenance)[0])
        members = artifact["model"].predict_members(ids)
        if not np.isfinite(members).all():
            raise ValueError("Non-finite candidate predictions")
        pool = pd.DataFrame({"variant": ids, "predicted_fitness": members.mean(axis=0)})
        actions, metadata, calls = [], {}, 0
        if method == "random":
            chosen = list(map(str, rng.choice(ids, size=batch_size, replace=False)))
        elif method == "full_greedy":
            chosen = pool.sort_values(["predicted_fitness", "variant"], ascending=[False, True]).head(batch_size).variant.tolist()
        elif method == "tool_rule":
            # Offline rule needs neither a tokenizer nor model weights. This standalone
            # adapter does not claim the P34 shared-token-pruned bank comparison.
            scoring = Scoring(pool, batch_size)
            bank = add_tools(build_bank(known, scoring, c, registry or {}), dict(zip(ids, members.T)), c)
            selector = SharedSelector(bank, scoring, c)
            for action in selector.choose_non_llm("adaptive_rule", None):
                intent = {"question": "contrast_backgrounds", "evidence_id": "MODEL_FORECAST",
                          "response_to": action["revision"]["revision_id"] if action.get("revision") else "NEW",
                          "reason": "Observed-evidence rule; numerical forecast belongs to the tool"}
                actions.append(attach_forecast(action, action["tool_forecast"], intent, "rule"))
            chosen = scoring.choose(actions)
            metadata = {"selector": selector.metadata(), "scope": "unpruned_offline_rule_demo"}
        else:
            from ..llama_baseline import decide
            planner = Planner(c["LLM"])
            chosen, actions, metadata = decide(known, pool, dict(zip(ids, members.T)), c,
                                               registry or {}, method, planner, output / "llm_calls")
            calls = sum(bool(call.get("inference_performed")) for call in planner.last_trace.get("calls", []))
        if len(chosen) != batch_size or len(set(chosen)) != batch_size or not set(chosen) <= available:
            raise ValueError("Selection violates candidate or assay budget")
        tests = {child: (action, parent) for action in actions for parent, child in zip(action["parents"], action["children"])}
        selected = pool.set_index("variant").loc[chosen].reset_index()
        selected["full_sequence"] = selected.variant.map(full_sequence)
        selected["mutations"] = selected.variant.map(lambda v: ";".join(mutation_names(v)))
        selected["role"] = ["hypothesis_test" if v in tests else "random" if method == "random" else "model_exploitation" for v in chosen]
        selected["parent"] = [tests[v][1] if v in tests else provenance.get(v, {}).get("generator_parent") for v in chosen]
        selected["reason"] = [tests[v][0]["intent"]["reason"] if v in tests else "Uniform random reference" if method == "random" else "Highest predicted fitness after reserved tests" for v in chosen]
        selected.to_csv(output / "recommendations.csv", index=False)
        pool.to_csv(output / "scored_candidates.csv", index=False)
        write_json(output / "analysis.json", DataAnalyst().analyze(known))
        write_json(output / "decision.json", {"method": method, "observed_sha256": state_hash(known),
                   "chosen": chosen, "actions": actions, "metadata": metadata,
                   "real_LLM_calls": calls, "measured_new_fitness": False, "model_ranking_is_LLM_design": False})
        write_json(output / "execution_state.json", {"status": "complete", "real_LLM_calls": calls,
                   "new_project_oracle_queries": 0, "recommendations": len(selected)})
        return selected, actions, c
    except Exception as exc:
        write_json(output / "execution_state.json", {"status": "failed", "error": repr(exc)})
        raise


def demo(source, output, methods=("random", "full_greedy", "tool_rule"), rounds=3):
    """Three-round engineering check using ONLY already-known input labels.

    Re-hiding part of a supplied observation table creates a toy oracle. It
    produces no new project measurement or claim about Llama effectiveness.
    """
    from ..workbench_v2.science import ledger_update
    from ..study_support import digest_ids
    source = observations(source)
    if len(source) < 160 or rounds not in (2, 3) or not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Demo needs >=160 known rows, 2-3 rounds and distinct supported methods")
    initial = source.loc[source.variant.map(mutation_count) <= 1].copy()
    if len(initial) < 30:
        initial = source.sort_values("variant").head(77).copy()
    hidden = source.loc[~source.variant.isin(initial.variant)].set_index("variant")
    if len(hidden) < rounds * 8:
        raise ValueError("Not enough already-known rows for toy feedback")
    output = fresh_directory(output)
    write_json(output / "design.json", {"scope": "engineering_demo_reusing_already_known_labels",
               "methods": list(methods), "rounds": rounds, "candidate_budget": 128, "batch_size": 8,
               "input_observations_sha256": state_hash(source), "initial_ids_sha256": digest_ids(initial.variant),
               "initial_n": len(initial), "predictor": "ridge", "seed": 20261031,
               "new_project_measurements": 0, "counts_as_formal_LLM_experiment": False})
    write_json(output / "execution_state.json", {"status": "running"})
    records, costs, feedbacks = [], [], []
    try:
        for method in methods:
            known, registry = initial.copy(), {}
            for turn in range(1, rounds + 1):
                artifact = fit_model(known, seed=20261031 + turn)
                folder = output / method / f"round_{turn}"
                chosen, actions, c = recommend(known, artifact, folder, method, set(hidden.index),
                                               128, 8, 20261031 + turn, registry)
                # Private lookup happens only AFTER selection; the planner never sees hidden.
                measured = hidden.loc[chosen.variant].reset_index()[["variant", "fitness"]]
                feedback = ledger_update(actions, known, measured, turn, registry, chosen.variant.tolist(), c)
                write_json(folder / "feedback.json", feedback)
                write_json(folder / "belief_state.json", registry)
                chosen["experimental_fitness"] = chosen.variant.map(measured.set_index("variant").fitness)
                chosen["method"], chosen["round"] = method, turn
                records.append(chosen)
                state = json.loads((folder / "execution_state.json").read_text())
                costs.append({"method": method, "round": turn, "observed_n": len(known),
                              "mean_fitness": float(measured.fitness.mean()), "batch_best": float(measured.fitness.max()),
                              "real_LLM_calls": state["real_LLM_calls"]})
                feedbacks.extend(feedback)
                known = observations(pd.concat([known, measured], ignore_index=True))
        pd.concat(records, ignore_index=True).to_csv(output / "recommendations.csv", index=False)
        pd.DataFrame(costs).to_csv(output / "round_metrics.csv", index=False)
        write_json(output / "execution_state.json", {"status": "complete", "engineering_reveals": len(records) * 8,
                   "new_project_measurements": 0, "real_LLM_calls": sum(r["real_LLM_calls"] for r in costs),
                   "formal_efficacy_result": False})
        return costs
    except Exception as exc:
        write_json(output / "execution_state.json", {"status": "failed", "error": repr(exc)})
        raise
