"""Current run/reproduction engine; frozen historical runners are unchanged."""
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from . import VERSION
from .science import Scoring, Selector, build_bank, ledger_update
from .llm import ScientificLLM, prepare_view
from ..config import ROOT, SPLITS, full_sequence, mutation_names
from ..candidates import generate_proposals, sample_uniform
from ..contextual_science import validate_action
from ..models import FitnessModel
from ..oracle import PanelRetrospectiveOracle
from ..p1_protocol import make_panel_ids
from ..local_gguf_client import DEFAULT_MODEL_PATH, EXPECTED_SHA256, file_sha256
from ..study_support import StudyAudit, digest_ids, write_json, discovery_for_orchestrator


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_digest(frame):
    return digest([[r.variant, float(r.fitness)] for r in frame.sort_values("variant").itertuples()])


def validate_config(c):
    required = {"seeds", "methods", "rounds", "candidate_budget", "batch_size", "predictor", "device", "allocation", "action_limit"}
    if not required <= set(c): raise ValueError("Incomplete run config")
    if not set(c["methods"]) <= {"random", "greedy", "rule", "llm", "knowledge_llm", "adaptive_llm"}: raise ValueError("Unknown method")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]) or len(set(c["methods"])) != len(c["methods"]): raise ValueError("Duplicate/empty design")
    if not 1 <= c["rounds"] <= 3 or not 4 <= c["batch_size"] <= c["candidate_budget"]: raise ValueError("Invalid budget")
    if c["allocation"] not in {"fixed4", "adaptive"} or c["predictor"] not in {"ridge", "blend_resmlp_pairwise"}: raise ValueError("Unknown strategy")


class Audit(StudyAudit):
    def freeze(self, design, source_files):
        design = dict(design, protocol=VERSION, code_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        self.output.mkdir(parents=True)
        write_json(self.output / "preregistered_design.json", design)
        with zipfile.ZipFile(self.output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for p in source_files: archive.write(p, p.relative_to(ROOT).as_posix())
        self.design = design
        return design


def panel_ids(c, audit, mode, manifest=None):
    if mode == "new":
        ids, hashes = audit.panels(tuple(c["seeds"]))[:2]
        index = ROOT/"results/workbench_run_index.json"
        previous = json.loads(index.read_text()) if index.exists() else []
        seen = {h for record in previous for h in record["panel_ids_sha256"].values()}
        if set(hashes.values()) & seen: raise ValueError("Initial IDs already reserved; use explicit reproduce mode")
        return ids, hashes
    if mode != "reproduce" or not manifest or manifest.get("protocol") != VERSION: raise ValueError("Compatible frozen manifest required")
    if digest(c) != manifest["config_sha256"]: raise ValueError("Config differs from frozen reproduction")
    required_sources = {"src/proteinevo/workbench/run.py", "src/proteinevo/workbench/science.py", "src/proteinevo/models.py"}
    if not required_sources <= {str(Path(n).as_posix()) for n in manifest["code_sha256"]}: raise ValueError("Incomplete source manifest")
    for name, sha in manifest["code_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha: raise ValueError(f"Source changed: {name}")
    if hashlib.sha256((SPLITS / "gb1_partition.csv.gz").read_bytes()).hexdigest() != manifest["partition_sha256"]: raise ValueError("Partition changed")
    part = pd.read_csv(SPLITS / "gb1_partition.csv.gz", usecols=["variant", "mutation_count", "split"])
    ids = {s: make_panel_ids(part, s, (512,))[512] for s in c["seeds"]}
    hashes = {str(s): digest_ids(v) for s, v in ids.items()}
    if hashes != manifest["panel_ids_sha256"]: raise ValueError("Reproduced initial IDs differ")
    return ids, hashes


def decide(observed, pool, method, allocation, registry, *, llm=None, trace_dir=None, action_limit=12, batch_size=16):
    if "llm" in method and llm is None: llm = ScientificLLM(knowledge=method == "knowledge_llm")
    scoring = Scoring(pool, batch_size)
    full = build_bank(observed, scoring, limit=action_limit, registry=registry)
    if llm is not None:
        bank, context, view = prepare_view(llm.client(), full, observed, registry)
    else: bank, context, view = full, {}, {"canonical_before_inference": True}
    selector = Selector(bank, scoring, "adaptive" if method == "adaptive_llm" else allocation)
    if method == "greedy": actions = []
    elif method == "rule": actions = selector.choose_rule()
    else:
        llm = llm or ScientificLLM()
        llm.knowledge, llm.trace_dir = method == "knowledge_llm", trace_dir
        actions = llm.choose(selector, context)
    for a in actions: validate_action(a, observed, set(pool.variant))
    chosen = scoring.choose(actions)
    tests = {v: (a["action_id"], p) for a in actions for p, v in zip(a["parents"], a["children"])}
    selected = pool.set_index("variant").loc[chosen].reset_index().copy()
    selected["role"] = ["hypothesis_test" if v in tests else "exploitation" for v in chosen]
    selected["action_id"] = [tests[v][0] if v in tests else None for v in chosen]
    selected["test_parent"] = [tests[v][1] if v in tests else None for v in chosen]
    return selected, actions, {"full_bank": full, "bank": bank, "context": context, "view": view,
        "selector": selector.metadata(), "predicted_mean_cost": scoring.cost(actions), "greedy_ids": scoring.greedy}


def run_study(c, output, *, mode="new", manifest=None):
    validate_config(c)
    os.environ["PROTEINEVO_TORCH_DEVICE"] = c["device"]
    uses_llm = any("llm" in m for m in c["methods"])
    if uses_llm: os.environ["PROTEINEVO_LOCAL_GPU_LAYERS"] = str(c.get("llm_gpu_layers", -1))
    if uses_llm and file_sha256(DEFAULT_MODEL_PATH) != EXPECTED_SHA256: raise ValueError("GGUF identity mismatch")
    audit = Audit(Path(output)); label = "freeze"
    try:
        ids, hashes = panel_ids(c, audit, mode, manifest)
        audit.freeze({"config": c, "config_sha256": digest(c), "mode": mode, "panel_ids_sha256": hashes,
            "partition_sha256": hashlib.sha256((SPLITS / "gb1_partition.csv.gz").read_bytes()).hexdigest(),
            "reproduces_manifest_sha256": digest(manifest) if manifest else None,
            "primary": ["Mean fitness of all acquired variants at frozen budget", "Final best gain over initial"],
            "secondary": ["round to new best", "hits fitness>=5", "conditional claim support", "query/inference costs"],
            "initial_labels": 512, "assay_events_expected": len(ids)*len(c["methods"])*c["rounds"]*c["batch_size"],
            "LLM_calls_max": len(ids)*sum("llm" in m for m in c["methods"])*c["rounds"]*2,
            "known_test_history": "Fixed final test was historically evaluated; this run does not read its labels",
            "new_final_holdout_reads": False, "knowledge_uses_extra_labels": False,
            "model_sha256": EXPECTED_SHA256 if uses_llm else None}, sorted((ROOT / "src/proteinevo").rglob("*.py")))
        if mode == "new":
            index = ROOT/"results/workbench_run_index.json"
            previous = json.loads(index.read_text()) if index.exists() else []
            previous.append({"manifest": str(audit.output.resolve()/"preregistered_design.json"), "panel_ids_sha256": hashes, "config_sha256": digest(c)})
            write_json(index, previous)
        secret = discovery_for_orchestrator()
        recs, metrics, ledgers, fits = [], [], [], []
        llm = ScientificLLM() if uses_llm else None
        def flush():
            if recs: pd.concat(recs, ignore_index=True).to_csv(audit.output / "recommendations.csv", index=False)
            pd.DataFrame(metrics).to_csv(audit.output / "round_metrics.csv", index=False)
            pd.DataFrame(fits).to_csv(audit.output / "training_cost.csv", index=False)
            write_json(audit.output / "hypothesis_ledger.json", ledgers)
        for seed in c["seeds"]:
            folder = audit.output / f"panel_{seed}"; folder.mkdir()
            initial = secret.loc[sorted(ids[seed])].reset_index()
            initial.to_csv(folder / "initial_observations.csv", index=False)
            states = {m: initial.copy() for m in c["methods"]}; registries = {m: {} for m in c["methods"]}
            gates = {m: PanelRetrospectiveOracle(ids[seed]) for m in c["methods"]}; cache = {}
            for turn in range(1, c["rounds"]+1):
                first_pools = []
                for method in c["methods"]:
                    label = f"{seed}:{turn}:{method}"; audit.state("running", label); print("START "+label, flush=True)
                    observed = states[method].sort_values("variant").reset_index(drop=True)
                    sha = state_digest(observed); directory = folder / method; directory.mkdir(exist_ok=True)
                    rng = np.random.default_rng(seed+1000*turn)
                    if method == "random":
                        proposals = sample_uniform(gates[method].available_variants(), c["candidate_budget"], rng)
                        chosen = list(map(str, rng.choice(proposals, size=c["batch_size"], replace=False)))
                        selected = pd.DataFrame({"variant": chosen, "predicted_fitness": np.nan, "role": "random", "action_id": None, "test_parent": None})
                        pool = pd.DataFrame({"variant": proposals, "predicted_fitness": np.nan}); actions, meta = [], {}
                    else:
                        proposals, _ = generate_proposals(gates[method].available_variants(), observed, [], c["candidate_budget"], rng)
                        key = f"{seed+turn}_{sha}"
                        if key not in cache:
                            start = time.perf_counter()
                            cache[key] = FitnessModel(c["predictor"], seed+turn).fit(observed.variant.tolist(), observed.fitness.to_numpy())
                            fits.append({"state_id": key, "labels": len(observed), "seconds": time.perf_counter()-start})
                        pool = pd.DataFrame({"variant": proposals, "predicted_fitness": cache[key].predict(proposals).astype(float)})
                        selected, actions, meta = decide(observed, pool, method, c["allocation"], registries[method],
                            llm=llm, trace_dir=directory/"llm_calls", action_limit=c["action_limit"], batch_size=c["batch_size"])
                        if turn == 1: first_pools.append(digest_ids(proposals))
                    assert len(selected) == c["batch_size"] and not set(selected.variant)&set(observed.variant)
                    pool["selected"] = pool.variant.isin(selected.variant)
                    pool.to_csv(directory/f"candidates_{turn}.csv.gz", index=False)
                    write_json(directory/f"decision_{turn}.json", {"observed_sha256": sha, "actions": actions, "chosen": selected.variant.tolist(), "metadata": meta,
                        "LLM_ref": llm.last_trace.get("decision_id") if "llm" in method else None})
                    measured = gates[method].measure(selected.variant.tolist())
                    feedback = ledger_update(actions, observed, measured, turn, registries[method], selected.variant.tolist())
                    write_json(directory/f"belief_state_{turn}.json", registries[method])
                    selected["experimental_fitness"] = selected.variant.map(measured.set_index("variant").fitness)
                    selected["panel_seed"], selected["method"], selected["round"] = seed, method, turn
                    selected["rank"] = range(1, len(selected)+1); selected["full_sequence"] = selected.variant.map(full_sequence)
                    selected["mutations"] = selected.variant.map(lambda v: ";".join(mutation_names(v))); recs.append(selected)
                    best = max(float(observed.fitness.max()), float(measured.fitness.max()))
                    metrics.append({"panel_seed": seed, "method": method, "round": turn, "observed_n": len(observed),
                        "mean_fitness": float(measured.fitness.mean()), "batch_best": float(measured.fitness.max()),
                        "initial_best": float(initial.fitness.max()), "cumulative_best": best, "gain_over_initial": best-float(initial.fitness.max()),
                        "hits_ge5": int((measured.fitness>=5).sum()), "improves_initial_best": float(measured.fitness.max())>float(initial.fitness.max())+1e-9,
                        "test_slots": 2*len(actions), "candidate_count": len(pool), "state_sha256": sha})
                    ledgers.extend(dict(r, panel_seed=seed, method=method) for r in feedback)
                    states[method] = pd.concat([observed, measured], ignore_index=True)
                    assert len(states[method]) == 512+c["batch_size"]*turn
                    audit.completed.append(label); flush(); print(f"DONE {label} mean={measured.fitness.mean():.4f} tests={2*len(actions)}", flush=True)
                if first_pools: assert len(set(first_pools)) == 1
        result = pd.concat(recs, ignore_index=True)
        panel = result.groupby(["panel_seed", "method"]).experimental_fitness.agg(["mean", "max"]).reset_index()
        panel.to_csv(audit.output/"panel_summary.csv", index=False)
        panel.groupby("method")["mean"].agg(["mean", "std", "count"]).to_csv(audit.output/"method_summary.csv")
        calls = []
        for file in audit.output.glob("panel_*/*/llm_calls/*.json"):
            d = json.loads(file.read_text()); assert d["status"] == "complete"; calls.extend(d["calls"])
        write_json(audit.output/"audit_summary.json", {"assay_events": len(result), "LLM_calls": len(calls), "predictor_fits": len(fits),
            "input_tokens": sum(d["usage"]["prompt_tokens"] for d in calls), "output_tokens": sum(d["usage"]["completion_tokens"] for d in calls),
            "new_final_holdout_reads": False, "claim_types": pd.Series([r["claim"]["kind"] for r in ledgers if "claim" in r], dtype=str).value_counts().to_dict(),
            "belief_refutations": sum(r.get("status") == "refuted" for r in ledgers)})
        audit.check_sources(); audit.state("complete"); print("COMPLETE "+str(audit.output), flush=True)
        return audit.output
    except Exception as exc:
        if audit.output.exists(): audit.state("failed", label, repr(exc))
        raise
    finally: audit.close()
