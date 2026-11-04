"""Canonical scientific decisions: observable claims, conditional knowledge, revisions."""
import copy
import json
import time
import uuid

from ..contextual_llm import format_prompt
from ..local_gguf_client import LocalGGUFHypothesisGenerator, _load
from ..scientific_action_view import canonical_action_bank
from ..knowledge import AA_GROUP
from ..study_support import write_json

SYSTEM = (
    "You plan virtual GB1 binding-fitness experiments. Mutable sites:39,40,41,54. "
    "Every action gives TWO already measured parents and two UNKNOWN one-edit children. "
    "Select ONE currently allowed scientific action, its hypothesis type and directions in canonical parent order A/B. "
    "beneficial_both predicts increase/increase. background_dependent predicts increase/decrease or decrease/increase; "
    "the scope lists the actual differing background positions, not a proven causal mechanism. uncertain_transfer "
    "predicts uncertain/uncertain and explicitly tests transfer. "
    "Evidence C:parent>child is a PREVIOUSLY OBSERVED contrast in a stated background. A positive observation elsewhere "
    "does not prove improvement in these parents. Median/context counts and nearest-background medians summarize only "
    "already known labels. child predictions are MODEL POINT ESTIMATES, not child measurements or confidence. "
    "Opportunity cost is predicted batch-mean loss relative to greedy selection, not an actual fitness bound. "
    "If a prior revision is supplied, cite its revision ID in response_to and reconsider its tested scope. "
    "Zero tests mean untested, not refuted. Knowledge cards are weak observed conditional associations, not universal rules. "
    "Use evidence and model magnitudes; prefer valuable candidates or discriminating context tests. "
    "Do not infer folding/stability from this binding assay. Reply only JSON matching the schema; reason at most120 characters."
)


def card(action, knowledge=False):
    result = {"id": action["action_id"], "edit": action["substitution"],
        "tests": [{"label": label, "parent": p, "parent_fitness": round(f, 5), "child": c,
                   "prediction": action["visible_stats"]["child_prediction"][i], "nearest_observed": action["nearest"][i]}
                  for i, (label, p, f, c) in enumerate(zip(("A", "B"), action["parents"], action["parent_fitness"], action["children"]))],
        "statistics": action["visible_stats"], "scope": action["scope"],
        "context_sign_counts": [action["positive_context_count"], action["negative_context_count"]],
        "evidence": [{"id": r["evidence_id"], "delta": round(r["delta"], 5)} for r in action["evidence"]],
        "prior_revision": action.get("revision")}
    if knowledge:
        result["knowledge"] = {"from_property": AA_GROUP[action["from_residue"]],
            "to_property": AA_GROUP[action["to_residue"]], "conditional_rules": action["knowledge_rules"],
            "limitation": "Same current observations, no extra labels; weak property/conditional priors"}
    return result


def prepare_view(client, bank, observed, registry, *, max_input=7168, knowledge_budget=True):
    # Normalize BEFORE shortening or constructing grammar, not just in artifact analysis.
    active, aliases = canonical_action_bank(bank)
    originals = {a["action_id"]: a for a in bank}
    for action in active:
        original = originals[aliases[action["action_id"]]]
        nearest_by_parent = dict(zip(original["parents"], original["nearest"]))
        action["nearest"] = [nearest_by_parent[p] for p in action["parents"]]
        action["scope"]["backgrounds"] = list(action["parents"])
        assert action["parents"] == sorted(action["parents"])
    recent = sorted(registry.values(), key=lambda r: (-r["revision_count"], r["revision_id"]))[:4]
    context = {"observed_n": len(observed), "best_known": round(float(observed.fitness.max()), 5),
               "belief_revisions": recent, "synthetic_contract_test": False}
    while True:
        payload = dict(context, actions=[card(a, knowledge_budget) for a in active], step=2,
                       prior_selected=[active[0]["action_id"]] if active else [])
        count = len(client.tokenize(format_prompt(SYSTEM, payload).encode(), add_bos=False, special=True))
        if count <= max_input: break
        if len(active) <= 2: raise ValueError("Scientific conditions do not fit the frozen prompt budget")
        counts = {s: sum(a["site"] == s for a in active) for s in {a["site"] for a in active}}
        remove = max(active, key=lambda a: (counts[a["site"]], a["visible_stats"]["batch_cost_alone"], a["action_id"]))
        active.remove(remove)
    return active, context, {"input_limit": max_input, "worst_knowledge_prompt_tokens": count,
        "full_actions": len(bank), "active_actions": len(active), "canonical_aliases": aliases,
        "knowledge_budget_shared": knowledge_budget}


def claim_schema(actions):
    branches = []
    options = (("beneficial_both", "increase", "increase"), ("background_dependent", "increase", "decrease"),
               ("background_dependent", "decrease", "increase"), ("uncertain_transfer", "uncertain", "uncertain"))
    for action in actions:
        for kind, a, b in options:
            properties = {"action_id": {"const": action["action_id"]}, "kind": {"const": kind},
                "expect_A": {"const": a}, "expect_B": {"const": b},
                "evidence_id": {"type": "string", "enum": [r["evidence_id"] for r in action["evidence"]] + ["EXPLORATION"]},
                "response_to": {"const": action["revision"]["revision_id"] if action.get("revision") else "NEW"},
                "reason": {"type": "string", "minLength": 1, "maxLength": 120}}
            branches.append({"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False})
    return {"oneOf": branches}


def validate_claim(raw, allowed):
    required = {"action_id", "kind", "expect_A", "expect_B", "evidence_id", "response_to", "reason"}
    if set(raw) != required: raise ValueError("Invalid claim fields")
    action = next((a for a in allowed if a["action_id"] == raw["action_id"]), None)
    if action is None: raise ValueError("Unknown/overlapping action")
    signs = (raw["expect_A"], raw["expect_B"])
    valid = ((raw["kind"] == "beneficial_both" and signs == ("increase", "increase")) or
             (raw["kind"] == "background_dependent" and set(signs) == {"increase", "decrease"}) or
             (raw["kind"] == "uncertain_transfer" and signs == ("uncertain", "uncertain")))
    revision = action["revision"]["revision_id"] if action.get("revision") else "NEW"
    if not valid or raw["response_to"] != revision: raise ValueError("Inconsistent hypothesis or missing revision response")
    if raw["evidence_id"] not in {r["evidence_id"] for r in action["evidence"]} | {"EXPLORATION"}: raise ValueError("False citation")
    if not isinstance(raw["reason"], str) or not 1 <= len(raw["reason"]) <= 120: raise ValueError("Invalid reason")
    chosen = copy.deepcopy(action)
    chosen["claim"] = {k: raw[k] for k in required if k != "action_id"}
    return chosen


class ScientificLLM:
    def __init__(self, *, knowledge=False, context_window=8192, max_input=7168, max_output=256, trace_dir=None):
        self.knowledge, self.context_window = knowledge, context_window
        self.max_input, self.max_output, self.trace_dir = max_input, max_output, trace_dir
        self.settings = LocalGGUFHypothesisGenerator(seed=20261030)
        self.settings.n_ctx = context_window
        self.last_trace = {}

    def client(self):
        self.settings.check_model()
        return _load(str(self.settings.model_path), self.context_window, self.settings.threads, 20261030, self.settings.gpu_layers)

    def choose(self, selector, context):
        from llama_cpp import LlamaGrammar
        trace = {"decision_id": uuid.uuid4().hex, "status": "prepared", "calls": [], "selected_actions": [],
            "knowledge": self.knowledge, "selector": selector.metadata(), "model": self.settings.check_model(),
            "normalized_before_inference": True}
        self.last_trace = trace
        def save():
            if self.trace_dir is not None: write_json(self.trace_dir / f"{trace['decision_id']}.json", trace)
        selected = []
        try:
            for step in range(1, selector.target + 1):
                allowed = selector.allowed(selected)
                payload = dict(context, actions=[card(a, self.knowledge) for a in allowed], step=step,
                               prior_selected=[a["action_id"] for a in selected])
                prompt = format_prompt(SYSTEM, payload)
                client = self.client()
                tokens = client.tokenize(prompt.encode(), add_bos=False, special=True)
                if len(tokens) > self.max_input: raise ValueError("Prompt exceeds frozen budget")
                schema = claim_schema(allowed)
                call = {"call_id": uuid.uuid4().hex, "step": step, "status": "prepared", "prompt": prompt,
                    "context": payload, "schema": schema, "prompt_tokens": len(tokens), "max_output": self.max_output,
                    "inference_performed": False}
                trace["calls"].append(call); save()
                grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
                start = time.perf_counter(); call["inference_performed"] = True
                response = client.create_completion(prompt=tokens, grammar=grammar, max_tokens=self.max_output, temperature=0,
                    seed=20261030, stop=["<|eot_id|>", "<|end_of_text|>"], echo=False)
                call.update(raw=response["choices"][0]["text"], finish=response["choices"][0]["finish_reason"],
                            usage=response.get("usage", {}), seconds=time.perf_counter() - start)
                save()
                if call["finish"] != "stop": raise ValueError("Truncated raw claim; no silent repair/retry")
                selected.append(validate_claim(json.loads(call["raw"]), allowed))
                call["status"] = "complete"; trace["selected_actions"] = selected; save()
            trace["status"] = "complete"; save()
            return selected
        except Exception as exc:
            trace.update(status="failed", error=repr(exc)); save(); raise
