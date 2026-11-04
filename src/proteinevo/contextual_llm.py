"""Real bounded GGUF inference selecting background-specific experiment actions."""
from __future__ import annotations

import copy
import json
import os
import time
import uuid

from .contextual_science import compatible_actions, property_card
from .local_gguf_client import DEFAULT_MODEL_PATH, LocalGGUFHypothesisGenerator, _load
from .study_support import write_json

MAX_INPUT_TOKENS = 3584
MAX_OUTPUT_TOKENS = 128

SYSTEM = (
    "You plan GB1 binding experiments at sites 39,40,41,54 (WT VDGV). "
    "Choose ONE allowed action_id for a hypothesis that the stated source-to-target substitution increases fitness "
    "in BOTH listed measured-parent backgrounds. Each child changes only that site. "
    "Parents carry known fitness; child fitness is UNKNOWN. Evidence IDs C:parent>child are ALREADY OBSERVED "
    "single-edit contrasts in other explicit backgrounds; delta means child minus parent fitness. "
    "The same substitution can have different signs across backgrounds. Never merge distinct source residues. "
    "Past pairs retain parent, child and delta; use them to revise the scope of a claim. "
    "Zero tests are untested, not refuted; no significance or universal-benefit claims. "
    "A source-to-WT reversion is allowed when it changes the current parent. "
    "Choose a supplied evidence_id or EXPLORATION; a negative contrast may motivate a context-dependence test "
    "but is not positive support. Choose only actions with children disjoint from this decision's previous choice. "
    "Reply only JSON with action_id, evidence_id and reason (1-80 characters)."
)


def format_prompt(system, context):
    user = json.dumps(context, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def compact_action(action):
    return {"id": action["action_id"], "edit": action["substitution"],
            "parents": [[p, round(f, 5)] for p, f in zip(action["parents"], action["parent_fitness"])],
            "children": action["children"],
            "evidence": [{"id": r["evidence_id"], "delta": round(r["delta"], 5)} for r in action["evidence"]],
            "context_sign_counts": [action["positive_context_count"], action["negative_context_count"]]}


def compact_memory(memory):
    return [{"round": r["round"], "edit": r["substitution"], "status": r["status"],
             "pairs": [{"parent": p["parent"], "child": p["child"], "delta": round(p["delta"], 5)}
                       for p in r["pairs"]]} for r in memory[-4:]]


def prepare_shared_view(client, bank, summary, memory):
    """Budget against the knowledge arm so plain/rule/random see the same IDs.

    Drop whole action cards, never their background conditions or outcome fields.
    The full action bank and complete memory remain saved in the run artifacts.
    """
    active = list(bank)
    context = {"n_observed": summary["n_observed"], "best_seen_fitness": round(summary["best_seen_fitness"], 5),
               "top_variants": [{"v": r["variant"], "f": round(r["fitness"], 5)} for r in summary["top_variants"][:4]],
               "memory": compact_memory(memory), "previously_selected_action_ids": [], "step": 2}
    while True:
        context["actions"] = [compact_action(a) for a in active]
        context["properties"] = [property_card(a) for a in active]
        # Reserve space for one selected ID and step marker before a second call.
        context["previously_selected_action_ids"] = ["H:ffffffffffff"]
        count = len(client.tokenize(format_prompt(SYSTEM, context).encode(), add_bos=False, special=True))
        if count <= MAX_INPUT_TOKENS:
            break
        if len(active) <= 2:
            raise ValueError("Complete background conditions do not fit the frozen LLM input budget")
        counts = {site: sum(a["site"] == site for a in active) for site in {a["site"] for a in active}}
        removable = sorted(active, key=lambda a: (-counts[a["site"]], a["action_id"]), reverse=False)
        removed = False
        for candidate in removable:
            proposed = [a for a in active if a is not candidate]
            first = compatible_actions(proposed, [])
            if any(compatible_actions(proposed, [a]) for a in first):
                active = proposed
                removed = True
                break
        if not removed:
            raise ValueError("Cannot preserve a disjoint action pair within the prompt budget")
    context.pop("properties")
    context["previously_selected_action_ids"] = []
    context["step"] = 1
    return active, context, {"full_actions": len(bank), "visible_actions": len(active),
                             "dropped_whole_action_ids": [a["action_id"] for a in bank if a not in active],
                             "memory_records_in_full": len(memory), "memory_records_visible": min(len(memory), 4),
                             "worst_case_knowledge_input_tokens": count, "input_token_limit": MAX_INPUT_TOKENS}


class ContextualActionLLM:
    def __init__(self, *, knowledge=False, trace_directory=None, seed=20261030):
        self.knowledge = knowledge
        self.trace_directory = trace_directory
        self.seed = seed
        self.settings = LocalGGUFHypothesisGenerator(model_path=DEFAULT_MODEL_PATH, seed=seed)
        if self.settings.n_ctx != 4096:
            raise ValueError("The contextual protocol fixes the context window at 4096")
        self.last_trace = {}

    def client(self):
        self.settings.check_model()
        return _load(str(self.settings.model_path), 4096, self.settings.threads, self.seed, self.settings.gpu_layers)

    def choose(self, bank, base_context):
        from llama_cpp import LlamaGrammar

        client = self.client()
        decision_id = uuid.uuid4().hex
        trace = {"decision_id": decision_id, "status": "prepared", "knowledge_enhanced": self.knowledge,
                 "model_metadata": self.settings.check_model(), "calls": [], "selected_actions": [],
                 "expected_direction": "increase", "temperature": 0, "uses_unmeasured_fitness": False}
        self.last_trace = trace
        def save():
            if self.trace_directory is not None:
                write_json(self.trace_directory / f"decision_{decision_id}.json", trace)
        selected = []
        try:
            for step in (1, 2):
                allowed = compatible_actions(bank, selected)
                if not allowed:
                    break
                context = copy.deepcopy(base_context)
                context["actions"] = [compact_action(a) for a in allowed]
                context["previously_selected_action_ids"] = [a["action_id"] for a in selected]
                context["step"] = step
                if self.knowledge:
                    context["properties"] = [property_card(a) for a in allowed]
                prompt = format_prompt(SYSTEM, context)
                tokens = client.tokenize(prompt.encode(), add_bos=False, special=True)
                if len(tokens) > MAX_INPUT_TOKENS:
                    raise ValueError("Actual inference prompt exceeds frozen budget")
                schema = {"oneOf": [{"type": "object", "properties": {
                    "action_id": {"const": a["action_id"]},
                    "evidence_id": {"type": "string", "enum": [r["evidence_id"] for r in a["evidence"]] + ["EXPLORATION"]},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 80}},
                    "required": ["action_id", "evidence_id", "reason"], "additionalProperties": False} for a in allowed]}
                call = {"call_id": uuid.uuid4().hex, "step": step, "status": "prepared", "prompt": prompt,
                        "prompt_context": context, "schema": schema, "exact_prompt_tokens": len(tokens),
                        "max_output_tokens": MAX_OUTPUT_TOKENS, "new_inference_performed": False}
                trace["calls"].append(call)
                save()
                start = time.perf_counter()
                grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
                call["new_inference_performed"] = True
                response = client.create_completion(prompt=tokens, grammar=grammar, max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0, seed=self.seed, stop=["<|eot_id|>", "<|end_of_text|>"], echo=False)
                call.update(raw_response=response["choices"][0]["text"], finish_reason=response["choices"][0]["finish_reason"],
                            usage=response.get("usage", {}), seconds=time.perf_counter() - start)
                save()
                parsed = json.loads(call["raw_response"])
                if call["finish_reason"] != "stop" or set(parsed) != {"action_id", "evidence_id", "reason"}:
                    raise ValueError("Incomplete or invalid raw LLM response; no silent repair/retry")
                by_id = {a["action_id"]: a for a in allowed}
                if parsed["action_id"] not in by_id or not isinstance(parsed["reason"], str) or not 1 <= len(parsed["reason"]) <= 80:
                    raise ValueError("Invalid action or rationale")
                action = copy.deepcopy(by_id[parsed["action_id"]])
                valid_evidence = {r["evidence_id"] for r in action["evidence"]} | {"EXPLORATION"}
                if parsed["evidence_id"] not in valid_evidence:
                    raise ValueError("Incompatible evidence citation")
                action["LLM_reason_original"] = parsed["reason"]
                action["cited_evidence_id"] = parsed["evidence_id"]
                action["expected_direction"] = "increase"
                cited = next((r for r in action["evidence"] if r["evidence_id"] == parsed["evidence_id"]), None)
                action["critic_evidence_note"] = ("Explicit exploration" if cited is None else
                    "Cited measured effect is nonpositive; not positive support" if cited["delta"] <= 0 else
                    "Positive effect is limited to the cited measured background")
                selected.append(action)
                call["status"] = "complete"
                trace["selected_actions"] = selected
                save()
            trace["status"] = "complete"
            save()
            return selected
        except Exception as exc:
            trace.update(status="failed", error=repr(exc))
            save()
            raise
