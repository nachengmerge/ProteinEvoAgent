"""Audited helpers for new studies; historical runners remain unchanged."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT, SPLITS
from .p1_protocol import make_panel_ids


def digest_ids(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("ascii")).hexdigest()


def frame_digest(frame):
    canonical = frame.sort_values("variant")[["variant", "fitness"]]
    return hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class StudyAudit:
    """Block new final-holdout label reads and refuse overwriting a frozen run."""

    def __init__(self, output):
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(f"Frozen output already exists: {self.output}")
        self.started = time.perf_counter()
        self.reads = []
        self.completed = []
        self.original_read = pd.read_csv
        pd.read_csv = self._read

    def _read(self, path, *args, **kwargs):
        name = Path(path).name if isinstance(path, (str, Path)) else ""
        columns = kwargs.get("usecols")
        if name in {"gb1_validation.csv.gz", "gb1_test.csv.gz", "gb1_measured.csv.gz"}:
            if columns is None or callable(columns) or {"fitness", "log1p_fitness"} & set(columns):
                raise RuntimeError("New final holdout/all-label fitness reads are forbidden")
        self.reads.append({"file": str(path), "columns": columns if isinstance(columns, list) else str(columns)})
        return self.original_read(path, *args, **kwargs)

    def panels(self, seeds):
        partition = pd.read_csv(SPLITS / "gb1_partition.csv.gz", usecols=["variant", "mutation_count", "split"])
        ids = {s: make_panel_ids(partition, s, (512,))[512] for s in seeds}
        previous = set()
        for file in (ROOT / "data").glob("**/initial_panel*.csv"):
            previous.add(digest_ids(pd.read_csv(file, usecols=["variant"]).variant))
        previous.add(digest_ids(pd.read_csv(SPLITS / "gb1_initial_labeled.csv", usecols=["variant"]).variant))
        for base in (ROOT / "examples", ROOT / "research", ROOT / "results/runs"):
            for file in base.glob("**/*preregistered_design.json"):
                record = json.loads(file.read_text(encoding="utf-8"))
                previous.update(record.get("panel_ids_sha256", {}).values())
                if record.get("initial_ids_sha256"):
                    previous.add(record["initial_ids_sha256"])
        hashes = {str(s): digest_ids(v) for s, v in ids.items()}
        if len(set(hashes.values())) != len(ids) or set(hashes.values()) & previous:
            raise ValueError("Initial panels overlap a previous panel ID hash")
        return ids, hashes, len(previous)

    def freeze(self, design, source_files):
        design = dict(design, frozen_at_utc=datetime.now(timezone.utc).isoformat())
        design["code_sha256"] = {
            str(Path(f).relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for f in sorted(set(map(Path, source_files)))
        }
        self.output.mkdir(parents=True)
        write_json(self.output / "preregistered_design.json", design)
        for file in source_files:
            destination = self.output / "source_snapshot" / Path(file).relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        self.design = design
        return design

    def check_sources(self):
        for name, expected in self.design["code_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen dependency changed during study: {name}")

    def state(self, status, current=None, error=None):
        write_json(self.output / "execution_state.json", {
            "status": status, "current": current, "completed": self.completed,
            "elapsed_seconds": time.perf_counter() - self.started, "error": error,
        })
        write_json(self.output / "read_access_log.json", self.reads)

    def close(self):
        pd.read_csv = self.original_read


def discovery_for_orchestrator():
    """Only orchestration calls this; no hidden frame is passed to an agent/model."""
    frame = pd.concat([
        pd.read_csv(SPLITS / "gb1_initial_labeled.csv", usecols=["variant", "fitness"]),
        pd.read_csv(SPLITS / "gb1_oracle.csv.gz", usecols=["variant", "fitness"]),
    ], ignore_index=True)
    if len(frame) != 112040 or frame.variant.duplicated().any():
        raise ValueError("Unexpected discovery partition")
    return frame.set_index("variant")


def paired_summary(panel, method, reference, value="mean", tolerance=1e-9):
    wide = panel.pivot(index="panel_seed", columns="method", values=value)
    differences = wide[method] - wide[reference]
    return {
        "method": method, "reference": reference, "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std()), "positive": int((differences > tolerance).sum()),
        "negative": int((differences < -tolerance).sum()), "tied": int((differences.abs() <= tolerance).sum()),
        "per_panel": {str(k): float(v) for k, v in differences.items()},
    }
