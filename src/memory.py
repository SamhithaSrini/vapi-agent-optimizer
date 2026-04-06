"""Optimization memory: append-only log injected into the rewriter prompt."""
from __future__ import annotations

import json
from pathlib import Path

from src.models import FailureGroup, MemoryEntry


class OptimizationMemory:
    def __init__(self, path: str = "results/memory.json"):
        self._path = Path(path)
        self._entries: list[MemoryEntry] = []
        if self._path.exists():
            self._load()

    # ── Mutation ──────────────────────────────────────────────────────────────────

    def append(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._save()

    # ── Formatting for rewriter ───────────────────────────────────────────────────

    def format_for_rewriter(self) -> str:
        if not self._entries:
            return "(no optimization history yet — this is the first iteration)"

        successful = [e for e in self._entries if e.accepted]
        failed = [e for e in self._entries if not e.accepted]

        parts: list[str] = []

        if successful:
            parts.append("PREVIOUS SUCCESSFUL CHANGES:")
            for e in successful:
                parts.append(f"  - iter {e.iteration} (+{e.score_delta:.3f}): {e.change_summary}")
        else:
            parts.append("PREVIOUS SUCCESSFUL CHANGES:\n  (none yet)")

        parts.append("")

        if failed:
            parts.append("FAILED CHANGES (DO NOT REPEAT):")
            for e in failed:
                parts.append(f"  - iter {e.iteration} (rejected): {e.change_summary}")
        else:
            parts.append("FAILED CHANGES:\n  (none yet)")

        parts.append("")
        parts.append("SCORE HISTORY:")
        parts.append("  iter | train  | holdout | accepted")
        for e in self._entries:
            acc = "yes" if e.accepted else " no"
            parts.append(
                f"   {e.iteration:3d} | {e.train_score:.3f}  |  {e.holdout_score:.3f}  |   {acc}"
            )

        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [_entry_to_dict(e) for e in self._entries]
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        with open(self._path) as f:
            data = json.load(f)
        self._entries = [_dict_to_entry(d) for d in data]

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)


# ── Serialization helpers ─────────────────────────────────────────────────────────

def _entry_to_dict(e: MemoryEntry) -> dict:
    return {
        "iteration": e.iteration,
        "prompt_version": e.prompt_version,
        "train_score": e.train_score,
        "holdout_score": e.holdout_score,
        "top_failures": [
            {
                "dimension": f.dimension,
                "count": f.count,
                "mean_severity_score": f.mean_severity_score,
                "weighted_priority": f.weighted_priority,
                "example_reasons": f.example_reasons,
            }
            for f in e.top_failures
        ],
        "accepted": e.accepted,
        "score_delta": e.score_delta,
        "change_summary": e.change_summary,
    }


def _dict_to_entry(d: dict) -> MemoryEntry:
    top_failures = [
        FailureGroup(
            dimension=f["dimension"],
            count=f["count"],
            mean_severity_score=f["mean_severity_score"],
            weighted_priority=f["weighted_priority"],
            example_reasons=f["example_reasons"],
        )
        for f in d.get("top_failures", [])
    ]
    return MemoryEntry(
        iteration=d["iteration"],
        prompt_version=d["prompt_version"],
        train_score=d["train_score"],
        holdout_score=d["holdout_score"],
        top_failures=top_failures,
        accepted=d["accepted"],
        score_delta=d["score_delta"],
        change_summary=d["change_summary"],
    )
