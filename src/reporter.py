"""Write iteration results, summary, and final markdown report to disk."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from src.models import CallRecord, FailureGroup, IterationResult


class Reporter:
    def __init__(self, results_dir: str = "results"):
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Per-iteration ─────────────────────────────────────────────────────────────

    def save_iteration(
        self,
        result: IterationResult,
        records: list[CallRecord],
    ) -> None:
        data = {
            "summary": _iter_to_dict(result),
            "calls": [_record_to_dict(r) for r in records],
        }
        path = self._dir / f"iteration_{result.iteration}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Running summary ───────────────────────────────────────────────────────────

    def save_summary(self, results: list[IterationResult]) -> None:
        path = self._dir / "summary.json"
        with open(path, "w") as f:
            json.dump([_iter_to_dict(r) for r in results], f, indent=2)

    # ── Final prompt ──────────────────────────────────────────────────────────────

    def save_final_prompt(self, prompt: str) -> None:
        path = self._dir / "final_prompt.txt"
        path.write_text(prompt)

    # ── Markdown report ───────────────────────────────────────────────────────────

    def write_report(
        self,
        results: list[IterationResult],
        initial_prompt: str,
        final_prompt: str,
        baseline_holdout_score: float | None = None,
    ) -> None:
        if not results:
            return

        baseline = results[0]
        final = results[-1]
        # Use the pre-optimization holdout baseline if provided (more accurate before/after)
        baseline_holdout = baseline_holdout_score if baseline_holdout_score is not None \
            else baseline.holdout_composite_score_current

        lines: list[str] = [
            "# Vapi Agent Optimizer — Results Report",
            "",
            "## Before / After",
            "",
            "| Metric | Baseline | Final | Delta |",
            "|--------|----------|-------|-------|",
        ]

        def row(label, b, f):
            delta = f - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {label} | {b:.3f} | {f:.3f} | {sign}{delta:.3f} |")

        row("Composite score (train)", baseline.train_composite_score, final.train_composite_score)
        row("Composite score (holdout)", baseline_holdout, final.holdout_composite_score_current)

        for dim in ["goal_completion", "tone_and_empathy", "information_accuracy", "efficiency", "edge_case_handling"]:
            b_dim = baseline.dimension_scores.get(dim, 0.0)
            f_dim = final.dimension_scores.get(dim, 0.0)
            row(f"  {dim}", b_dim, f_dim)

        lines += [
            "",
            "## Iteration-by-Iteration",
            "",
            "| Iter | Train | Holdout | Accepted | Cost ($) | Cumulative ($) |",
            "|------|-------|---------|----------|----------|----------------|",
        ]
        for r in results:
            acc = "yes" if r.candidate_prompt_accepted else "no"
            lines.append(
                f"| {r.iteration} | {r.train_composite_score:.3f} | "
                f"{r.holdout_composite_score_current:.3f} | {acc} | "
                f"{r.iteration_cost_usd:.4f} | {r.cumulative_cost_usd:.4f} |"
            )

        lines += [
            "",
            "## Top Failure Patterns (final iteration)",
            "",
        ]
        for g in final.failure_digest[:3]:
            lines.append(f"**{g.dimension}** — count {g.count}, priority {g.weighted_priority:.1f}")
            for ex in g.example_reasons:
                lines.append(f"  - {ex}")
            lines.append("")

        lines += [
            "## Initial Prompt (baseline)",
            "```",
            initial_prompt.strip(),
            "```",
            "",
            "## Final Prompt (after optimization)",
            "```",
            final_prompt.strip(),
            "```",
        ]

        path = self._dir / "report.md"
        path.write_text("\n".join(lines))
        print(f"Report written to {path}")


# ── Serialization helpers ─────────────────────────────────────────────────────────

def _iter_to_dict(r: IterationResult) -> dict:
    return {
        "iteration": r.iteration,
        "prompt_version": r.prompt_version,
        "train_composite_score": r.train_composite_score,
        "holdout_composite_score_current": r.holdout_composite_score_current,
        "holdout_composite_score_best_candidate": r.holdout_composite_score_best_candidate,
        "dimension_scores": r.dimension_scores,
        "failure_digest": [
            {
                "dimension": g.dimension,
                "count": g.count,
                "mean_severity_score": g.mean_severity_score,
                "weighted_priority": g.weighted_priority,
                "example_reasons": g.example_reasons,
            }
            for g in r.failure_digest
        ],
        "candidates_generated": r.candidates_generated,
        "candidate_prompt_accepted": r.candidate_prompt_accepted,
        "accepted_candidate_index": r.accepted_candidate_index,
        "call_count": r.call_count,
        "iteration_cost_usd": r.iteration_cost_usd,
        "cumulative_cost_usd": r.cumulative_cost_usd,
    }


def _record_to_dict(r: CallRecord) -> dict:
    d: dict = {
        "call_id": r.call_id,
        "scenario_id": r.scenario_id,
        "iteration": r.iteration,
        "candidate_index": r.candidate_index,
        "assistant_id": r.assistant_id,
        "prompt_version": r.prompt_version,
        "transcript": [{"role": m.role, "message": m.message, "time": m.time} for m in r.transcript],
        "duration_seconds": r.duration_seconds,
        "ended_reason": r.ended_reason,
        "vapi_analysis": r.vapi_analysis,
        "structured_score": r.structured_score,
        "final_score": r.final_score,
    }
    if r.llm_judge_result:
        j = r.llm_judge_result
        d["llm_judge_result"] = {
            "composite_score": j.composite_score,
            "passed": j.passed,
            "dimension_scores": j.dimension_scores.as_dict(),
            "failures": [{"dimension": f.dimension, "reason": f.reason, "severity": f.severity} for f in j.failures],
        }
    return d
