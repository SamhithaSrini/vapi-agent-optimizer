"""Aggregate judge failures across a batch into ranked FailureGroup objects."""
from __future__ import annotations

from collections import defaultdict

from src.models import CallRecord, FailureGroup, JudgeFailure

_SEVERITY_SCORE = {"low": 1.0, "med": 2.0, "high": 3.0}


def aggregate_failures(records: list[CallRecord], top_k: int = 5) -> list[FailureGroup]:
    """
    Group all failures by dimension, rank by count × mean_severity, return top_k.
    """
    buckets: dict[str, list[JudgeFailure]] = defaultdict(list)

    for r in records:
        if r.llm_judge_result is None:
            continue
        for f in r.llm_judge_result.failures:
            buckets[f.dimension].append(f)

    groups: list[FailureGroup] = []
    for dim, failures in buckets.items():
        scores = [_SEVERITY_SCORE.get(f.severity, 2.0) for f in failures]
        mean_sev = sum(scores) / len(scores)
        priority = len(failures) * mean_sev

        # Deduplicate example reasons
        seen: set[str] = set()
        examples: list[str] = []
        for f in failures:
            if f.reason not in seen and len(examples) < 3:
                examples.append(f.reason)
                seen.add(f.reason)

        groups.append(
            FailureGroup(
                dimension=dim,
                count=len(failures),
                mean_severity_score=round(mean_sev, 2),
                weighted_priority=round(priority, 2),
                example_reasons=examples,
            )
        )

    groups.sort(key=lambda g: g.weighted_priority, reverse=True)
    return groups[:top_k]


def format_digest_yaml(groups: list[FailureGroup]) -> str:
    """Format failure groups as YAML for injection into the rewriter prompt."""
    if not groups:
        return "  (no failures detected)"
    lines = ["failures:"]
    for g in groups:
        lines.append(f"  - dimension: {g.dimension}")
        lines.append(f"    count: {g.count}")
        lines.append(f"    mean_severity: {g.mean_severity_score}")
        lines.append(f"    priority: {g.weighted_priority}")
        lines.append(f"    examples:")
        for ex in g.example_reasons:
            lines.append(f"      - \"{ex}\"")
    return "\n".join(lines)
