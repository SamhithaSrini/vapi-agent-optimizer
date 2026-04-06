"""Shared data models for the Vapi Agent Optimizer."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ── Scenario library ─────────────────────────────────────────────────────────────

@dataclass
class Persona:
    name: str
    intent: str
    behaviour: str


@dataclass
class ScriptTurn:
    turn: int
    line: str


@dataclass
class Scenario:
    id: str
    split: str                          # "train" | "holdout"
    category: str
    difficulty: str
    persona: Persona
    expected_outcome: str
    expected_structured_fields: list[str]
    known_edge_cases: list[str]
    script: list[ScriptTurn]


# ── Call artifacts ───────────────────────────────────────────────────────────────

@dataclass
class TranscriptMessage:
    role: str           # "user" | "assistant"
    message: str
    time: float = 0.0


@dataclass
class CallRecord:
    call_id: str
    scenario_id: str
    iteration: int
    candidate_index: int        # -1 for baseline / training run
    assistant_id: str
    prompt_version: str
    transcript: list[TranscriptMessage]
    duration_seconds: float
    ended_reason: str
    vapi_analysis: Optional[dict] = None
    llm_judge_result: Optional["JudgeResult"] = None
    structured_score: Optional[float] = None
    final_score: Optional[float] = None


# ── Judge outputs ─────────────────────────────────────────────────────────────────

@dataclass
class JudgeDimensionScores:
    goal_completion: float
    tone_and_empathy: float
    information_accuracy: float
    efficiency: float
    edge_case_handling: float

    def weighted_composite(self) -> float:
        # Weights calibrated against MultiWOZ 2.2 benchmark:
        # goal_completion raised (most discriminative), tone_and_empathy lowered (least discriminative)
        return (
            self.goal_completion        * 0.40
            + self.tone_and_empathy     * 0.15
            + self.information_accuracy * 0.20
            + self.efficiency           * 0.15
            + self.edge_case_handling   * 0.10
        )

    def as_dict(self) -> dict:
        return {
            "goal_completion": self.goal_completion,
            "tone_and_empathy": self.tone_and_empathy,
            "information_accuracy": self.information_accuracy,
            "efficiency": self.efficiency,
            "edge_case_handling": self.edge_case_handling,
        }


@dataclass
class JudgeFailure:
    dimension: str
    reason: str
    severity: str       # "low" | "med" | "high"


@dataclass
class JudgeResult:
    composite_score: float
    passed: bool
    dimension_scores: JudgeDimensionScores
    failures: list[JudgeFailure]


# ── Aggregator outputs ───────────────────────────────────────────────────────────

@dataclass
class FailureGroup:
    dimension: str
    count: int
    mean_severity_score: float
    weighted_priority: float
    example_reasons: list[str]


# ── Optimization memory ──────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    iteration: int
    prompt_version: str
    train_score: float
    holdout_score: float
    top_failures: list[FailureGroup]
    accepted: bool
    score_delta: float
    change_summary: str


# ── Iteration results ─────────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    iteration: int
    prompt_version: str
    train_composite_score: float
    holdout_composite_score_current: float
    holdout_composite_score_best_candidate: float
    dimension_scores: dict
    failure_digest: list[FailureGroup]
    candidates_generated: int
    candidate_prompt_accepted: bool
    accepted_candidate_index: Optional[int]
    call_count: int
    iteration_cost_usd: float
    cumulative_cost_usd: float


# ── Helpers ───────────────────────────────────────────────────────────────────────

def prompt_version(prompt: str) -> str:
    """Short SHA-256 fingerprint of a prompt string."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]
