"""Load scenarios from JSON and return typed Scenario objects."""
from __future__ import annotations

import json
from src.models import Persona, Scenario, ScriptTurn


def load_scenarios(path: str) -> list[Scenario]:
    with open(path) as f:
        raw = json.load(f)
    return [_parse(s) for s in raw]


def _parse(s: dict) -> Scenario:
    persona = Persona(**s["persona"])
    script = [ScriptTurn(**t) for t in s.get("script", [])]
    return Scenario(
        id=s["id"],
        split=s["split"],
        category=s.get("category", ""),
        difficulty=s.get("difficulty", ""),
        persona=persona,
        expected_outcome=s["expected_outcome"],
        expected_structured_fields=s["expected_structured_fields"],
        known_edge_cases=s.get("known_edge_cases", []),
        script=script,
    )


def train_scenarios(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if s.split == "train"]


def holdout_scenarios(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios if s.split == "holdout"]
