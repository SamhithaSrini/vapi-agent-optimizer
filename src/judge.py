"""Hybrid judge: LLM rubric (0.7) + structured task completion (0.3)."""
from __future__ import annotations

import json

from src.llm_client import LLMClient
from src.models import (
    CallRecord,
    JudgeDimensionScores,
    JudgeFailure,
    JudgeResult,
    Scenario,
    TranscriptMessage,
)

_JUDGE_SYSTEM = """\
You are an expert evaluator for voice AI agents serving dental offices.
You will receive a conversation transcript and a scenario description.
Score the agent on the rubric below. Return ONLY valid JSON — no prose, no markdown fences.

IMPORTANT CALIBRATION NOTES (derived from MultiWOZ benchmark evaluation):
- An agent that sounds polite but fails to complete the task should score LOW on goal_completion.
  Do NOT let a warm tone inflate goal_completion. They are independent dimensions.
- A long call is only penalised on efficiency if length did NOT serve the caller's needs.
  Complex calls (emergency, multi-person booking) may legitimately need more turns.
- Score information_accuracy low (< 0.5) when the agent gives no info or wrong info —
  saying "I don't know, call back" is an accuracy failure if the agent should know.

Rubric dimensions (each scored 0.0–1.0):
  goal_completion      — Did the agent achieve the caller's stated goal? (most important)
  tone_and_empathy     — Was the agent warm, professional, and reassuring?
  information_accuracy — Were all facts (hours, services, prices) correct or appropriately caveated?
  efficiency           — Was the call resolved efficiently given its complexity?
  edge_case_handling   — Did the agent handle objections / unusual requests gracefully?

Composite = 0.40*goal_completion + 0.15*tone + 0.20*accuracy + 0.15*efficiency + 0.10*edge_case
passed = composite >= 0.70

For each dimension scoring below 0.7, add a failure entry.

Return exactly this JSON structure:
{
  "composite_score": <float>,
  "passed": <bool>,
  "dimension_scores": {
    "goal_completion": <float>,
    "tone_and_empathy": <float>,
    "information_accuracy": <float>,
    "efficiency": <float>,
    "edge_case_handling": <float>
  },
  "failures": [
    {"dimension": <str>, "reason": <str>, "severity": "low"|"med"|"high"}
  ]
}"""


class HybridJudge:
    def __init__(self, llm: LLMClient, config):
        self._llm = llm
        self._cfg = config

    async def score(self, record: CallRecord, scenario: Scenario) -> CallRecord:
        """Score a call record in-place; returns the same record with scores filled."""
        transcript_text = _format_transcript(record.transcript)

        # ── Signal 1: LLM rubric ──────────────────────────────────────────────────
        user_msg = (
            f"SCENARIO: {scenario.persona.intent}\n"
            f"EXPECTED OUTCOME: {scenario.expected_outcome}\n"
            f"DIFFICULTY: {scenario.difficulty}\n\n"
            f"TRANSCRIPT:\n{transcript_text}"
        )
        try:
            raw = await self._llm.complete(
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                model=self._cfg.judge_model,
                temperature=self._cfg.judge_temperature,
                max_tokens=600,
            )
            raw = _strip_fences(raw)
            parsed = json.loads(raw)
            dim = parsed["dimension_scores"]
            judge_result = JudgeResult(
                composite_score=float(parsed["composite_score"]),
                passed=bool(parsed["passed"]),
                dimension_scores=JudgeDimensionScores(
                    goal_completion=float(dim["goal_completion"]),
                    tone_and_empathy=float(dim["tone_and_empathy"]),
                    information_accuracy=float(dim["information_accuracy"]),
                    efficiency=float(dim["efficiency"]),
                    edge_case_handling=float(dim["edge_case_handling"]),
                ),
                failures=[
                    JudgeFailure(
                        dimension=f["dimension"],
                        reason=f["reason"],
                        severity=f["severity"],
                    )
                    for f in parsed.get("failures", [])
                ],
            )
        except Exception as e:
            # Fallback: neutral score if judge fails
            judge_result = JudgeResult(
                composite_score=0.5,
                passed=False,
                dimension_scores=JudgeDimensionScores(0.5, 0.5, 0.5, 0.5, 0.5),
                failures=[JudgeFailure("system", f"Judge parse error: {e}", "med")],
            )

        # ── Signal 2: structured task completion ──────────────────────────────────
        structured = _structured_score(record.vapi_analysis, scenario.expected_structured_fields)

        # ── Hybrid composite ──────────────────────────────────────────────────────
        w_llm = self._cfg.llm_judge_weight
        w_str = self._cfg.structured_score_weight
        final = w_llm * judge_result.composite_score + w_str * structured

        record.llm_judge_result = judge_result
        record.structured_score = round(structured, 4)
        record.final_score = round(final, 4)
        return record

    async def score_batch(
        self, records: list[CallRecord], scenario_map: dict[str, Scenario]
    ) -> list[CallRecord]:
        import asyncio
        tasks = [self.score(r, scenario_map[r.scenario_id]) for r in records]
        return list(await asyncio.gather(*tasks))


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _format_transcript(transcript: list[TranscriptMessage]) -> str:
    lines = []
    for m in transcript:
        label = "AGENT" if m.role == "assistant" else "CALLER"
        lines.append(f"{label}: {m.message}")
    return "\n".join(lines)


def _structured_score(vapi_analysis: dict | None, expected_fields: list[str]) -> float:
    if not vapi_analysis or not expected_fields:
        return 0.5  # neutral fallback
    hits = sum(1 for f in expected_fields if vapi_analysis.get(f) is True)
    return hits / len(expected_fields)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ── Quick test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from src.config import load_config
    from src.models import TranscriptMessage

    async def _smoke():
        cfg = load_config()
        from src.llm_client import LLMClient
        from src.scenarios import load_scenarios
        llm = LLMClient(api_key=cfg.openrouter_api_key, max_concurrent=cfg.max_concurrent_llm_calls)
        judge = HybridJudge(llm, cfg)

        scenarios = load_scenarios(cfg.scenario_file)
        scenario = next(s for s in scenarios if s.id == "dental_new_patient_01")

        # Fabricate a sample call record
        from src.models import CallRecord
        record = CallRecord(
            call_id="test-001",
            scenario_id=scenario.id,
            iteration=0,
            candidate_index=-1,
            assistant_id="test",
            prompt_version="abcd1234",
            transcript=[
                TranscriptMessage("assistant", "Thank you for calling Bright Smile Dental. How can I help?"),
                TranscriptMessage("user", "Hi, I'd like to book a cleaning appointment please."),
                TranscriptMessage("assistant", "Of course! Can I get your name?"),
                TranscriptMessage("user", "Sure, it's Sarah Johnson."),
                TranscriptMessage("assistant", "Great, Sarah. What date works for you?"),
                TranscriptMessage("user", "How about next Tuesday afternoon?"),
                TranscriptMessage("assistant", "We have 2pm or 3pm on Tuesday. Which do you prefer?"),
                TranscriptMessage("user", "2pm please."),
                TranscriptMessage("assistant", "Perfect, I've booked you for Tuesday at 2pm. Do you have dental insurance?"),
                TranscriptMessage("user", "Yes, Delta Dental."),
                TranscriptMessage("assistant", "Great, we accept Delta Dental. See you Tuesday at 2pm!"),
                TranscriptMessage("user", "Thank you, goodbye!"),
            ],
            vapi_analysis={
                "appointment_booked": True,
                "collected_name": True,
                "collected_date": True,
                "collected_time": True,
                "collected_insurance": True,
                "escalation_offered": False,
                "caller_satisfied": True,
            },
            duration_seconds=45.0,
            ended_reason="completed",
        )

        scored = await judge.score(record, scenario)
        print(f"Final score: {scored.final_score}")
        print(f"LLM score:   {scored.llm_judge_result.composite_score}")
        print(f"Structured:  {scored.structured_score}")
        print(f"Passed:      {scored.llm_judge_result.passed}")
        print(f"Failures:    {[f.dimension + ': ' + f.reason for f in scored.llm_judge_result.failures]}")

    asyncio.run(_smoke())
