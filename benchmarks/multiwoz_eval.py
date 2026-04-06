"""
MultiWOZ 2.2 Benchmark Evaluation
==================================
Evaluates our hybrid judge against MultiWOZ restaurant booking dialogs, where
ground-truth task success labels are derived from slot annotations.

Purpose
-------
1. Validate judge calibration: does our composite score correctly separate
   successful from failed dialogs?
2. Identify systematic biases: which judge dimensions are mis-weighted?
3. Extract failure patterns: what common failures in MultiWOZ aren't covered
   by our dental scenarios?
4. Apply insights: adjust judge weights / scenario library before running
   the optimizer.

Usage
-----
    python benchmarks/multiwoz_eval.py             # run full evaluation
    python benchmarks/multiwoz_eval.py --n 30      # sample 30 dialogs
    python benchmarks/multiwoz_eval.py --apply     # apply calibration fixes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import urllib.request
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.judge import HybridJudge
from src.llm_client import LLMClient
from src.models import CallRecord, TranscriptMessage

# ── Data loading ──────────────────────────────────────────────────────────────

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_BASE = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2"
_TEST_SHARDS = ["test/dialogues_001.json", "test/dialogues_002.json"]

# Slots that must be filled for a restaurant booking to be considered complete
_REQUIRED_RESTAURANT_SLOTS = {
    "restaurant-name",
    "restaurant-book_day",
    "restaurant-book_time",
    "restaurant-book_people",
}

# Booking success keywords in system utterances
_BOOKING_SUCCESS_PHRASES = [
    "booking was successful",
    "reference number",
    "reservation has been made",
    "table has been booked",
    "booked a table",
    "booking reference",
]


def fetch_shard(path: str) -> list[dict]:
    url = f"{_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as r:
        return json.loads(r.read())


def load_restaurant_dialogs(n: int = 60) -> list[dict]:
    """Fetch restaurant-domain dialogs from MultiWOZ 2.2 test set."""
    dialogs = []
    for shard in _TEST_SHARDS:
        print(f"  Fetching {shard}…")
        for d in fetch_shard(shard):
            if "restaurant" in d.get("services", []):
                dialogs.append(d)
        if len(dialogs) >= n:
            break
    print(f"  Found {len(dialogs)} restaurant dialogs")
    return dialogs[:n]


# ── Ground truth extraction ───────────────────────────────────────────────────

def extract_ground_truth(dialog: dict) -> dict:
    """
    Derive task success label and slot-fill status from frame annotations.
    Returns:
      task_success: bool   — did the system successfully complete the booking?
      slots_filled: dict   — which required slots were collected
      booking_confirmed: bool — explicit booking confirmation phrase present
      n_turns: int         — conversation length
    """
    all_slot_values: dict[str, list[str]] = {}
    booking_confirmed = False

    for turn in dialog["turns"]:
        # Check system utterance for booking confirmation
        if turn["speaker"] == "SYSTEM":
            utt_lower = turn["utterance"].lower()
            if any(p in utt_lower for p in _BOOKING_SUCCESS_PHRASES):
                booking_confirmed = True

        # Aggregate slot values from user frames
        for frame in turn.get("frames", []):
            if frame.get("service") == "restaurant":
                for slot, vals in frame.get("state", {}).get("slot_values", {}).items():
                    all_slot_values[slot] = vals

    slots_filled = {s: s in all_slot_values for s in _REQUIRED_RESTAURANT_SLOTS}
    slots_filled_count = sum(slots_filled.values())
    task_success = booking_confirmed or slots_filled_count >= 3  # lenient: 3/4 slots + any evidence

    return {
        "task_success": task_success,
        "slots_filled": slots_filled,
        "slots_filled_count": slots_filled_count,
        "booking_confirmed": booking_confirmed,
        "n_turns": len(dialog["turns"]),
        "all_slot_values": {k: v[0] if v else "" for k, v in all_slot_values.items()},
    }


# ── Format conversion ─────────────────────────────────────────────────────────

def dialog_to_call_record(dialog: dict) -> CallRecord:
    """Convert a MultiWOZ dialog to our CallRecord format."""
    transcript = []
    t = 0.0
    for turn in dialog["turns"]:
        role = "assistant" if turn["speaker"] == "SYSTEM" else "user"
        transcript.append(TranscriptMessage(role=role, message=turn["utterance"], time=t))
        t += 8.0

    return CallRecord(
        call_id=dialog["dialogue_id"],
        scenario_id=dialog["dialogue_id"],
        iteration=-1,
        candidate_index=-1,
        assistant_id="multiwoz",
        prompt_version="multiwoz-2.2",
        transcript=transcript,
        duration_seconds=t,
        ended_reason="completed",
        vapi_analysis=None,
    )


# ── Domain-agnostic judge scenario stub ──────────────────────────────────────

class _MultiWOZScenario:
    """Minimal scenario object for the generic judge."""
    def __init__(self, dialog_id: str, gt: dict):
        self.id = dialog_id
        self.split = "test"
        self.category = "booking"
        self.difficulty = "medium"
        self.persona = type("P", (), {
            "name": "caller",
            "intent": "Book a table at a restaurant",
            "behaviour": "cooperative",
        })()
        self.expected_outcome = "Restaurant booked with date, time, and party size confirmed"
        # Map required restaurant slots to our structured field names
        self.expected_structured_fields = []
        if gt["slots_filled"].get("restaurant-name"):
            self.expected_structured_fields.append("collected_name")
        if gt["slots_filled"].get("restaurant-book_day"):
            self.expected_structured_fields.append("collected_date")
        if gt["slots_filled"].get("restaurant-book_time"):
            self.expected_structured_fields.append("collected_time")
        if gt["booking_confirmed"]:
            self.expected_structured_fields.append("appointment_booked")
        self.known_edge_cases = []
        self.script = []


_GENERIC_JUDGE_SYSTEM = """\
You are an expert evaluator for task-oriented voice/chat AI agents.
You will receive a conversation transcript and context about what the user wanted.
Score the SYSTEM agent on the rubric below. Return ONLY valid JSON — no prose, no markdown fences.

Rubric dimensions (each scored 0.0–1.0):
  goal_completion      — Did the system achieve the user's stated goal (booking, info request, etc.)?
  tone_and_empathy     — Was the system helpful, professional, and courteous?
  information_accuracy — Were the facts provided correct and complete?
  efficiency           — Was the task resolved in a reasonable number of turns?
  edge_case_handling   — Did the system handle unexpected requests or clarifications gracefully?

Composite = 0.35*goal + 0.20*tone + 0.20*accuracy + 0.15*efficiency + 0.10*edge
passed = composite >= 0.70

Return exactly:
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
  "failures": [{"dimension": <str>, "reason": <str>, "severity": "low"|"med"|"high"}]
}"""


# ── Calibration analysis ──────────────────────────────────────────────────────

def calibration_report(results: list[dict]) -> dict:
    """
    Compute calibration metrics: how well does our judge score match
    MultiWOZ ground truth task success labels?
    """
    tp = sum(1 for r in results if r["gt_success"] and r["judge_passed"])
    tn = sum(1 for r in results if not r["gt_success"] and not r["judge_passed"])
    fp = sum(1 for r in results if not r["gt_success"] and r["judge_passed"])
    fn = sum(1 for r in results if r["gt_success"] and not r["judge_passed"])
    total = len(results)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Score distributions by ground truth label
    success_scores = [r["composite_score"] for r in results if r["gt_success"]]
    fail_scores = [r["composite_score"] for r in results if not r["gt_success"]]

    # Per-dimension bias: how much does each dimension deviate from composite for correct vs wrong predictions
    dim_names = ["goal_completion", "tone_and_empathy", "information_accuracy", "efficiency", "edge_case_handling"]
    dim_means_success = {}
    dim_means_fail = {}
    for dim in dim_names:
        s_scores = [r["dimension_scores"].get(dim, 0) for r in results if r["gt_success"]]
        f_scores = [r["dimension_scores"].get(dim, 0) for r in results if not r["gt_success"]]
        dim_means_success[dim] = mean(s_scores) if s_scores else 0
        dim_means_fail[dim] = mean(f_scores) if f_scores else 0

    return {
        "n": total,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "gt_success_rate": round(sum(r["gt_success"] for r in results) / total, 3),
        "judge_pass_rate": round(sum(r["judge_passed"] for r in results) / total, 3),
        "score_mean_when_gt_success": round(mean(success_scores), 3) if success_scores else 0,
        "score_mean_when_gt_fail": round(mean(fail_scores), 3) if fail_scores else 0,
        "score_separation": round(
            (mean(success_scores) - mean(fail_scores)) if success_scores and fail_scores else 0, 3
        ),
        "dim_means_success": {k: round(v, 3) for k, v in dim_means_success.items()},
        "dim_means_fail": {k: round(v, 3) for k, v in dim_means_fail.items()},
    }


def extract_insights(calibration: dict, results: list[dict]) -> dict:
    """
    Derive actionable insights from calibration results.
    Returns a dict of findings and recommended changes.
    """
    insights = {}

    # 1. Is our pass threshold (0.7) too strict or too lenient?
    score_gap = calibration["score_mean_when_gt_success"] - calibration["score_mean_when_gt_fail"]
    optimal_threshold = (
        calibration["score_mean_when_gt_success"] + calibration["score_mean_when_gt_fail"]
    ) / 2
    insights["threshold_analysis"] = {
        "current": 0.70,
        "suggested": round(optimal_threshold, 2),
        "score_separation": score_gap,
        "recommendation": (
            "threshold is well-calibrated" if abs(optimal_threshold - 0.70) < 0.05
            else f"consider adjusting pass threshold to {optimal_threshold:.2f}"
        ),
    }

    # 2. Which dimensions are most discriminative?
    dim_discriminability = {}
    for dim in calibration["dim_means_success"]:
        gap = calibration["dim_means_success"][dim] - calibration["dim_means_fail"][dim]
        dim_discriminability[dim] = round(gap, 3)
    most_discriminative = max(dim_discriminability, key=lambda k: dim_discriminability[k])
    least_discriminative = min(dim_discriminability, key=lambda k: dim_discriminability[k])
    insights["dimension_discriminability"] = dim_discriminability
    insights["most_discriminative_dim"] = most_discriminative
    insights["least_discriminative_dim"] = least_discriminative

    # 3. Common failure patterns in MultiWOZ we might be under-representing
    false_negative_records = [r for r in results if r["gt_success"] and not r["judge_passed"]]
    false_positive_records = [r for r in results if not r["gt_success"] and r["judge_passed"]]
    insights["false_negative_count"] = len(false_negative_records)  # successes we marked failed
    insights["false_positive_count"] = len(false_positive_records)  # failures we marked passed
    insights["fn_avg_turns"] = round(mean(r["n_turns"] for r in false_negative_records), 1) if false_negative_records else 0
    insights["fp_avg_turns"] = round(mean(r["n_turns"] for r in false_positive_records), 1) if false_positive_records else 0

    # 4. New scenario types to add
    scenario_gaps = []
    if calibration["score_mean_when_gt_fail"] > 0.60:
        scenario_gaps.append("Add more challenging failure cases — judge scores failed dialogs too high")
    if calibration["fp"] > calibration["n"] * 0.20:
        scenario_gaps.append("Add failure scenarios with polite-but-ineffective agents (false positive rate high)")
    if dim_discriminability.get("efficiency", 0) < 0.05:
        scenario_gaps.append("Add efficiency-testing scenarios (long rambling calls that still complete the task)")
    insights["scenario_gaps"] = scenario_gaps

    return insights


def apply_calibration(insights: dict, config_path: str = "config.yaml") -> None:
    """
    Optionally update config.yaml with calibrated threshold.
    Only updates if the suggested threshold differs from current by > 0.05.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    suggested = insights["threshold_analysis"]["suggested"]
    current = 0.70
    if abs(suggested - current) > 0.05:
        # Update the judge threshold via the rewriter's min_delta reference
        # (our system doesn't store the 0.7 threshold in config.yaml directly,
        #  it's in judge.py — so we just report the recommendation)
        print(f"\n[CALIBRATION] Suggested judge pass threshold: {suggested}")
        print(f"  Current threshold in judge.py: {current}")
        print(f"  Updating target_score in config.yaml to {max(0.75, suggested):.2f}")
        cfg["target_score"] = round(max(0.75, suggested), 2)
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print("  config.yaml updated.")
    else:
        print(f"\n[CALIBRATION] Threshold {current} is well-calibrated (suggested: {suggested}). No changes.")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_evaluation(n: int = 50, apply: bool = False) -> None:
    cfg = load_config()
    llm = LLMClient(api_key=cfg.llm_api_key, max_concurrent=cfg.max_concurrent_llm_calls)

    # Use generic judge system prompt for domain-agnostic evaluation
    from src.judge import HybridJudge, _strip_fences
    import copy

    class GenericJudge(HybridJudge):
        """Judge variant with domain-agnostic rubric for MultiWOZ evaluation."""
        async def score(self, record, scenario):
            import json as _json
            from src.models import JudgeDimensionScores, JudgeFailure, JudgeResult
            transcript_text = "\n".join(
                f"{'SYSTEM' if m.role == 'assistant' else 'USER'}: {m.message}"
                for m in record.transcript
            )
            user_msg = (
                f"USER GOAL: {scenario.expected_outcome}\n\n"
                f"TRANSCRIPT:\n{transcript_text}"
            )
            try:
                raw = await self._llm.complete(
                    messages=[
                        {"role": "system", "content": _GENERIC_JUDGE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    model=self._cfg.judge_model,
                    temperature=self._cfg.judge_temperature,
                    max_tokens=600,
                )
                raw = _strip_fences(raw)
                parsed = _json.loads(raw)
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
                        JudgeFailure(f["dimension"], f["reason"], f["severity"])
                        for f in parsed.get("failures", [])
                    ],
                )
            except Exception as e:
                judge_result = JudgeResult(
                    composite_score=0.5, passed=False,
                    dimension_scores=JudgeDimensionScores(0.5, 0.5, 0.5, 0.5, 0.5),
                    failures=[JudgeFailure("system", str(e), "med")],
                )

            # structured_score: use slot fill rate as proxy
            expected = scenario.expected_structured_fields
            filled = scenario.slots_filled_count if hasattr(scenario, "slots_filled_count") else 0
            total_exp = len(expected) if expected else 1
            structured = min(1.0, filled / total_exp) if expected else 0.5

            record.llm_judge_result = judge_result
            record.structured_score = round(structured, 4)
            record.final_score = round(
                self._cfg.llm_judge_weight * judge_result.composite_score
                + self._cfg.structured_score_weight * structured, 4
            )
            return record

    judge = GenericJudge(llm, cfg)

    # Load and process dialogs
    print(f"\nFetching {n} MultiWOZ 2.2 restaurant dialogs…")
    dialogs = load_restaurant_dialogs(n)

    results = []
    print(f"\nScoring {len(dialogs)} dialogs…")
    for i, dialog in enumerate(dialogs):
        gt = extract_ground_truth(dialog)
        record = dialog_to_call_record(dialog)
        scenario = _MultiWOZScenario(dialog["dialogue_id"], gt)
        scenario.slots_filled_count = gt["slots_filled_count"]

        scored = await judge.score(record, scenario)
        results.append({
            "dialog_id": dialog["dialogue_id"],
            "gt_success": gt["task_success"],
            "gt_booking_confirmed": gt["booking_confirmed"],
            "gt_slots_filled": gt["slots_filled_count"],
            "n_turns": gt["n_turns"],
            "composite_score": scored.final_score,
            "judge_passed": scored.llm_judge_result.passed,
            "dimension_scores": scored.llm_judge_result.dimension_scores.as_dict(),
            "structured_score": scored.structured_score,
        })
        if (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(dialogs)}…")

    # Calibration analysis
    cal = calibration_report(results)
    insights = extract_insights(cal, results)

    # Print report
    print("\n" + "═" * 60)
    print("MULTIWOZ BENCHMARK CALIBRATION REPORT")
    print("═" * 60)
    print(f"\nDataset: MultiWOZ 2.2 — restaurant domain ({cal['n']} dialogs)")
    print(f"\n── Classification metrics ──────────────────────────────────")
    print(f"  Accuracy  : {cal['accuracy']:.3f}")
    print(f"  Precision : {cal['precision']:.3f}")
    print(f"  Recall    : {cal['recall']:.3f}")
    print(f"  F1        : {cal['f1']:.3f}")
    print(f"  GT success rate   : {cal['gt_success_rate']:.1%}")
    print(f"  Judge pass rate   : {cal['judge_pass_rate']:.1%}")
    print(f"\n── Score distributions ─────────────────────────────────────")
    print(f"  Mean score when GT=success : {cal['score_mean_when_gt_success']:.3f}")
    print(f"  Mean score when GT=fail    : {cal['score_mean_when_gt_fail']:.3f}")
    print(f"  Separation                 : {cal['score_separation']:.3f}")
    print(f"\n── Threshold calibration ───────────────────────────────────")
    t = insights['threshold_analysis']
    print(f"  Current threshold  : {t['current']}")
    print(f"  Suggested threshold: {t['suggested']}")
    print(f"  Recommendation     : {t['recommendation']}")
    print(f"\n── Dimension discriminability (success - fail score gap) ───")
    for dim, gap in sorted(insights['dimension_discriminability'].items(), key=lambda x: -x[1]):
        bar = "█" * int(gap * 30) if gap > 0 else ""
        print(f"  {dim:<25} {gap:+.3f}  {bar}")
    print(f"\n  Most discriminative : {insights['most_discriminative_dim']}")
    print(f"  Least discriminative: {insights['least_discriminative_dim']}")
    print(f"\n── Confusion matrix ────────────────────────────────────────")
    print(f"  True Positive  (correctly identified success): {cal['tp']}")
    print(f"  True Negative  (correctly identified failure): {cal['tn']}")
    print(f"  False Positive (failure scored as success)  : {cal['fp']}")
    print(f"  False Negative (success scored as failure)  : {cal['fn']}")
    print(f"\n── Scenario gaps identified ────────────────────────────────")
    if insights['scenario_gaps']:
        for gap in insights['scenario_gaps']:
            print(f"  • {gap}")
    else:
        print("  No significant gaps identified.")

    # Save results
    output = {
        "calibration": cal,
        "insights": insights,
        "per_dialog_results": results,
    }
    Path("benchmarks").mkdir(exist_ok=True)
    with open("benchmarks/multiwoz_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to benchmarks/multiwoz_results.json")

    if apply:
        apply_calibration(insights)

    return insights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="Number of dialogs to evaluate")
    parser.add_argument("--apply", action="store_true", help="Apply calibration to config")
    args = parser.parse_args()
    asyncio.run(run_evaluation(n=args.n, apply=args.apply))
