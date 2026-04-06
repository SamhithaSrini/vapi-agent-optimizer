"""Test 4: Run one simulated call AND score it with the hybrid judge.

Run: python tests/test_judge.py [scenario_id]
"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.call_runner import CallRunner
from src.config import load_config
from src.judge import HybridJudge
from src.llm_client import LLMClient
from src.scenarios import load_scenarios


async def main():
    scenario_id = sys.argv[1] if len(sys.argv) > 1 else "dental_new_patient_01"
    cfg = load_config()
    llm = LLMClient(api_key=cfg.llm_api_key, max_concurrent=cfg.max_concurrent_llm_calls)
    runner = CallRunner(llm, cfg)
    judge = HybridJudge(llm, cfg)

    scenarios = load_scenarios(cfg.scenario_file)
    scenario = next((s for s in scenarios if s.id == scenario_id), None)
    if scenario is None:
        print(f"Scenario '{scenario_id}' not found.")
        sys.exit(1)

    with open(cfg.initial_prompt_file) as f:
        prompt = f.read().strip()

    print(f"Running + judging scenario: {scenario.id}\n")
    record = await runner._run_one(scenario, prompt, "test-assistant", 0, -1)
    record = await judge.score(record, scenario)

    j = record.llm_judge_result
    print(f"{'─'*50}")
    print(f"Final score (hybrid) : {record.final_score:.3f}")
    print(f"LLM rubric score     : {j.composite_score:.3f}")
    print(f"Structured score     : {record.structured_score:.3f}")
    print(f"Passed               : {j.passed}")
    print(f"\nDimension scores:")
    for dim, val in j.dimension_scores.as_dict().items():
        bar = "█" * int(val * 20)
        print(f"  {dim:<25} {val:.3f}  {bar}")

    if j.failures:
        print(f"\nFailures:")
        for f in j.failures:
            print(f"  [{f.severity.upper()}] {f.dimension}: {f.reason}")
    else:
        print("\nNo failures detected.")


if __name__ == "__main__":
    asyncio.run(main())
