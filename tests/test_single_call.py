"""Test 3: Run one simulated conversation and print the transcript.

Run: python tests/test_single_call.py [scenario_id]
Default scenario: dental_new_patient_01
"""
import asyncio
import json
import sys
sys.path.insert(0, ".")

from src.call_runner import CallRunner
from src.config import load_config
from src.llm_client import LLMClient
from src.scenarios import load_scenarios


async def main():
    scenario_id = sys.argv[1] if len(sys.argv) > 1 else "dental_new_patient_01"
    cfg = load_config()
    llm = LLMClient(api_key=cfg.llm_api_key, max_concurrent=cfg.max_concurrent_llm_calls)
    runner = CallRunner(llm, cfg)

    scenarios = load_scenarios(cfg.scenario_file)
    scenario = next((s for s in scenarios if s.id == scenario_id), None)
    if scenario is None:
        print(f"Scenario '{scenario_id}' not found. Available IDs:")
        for s in scenarios:
            print(f"  {s.id}")
        sys.exit(1)

    with open(cfg.initial_prompt_file) as f:
        prompt = f.read().strip()

    print(f"\nRunning scenario: {scenario.id} ({scenario.difficulty})")
    print(f"Persona: {scenario.persona.name} — {scenario.persona.intent}\n")

    record = await runner._run_one(scenario, prompt, "test-assistant", 0, -1)

    print(f"{'─'*60}")
    for msg in record.transcript:
        label = "AGENT " if msg.role == "assistant" else "CALLER"
        print(f"[{label}] {msg.message}")
    print(f"{'─'*60}")
    print(f"Duration : {record.duration_seconds:.1f}s  |  Turns: {len(record.transcript)}")
    print(f"Vapi analysis: {json.dumps(record.vapi_analysis, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
