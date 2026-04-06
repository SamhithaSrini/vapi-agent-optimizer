"""Simulates voice call conversations between an LLM agent and an LLM caller.

Architecture note
-----------------
Real Vapi calls require phone/WebRTC infrastructure. To keep the system fully
automated and free to run, we simulate conversations by having two LLMs talk to
each other:

  - AGENT LLM  — uses the system prompt we are optimizing
  - CALLER LLM — uses the scenario persona as its system prompt

The Vapi assistant is still created via the real API (demonstrating assistant
management) but `POST /call` is not used. After the conversation we extract
structured data with a short LLM call (simulating Vapi's analysisPlan).
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict

from src.llm_client import LLMClient
from src.models import CallRecord, Scenario, TranscriptMessage


# ── Structured-data extraction prompt ────────────────────────────────────────────

_EXTRACT_SYSTEM = """You are an analyst reviewing a dental office call transcript.
Extract the following boolean fields from the conversation. Return ONLY valid JSON.

Fields:
  appointment_booked  — was an appointment actually booked or confirmed?
  collected_name      — did the agent collect the caller's name?
  collected_date      — was a specific date discussed or confirmed?
  collected_time      — was a specific time discussed or confirmed?
  collected_insurance — was insurance mentioned or collected?
  escalation_offered  — was the caller offered to speak with someone else or given special help?
  caller_satisfied    — did the caller appear satisfied at call end (said thanks, sounded positive)?

Return exactly: {"appointment_booked": bool, "collected_name": bool, ...}
"""

# Phrases that signal the conversation has naturally ended
_END_PHRASES = [
    "goodbye", "good bye", "bye bye", "take care", "have a great day",
    "have a good day", "thanks, bye", "thank you, bye", "that's all",
    "i think that covers it", "i'll call back", "wrong number",
]


class CallRunner:
    def __init__(self, llm: LLMClient, config):
        self._llm = llm
        self._cfg = config

    # ── Public API ────────────────────────────────────────────────────────────────

    async def run_batch(
        self,
        scenarios: list[Scenario],
        system_prompt: str,
        assistant_id: str,
        iteration: int,
        candidate_index: int = -1,
    ) -> list[CallRecord]:
        """Run all scenarios concurrently and return call records."""
        tasks = [
            self._run_one(s, system_prompt, assistant_id, iteration, candidate_index)
            for s in scenarios
        ]
        return list(await asyncio.gather(*tasks))

    # ── Single call ───────────────────────────────────────────────────────────────

    async def _run_one(
        self,
        scenario: Scenario,
        system_prompt: str,
        assistant_id: str,
        iteration: int,
        candidate_index: int,
    ) -> CallRecord:
        from src.models import prompt_version

        rng = random.Random(self._cfg.scenario_seed + hash(scenario.id))
        call_id = str(uuid.uuid4())
        start = time.monotonic()

        transcript = await self._simulate_conversation(scenario, system_prompt, rng)
        duration = time.monotonic() - start

        vapi_analysis = await self._extract_structured_data(transcript)

        return CallRecord(
            call_id=call_id,
            scenario_id=scenario.id,
            iteration=iteration,
            candidate_index=candidate_index,
            assistant_id=assistant_id,
            prompt_version=prompt_version(system_prompt),
            transcript=transcript,
            duration_seconds=round(duration, 2),
            ended_reason="completed",
            vapi_analysis=vapi_analysis,
        )

    # ── Conversation simulation ───────────────────────────────────────────────────

    async def _simulate_conversation(
        self, scenario: Scenario, system_prompt: str, rng: random.Random
    ) -> list[TranscriptMessage]:
        transcript: list[TranscriptMessage] = []
        t = 0.0  # simulated time in seconds

        # Build script hint for caller
        script_hint = "\n".join(f"  Turn {s.turn}: {s.line}" for s in scenario.script)

        caller_system = (
            f"You are {scenario.persona.name}, a real person calling a dental office.\n"
            f"Your goal: {scenario.persona.intent}\n"
            f"Your personality/behaviour: {scenario.persona.behaviour}\n\n"
            f"Loose script to guide you (don't follow robotically — react naturally):\n"
            f"{script_hint}\n\n"
            "Rules:\n"
            "1. Respond naturally in 1-3 sentences as the caller.\n"
            "2. When your goal is clearly achieved or clearly impossible, say goodbye and end the call.\n"
            "3. Do NOT narrate or explain — just speak as the caller would.\n"
            "4. If the agent asks a question, answer it realistically.\n"
            "5. Stay in character throughout."
        )

        agent_history: list[dict] = [{"role": "system", "content": system_prompt}]
        caller_history: list[dict] = [{"role": "system", "content": caller_system}]

        # Agent speaks first (greeting)
        agent_greeting = await self._llm.complete(
            messages=agent_history + [{"role": "user", "content": "[Call connected. Greet the caller.]"}],
            model=self._cfg.agent_model,
            temperature=0.5,
            max_tokens=120,
        )
        agent_greeting = agent_greeting.strip()
        transcript.append(TranscriptMessage("assistant", agent_greeting, t))
        agent_history.append({"role": "assistant", "content": agent_greeting})

        for _turn in range(self._cfg.max_turns_per_call):
            t += rng.uniform(4.0, 10.0)

            # Caller responds
            caller_history.append({"role": "user", "content": agent_greeting})
            caller_reply = await self._llm.complete(
                messages=caller_history,
                model=self._cfg.caller_model,
                temperature=0.7,
                max_tokens=150,
            )
            caller_reply = caller_reply.strip()
            transcript.append(TranscriptMessage("user", caller_reply, t))
            caller_history.append({"role": "assistant", "content": caller_reply})
            t += rng.uniform(3.0, 8.0)

            # Check for natural call end
            if _call_ended(caller_reply):
                # Let agent say goodbye
                agent_history.append({"role": "user", "content": caller_reply})
                farewell = await self._llm.complete(
                    messages=agent_history,
                    model=self._cfg.agent_model,
                    temperature=0.3,
                    max_tokens=60,
                )
                transcript.append(TranscriptMessage("assistant", farewell.strip(), t))
                break

            # Agent responds
            agent_history.append({"role": "user", "content": caller_reply})
            agent_reply = await self._llm.complete(
                messages=agent_history,
                model=self._cfg.agent_model,
                temperature=0.5,
                max_tokens=200,
            )
            agent_reply = agent_reply.strip()
            transcript.append(TranscriptMessage("assistant", agent_reply, t))
            agent_history.append({"role": "assistant", "content": agent_reply})
            agent_greeting = agent_reply  # next iteration: agent's last line is what caller sees

        return transcript

    # ── Structured data extraction (simulates Vapi analysisPlan) ─────────────────

    async def _extract_structured_data(
        self, transcript: list[TranscriptMessage]
    ) -> dict:
        convo = "\n".join(f"{m.role.upper()}: {m.message}" for m in transcript)
        try:
            raw = await self._llm.complete(
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": f"TRANSCRIPT:\n{convo}"},
                ],
                model=self._cfg.summarizer_model,
                temperature=0.0,
                max_tokens=200,
            )
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception:
            return {}


def _call_ended(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _END_PHRASES)


# ── Quick test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio, json
    from src.config import load_config
    from src.scenarios import load_scenarios

    async def _smoke():
        cfg = load_config()
        llm = LLMClient(api_key=cfg.llm_api_key, max_concurrent=cfg.max_concurrent_llm_calls)
        runner = CallRunner(llm, cfg)

        scenarios = load_scenarios(cfg.scenario_file)
        scenario = next(s for s in scenarios if s.id == "dental_new_patient_01")

        with open(cfg.initial_prompt_file) as f:
            prompt = f.read()

        print(f"Running scenario: {scenario.id}")
        record = await runner._run_one(scenario, prompt, "test-assistant-id", 0, -1)

        print(f"\nTranscript ({len(record.transcript)} turns, {record.duration_seconds:.1f}s simulated):")
        for msg in record.transcript:
            label = "AGENT" if msg.role == "assistant" else "CALLER"
            print(f"  [{label}] {msg.message[:120]}")

        print(f"\nVapi analysis: {json.dumps(record.vapi_analysis, indent=2)}")

    asyncio.run(_smoke())
