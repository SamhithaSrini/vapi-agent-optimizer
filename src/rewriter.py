"""Multi-candidate prompt rewriter with RLAIF self-critique.

RLAIF loop
----------
1. Self-critique  — the agent critiques its own worst transcripts to generate
                    a structured list of failure reasons (Constitutional AI style)
2. Digest merge   — external judge failures + self-critique failures are merged
3. Rewrite        — 3 candidates generated at different temperatures
4. Validate       — holdout gate selects or rejects the best candidate

The self-critique step makes the system explicitly RLAIF: AI feedback drives
the reward signal rather than hand-crafted heuristics or human labels.
"""
from __future__ import annotations

import asyncio

from src.aggregator import format_digest_yaml
from src.llm_client import LLMClient
from src.models import CallRecord, FailureGroup

_SELF_CRITIQUE_SYSTEM = """\
You are a voice AI agent that has just reviewed recordings of your own conversations.
Identify the most important mistakes YOU made as the agent.
Be specific and honest. Focus on what you could have done differently.
Return a plain bullet list (max 6 bullets). No preamble, no markdown headers.
Each bullet: one concrete failure and how to fix it.
"""


class PromptRewriter:
    def __init__(self, llm: LLMClient, config, rewriter_system_prompt: str):
        self._llm = llm
        self._cfg = config
        self._system = rewriter_system_prompt

    # ── RLAIF self-critique ───────────────────────────────────────────────────────

    async def self_critique(
        self,
        failed_records: list[CallRecord],
        current_prompt: str,
        max_transcripts: int = 3,
    ) -> str:
        """
        Ask the agent to critique its own worst transcripts.
        Returns a bullet-list string that gets merged into the rewriter context.

        This is the 'AI Feedback' in RLAIF: the model generates reward signal
        about its own outputs, which then guides the policy (prompt) update.
        """
        if not failed_records:
            return "(no failed calls to critique)"

        # Take the worst-scoring calls
        worst = sorted(
            [r for r in failed_records if r.final_score is not None],
            key=lambda r: r.final_score,
        )[:max_transcripts]

        transcript_blocks = []
        for r in worst:
            lines = "\n".join(
                f"  {'AGENT' if m.role == 'assistant' else 'CALLER'}: {m.message}"
                for m in r.transcript
            )
            transcript_blocks.append(
                f"[Scenario: {r.scenario_id}  Score: {r.final_score:.2f}]\n{lines}"
            )

        user_msg = (
            f"Here is your current system prompt:\n\n{current_prompt}\n\n"
            "Here are your worst-performing conversations from this iteration:\n\n"
            + "\n\n---\n\n".join(transcript_blocks)
            + "\n\nWhat did you do wrong? How should you change your behavior?"
        )

        try:
            critique = await self._llm.complete(
                messages=[
                    {"role": "system", "content": _SELF_CRITIQUE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                model=self._cfg.rewriter_model,
                temperature=0.3,
                max_tokens=400,
            )
            return critique.strip()
        except Exception:
            return "(self-critique unavailable)"

    # ── Candidate generation ──────────────────────────────────────────────────────

    async def generate_candidates(
        self,
        current_prompt: str,
        failure_digest: list[FailureGroup],
        memory_text: str,
        self_critique_text: str = "",
    ) -> list[str]:
        """Return N candidate prompts in parallel via temperature sweep."""
        digest_yaml = format_digest_yaml(failure_digest)
        user_msg = _build_rewriter_user_message(
            current_prompt, digest_yaml, memory_text, self_critique_text
        )

        payloads = [
            {"messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": user_msg},
            ]}
            for _ in range(self._cfg.n_candidates)
        ]

        candidates = await self._llm.complete_many(
            payloads=payloads,
            model=self._cfg.rewriter_model,
            temperatures=self._cfg.candidate_temperatures,
            max_tokens=1800,
        )
        return [c.strip() for c in candidates]

    # ── Change summarizer ─────────────────────────────────────────────────────────

    async def summarize_change(self, old_prompt: str, new_prompt: str) -> str:
        """One-sentence summary of the key change between two prompts."""
        msg = (
            "Summarize the single most important change between OLD and NEW system prompts "
            "in ONE concise sentence (max 20 words). Focus on what was added or changed.\n\n"
            f"OLD PROMPT:\n{old_prompt[:600]}\n\n"
            f"NEW PROMPT:\n{new_prompt[:600]}"
        )
        try:
            result = await self._llm.complete(
                messages=[{"role": "user", "content": msg}],
                model=self._cfg.summarizer_model,
                temperature=0.0,
                max_tokens=60,
            )
            return result.strip().rstrip(".")
        except Exception:
            return "Prompt updated"


def _build_rewriter_user_message(
    current_prompt: str,
    digest_yaml: str,
    memory_text: str,
    self_critique_text: str,
) -> str:
    critique_section = (
        f"\nAGENT SELF-CRITIQUE (AI feedback on its own worst calls):\n"
        f"{self_critique_text}\n"
        if self_critique_text and self_critique_text != "(no failed calls to critique)"
        else ""
    )
    return f"""\
CURRENT SYSTEM PROMPT:
---
{current_prompt}
---

CURRENT FAILURE PATTERNS (ranked by frequency × severity):
{digest_yaml}
{critique_section}
OPTIMIZATION MEMORY:
---
{memory_text}
---

Rewrite the system prompt to address the current failures.
"""
