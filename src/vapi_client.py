"""Thin async Vapi REST client with retry logic.

We use Vapi to:
  - CREATE assistants (POST /assistant) — stores the optimized prompt config
  - DELETE assistants (DELETE /assistant/{id}) — cleanup after each iteration
  - GET assistant list (GET /assistant) — verification

Conversations are simulated locally (see call_runner.py), so we do NOT call
POST /call. This means no call credits are consumed. Assistant CRUD is free.
"""
from __future__ import annotations

import os
import uuid

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

VAPI_BASE = "https://api.vapi.ai"


class VapiClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("VAPI_API_KEY")
        if not key:
            raise ValueError("VAPI_API_KEY not set")
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    # ── Assistant management ──────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def create_assistant(self, system_prompt: str, label: str = "") -> dict:
        """Create a named assistant with the given system prompt. Returns the full assistant object."""
        name = f"optimizer-{label or uuid.uuid4().hex[:6]}"
        body = {
            "name": name,
            "model": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt}
                ],
                "temperature": 0.7,
                "maxTokens": 500,
            },
            "voice": {
                "provider": "openai",
                "voiceId": "nova",
            },
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-2",
                "language": "en",
            },
            "firstMessage": "Thank you for calling Bright Smile Dental. How can I help you today?",
            "analysisPlan": {
                "structuredDataSchema": {
                    "type": "object",
                    "properties": {
                        "appointment_booked":  {"type": "boolean"},
                        "collected_name":      {"type": "boolean"},
                        "collected_date":      {"type": "boolean"},
                        "collected_time":      {"type": "boolean"},
                        "collected_insurance": {"type": "boolean"},
                        "escalation_offered":  {"type": "boolean"},
                        "caller_satisfied":    {"type": "boolean"},
                    },
                },
                "successEvaluationPlan": {
                    "rubric": "NumericScale",
                },
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{VAPI_BASE}/assistant", headers=self._headers, json=body)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def delete_assistant(self, assistant_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.delete(
                f"{VAPI_BASE}/assistant/{assistant_id}", headers=self._headers
            )
            # 404 is fine — already deleted
            if r.status_code not in (200, 204, 404):
                r.raise_for_status()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def list_assistants(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{VAPI_BASE}/assistant", headers=self._headers)
            r.raise_for_status()
            return r.json()


if __name__ == "__main__":
    import asyncio

    async def _smoke():
        client = VapiClient()
        print("Creating test assistant…")
        assistant = await client.create_assistant(
            "You are a helpful dental office receptionist.", label="smoke-test"
        )
        aid = assistant["id"]
        print(f"  Created: {aid}  name={assistant['name']}")

        assistants = await client.list_assistants()
        ids = [a["id"] for a in assistants]
        print(f"  Total assistants in account: {len(ids)}")
        assert aid in ids, "Newly created assistant not found in list"

        await client.delete_assistant(aid)
        print(f"  Deleted: {aid}")
        print("Vapi smoke test passed.")

    asyncio.run(_smoke())
