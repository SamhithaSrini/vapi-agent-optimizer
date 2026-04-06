"""Async LLM client — supports OpenAI directly or OpenRouter as fallback.

Provider selection (checked in order):
  1. If OPENAI_API_KEY is set in env → use OpenAI API (api.openai.com)
  2. Else → use OpenRouter (openrouter.ai/api/v1)

This means you never need to change this file — just set the right env var.
"""
from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


def _make_client() -> AsyncOpenAI:
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        return AsyncOpenAI(api_key=openai_key)  # native OpenAI endpoint
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
    )


class LLMClient:
    def __init__(self, api_key: str | None = None, max_concurrent: int = 4):
        if api_key:
            # Explicit key: assume OpenAI if it starts with sk-proj, else OpenRouter
            if api_key.startswith("sk-proj") or api_key.startswith("sk-"):
                self._client = AsyncOpenAI(api_key=api_key)
            else:
                self._client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1", api_key=api_key
                )
        else:
            self._client = _make_client()
        self._sem = asyncio.Semaphore(max_concurrent)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        if _no_system_role(model):
            messages = _merge_system_into_user(messages)
        async with self._sem:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

    async def complete_many(
        self,
        payloads: list[dict],
        model: str,
        temperatures: list[float],
        max_tokens: int = 2048,
    ) -> list[str]:
        """Run N completions in parallel with different temperatures."""
        tasks = [
            self.complete(p["messages"], model, t, max_tokens)
            for p, t in zip(payloads, temperatures)
        ]
        return list(await asyncio.gather(*tasks))


_NO_SYSTEM_ROLE_PREFIXES = ("google/gemma",)


def _no_system_role(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _NO_SYSTEM_ROLE_PREFIXES)


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    result = []
    system_parts: list[str] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            result.append(m)
    if system_parts and result:
        preamble = "\n\n".join(system_parts)
        first = result[0]
        if first["role"] == "user":
            result[0] = {"role": "user", "content": preamble + "\n\n" + first["content"]}
        else:
            result.insert(0, {"role": "user", "content": preamble})
    return result


if __name__ == "__main__":
    import asyncio

    async def _smoke():
        client = LLMClient()
        result = await client.complete(
            messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=20,
        )
        print("LLM response:", repr(result))

    asyncio.run(_smoke())
