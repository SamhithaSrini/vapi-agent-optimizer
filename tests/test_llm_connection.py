"""Test 1: Verify OpenRouter connection with a free model.

Run: python tests/test_llm_connection.py
"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.config import load_config
from src.llm_client import LLMClient


async def main():
    cfg = load_config()
    provider = "OpenAI" if cfg.openai_api_key else "OpenRouter"
    print(f"Testing {provider} with model: {cfg.caller_model}")

    llm = LLMClient(api_key=cfg.llm_api_key, max_concurrent=2)
    response = await llm.complete(
        messages=[{"role": "user", "content": "Reply with exactly: CONNECTION_OK"}],
        model=cfg.caller_model,
        temperature=0.0,
        max_tokens=20,
    )
    print(f"Response: {repr(response)}")
    assert "CONNECTION_OK" in response or len(response) > 0, "Empty response"
    print("PASS: LLM connection working.")


if __name__ == "__main__":
    asyncio.run(main())
