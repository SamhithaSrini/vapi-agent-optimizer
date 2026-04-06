"""Test 2: Verify Vapi connection — create and delete a test assistant.

Run: python tests/test_vapi_connection.py
"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.config import load_config
from src.vapi_client import VapiClient


async def main():
    cfg = load_config()
    print("Testing Vapi API connection…")

    vapi = VapiClient(api_key=cfg.vapi_api_key)

    assistant = await vapi.create_assistant(
        "You are a friendly dental office receptionist.",
        label="connection-test",
    )
    aid = assistant["id"]
    print(f"  Created assistant: {aid}  name={assistant['name']}")

    assistants = await vapi.list_assistants()
    ids = [a["id"] for a in assistants]
    assert aid in ids, f"Assistant {aid} not found in list"
    print(f"  Verified in account ({len(ids)} total assistants)")

    await vapi.delete_assistant(aid)
    print(f"  Deleted assistant: {aid}")
    print("PASS: Vapi connection working.")


if __name__ == "__main__":
    asyncio.run(main())
