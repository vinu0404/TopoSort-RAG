"""
Quick HITL test — streams a query, catches hitl_required, approves it, waits for result.
Usage: python test_hitl.py
"""
import httpx
import json
import threading
import time

BASE = "http://127.0.0.1:8000/api/v1"
TOKEN = "eyJ1c2VyX2lkIjogIjMzZDIyMDMzLTFjYWEtNDc0ZS1hOWFkLWM4ZTJhMjA4YmY0YiIsICJleHAiOiAxNzcxMzcyNjg1fQ==.5ce055060e14c9a4dd1fdcf7a590b0a4b28c1d6a3855bee70b5445fbc4a33e96"
USER_ID = "33d22033-1caa-474e-a9ad-c8e2a208bf4b"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# Will be set when we get the hitl_required event
hitl_request_id = None
hitl_received = threading.Event()


def approve_hitl():
    """Waits for hitl_required, then POSTs approval."""
    print("\n⏳ Waiting for HITL event...")
    hitl_received.wait(timeout=60)
    if not hitl_request_id:
        print("❌ Never got hitl_required event")
        return

    time.sleep(1)  # small delay so we can see the flow
    print(f"\n❌ Denying request: {hitl_request_id}")
    r = httpx.post(
        f"{BASE}/hitl/respond",
        headers=HEADERS,
        json={
            "request_id": hitl_request_id,
            "decision": "denied",
            "instructions": None,
        },
    )
    print(f"   Response: {r.status_code} {r.json()}")


def main():
    global hitl_request_id

    # Start approval thread
    approver = threading.Thread(target=approve_hitl, daemon=True)
    approver.start()

    print("🚀 Sending streaming query: 'search for latest AI news'")
    print("=" * 60)

    with httpx.stream(
        "POST",
        f"{BASE}/query/stream",
        headers=HEADERS,
        json={"query": "search for latest AI news", "user_id": USER_ID},
        timeout=180,
    ) as response:
        event_type = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: ") and event_type:
                raw = line[6:]
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = raw

                # Print all events
                if event_type == "token":
                    print(data.get("text", ""), end="", flush=True)
                elif event_type == "hitl_required":
                    print(f"\n🔒 HITL REQUIRED:")
                    print(f"   request_id:  {data['request_id']}")
                    print(f"   agent:       {data['agent_name']}")
                    print(f"   tools:       {data['tool_names']}")
                    print(f"   task:        {data['task_description']}")
                    print(f"   timeout:     {data['timeout_seconds']}s")
                    hitl_request_id = data["request_id"]
                    hitl_received.set()
                elif event_type == "hitl_approved":
                    print(f"\n✅ HITL APPROVED — agent resuming")
                elif event_type == "hitl_denied":
                    print(f"\n❌ HITL DENIED — agent skipped")
                elif event_type == "hitl_timeout":
                    print(f"\n⏰ HITL TIMEOUT — agent skipped")
                elif event_type == "done":
                    print(f"\n\n{'=' * 60}")
                    print("✅ Stream complete")
                else:
                    print(f"[{event_type}] {json.dumps(data, indent=2) if isinstance(data, dict) else data}")

                event_type = None

    approver.join(timeout=5)


if __name__ == "__main__":
    main()
