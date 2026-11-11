"""
Smoke test for a running Yatra instance — no Groq key needed.

Checks the deployment plumbing, not answer quality (that is what
run_eval.py + eval_answers.py are for): liveness, readiness (Redis + Chroma
answering), session minting, auth enforcement, and the OpenAPI surface.

Run as `python scripts/smoke_test.py` (add --api-url to target a remote
instance); exits nonzero on the first failed check.
"""

import argparse
import sys

import httpx


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running instance (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    client = httpx.Client(base_url=args.api_url, timeout=args.timeout)
    failures = 0

    print(f"Smoke-testing {args.api_url} (no LLM calls)\n")

    response = client.get("/health")
    failures += not check(
        "/health liveness", response.status_code == 200, response.text.strip()
    )

    response = client.get("/ready")
    failures += not check(
        "/ready (Redis + Chroma answer)",
        response.status_code == 200,
        response.text.strip(),
    )

    response = client.post("/api/v1/session")
    token_ok = response.status_code == 200 and response.json().get("session_token")
    failures += not check("POST /session mints a token", bool(token_ok))
    token = response.json().get("session_token") if token_ok else None

    response = client.post("/api/v1/chat", json={"query": "hi", "stream": False})
    failures += not check(
        "/chat without a session token is rejected", response.status_code == 401
    )

    response = client.get("/openapi.json")
    schema_ok = response.status_code == 200 and "api/v1" in response.text
    failures += not check("OpenAPI schema served", schema_ok)

    if token:
        # A session exists — /chat would now call Groq. Rate-limit plumbing
        # is already covered by the unit tests; stop before LLM calls so
        # this stays runnable without a valid key.
        print("\nSession token verified; skipping /chat (needs a valid GROQ_API_KEY).")

    print(f"\n{'FAILED' if failures else 'All checks passed'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
