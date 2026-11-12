"""
Runs eval/cases/*.jsonl against the running system over HTTP, exercising the
full stack (Turn Handler, agent loop, tools).

Each case runs in its own session (fresh history) — POST /api/v1/session,
then POST /api/v1/chat with the X-Session-Token header. The server must run
with EXPOSE_EVAL_CONTEXT=true for retrieved_context to be captured.

Captures each case's actual output, retrieved context, tools called and
latency for the judge/metrics step to score. It does not auto-grade
expected_answer — that is free text and hard to compare exactly.

All paths are anchored to the project root, so the script behaves the same
no matter where it is launched from; every knob is a CLI flag.
"""

import argparse
import ipaddress
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpcore
import httpx

from scripts.eval_paths import input_path, output_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _PinnedNetworkBackend(httpcore.SyncBackend):
    """Resolve each socket destination once, validate it, and connect to that IP."""

    def __init__(self, allow_private: bool) -> None:
        self.allow_private = allow_private

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        addresses = list(
            dict.fromkeys(
                item[4][0]
                for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            )
        )
        if not addresses:
            raise OSError(f"API host resolved to no addresses: {host}")
        resolved = [
            ipaddress.ip_address(address.split("%", 1)[0]) for address in addresses
        ]
        if not self.allow_private and any(
            not address.is_global for address in resolved
        ):
            raise OSError(f"API host resolves to a non-public address: {host}")
        # Passing a numeric address to SyncBackend prevents a second DNS lookup
        # between validation and connect (the TLS layer still uses the URL host).
        last_error = None
        for address in resolved:
            try:
                return super().connect_tcp(
                    str(address), port, timeout, local_address, socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error


class _PinnedHTTPTransport(httpx.HTTPTransport):
    def __init__(self, allow_private: bool) -> None:
        super().__init__(trust_env=False)
        self._pool._network_backend = _PinnedNetworkBackend(allow_private)


def _public_addresses(host: str, addresses: set[str]) -> None:
    if host.lower().rstrip(".") in {"localhost", "metadata.google.internal"}:
        raise ValueError("API host is not permitted")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global:
            raise ValueError(f"API host resolves to a non-public address: {ip}")


def validate_api_url(url: str, *, allow_private: bool = False) -> str:
    """Require HTTP(S) and reject DNS answers that are not public by default."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL must use http:// or https:// and include a host")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("API URL must not contain credentials or a fragment")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname, port, type=socket.SOCK_STREAM
            )
        }
    except (OSError, ValueError) as exc:
        raise ValueError(f"API host could not be resolved: {parsed.hostname}") from exc
    if not addresses:
        raise ValueError("API host resolved to no addresses")
    if not allow_private:
        _public_addresses(parsed.hostname, addresses)
    return url.rstrip("/")


def load_cases(eval_dir: Path) -> list[dict]:
    """Load all evaluation cases from JSONL files."""
    cases = []
    for jsonl_file in sorted(eval_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def run_case(
    case: dict,
    client: httpx.Client,
    base_url: str,
    timeout: float,
    *,
    allow_private_api: bool = False,
) -> dict:
    start = time.perf_counter()
    try:
        base_url = validate_api_url(base_url, allow_private=allow_private_api)
        session_response = client.post(f"{base_url}/session", timeout=timeout)
        if 300 <= session_response.status_code < 400:
            raise ValueError("API redirects are not followed")
        session_response.raise_for_status()
        session_token = session_response.json()["session_token"]

        response = client.post(
            f"{base_url}/chat",
            json={"query": case["question"], "stream": False},
            headers={"X-Session-Token": session_token},
            timeout=timeout,
        )
        if 300 <= response.status_code < 400:
            raise ValueError("API redirects are not followed")
        response.raise_for_status()
        body = response.json()
        answer = body.get("answer")
        retrieved_context = body.get("retrieved_context", [])
        tools_called = body.get("tools_called", [])
        llm_calls = body.get("llm_calls")
        error = None
    except Exception as e:
        answer = None
        retrieved_context = []
        tools_called = []
        llm_calls = None
        error = str(e)
    elapsed_s = round(time.perf_counter() - start, 2)

    return {
        "id": case["id"],
        "category": case.get("category"),
        "difficulty": case.get("difficulty"),
        "question": case["question"],
        "expected_answer": case.get("expected_answer"),
        "expected_tools": case.get("expected_tools", []),
        "requires_refusal": case.get("requires_refusal", False),
        "hedge": case.get("hedge", False),
        "actual_answer": answer,
        "retrieved_context": retrieved_context,
        "tools_called": tools_called,
        "llm_calls": llm_calls,
        "error": error,
        "time_secs": elapsed_s,
    }


def _print_case_result(result: dict) -> None:
    if result["error"]:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  time={result['time_secs']}s tools={result['tools_called']}")
    print(f"  expected: {(result['expected_answer'] or '')[:100]}")
    print(f"  actual:   {(result['actual_answer'] or '')[:100]}")


def _run_cases(
    cases: list[dict],
    api_url: str,
    delay: float,
    timeout: float,
    allow_private_api: bool,
) -> list[dict]:
    results = []
    with httpx.Client(
        transport=_PinnedHTTPTransport(allow_private_api),
        follow_redirects=False,
        trust_env=False,
        timeout=timeout,
    ) as client:
        for index, case in enumerate(cases):
            print(f"[{case['id']}] ({case.get('category')}) {case['question']}")
            result = run_case(
                case,
                client,
                api_url,
                timeout,
                allow_private_api=allow_private_api,
            )
            results.append(result)
            _print_case_result(result)
            print()
            if index < len(cases) - 1 and delay > 0:
                time.sleep(delay)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1",
        help="Base URL of the running Yatra API (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-private-api",
        action="store_true",
        help="Allow private API addresses for an explicitly trusted local evaluation server",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "cases",
        help="Directory of *.jsonl eval cases",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results.jsonl",
        help="Output results file (JSONL)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between cases (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Per-request HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases"
    )
    args = parser.parse_args()

    try:
        api_url = validate_api_url(args.api_url, allow_private=args.allow_private_api)
        eval_dir = input_path(args.eval_dir)
        results_path = output_path(args.results)
    except ValueError as exc:
        parser.error(str(exc))

    cases = load_cases(eval_dir)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No cases found in {eval_dir}", file=sys.stderr)
        return 1

    print(f"Running {len(cases)} eval cases against {api_url}...\n")

    results = _run_cases(
        cases,
        api_url,
        args.delay,
        args.timeout,
        args.allow_private_api,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(results)} results to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
