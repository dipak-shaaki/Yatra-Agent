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
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpcore
import httpx

from scripts.eval_paths import input_path, output_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRUSTED_API_HOSTS_ENV = "YATRA_EVAL_ALLOWED_API_HOSTS"


@dataclass(frozen=True)
class ValidatedApiTarget:
    url: str
    host: str
    port: int
    addresses: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]


def _normalize_host(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if not host or "%" in host:
            raise ValueError("Invalid API host")
        return host.encode("idna").decode("ascii")


def _trusted_api_hosts() -> frozenset[str]:
    configured = os.environ.get(TRUSTED_API_HOSTS_ENV, "")
    return frozenset(
        _normalize_host(host) for host in configured.split(",") if host.strip()
    )


class _PinnedNetworkBackend(httpcore.SyncBackend):
    """Connect only to the validated DNS answers captured before the request."""

    def __init__(self, target: ValidatedApiTarget, allow_loopback: bool) -> None:
        self.target = target
        self.allow_loopback = allow_loopback

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        if _normalize_host(host) != self.target.host or port != self.target.port:
            raise OSError("Request destination does not match its validated API target")
        non_public = any(not address.is_global for address in self.target.addresses)
        loopback_target = all(address.is_loopback for address in self.target.addresses)
        if non_public and not (self.allow_loopback and loopback_target):
            raise OSError("API target is no longer permitted")
        last_error = None
        for address in self.target.addresses:
            try:
                return super().connect_tcp(
                    str(address), port, timeout, local_address, socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error


class _PinnedHTTPTransport(httpx.HTTPTransport):
    def __init__(self, target: ValidatedApiTarget, allow_loopback: bool) -> None:
        super().__init__(trust_env=False)
        self._pool._network_backend = _PinnedNetworkBackend(target, allow_loopback)


class _ApiOriginTransport(httpx.BaseTransport):
    """Route fixed relative API paths through a previously validated origin."""

    def __init__(self, target: ValidatedApiTarget, allow_loopback: bool) -> None:
        self._origin = httpx.URL(target.url)
        self._transport = _PinnedHTTPTransport(target, allow_loopback)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.url = request.url.copy_with(
            scheme=self._origin.scheme,
            host=self._origin.host,
            port=self._origin.port,
        )
        request.headers["Host"] = self._origin.netloc.decode("ascii")
        return self._transport.handle_request(request)

    def close(self) -> None:
        self._transport.close()


def validate_api_url(
    url: str,
    *,
    allow_loopback: bool = False,
    allowed_hosts: frozenset[str] | None = None,
) -> ValidatedApiTarget:
    """Require an allowlisted HTTPS API, except explicitly enabled loopback."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL must use http:// or https:// and include a host")
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError("API URL must not contain credentials, query, or fragment")
    host = _normalize_host(parsed.hostname)
    if host in {"metadata", "metadata.google.internal"}:
        raise ValueError("Cloud metadata hosts are not permitted")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        address_texts = dict.fromkeys(
            item[4][0]
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"API host could not be resolved: {host}") from exc
    if not address_texts:
        raise ValueError("API host resolved to no addresses")
    addresses = tuple(
        ipaddress.ip_address(address.split("%", 1)[0]) for address in address_texts
    )
    is_loopback_host = host == "localhost" or _is_loopback_ip(host)
    if is_loopback_host and allow_loopback and all(ip.is_loopback for ip in addresses):
        return ValidatedApiTarget(url.rstrip("/"), host, port, addresses)
    if is_loopback_host:
        raise ValueError("Loopback API hosts require --allow-local-api")
    if any(not ip.is_global for ip in addresses):
        raise ValueError(
            "API host resolves to localhost, private, link-local, or reserved IP"
        )
    if parsed.scheme != "https":
        raise ValueError("Public API URLs must use HTTPS")
    trusted_hosts = allowed_hosts if allowed_hosts is not None else _trusted_api_hosts()
    if host not in trusted_hosts:
        raise ValueError(
            f"API host is not allowlisted; configure {TRUSTED_API_HOSTS_ENV}"
        )
    return ValidatedApiTarget(url.rstrip("/"), host, port, addresses)


def _is_loopback_ip(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
    timeout: float,
) -> dict:
    start = time.perf_counter()
    try:
        session_response = client.post("/session", timeout=timeout)
        if 300 <= session_response.status_code < 400:
            raise ValueError("API redirects are not followed")
        session_response.raise_for_status()
        session_token = session_response.json()["session_token"]

        response = client.post(
            "/chat",
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
    target: ValidatedApiTarget,
    delay: float,
    timeout: float,
    allow_loopback: bool,
) -> list[dict]:
    results = []
    with httpx.Client(
        base_url="http://eval.invalid/api/v1",
        transport=_ApiOriginTransport(target, allow_loopback),
        follow_redirects=False,
        trust_env=False,
        timeout=timeout,
    ) as client:
        for index, case in enumerate(cases):
            print(f"[{case['id']}] ({case.get('category')}) {case['question']}")
            result = run_case(
                case,
                client,
                timeout,
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
        help=(
            "Base URL of the running Yatra API. Public hosts must be HTTPS and "
            f"listed in {TRUSTED_API_HOSTS_ENV}; default: %(default)s"
        ),
    )
    parser.add_argument(
        "--allow-local-api",
        "--allow-private-api",
        dest="allow_loopback",
        action="store_true",
        help="Allow a loopback API server; private LAN destinations stay blocked",
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
        target = validate_api_url(args.api_url, allow_loopback=args.allow_loopback)
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

    print(f"Running {len(cases)} eval cases against {target.url}...\n")

    results = _run_cases(
        cases,
        target,
        args.delay,
        args.timeout,
        args.allow_loopback,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(results)} results to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
