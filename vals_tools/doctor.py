"""Check your setup, one line per check: credentials, the AC2 project, its datasets, the harness and the connector.

    uv run python -m vals_tools.doctor                                   # credentials and the project
    uv run python -m vals_tools.doctor --datasets                        # ... and the datasets from rl/make_dataset.py
    uv run python -m vals_tools.doctor --harness http://127.0.0.1:8765   # ... and a running harness and its connector

Each line starts with OK or FAIL and says what to do about a failure. It also prints your project id (`prj_...`),
which `ac2 harness connect` and `ac2 harness test` need. Exits non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib.metadata import version
from typing import Any

import httpx
from ac2.sdk import Client

DATASETS = ("train", "select", "eval", "smoke", "table-smoke")
MIN_AC2 = (0, 3, 97)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3] if part.isdigit())


class Report:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, message: str) -> None:
        print(f"OK    {message}")

    def fail(self, message: str) -> None:
        self.failed = True
        print(f"FAIL  {message}")


def check(args: argparse.Namespace) -> bool:
    report = Report()
    sdk = version("ac2")
    if _version_tuple(sdk) >= MIN_AC2:
        report.ok(f"ac2 SDK {sdk}")
    else:
        report.fail(f"ac2 SDK {sdk} is older than 0.3.97: run `uv sync` (and `ac2 upgrade` for the CLI)")
    if shutil.which("ac2"):
        report.ok("ac2 CLI on PATH")
    else:
        report.fail("ac2 CLI not found: install it with `curl -fsSL https://api.appliedcompute.com/install.sh | sh`")

    try:
        client = Client(project=args.project)
    except Exception as exc:  # noqa: BLE001 - every failure here is reported, not raised
        name = type(exc).__name__
        if name == "ProjectNotFoundError":
            report.fail(f"credentials work, but project {args.project!r} doesn't exist: run `ac2 project init`")
        else:
            report.fail(f"can't reach AC2 ({name}: {str(exc)[:200]}): run `ac2 login`, or export AC2_API_KEY")
        return not report.failed
    project_id = client._project_id  # noqa: SLF001
    report.ok(f"project {args.project} is {project_id}")

    if args.datasets:
        for split in DATASETS:
            name = f"{args.prefix}-{split}"
            try:
                client.datasets.get(name)
                report.ok(f"dataset {name}")
            except Exception:  # noqa: BLE001
                report.fail(f"dataset {name} is missing: run `uv run python rl/make_dataset.py ...`")

    if args.harness:
        token = os.environ.get("POKER_HARNESS_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            health = httpx.get(args.harness.rstrip("/") + "/health", headers=headers, timeout=10).json()
            if health.get("status") == "ok":
                report.ok(f"harness at {args.harness} is up ({health.get('rollouts', {})})")
            else:
                report.fail(f"harness at {args.harness} answered {health}: check its terminal for errors")
        except (httpx.HTTPError, ValueError) as exc:
            report.fail(f"harness at {args.harness} unreachable ({type(exc).__name__}): start it with run_sharded.sh")
        try:
            statuses: list[dict[str, Any]] = (
                client._require_api()  # noqa: SLF001
                .request("GET", "/harness-gateway/status", params={"project_id": project_id})
                .json()
            )
            if any(s.get("connected") for s in statuses):
                report.ok(f"a connector is connected to {args.project}")
            else:
                report.fail(f"no connector is connected to {args.project}: run `ac2 harness connect` (step 4)")
        except Exception as exc:  # noqa: BLE001
            report.fail(f"couldn't read the connector status ({type(exc).__name__})")
    return not report.failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="pokerbots")
    parser.add_argument("--prefix", default="pokerbots", help="dataset prefix")
    parser.add_argument("--datasets", action="store_true", help="also check the datasets exist")
    parser.add_argument("--harness", help="also check this harness URL and the project's connector")
    sys.exit(0 if check(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()
