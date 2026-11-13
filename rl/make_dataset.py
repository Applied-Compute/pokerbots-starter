"""Make the RL datasets: single 6-max hands at 100 BB on fresh seeds, one AC2 task per hand.

Writes <prefix>-train, -select (for picking checkpoints), -eval (held out), -smoke and -table-smoke. Every task
starts from the harness's "event" preset. Runs on these datasets name no grader, so the harness's sample
grader (harness/src/poker_harness/grading.py) scores each hand by its EV in big blinds.

    uv run python rl/make_dataset.py --project <project> --opponent-model <project>/<endpoint> --dry-run
    uv run python rl/make_dataset.py --project <project> --opponent-model <project>/<endpoint>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vals_tools.tasks import SEATS, Opponents, datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="pokerbots")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--opponent-model", required=True, help="model name the harness calls for opponent seats")
    parser.add_argument("--opponent-api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--opponent-effort", default="medium", help="reasoning effort; 'none' leaves it unset")
    parser.add_argument("--seats", type=int, default=SEATS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    effort = None if args.opponent_effort == "none" else args.opponent_effort
    built = datasets(args.prefix, Opponents(args.opponent_model, args.opponent_api, effort), args.seats)
    if args.dry_run:
        for name, tasks in built.items():
            print(name, len(tasks), tasks[0].env_params)
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} {args.seats}-max poker tasks.", exist_ok=True)
        for start in range(0, len(tasks), 500):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
