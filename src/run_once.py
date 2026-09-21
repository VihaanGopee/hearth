"""Run one prompt non-interactively (used by scheduled jobs)."""
from __future__ import annotations
import sys
from .config import load_config
from .agent import Agent


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    if not prompt.strip():
        print('usage: python -m src.run_once "prompt"')
        sys.exit(2)
    agent = Agent(load_config())
    print(agent.run(prompt))


if __name__ == "__main__":
    main()
