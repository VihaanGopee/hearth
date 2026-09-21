"""Interactive chat in the terminal."""
from __future__ import annotations
from .config import load_config
from .agent import Agent


def main() -> None:
    cfg = load_config()
    agent = Agent(cfg)
    print(f"Hearth \u2014 local agent ({agent.model_label}). "
          f"Type /quit to exit, /new for a fresh chat.")
    messages: list[dict] = []
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            print("Bye.")
            break
        if text == "/new":
            messages = []
            print("(fresh chat)")
            continue
        reply = agent.run(text, messages)
        print(f"\nHearth: {reply}")


if __name__ == "__main__":
    main()
