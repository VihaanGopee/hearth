#!/bin/bash
# Hearth launcher. Usage: ./run.sh        -> terminal chat
#                        ./run.sh web    -> local web UI at http://127.0.0.1:8765
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtualenv and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

if [ "${1:-}" = "web" ]; then
  exec .venv/bin/python -m src.server
else
  exec .venv/bin/python -m src.cli "$@"
fi
