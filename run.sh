#!/usr/bin/env bash
# Start Drum Queue. Extra flags pass through, e.g.:
#   ./run.sh --out ~/DrumSplits --inbox ~/Drum_Inbox
cd "$(dirname "$0")"
if [ ! -d .venv ]; then echo "Run ./setup.sh first."; exit 1; fi
source .venv/bin/activate
(sleep 2 && open "http://127.0.0.1:${PORT:-8766}") >/dev/null 2>&1 &
exec python app.py "$@"
