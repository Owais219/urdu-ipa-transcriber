#!/bin/bash
# Start the Urdu Phonetic Transcription Tool. Double-click in Finder, or run:  ./run_mac.command
# Extra arguments are passed on, e.g.  ./run_mac.command --asr manual
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Run setup_mac.command first."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi
source .venv/bin/activate
python -m urdu_ipa "$@"
