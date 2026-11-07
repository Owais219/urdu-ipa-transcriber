#!/bin/bash
# One-time setup for macOS. Double-click in Finder, or run:  bash setup_mac.command
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.11 \
         /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.11/bin/python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 or 3.11 was not found."
  echo "Install Python 3.12 from https://www.python.org/downloads/macos/ and run this again."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi
echo "Using $PY ($("$PY" --version))"

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if [ "$(uname -m)" = "x86_64" ]; then
  echo "Intel Mac detected: installing pinned versions."
  pip install -r requirements-mac-intel.txt
else
  echo "Apple Silicon detected."
  pip install -r requirements.txt
fi

echo "Downloading the phone model (about 41 MB)..."
python -c "from allosaurus.app import read_recognizer; read_recognizer()"

echo
echo "Downloading the Urdu Whisper speech recogniser (1.6 GB or more; this can take a while)..."
echo "If this fails or you want to skip it, the app still works with:  ./run_mac.command --asr manual"
python - <<'PY' || echo "Whisper download did not finish; it will be retried when the app starts."
from urdu_ipa.config import load_config
from urdu_ipa.asr import make_asr
c = load_config()
kind, model = c.resolved_asr()
if kind != "manual":
    a = make_asr(kind, model); a.load(); print("Downloaded", a.version)
PY

echo
echo "Setup finished. Start the app with:  ./run_mac.command"
read -n 1 -s -r -p "Press any key to close."
