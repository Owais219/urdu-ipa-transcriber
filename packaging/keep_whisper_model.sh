#!/bin/bash
# Copy the Whisper model out of an installed Urdu IPA app into ~/UrduIPA/models, so that the
# small update app (which has no model inside) can use it.
#   bash keep_whisper_model.sh                      # uses "/Applications/Urdu IPA.app"
#   bash keep_whisper_model.sh "/path/to/Urdu IPA.app"
set -e
APP="${1:-/Applications/Urdu IPA.app}"
DEST="$HOME/UrduIPA/models/whisper-urdu-ct2"
if [ -f "$DEST/model.bin" ]; then echo "Model already kept in $DEST"; exit 0; fi
SRC="$(find "$APP" -path '*whisper-urdu-ct2/model.bin' 2>/dev/null | head -1)"
if [ -z "$SRC" ]; then
  echo "No Whisper model found in $APP."
  echo "Install the full UrduIPA-Intel.dmg instead of the update."
  exit 1
fi
mkdir -p "$DEST"
cp -RL "$(dirname "$SRC")/." "$DEST/"
echo "Kept the Whisper model in $DEST ($(du -sh "$DEST" | cut -f1)). You can now install the update."
