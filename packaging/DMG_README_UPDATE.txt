URDU IPA — UPDATE (small download, no Whisper model inside)

Use this only if "Urdu IPA" is already installed in Applications.
It reuses the speech recogniser (Whisper) from your installed copy, so you
do not have to download 1 GB again.

1. BEFORE installing, keep the Whisper model from your current app.
   Open Terminal and run (copy the whole line):

     bash "/Volumes/Urdu IPA update/keep_whisper_model.sh"

   It should say "Kept the Whisper model in .../UrduIPA/models/...".
   (Only needed once. If it says the model is already kept, that is fine.)

2. Quit Urdu IPA if it is open.

3. Drag "Urdu IPA" from THIS window ("Urdu IPA update") onto Applications and
   choose "Replace". Make sure no other "Urdu IPA" disk image is open, so you
   do not drag the old app by mistake. The new window title says
   "Urdu IPA 0.4.0".

4. Open the app. The first time, macOS may ask you to confirm it again:
   macOS 15: System Settings > Privacy & Security > "Open Anyway".
   macOS 12-14: right-click the app > Open > Open.
   macOS may also ask again for microphone access; click OK.

If the app later says the speech recogniser is unavailable, the model was
not kept: install the full UrduIPA-Intel.dmg instead.
