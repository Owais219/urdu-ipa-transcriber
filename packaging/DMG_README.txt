URDU IPA — Urdu single-word speech-to-IPA transcription
For Intel Macs, macOS 12 (Monterey) or later.

INSTALL
1. Drag "Urdu IPA" onto the Applications folder in this window.
2. Eject the "Urdu IPA" disk.

FIRST OPENING (one time only)
The app is not from the App Store, so macOS asks you to confirm once:
  macOS 15 Sequoia:   open the app; when macOS refuses, go to
                      System Settings > Privacy & Security, scroll down,
                      click "Open Anyway" next to "Urdu IPA", then confirm.
  macOS 12-14:        in Applications, right-click (or Control-click)
                      "Urdu IPA" > Open > Open.
If macOS says the app "is damaged", open Terminal and run once:
  xattr -dr com.apple.quarantine "/Applications/Urdu IPA.app"

MICROPHONE
The first time you click Start Recording, macOS asks whether Urdu IPA may use
the microphone. Click OK. If you clicked "Don't Allow": System Settings >
Privacy & Security > Microphone > turn on "Urdu IPA", then reopen the app.

USING IT
Wait until the status bar at the bottom says "Ready" (a few seconds).
Type the expected word in Urdu script (optional), click Start Recording, say
one word, and leave about half a second of silence before you click Stop.
Everything works offline; no models are downloaded.

SPEED
"Fast mode" (on by default): when you type the expected word, the app skips
speech recognition and analyses the pronunciation directly from the audio;
the result appears in under a second.
With the expected-word box empty, the app first recognises the word with
Whisper, which takes roughly 5-15 seconds per word on an Intel Mac.

YOUR FILES
Recordings, the dataset, your own lexicon entries (my_lexicon.tsv) and a log
file are in the "UrduIPA" folder in your home folder. If something goes wrong,
send UrduIPA/logs/app.log.

Read the known limitations (Help > About and Limitations) before using results
as research data.
