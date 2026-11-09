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

AUDIO FILES
Click "Open Audio File(s)…" (or File > Open Audio Files…, Command-O) to transcribe
recordings you already have: WAV, MP3, M4A, FLAC, AIFF or OGG, one word per file.
Select several files at once for a batch; the results table can be saved to the
dataset or exported as CSV. Name a file with the Urdu word (for example
جہاد.m4a) and that name is used as the expected word (fast, no Whisper needed).

WHICH VERSION AM I RUNNING?
The version is shown in the window title ("Urdu IPA 0.4.0 ...") and in the
status bar. Always start the app from Applications, not from the disk image.

YOUR FILES
Recordings, the dataset, your own lexicon entries (my_lexicon.tsv) and a log
file are in the "UrduIPA" folder in your home folder. If something goes wrong,
send UrduIPA/logs/app.log.

Read the known limitations (Help > About and Limitations) before using results
as research data.
