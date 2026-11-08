"""Time faster-whisper settings on this machine (build-time benchmark, not part of the app)."""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

model_dir, wav = sys.argv[1], sys.argv[2]
x, sr = sf.read(wav, dtype="float32")
if x.ndim > 1:
    x = x.mean(axis=1)
if sr != 16000:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype("float32")
x = np.concatenate([np.zeros(4000, "float32"), x, np.zeros(4000, "float32")])
results = []
for threads in (0, os.cpu_count() or 4):
    for ctype in ("int8", "int8_float32"):
        t = time.time()
        m = WhisperModel(model_dir, device="cpu", compute_type=ctype, cpu_threads=threads, local_files_only=True)
        load = time.time() - t
        for beam in (3, 1):
            for chunk in (None, 10, 5):
                kw = dict(language="ur", task="transcribe", beam_size=beam, temperature=0.0,
                          condition_on_previous_text=False, without_timestamps=True, max_new_tokens=24)
                if chunk:
                    kw["chunk_length"] = chunk
                try:
                    times, text = [], ""
                    for _ in range(2):
                        t = time.time(); segs, _ = m.transcribe(x, **kw); text = " ".join(s.text for s in segs).strip()
                        times.append(time.time() - t)
                    r = dict(threads=threads, compute=ctype, beam=beam, chunk=chunk, load_s=round(load, 2),
                             first_s=round(times[0], 2), warm_s=round(times[1], 2), text=text)
                except Exception as e:  # noqa: BLE001
                    r = dict(threads=threads, compute=ctype, beam=beam, chunk=chunk, error=f"{type(e).__name__}: {e}"[:200])
                print(json.dumps(r, ensure_ascii=False), flush=True)
                results.append(r)
        del m
json.dump(results, open("bench.json", "w"), ensure_ascii=False, indent=1)
