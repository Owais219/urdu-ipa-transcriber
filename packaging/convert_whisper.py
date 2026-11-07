"""Convert a Hugging Face Whisper checkpoint to a self-contained CTranslate2 (int8) folder.

    python packaging/convert_whisper.py kingabzpro/whisper-large-v3-turbo-urdu models/whisper-urdu-ct2

The output contains everything faster-whisper needs offline (model.bin, config.json,
vocabulary, tokenizer.json, preprocessor_config.json) plus SOURCE.txt recording the
source model and revision. After conversion it is loaded and run once offline.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def main(model_id: str, out: str, quantization: str = "int8") -> None:
    from huggingface_hub import HfApi, snapshot_download
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperProcessor

    out_dir = Path(out)
    info = HfApi().model_info(model_id)
    print(f"source: {model_id} @ {info.sha}")
    files = [s.rfilename for s in info.siblings]
    print("files:", files)
    local = snapshot_download(model_id, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.bin"]
                              if not any(f.endswith(".safetensors") for f in files)
                              else ["*.json", "*.txt", "*.safetensors", "*.model"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t = time.time()
    TransformersConverter(local, low_cpu_mem_usage=True).convert(str(out_dir), quantization=quantization, force=True)
    print(f"converted in {time.time() - t:.0f} s")

    # tokenizer.json and preprocessor_config.json so that faster-whisper never downloads anything
    proc = WhisperProcessor.from_pretrained(local)
    with tempfile.TemporaryDirectory() as tmp:
        proc.save_pretrained(tmp)
        for name in ("tokenizer.json", "preprocessor_config.json"):
            src = Path(tmp) / name
            if not src.exists():
                sys.exit(f"could not produce {name}")
            shutil.copy(src, out_dir / name)
    n_mels = json.loads((out_dir / "preprocessor_config.json").read_text())["feature_size"]
    (out_dir / "SOURCE.txt").write_text(f"{model_id}@{info.sha[:10]}\nquantization={quantization}\nn_mels={n_mels}\n",
                                        encoding="utf-8")

    # offline load + one transcription
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    from faster_whisper import WhisperModel
    m = WhisperModel(str(out_dir), device="cpu", compute_type=quantization, local_files_only=True)
    t = time.time()
    segs, _ = m.transcribe(np.zeros(16000, dtype=np.float32), language="ur", beam_size=3,
                           without_timestamps=True, condition_on_previous_text=False)
    list(segs)
    print(f"offline test transcription ok ({time.time() - t:.1f} s); n_mels={n_mels}")
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"output {out_dir}: {size:.0f} MB:", sorted(p.name for p in out_dir.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
