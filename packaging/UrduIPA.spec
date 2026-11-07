# PyInstaller recipe for the Intel-Mac app:  pyinstaller --noconfirm packaging/UrduIPA.spec
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = os.environ.get("APP_VERSION", "0.2.0")
WHISPER = ROOT / "models" / "whisper-urdu-ct2"
if not (WHISPER / "model.bin").exists():
    raise SystemExit(f"converted Whisper model missing: {WHISPER}")

datas = [
    (str(ROOT / "data"), "data"),
    (str(WHISPER), "models/whisper-urdu-ct2"),
    (str(ROOT / "docs"), "docs"),
    (str(ROOT / "README.md"), "."),
]
datas += collect_data_files("allosaurus")          # includes pretrained/uni2005 (phone model)
datas += collect_data_files("panphon")
datas += collect_data_files("faster_whisper")
if not any("uni2005" in d[0] for d in datas):
    raise SystemExit("Allosaurus model uni2005 not found in the allosaurus package")

hidden = (collect_submodules("allosaurus") + collect_submodules("faster_whisper") + collect_submodules("urdu_ipa")
          + ["ctranslate2", "openpyxl", "soundfile", "sounddevice", "scipy.signal"])
excludes = ["resampy", "numba", "llvmlite", "onnxruntime", "transformers", "tensorflow", "matplotlib",
            "tkinter", "IPython", "jupyter", "pytest", "torchvision", "torchaudio", "torch.utils.tensorboard",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
            "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf"]

a = Analysis([str(ROOT / "packaging" / "launcher.py")], pathex=[str(ROOT)], datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="UrduIPA", console=False, upx=False,
          target_arch="x86_64", codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="UrduIPA")
app = BUNDLE(
    coll,
    name="Urdu IPA.app",
    icon=None,
    bundle_identifier="pk.owais.urduipa",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": "Urdu IPA",
        "CFBundleName": "Urdu IPA",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "NSMicrophoneUsageDescription":
            "Urdu IPA records one spoken Urdu word at a time to transcribe its pronunciation. "
            "Audio stays on this Mac.",
        "LSApplicationCategoryType": "public.app-category.education",
    },
)
