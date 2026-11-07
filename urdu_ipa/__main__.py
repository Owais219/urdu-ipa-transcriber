"""Entry point: `python -m urdu_ipa`, and the packaged Mac app."""
import multiprocessing
import os
import sys


def _prepare_environment() -> None:
    from .config import FROZEN, BUNDLED_WHISPER
    if FROZEN or BUNDLED_WHISPER.exists():
        # every model ships with the app: never try the network
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + ctranslate2 both ship OpenMP on macOS


def _setup_logging() -> str:
    """The packaged app has no terminal, so stdout/stderr and crashes go to ~/UrduIPA/logs/app.log."""
    import faulthandler
    import logging
    from pathlib import Path
    from .config import FROZEN
    log_dir = Path.home() / "UrduIPA" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "app.log"
    if path.exists() and path.stat().st_size > 2_000_000:
        path.replace(log_dir / "app.previous.log")
    f = open(path, "a", encoding="utf-8", buffering=1)
    if FROZEN or sys.stderr is None:
        sys.stdout = sys.stderr = f
    faulthandler.enable(f)
    logging.basicConfig(stream=f, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urdu_ipa").info("start, python %s, frozen=%s, argv=%s", sys.version.split()[0], FROZEN, sys.argv)
    return str(path)


def main() -> int:
    multiprocessing.freeze_support()
    _prepare_environment()
    if "--selftest" in sys.argv:
        from .selftest import run
        return run(sys.argv[sys.argv.index("--selftest") + 1:])
    _setup_logging()
    from .ui import main as ui_main
    return ui_main([a for a in sys.argv[1:] if not a.startswith("-psn_")])   # Finder adds -psn_… on old macOS


if __name__ == "__main__":
    sys.exit(main())
