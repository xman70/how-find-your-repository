"""Installer / launcher for the VUSA AI Quant Terminal (standard library only).

Used by install.bat and run.bat on Windows (and usable on Linux/macOS):

    python bootstrap.py --install     install or repair the environment
    python bootstrap.py --run [args]  install if needed, then start main.py with [args]

Why the environment does NOT live inside the project folder
-----------------------------------------------------------
* Windows limits paths to 260 characters unless "long paths" are enabled. PySide6 ships files
  whose path inside site-packages is up to ~151 characters, so a project folder deeper than
  ~80 characters (e.g. Desktop\\New folder\\how-find-your-repository-<40-char-sha>\\...) cannot
  hold the GUI library.
* Some libraries (curl_cffi, used by yfinance) cannot open files under non-ASCII paths such as a
  Greek user name.
The environment is therefore created at a short ASCII location: %ProgramData%\\VUSA-Quant\\venv
(fallbacks: C:\\VUSA-Quant\\venv, then %USERPROFILE%\\.vusa-quant\\venv).
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "install_ok.txt"
LONGEST_WHEEL_PATH = 155  # longest file path inside PySide6 wheels (+ margin)
MAX_PATH = 259
CORE = HERE / "requirements-core.txt"
OPTIONAL = ["xgboost", "lightgbm", "catboost", "optuna", "shap", "hmmlearn", "anthropic"]
MIN_PY, MAX_PY = (3, 10), (3, 14)
IS_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------- locations
def _site_packages_prefix(venv: Path) -> str:
    return str(venv / ("Lib/site-packages" if IS_WINDOWS else "lib/python3.x/site-packages")) + os.sep


def _windows_long_paths_enabled() -> bool:
    if not IS_WINDOWS:
        return True
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def _usable_location(venv: Path) -> bool:
    if IS_WINDOWS:
        if not str(venv).isascii():
            return False
        if not _windows_long_paths_enabled() and len(_site_packages_prefix(venv)) + LONGEST_WHEEL_PATH > MAX_PATH:
            return False
    try:
        venv.parent.mkdir(parents=True, exist_ok=True)
        probe = venv.parent / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def candidate_venvs() -> list[Path]:
    if not IS_WINDOWS:
        return [HERE / ".venv"]
    c = []
    if os.environ.get("ProgramData"):
        c.append(Path(os.environ["ProgramData"]) / "VUSA-Quant" / "venv")
    c.append(Path(os.environ.get("SystemDrive", "C:") + os.sep) / "VUSA-Quant" / "venv")
    c.append(Path(os.environ.get("USERPROFILE", str(HERE))) / ".vusa-quant" / "venv")
    return c


def existing_venv() -> Path | None:
    """The venv a previous successful install created (marker present)."""
    for v in candidate_venvs():
        if (v / MARKER).exists() and venv_python(v).exists():
            return v
    return None


def choose_venv() -> Path:
    for v in candidate_venvs():
        if _usable_location(v):
            return v
    raise SystemExit(
        "\n[ERROR] No suitable folder for the Python environment was found.\n"
        "        Tried: " + ", ".join(str(v) for v in candidate_venvs()) + "\n"
        "        Create the folder C:\\VUSA-Quant (or enable Windows long paths) and run install.bat again.")


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


# ---------------------------------------------------------------------------- checks
def check_base_python() -> None:
    v = sys.version_info[:2]
    if not (MIN_PY <= v <= MAX_PY):
        raise SystemExit(f"[ERROR] Python {v[0]}.{v[1]} is not supported. Install 64-bit Python 3.12 from python.org.")
    if struct.calcsize("P") != 8:
        raise SystemExit("[ERROR] 32-bit Python is not supported (the GUI library needs 64-bit). "
                         "Install the 64-bit Python 3.12 from python.org.")


def _run(cmd: list[str], quiet: bool = False) -> int:
    kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
    try:
        return subprocess.call(cmd, **kw)
    except OSError:  # e.g. a venv python whose base interpreter was uninstalled or is corrupt
        return 1


def venv_healthy(venv: Path) -> bool:
    """Venv exists, its Python starts, is 64-bit and has the same minor version as this Python."""
    py = venv_python(venv)
    if not py.exists():
        return False
    want = "%d.%d" % sys.version_info[:2]
    code = ("import sys,struct; sys.exit(0 if '%d.%d' % sys.version_info[:2] == '" + want +
            "' and struct.calcsize('P') == 8 else 1)")
    return _run([str(py), "-c", code], quiet=True) == 0


def essentials_ok(py: Path) -> bool:
    code = ("import numpy, pandas, scipy, sklearn, statsmodels, plotly, requests\n"
            "from PySide6.QtWidgets import QApplication")
    return _run([str(py), "-c", code], quiet=True) == 0


# ---------------------------------------------------------------------------- install
def install(verbose_header: bool = True) -> Path:
    check_base_python()
    venv = choose_venv()
    print(f"[1/5] Python {sys.version.split()[0]} (64-bit) at {sys.executable}")
    if venv.exists() and not venv_healthy(venv):
        print(f"[2/5] Existing environment {venv} is broken or was made with another Python - recreating it ...")
        shutil.rmtree(venv, ignore_errors=True)
    if not venv_python(venv).exists():
        print(f"[2/5] Creating the Python environment in {venv} ...")
        if _run([sys.executable, "-m", "venv", str(venv)]) != 0 or not venv_python(venv).exists():
            raise SystemExit(f"[ERROR] Could not create the environment in {venv}.")
    else:
        print(f"[2/5] Using the environment in {venv}")
    (venv / MARKER).unlink(missing_ok=True)
    py = str(venv_python(venv))
    pip = [py, "-m", "pip", "--disable-pip-version-check"]
    _run(pip + ["install", "-q", "--upgrade", "pip", "wheel"])

    print("[3/5] Installing essential packages (this can take several minutes) ...")
    if _run(pip + ["install", "--prefer-binary", "-r", str(CORE)]) != 0:
        raise SystemExit("\n[ERROR] Installing the essential packages failed (see the messages above).\n"
                         "        Most common causes: no internet connection, or a firewall/antivirus blocking pip.\n"
                         "        Run install.bat again once the cause is fixed.")

    print("[4/5] Installing optional packages (skipped if not available for this Python version) ...")
    for pkg in OPTIONAL:
        ok = _run(pip + ["install", "-q", "--only-binary=:all:", pkg], quiet=True) == 0
        print(f"      [{'OK' if ok else 'SKIPPED'}] {pkg}" +
              ("" if ok else " - no pre-built package for this Python version; the app works without it"))
    print("      Installing PyTorch (CPU) for the deep-learning models ...")
    ok = _run(pip + ["install", "-q", "--only-binary=:all:", "torch", "--index-url",
                     "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple"],
              quiet=True) == 0
    if ok and _run([py, "-c", "import torch"], quiet=True) != 0:
        _run(pip + ["uninstall", "-y", "-q", "torch"], quiet=True)  # never leave a half-working torch
        ok = False
    print(f"      [{'OK' if ok else 'SKIPPED'}] torch" + ("" if ok else " - deep-learning models will be reported as unavailable"))

    print("[5/5] Verifying ...")
    if not essentials_ok(venv_python(venv)):
        raise SystemExit("\n[ERROR] The packages were installed but the GUI library (PySide6) cannot be loaded.\n"
                         "        Install the Microsoft Visual C++ Redistributable (x64):\n"
                         "        https://aka.ms/vs/17/release/vc_redist.x64.exe  - then run install.bat again.")
    env_file, example = HERE / ".env", HERE / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copyfile(example, env_file)
    (venv / MARKER).write_text("ok", encoding="utf-8")
    print(f"\nInstallation complete (environment: {venv}).")
    return venv


def ensure() -> Path:
    v = existing_venv()
    if v is not None and venv_healthy(v) and essentials_ok(venv_python(v)):
        return v
    print("First start (or the previous installation did not finish): installing now ...\n")
    return install()


def main(argv: list[str]) -> int:
    try:
        if argv[:1] == ["--install"]:
            install()
            return 0
        if argv[:1] == ["--run"]:
            v = ensure()
            return subprocess.call([str(venv_python(v)), str(HERE / "main.py"), *argv[1:]],
                                   env=dict(os.environ, VUSA_NO_REEXEC="1"))
        print(__doc__)
        return 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
            return 1
        return int(e.code or 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
