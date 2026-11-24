"""Run the desktop pilot's automated checks on the checked-in synthetic pair."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 120
DEPENDENCIES = ("numpy", "obspy", "pyrocko", "PyQt5", "matplotlib")

QT_INFO = """
from PyQt5.QtCore import PYQT_VERSION_STR, qVersion
from PyQt5.QtWidgets import QApplication

app = QApplication([])
backend = app.platformName()
print(f"Qt {qVersion()}, PyQt {PYQT_VERSION_STR}, platform {backend}")
if backend in {"offscreen", "minimal", "minimalegl"}:
    print("Headless Qt synthetic check. Native Windows desktop evidence remains separate.")
"""

RUN_TESTS = """
import sys
import unittest

suite = unittest.defaultTestLoader.discover(sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(f"{result.testsRun} run, {len(result.skipped)} skipped")
if result.skipped:
    print("The pilot check requires every discovered test to run.")
sys.exit(not (result.wasSuccessful() and result.testsRun and not result.skipped))
"""

CHECKS = (
    ("Qt runtime", ("-c", QT_INFO), None),
    ("Synthetic fixture", ("-c", RUN_TESTS, "fixtures"), None),
    ("Application tests including the desktop", ("-c", RUN_TESTS, "tests"), None),
    ("Snuffler sample nudge", ("tools/snuffler_probe.py", "--check"),
     "PASS: native marker selection, exact sample nudge"),
)


def windows_architectures():
    if sys.platform != "win32":
        raise RuntimeError("Windows architecture detection requires Windows.")

    import ctypes
    from ctypes import wintypes

    build = sysconfig.get_platform()
    interpreter = {"win-amd64": "x64", "win-arm64": "ARM64", "win32": "x86"}.get(
        build, build
    )
    host = os.environ.get("PROCESSOR_ARCHITEW6432", "unknown")
    kernel = ctypes.WinDLL("kernel32")
    probe = getattr(kernel, "IsWow64Process2", None)
    if probe is not None:
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        probe.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT))
        probe.restype = wintypes.BOOL
        guest, native = wintypes.USHORT(), wintypes.USHORT()
        if probe(kernel.GetCurrentProcess(), ctypes.byref(guest), ctypes.byref(native)):
            host = {0x014C: "x86", 0x8664: "x64", 0xAA64: "ARM64"}.get(native.value, host)

    # The Python build identifies x64 even if Windows reports no WOW process.
    if interpreter == "x64" and host.upper() == "ARM64":
        interpreter += " (emulated on Windows Arm)"
    return interpreter, host


def report_environment():
    interpreter = host = platform.machine()
    if sys.platform == "win32":
        interpreter, host = windows_architectures()
    elif sys.platform == "darwin" and interpreter == "x86_64":
        try:
            arm64 = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.optional.arm64"],
                stderr=subprocess.DEVNULL, timeout=2,
            )
            if arm64.strip() == b"1":
                host = "arm64"
        except (OSError, subprocess.SubprocessError):
            pass

    print(f"SeisSlate synthetic pilot check, {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"OS {platform.platform()}, OS architecture {host}")
    print(f"Python {platform.python_version()}, interpreter architecture {interpreter}")
    print(f"Interpreter {sys.executable}")
    print(f"QT_QPA_PLATFORM = {os.environ.get('QT_QPA_PLATFORM', '(default)')}")

    missing = []
    for name in DEPENDENCIES:
        try:
            print(f"{name} {version(name)}")
        except PackageNotFoundError:
            print(f"{name} missing")
            missing.append(name)
    return not missing


def run_check(label, args, required_output):
    print(f"\nRunning {label}", flush=True)
    start = monotonic()
    with TemporaryDirectory(prefix="seisslate-pilot-") as directory, TemporaryFile() as log:
        env = dict(
            os.environ,
            PYROCKO_DIR=str(Path(directory) / "pyrocko"),
            MPLCONFIGDIR=str(Path(directory) / "matplotlib"),
        )
        env.pop("SEISSLATE_TEST_SCREENSHOT", None)
        env.pop("PYTHONOPTIMIZE", None)
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-u", "-X", "utf8", *args], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS,
            )
            failure = f"exit {result.returncode}" if result.returncode else None
        except subprocess.TimeoutExpired:
            failure = f"timed out after {TIMEOUT_SECONDS} seconds"
        except OSError as error:
            failure = str(error)

        log.seek(0)
        output = log.read().decode("utf-8", errors="replace")
        if output:
            print(output.rstrip())
        if failure is None and required_output and required_output not in output:
            failure = "missing Snuffler interaction result"

    elapsed = monotonic() - start
    if failure is not None:
        print(f"FAIL {label} ({elapsed:.1f} s, {failure})")
        return False
    print(f"PASS {label} ({elapsed:.1f} s)")
    return True


def main():
    if not report_environment():
        print('Install the checkout with its desktop extra using python -m pip install -e ".[desktop]".')
        for label, _, _ in CHECKS:
            print(f"SKIP {label} (required dependencies missing)")
        return 1

    results = [run_check(*check) for check in CHECKS]
    passed = sum(results)
    print(f"\nSynthetic check groups: {passed} passed, {len(results) - passed} failed.")
    print("Record manual Windows workstation and field validation separately.")
    return int(not all(results))


if __name__ == "__main__":
    raise SystemExit(main())
