"""Wait for the exact registration controller, then run serial device checks."""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys

import check_hardware as h

STATE = h.t.HERE / "hardware_dependency_status.json"


def wait_for_process(pid, expected_creation):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise RuntimeError("Dependency handle unavailable; inspect before launching")
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        if created != expected_creation:
            raise RuntimeError("Dependency PID was reused")
        while True:
            h.t.write_json(STATE, {"state": "waiting_on_live_registration_controller", "pid": os.getpid(),
                "dependency_pid": pid, "dependency_creation": created, "updated_utc": h.t.utc_now()})
            state = kernel.WaitForSingleObject(handle, 30000)
            if state == 0:
                return
            if state != 258:
                raise RuntimeError(f"Process wait failed: {state}")
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-pid", type=int, required=True)
    parser.add_argument("--dependency-creation", type=int, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    process = None
    try:
        if h.t.sha(h.PROTOCOL) != args.protocol_sha256:
            raise ValueError("Hardware protocol differs from launch identity")
        wait_for_process(args.dependency_pid, args.dependency_creation)
        if h.t.sha(h.PROTOCOL) != args.protocol_sha256:
            raise ValueError("Hardware protocol changed while waiting")
        protocol = json.loads(h.PROTOCOL.read_text())
        h.validate_protocol(protocol)
        h.verify_dependency(protocol)
        with (h.t.HERE / "hardware.stdout.log").open("a", encoding="utf-8") as stdout, \
             (h.t.HERE / "hardware.stderr.log").open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen([sys.executable, "-B", str(h.t.HERE / "check_hardware.py"), "run"],
                                       cwd=h.t.HERE, stdout=stdout, stderr=stderr)
            h.t.write_json(STATE, {"state": "hardware_checks_running", "pid": os.getpid(),
                "child_pid": process.pid, "updated_utc": h.t.utc_now()})
            code = process.wait()
            if code:
                raise RuntimeError(f"Hardware acceptance failed: exit {code}")
        result = json.loads(h.REPORT.read_text())
        if not result["passed"] or len(result["gpu_resume"]) != 4 or result["protocol_sha256"] != args.protocol_sha256:
            raise ValueError("Hardware completion evidence is incomplete")
        h.t.write_json(STATE, {"state": "completed", "updated_utc": h.t.utc_now(),
                              "formal_training_started": False})
    except BaseException as error:
        if process is not None and process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.wait()
        h.t.write_json(STATE, {"state": "stopped_with_error", "pid": os.getpid(), "error": str(error),
                              "automatic_retry": False, "updated_utc": h.t.utc_now()})
        raise


if __name__ == "__main__":
    main()
