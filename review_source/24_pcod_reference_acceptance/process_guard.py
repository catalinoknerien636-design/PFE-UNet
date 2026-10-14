"""Exact Windows process binding and conservative project-process exclusion."""
import ctypes
from ctypes import wintypes
import json
import os
import subprocess


def kernel_api():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel


def creation(kernel, handle):
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def process_identity(pid):
    kernel = kernel_api()
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError('Process handle unavailable')
    try:
        return {'pid': pid, 'creation_ticks': creation(kernel, handle)}
    finally:
        kernel.CloseHandle(handle)


def wait_for_exact_process(pid, expected_creation, update):
    kernel = kernel_api()
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise RuntimeError('Dependency handle unavailable; inspect instead of launching')
    try:
        actual = creation(kernel, handle)
        if actual != expected_creation:
            raise RuntimeError('Dependency PID was reused')
        while True:
            update({'pid': pid, 'creation_ticks': actual})
            result = kernel.WaitForSingleObject(handle, 30000)
            if result == 0:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if code.value != 0:
                    raise RuntimeError(f'Dependency exited unsuccessfully: {code.value}')
                return
            if result != 258:
                raise RuntimeError(f'Dependency wait failed: {result}')
    finally:
        kernel.CloseHandle(handle)


def python_processes():
    command = "$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | Select-Object ProcessId,ParentProcessId,CommandLine); ConvertTo-Json -InputObject $p -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    records = json.loads(result.stdout)
    if not isinstance(records, list):
        raise ValueError('Process inventory was not an array')
    return records


def other_project_processes(records, own_pid):
    by_pid = {r['ProcessId']: r for r in records}
    ancestors, current = set(), own_pid
    while current in by_pid and current not in ancestors:
        ancestors.add(current)
        current = by_pid[current]['ParentProcessId']
    suspects = []
    for record in records:
        command = (record.get('CommandLine') or '').lower()
        if record['ProcessId'] not in ancestors and (
                'pfe_unet_research_extension_20260923' in command or
                'polar_camouflage_detection' in command):
            suspects.append(record)
    return suspects


def verify_dependency_command(records, pid):
    matches = [r for r in records if r['ProcessId'] == pid]
    if len(matches) != 1:
        raise ValueError('Dependency absent from process inventory')
    command = (matches[0].get('CommandLine') or '').replace('\\', '/').lower()
    if '/21_portable_rgbp_development/run_queue.py' not in command:
        raise ValueError('Dependency is not the registered development controller')


def require_idle():
    suspects = other_project_processes(python_processes(), os.getpid())
    if suspects:
        raise RuntimeError(f'Other project Python processes still exist: {suspects}')
