"""Atomic process-wide guard for the desktop GUI."""

from __future__ import annotations

import os
import re
import sys
import tempfile


class SingleInstanceGuard:
    """Own an OS lock for as long as this object is alive.

    A Windows named mutex is released by the OS even after a crash or forced
    termination, so it cannot leave a stale lock file behind. The POSIX path
    is used by development/test environments.
    """

    def __init__(self, name: str):
        self.acquired = False
        self._handle = None
        self._file = None

        if sys.platform == "win32":
            self._acquire_windows(name)
        else:
            self._acquire_posix(name)

    def _acquire_windows(self, name: str):
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, f"Local\\{name}")
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")

        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return

        self._kernel32 = kernel32
        self._handle = handle
        self.acquired = True

    def _acquire_posix(self, name: str):
        import fcntl

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        path = os.path.join(tempfile.gettempdir(), f"{safe_name}.lock")
        lock_file = open(path, "a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return
        self._file = lock_file
        self.acquired = True

    def close(self):
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self.acquired = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
