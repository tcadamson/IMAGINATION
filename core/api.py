"""The IMAGINE client window, and what IMAGINATION's commands share."""

import ctypes
import ctypes.wintypes
import pathlib
import sys
import time
import typing

ROOT_DIRECTORY: typing.Final = (
    pathlib.Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else pathlib.Path(__file__).resolve().parents[1]
)

_WNDENUMPROC: typing.Final = ctypes.WINFUNCTYPE(
    ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
)

_user32: typing.Final = ctypes.WinDLL("user32")
_user32.EnumWindows.argtypes = [_WNDENUMPROC, ctypes.wintypes.LPARAM]
_user32.IsWindowVisible.argtypes = [ctypes.wintypes.HWND]
_user32.GetWindowTextLengthW.argtypes = [ctypes.wintypes.HWND]
_user32.GetWindowTextW.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.wintypes.LPWSTR,
    ctypes.c_int,
]
_user32.GetWindowThreadProcessId.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.POINTER(ctypes.wintypes.DWORD),
]
_user32.GetWindowThreadProcessId.restype = ctypes.wintypes.DWORD

ABORT_MESSAGE: typing.Final = "Aborted by user"


def safe_abort() -> None:
    """Perform an uninterrupted sleep, resetting the SIGINT event.

    An interrupted sleep leaves the interpreter's SIGINT event signaled and sends two
    Ctrl+C inputs to the REPL. Must be called while intercepting a KeyboardInterrupt
    raised from a blocking call.
    """
    while True:
        try:
            time.sleep(0.01)
            break
        except KeyboardInterrupt:
            pass


def _window_title(handle: int) -> str:
    """Title of the window `handle`."""
    length = _user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


class Client:
    """Handle of a single IMAGINE client window."""

    CLIENT_IDENTIFIER: typing.Final = "IMAGINE Version"

    def __init__(self, handle: int):
        self._handle = handle

    @classmethod
    def locate_all(cls, identifier: str = CLIENT_IDENTIFIER) -> tuple[Client, ...]:
        """Return all visible client windows whose titles begin with the identifier."""
        handles = []

        @_WNDENUMPROC
        def collect(handle, _):
            if _user32.IsWindowVisible(handle) and _window_title(handle).startswith(
                identifier
            ):
                handles.append(handle)
            return True

        _user32.EnumWindows(collect, 0)

        if not handles:
            raise RuntimeError("no IMAGINE client window is open")

        return tuple(cls(handle) for handle in handles)

    @property
    def handle(self) -> int:
        """Client window handle."""
        return self._handle

    @property
    def pid(self) -> int:
        """ID of the client's process."""
        pid = ctypes.wintypes.DWORD()

        if not _user32.GetWindowThreadProcessId(self._handle, ctypes.byref(pid)):
            raise RuntimeError("the IMAGINE client window has closed")

        return pid.value
