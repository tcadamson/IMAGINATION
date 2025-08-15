import mss
import os
import time
import pygetwindow as gw
from PIL import Image


def capture_window(title: str = "IMAGINE Version") -> Image.Image:
    """
    Capture a window by title and return as PIL Image.

    Args:
        title: Substring to search for in window titles

    Returns:
        PIL Image of the window contents

    Raises:
        ValueError: If no window with matching title is found
    """
    windows = gw.getWindowsWithTitle(title)
    if not windows:
        raise ValueError(f"No window found with title containing '{title}'")

    window = windows[0]
    window.activate()

    with mss.mss() as sct:
        region = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }
        capture = sct.grab(region)

    return Image.frombytes("RGB", capture.size, capture.bgra, "raw", "BGRX")


def test_capture() -> None:
    """Test function to capture window and save to file."""
    try:
        image = capture_window()
        timestamp = int(time.time())
        filename = f"{timestamp}.png"
        filepath = os.path.join("tests", filename)
        image.save(filepath)
        print(f"Capture saved as {filepath}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_capture()
