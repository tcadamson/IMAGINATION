import mss
import os
import time
import pygetwindow as gw
import cv2
import numpy as np
import pyautogui


def capture_window(title: str = "IMAGINE Version") -> np.ndarray:
    """
    Capture a window by title and return as OpenCV numpy array.

    Args:
        title: Substring to search for in window titles

    Returns:
        OpenCV numpy array (BGR format) of the window contents

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

    # Convert MSS capture to OpenCV format (BGR numpy array)
    img_array = np.array(capture)
    return cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)


def match_template(
    image: np.ndarray, template_filename: str
) -> tuple[float, tuple[int, int]]:
    """
    Match a template against the captured image.

    Args:
        image: OpenCV numpy array (BGR format) from capture_window()
        template_filename: Filename of template image in templates/ directory

    Returns:
        Tuple of (confidence_score, (x, y)) where confidence is 0-1 and
        (x, y) is the top-left corner of the best match

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_path = os.path.join("templates", template_filename)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Could not load template: {template_path}")

    # Perform template matching using normalized cross correlation
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # Find the best match location
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return max_val, max_loc


def test_capture() -> None:
    """Test function to capture window and save to file."""
    try:
        image = capture_window()
        timestamp = int(time.time())
        filename = f"{timestamp}.png"
        filepath = os.path.join("tests", filename)
        cv2.imwrite(filepath, image)
        print(f"Capture saved as {filepath}")
    except ValueError as e:
        print(f"Error: {e}")


def test_match(template_filename: str) -> None:
    """Test function to match template and move mouse to location."""
    try:
        image = capture_window()
        confidence, (x, y) = match_template(image, template_filename)

        print(f"Template '{template_filename}' found with confidence: {confidence:.3f}")

        if confidence > 0.8:
            pyautogui.moveTo(x, y)
            print(f"Mouse moved to coordinates: ({x}, {y})")
        else:
            print("Confidence too low - mouse not moved")

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_match("thread.png")
