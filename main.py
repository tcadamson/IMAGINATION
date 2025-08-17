import mss
import os
import time
import pygetwindow as gw
import cv2
import numpy as np
import pydirectinput

# Global delay between each pydirectinput call
pydirectinput.PAUSE = 0.25


class Controller:
    def __init__(self, epsilon: float = 0.015):
        self.epsilon = epsilon

    def press(self, key: str) -> None:
        """Press a key with epsilon timing to behave well with DirectX."""
        pydirectinput.press(key, duration=self.epsilon)

    def hotkey(self, *keys: str) -> None:
        """Execute hotkey combination with epsilon timing."""
        pydirectinput.hotkey(*keys, wait=self.epsilon)

    def scroll(self, clicks: int) -> None:
        """Scroll with epsilon timing."""
        pydirectinput.scroll(clicks, interval=self.epsilon)

    def leftClick(
        self,
        x: int,
        y: int,
    ) -> None:
        """Left click at coordinates."""
        pydirectinput.leftClick(x, y)
        pydirectinput.mouseUp()  # Ensure click always terminates

    def rightClick(self, x: int, y: int) -> None:
        """Right click at coordinates."""
        pydirectinput.rightClick(x, y)
        pydirectinput.mouseUp(button=pydirectinput.MOUSE_SECONDARY)

    def doubleClick(self, x: int, y: int) -> None:
        """Double click at coordinates."""
        pydirectinput.doubleClick(x, y, button=pydirectinput.MOUSE_PRIMARY)
        pydirectinput.mouseUp()


class ImagineWindow:
    def __init__(self):
        windows = gw.getWindowsWithTitle("IMAGINE Version 1.")
        if not windows:
            raise ValueError("No IMAGINE windows found")
        self._window = windows[0]

    @property
    def left(self) -> int:
        return self._window.left

    @property
    def top(self) -> int:
        return self._window.top

    @property
    def width(self) -> int:
        return self._window.width

    @property
    def height(self) -> int:
        return self._window.height

    @property
    def image(self) -> np.ndarray:
        """
        Capture the window contents as OpenCV numpy array.

        Returns:
            OpenCV numpy array (BGR format) of the window contents
        """
        self._window.activate()
        time.sleep(0.1)  # Allow window activation to complete

        with mss.mss() as sct:
            region = {
                "left": self._window.left,
                "top": self._window.top,
                "width": self._window.width,
                "height": self._window.height,
            }
            capture = sct.grab(region)

        # Convert MSS capture to OpenCV format (BGR numpy array)
        img_array = np.array(capture)
        return cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    def match_template(
        self, template_filename: str, confidence: float = 0.8
    ) -> tuple[int, int] | None:
        """
        Match a template against the window contents.

        Args:
            template_filename: Filename of template image in templates/ directory
            confidence: Minimum confidence threshold (0-1), defaults to 0.8

        Returns:
            Tuple of (screen_x, screen_y) absolute screen coordinates if match found,
            None if no match above confidence threshold

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
        result = cv2.matchTemplate(self.image, template, cv2.TM_CCOEFF_NORMED)

        # Find the best match location
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Check if confidence meets threshold
        if max_val < confidence:
            return None

        # Convert window-relative coordinates to screen-absolute coordinates
        window_x, window_y = max_loc
        screen_x = self.left + window_x
        screen_y = self.top + window_y

        return (screen_x, screen_y)


def test_match(template_filename: str) -> None:
    """Test function to match template and move mouse to location."""
    try:
        window = ImagineWindow()
        match_pos = window.match_template(template_filename)

        if match_pos:
            x, y = match_pos
            pydirectinput.moveTo(x, y)
            print(f"Template '{template_filename}' found - mouse moved to: ({x}, {y})")
        else:
            print(
                f"Template '{template_filename}' not found with sufficient confidence"
            )

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_match("thread.png")
