"""Automation framework for IMAGINE game interaction.

Provides command pattern-based automation for window capture, template matching,
and input control for the IMAGINE game client.
"""

import mss
import os
import time
import pygetwindow as gw
import cv2
import numpy as np
import pydirectinput
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


class CommandStatus(Enum):
    """Status enumeration for command execution results."""

    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class CommandResult:
    """Result of command execution with status and optional message."""

    status: CommandStatus
    message: str = ""


class Command(ABC):
    """Abstract base class for automation commands."""

    @abstractmethod
    def execute(self, context: dict) -> CommandResult:
        """Execute the command with given context.

        Args:
            context: Shared execution context dictionary

        Returns:
            CommandResult indicating success or failure
        """
        pass


class ScrollCommand(Command):
    """Command to perform mouse scroll operations."""

    def __init__(self, clicks: int):
        """Initialize scroll command with click count."""
        self.clicks = clicks

    def execute(self, context: dict) -> CommandResult:
        """Execute scroll operation."""
        pydirectinput.scroll(self.clicks, interval=0.015)
        return CommandResult(CommandStatus.SUCCESS)


class WaitCommand(Command):
    """Command to pause execution for a specified duration."""

    def __init__(self, seconds: float):
        """Initialize wait command with duration in seconds."""
        self.seconds = seconds

    def execute(self, context: dict) -> CommandResult:
        """Execute wait operation."""
        time.sleep(self.seconds)
        return CommandResult(CommandStatus.SUCCESS)


class HotkeyCommand(Command):
    """Command to send keyboard hotkey combinations."""

    def __init__(self, *keys: str):
        """Initialize hotkey command with key combination."""
        self.keys = keys

    def execute(self, context: dict) -> CommandResult:
        """Execute hotkey operation."""
        pydirectinput.hotkey(*self.keys, wait=0.05)
        return CommandResult(CommandStatus.SUCCESS)


class ClickCommand(Command):
    """Command to perform mouse click operations."""

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        button=pydirectinput.MOUSE_LEFT,
        clicks: int = 1,
    ):
        """Initialize click command.

        Args:
            x: X coordinate to click (uses last template location if None)
            y: Y coordinate to click (uses last template location if None)
            button: Mouse button to click (default: left button)
            clicks: Number of clicks to perform
        """
        self.x = x
        self.y = y
        self.button = button
        self.clicks = clicks

    def execute(self, context: dict) -> CommandResult:
        """Execute click command.

        Args:
            context: Shared context dict containing:
                - last_template_location: (x, y) tuple if coordinates not provided

        Returns:
            CommandResult with SUCCESS status
        """
        if self.x is not None and self.y is not None:
            x, y = self.x, self.y
        else:
            x, y = context["last_template_location"]

        pydirectinput.click(x, y, clicks=self.clicks, button=self.button)
        pydirectinput.mouseUp(button=self.button)
        return CommandResult(CommandStatus.SUCCESS)


class LocateTemplateCommand(Command):
    """Command to locate template images within the game window."""

    def __init__(
        self, template_id: str, confidence: float = 0.8, grayscale: bool = False
    ):
        """Initialize template matching command.

        Args:
            template_id: Name of template file (without .png extension)
            confidence: Minimum confidence threshold for template matching
            grayscale: Whether to perform matching in grayscale
        """
        self.template_id = template_id
        self.confidence = confidence
        self.grayscale = grayscale

    def execute(self, context: dict) -> CommandResult:
        """
        Execute template matching command.

        Args:
            context: Shared context dict containing:
                - window_capture: np.ndarray of captured window
                - window_left: int, window's left screen coordinate
                - window_top: int, window's top screen coordinate

        Returns:
            CommandResult with SUCCESS if template found above confidence threshold,
            FAILURE otherwise. On success, sets context["last_template_location"] to
            (screen_x, screen_y) tuple of absolute screen coordinates.
        """
        window_capture = context["window_capture"]
        window_left = context["window_left"]
        window_top = context["window_top"]

        template_path = os.path.join("templates", f"{self.template_id}.png")
        if not os.path.exists(template_path):
            return CommandResult(
                CommandStatus.FAILURE, f"Template not found: {template_path}"
            )

        template = cv2.imread(template_path)
        if template is None:
            return CommandResult(
                CommandStatus.FAILURE, f"Could not load template: {template_path}"
            )

        if self.grayscale:
            window_capture = cv2.cvtColor(window_capture, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(window_capture, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.confidence:
            return CommandResult(
                CommandStatus.FAILURE,
                f"Template '{self.template_id}' not found with confidence {self.confidence}",
            )

        window_x, window_y = max_loc
        screen_x = window_left + window_x
        screen_y = window_top + window_y

        context["last_template_location"] = (screen_x, screen_y)
        return CommandResult(CommandStatus.SUCCESS)


class ImagineWindow:
    """Wrapper class for IMAGINE game window interaction and automation."""

    TITLE = "IMAGINE Version 1."

    def __init__(self):
        """Initialize IMAGINE window interface."""
        windows = gw.getWindowsWithTitle(self.TITLE)
        if not windows:
            raise ValueError("No IMAGINE windows found")
        self._window = windows[0]

    @property
    def left(self) -> int:
        """Left screen coordinate of the window."""
        return self._window.left

    @property
    def top(self) -> int:
        """Top screen coordinate of the window."""
        return self._window.top

    @property
    def width(self) -> int:
        """Width of the window in pixels."""
        return self._window.width

    @property
    def height(self) -> int:
        """Height of the window in pixels."""
        return self._window.height

    @property
    def capture(self) -> np.ndarray:
        """
        Capture the window contents as OpenCV numpy array.

        Returns:
            OpenCV numpy array (BGR format) of the window contents
        """
        with mss.mss() as sct:
            capture_region = {
                "left": self._window.left,
                "top": self._window.top,
                "width": self._window.width,
                "height": self._window.height,
            }
            capture = sct.grab(capture_region)

        # Convert MSS capture to OpenCV format (BGR numpy array)
        capture_array = np.array(capture)
        return cv2.cvtColor(capture_array, cv2.COLOR_BGRA2BGR)

    def focus(self) -> None:
        """Bring the IMAGINE window to the foreground."""
        while True:
            active_window = gw.getActiveWindow()
            if active_window is not None and self.TITLE in active_window.title:
                break
            self._window.activate()


class Bot:
    """Bot class that manages states and executes automation commands."""

    def __init__(self):
        """Initialize bot with window and starting state."""
        self.window = ImagineWindow()
        self.state = NormalizeState(self)

    def execute_commands(self, *commands: Command) -> bool:
        """
        Execute commands sequentially with shared context.
        Ensures window is focused before each command execution.

        Args:
            *commands: Variable number of Command objects to execute

        Returns:
            True if all commands succeeded, False if any command failed
        """
        context = {}
        for command in commands:
            self.window.focus()
            context["window_capture"] = self.window.capture
            context["window_left"] = self.window.left
            context["window_top"] = self.window.top
            result = command.execute(context)
            if result.status == CommandStatus.FAILURE:
                print(
                    f"Command failed: {command.__class__.__name__}{f' - {result.message}' if result.message else ''}"
                )
                return False
            time.sleep(0.2)
        return True

    def run(self) -> None:
        """Run states until None is encountered."""
        while self.state:
            self.state = self.state.run()

    def stop(self) -> None:
        """Stop the bot by setting state to None."""
        self.state = None


class State(ABC):
    """Abstract base class for bot states."""

    def __init__(self, bot: Bot):
        """Initialize state with reference to bot."""
        self.bot = bot

    @abstractmethod
    def run(self) -> "State | None":
        """Execute state logic.

        Returns:
            Next state to transition to, or None to stop the bot
        """
        pass


class CheckLocationState(State):
    def run(self) -> State | None:
        if self.bot.execute_commands(
            LocateTemplateCommand("map", grayscale=True),
            ClickCommand(),
            LocateTemplateCommand("map_cathedral"),
            HotkeyCommand("shift", "c"),
        ):
            return None
        else:
            return ThreadToCathedralState(self.bot)


class ThreadToCathedralState(State):
    def run(self) -> State | None:
        if self.bot.execute_commands(
            LocateTemplateCommand("thread"),
            ClickCommand(clicks=2),
            LocateTemplateCommand("thread_cathedral"),
            ClickCommand(clicks=2),
            LocateTemplateCommand("thread_yes"),
            ClickCommand(),
            WaitCommand(1),
        ):
            return CheckLocationState(self.bot)
        else:
            return self


class NormalizeState(State):
    def run(self) -> State | None:
        if self.bot.execute_commands(
            HotkeyCommand("shift", "h"),
            LocateTemplateCommand("show_ui", grayscale=True),
            ScrollCommand(75),
            ScrollCommand(-15),
            HotkeyCommand("shift", "h"),
            HotkeyCommand("shift", "c"),
        ):
            return ThreadToCathedralState(self.bot)
        else:
            return self


if __name__ == "__main__":
    bot = Bot()
    bot.run()
