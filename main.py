"""Automation framework for the IMAGINE game client.

Provides tools for matching templates, dispatching commands in isolation and in sequence,
and managing state. The framework is built to have moderate resilience against network
instability and aberrant client responses. Created strictly for rebirthing, it may be extended
to fulfill other automation tasks (especially UI tasks).
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import mss
import numpy
import pydirectinput
import pywinctl

# Global delay between each pydirectinput call
pydirectinput.PAUSE = 0.06


class CommandStatus(Enum):
    """Status enumeration for command execution results."""

    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class CommandResult:
    """Result of command execution with status and optional message."""

    status: CommandStatus
    message: str = ""


@dataclass
class CommandContext:
    """Execution context containing client data and state for commands."""

    capture: numpy.ndarray | None = None
    origin: tuple[int, int] | None = None
    center: tuple[int, int] | None = None
    last_template_location: tuple[int, int] | None = None


class Command(ABC):
    """Abstract base class for automation commands."""

    @abstractmethod
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the command with given context.

        Args:
            context: Shared execution context containing client data

        Returns:
            CommandResult indicating success or failure
        """
        pass


class WaitCommand(Command):
    """Command to pause execution for a specified duration."""

    def __init__(self, seconds: float):
        """Initialize wait command with duration in seconds."""
        super().__init__()
        self.seconds = seconds

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute wait operation."""
        time.sleep(self.seconds)
        return CommandResult(CommandStatus.SUCCESS)


class HotkeyCommand(Command):
    """Command to send keyboard hotkey combinations."""

    def __init__(self, *keys: str, presses: int = 1):
        """Initialize hotkey command with key combination."""
        super().__init__()
        self.keys = keys
        self.presses = presses

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute hotkey operation."""
        for _ in range(self.presses):
            pydirectinput.hotkey(*self.keys, wait=0.05)
        return CommandResult(CommandStatus.SUCCESS)


class ClickCommand(Command):
    """Command to perform mouse click operations."""

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = pydirectinput.MOUSE_PRIMARY,
        click_count: int = 1,
    ):
        """Initialize click command.

        Args:
            x: X coordinate to click (uses last template location if None)
            y: Y coordinate to click (uses last template location if None)
            button: Mouse button to click (default: left button)
            click_count: Number of clicks to perform
        """
        super().__init__()
        self.x = x
        self.y = y
        self.button = button
        self.click_count = click_count

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute click command.

        Args:
            context: Shared context containing last_template_location if coordinates not provided

        Returns:
            CommandResult with SUCCESS status
        """
        if self.x is not None and self.y is not None:
            x, y = self.x, self.y
        else:
            x, y = context.last_template_location or context.center

        for _ in range(self.click_count):
            pydirectinput.moveTo(x, y, attempt_pixel_perfect=True)
            pydirectinput.mouseDown(x, y, button=self.button)
            pydirectinput.mouseUp(button=self.button)
        return CommandResult(CommandStatus.SUCCESS)


class DragCommand(Command):
    """Command to perform mouse drag operations."""

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = pydirectinput.MOUSE_SECONDARY,
        drag_count: int = 1,
    ):
        """Initialize drag command.

        Args:
            x: X offset to drag relative to current position
            y: Y offset to drag relative to current position
            button: Mouse button to use for dragging (default: secondary button)
        """
        super().__init__()
        self.x = x
        self.y = y
        self.button = button
        self.drag_count = drag_count

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute drag command.

        Returns:
            CommandResult with SUCCESS status
        """
        center_x, center_y = context.center
        for _ in range(self.drag_count):
            pydirectinput.moveTo(center_x, center_y, attempt_pixel_perfect=True)
            pydirectinput.mouseDown(button=pydirectinput.MOUSE_SECONDARY)
            pydirectinput.moveTo(
                center_x + self.x,
                center_y + self.y,
                attempt_pixel_perfect=True,
            )
            pydirectinput.mouseUp(button=pydirectinput.MOUSE_SECONDARY)
        return CommandResult(CommandStatus.SUCCESS)


class LocateTemplateCommand(Command):
    """Command to locate template images within the client capture."""

    def __init__(
        self,
        template_id: str,
        confidence: float = 0.8,
        grayscale: bool = True,
    ):
        """Initialize template matching command.

        Args:
            template_id: Name of template file (without .png extension)
            confidence: Minimum confidence threshold for template matching
            grayscale: Whether to perform matching in grayscale
        """
        super().__init__()
        self.template_id = template_id
        self.confidence = confidence
        self.grayscale = grayscale

    def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute template matching command.

        Args:
            context: Shared context containing client capture and coordinates

        Returns:
            CommandResult with SUCCESS if template found above confidence threshold,
            FAILURE otherwise. On success, sets context.last_template_location to
            (screen_x, screen_y) tuple of absolute screen coordinates for template center.
        """
        capture = context.capture
        x, y = context.origin

        template_path = os.path.join("templates", f"{self.template_id}.png")
        template = cv2.imread(template_path)
        if template is None:
            return CommandResult(
                CommandStatus.FAILURE, f"Could not load template: {template_path}"
            )

        if self.grayscale:
            capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(capture, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.confidence:
            return CommandResult(
                CommandStatus.FAILURE,
                f"Template '{self.template_id}' not found with confidence {self.confidence}",
            )

        template_height, template_width = template.shape[:2]
        match_x, match_y = max_loc
        screen_x = x + match_x + template_width // 2
        screen_y = y + match_y + template_height // 2

        context.last_template_location = (screen_x, screen_y)
        return CommandResult(CommandStatus.SUCCESS)


class ImagineClient:
    """Wrapper class for IMAGINE client data and interaction."""

    IMAGINE_CLIENT_IDENTIFIER = "IMAGINE Version 1."

    def __init__(self):
        """Initialize the IMAGINE client window reference."""
        windows = pywinctl.getWindowsWithTitle(
            self.IMAGINE_CLIENT_IDENTIFIER, condition=pywinctl.Re.STARTSWITH
        )
        self._window = windows[0]

    @property
    def frame(self) -> tuple[int, int, int, int]:
        """Bounding box of the client frame (x, y, right, bottom)."""
        return self._window.getClientFrame()

    @property
    def x(self) -> int:
        """Left coordinate of the client frame."""
        return self.frame[0]

    @property
    def y(self) -> int:
        """Top coordinate of the client frame."""
        return self.frame[1]

    @property
    def right(self) -> int:
        """Right coordinate of the client frame."""
        return self.frame[2]

    @property
    def bottom(self) -> int:
        """Bottom coordinate of the client frame."""
        return self.frame[3]

    @property
    def center_x(self) -> int:
        """Horizontal center coordinate of the client frame."""
        return (self.x + self.right) // 2

    @property
    def center_y(self) -> int:
        """Vertical center coordinate of the client frame."""
        return (self.y + self.bottom) // 2

    @property
    def capture(self) -> numpy.ndarray:
        """
        Capture the client frame contents as OpenCV numpy array.

        Returns:
            OpenCV numpy array (BGR format) of the client frame contents
        """
        with mss.mss() as screenshot:
            capture = screenshot.grab(self.frame)
        return cv2.cvtColor(numpy.array(capture), cv2.COLOR_BGRA2BGR)

    @property
    def focused(self) -> bool:
        """Check if the client window is currently focused."""
        return self.IMAGINE_CLIENT_IDENTIFIER in pywinctl.getActiveWindowTitle()

    def focus(self) -> None:
        """Bring the client window to the foreground."""
        self._window.activate()


class Bot:
    """Base bot class that manages state and executes automation commands."""

    def __init__(self, state: "State | None" = None):
        """Initialize bot with starting state and IMAGINE client reference."""
        self.state = state
        self.client = ImagineClient()

    def execute_commands(self, *commands: Command) -> bool:
        """
        Execute commands sequentially with shared context.
        Ensures client is focused before each command execution.

        Args:
            *commands: Variable number of Command objects to execute

        Returns:
            True if all commands succeeded, False if any command failed
        """
        context = CommandContext()
        for command in commands:
            time.sleep(0.05)

            if not self.client.focused:
                sys.exit()

            context.capture = self.client.capture
            context.origin = (self.client.x, self.client.y)
            context.center = (self.client.center_x, self.client.center_y)
            result = command.execute(context)

            if result.status == CommandStatus.FAILURE:
                print(
                    f"Command failed: {command.__class__.__name__}{f' - {result.message}' if result.message else ''}"
                )
                return False
        return True

    def run(self) -> None:
        """Run states until None is encountered."""
        self.client.focus()
        elapsed = 0
        while self.state:
            start_time = time.time()
            result = self.state.run(elapsed)
            end_time = time.time()

            if result.next_state is not None:
                if not isinstance(self.state, result.next_state):
                    elapsed = 0
                else:
                    elapsed += end_time - start_time

                self.state = result.next_state(self, **(result.next_state_kwargs or {}))
            else:
                self.state = None


class RebirthBot(Bot):
    """Bot for automating rebirths."""

    def __init__(self):
        """Initialize bot with starting state."""
        super().__init__(RelogState(self))

    def count_rebirths(self) -> list[int] | None:
        """Count the number of rebirths on each rebirth path."""
        capture = self.client.capture

        border_bgr = numpy.array([151, 107, 13])
        empty_slot_bgr = numpy.array([67, 47, 8])

        border_mask = cv2.inRange(capture, border_bgr, border_bgr)
        empty_slot_mask = cv2.inRange(capture, empty_slot_bgr, empty_slot_bgr)

        try:
            # Origin coordinate (uppermost, leftmost border pixel)
            x, y = numpy.argwhere(border_mask > 0)[0][::-1]
        except IndexError:
            return None

        rebirths = [8] * 12
        roi_width = 162
        roi_height = 19
        for i in range(len(rebirths)):
            roi_y = y + (i * roi_height)
            roi_mask = empty_slot_mask[roi_y : roi_y + roi_height, x : x + roi_width]

            contours, _ = cv2.findContours(
                roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            empty_slots = sum(1 for contour in contours if cv2.contourArea(contour) > 0)

            rebirths[i] -= empty_slots
        return rebirths


class StateStatus(Enum):
    """Status enumeration for state execution results."""

    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class StateResult:
    """Result of state execution with status and next state transition information."""

    status: StateStatus
    message: str = ""
    next_state: "State | None" = None
    next_state_kwargs: dict | None = None


class State(ABC):
    """Abstract base class for bot states."""

    def __init__(
        self,
        bot: Bot,
        next_state: "State | None" = None,
        next_state_kwargs: dict | None = None,
        max_elapsed: float = numpy.inf,
    ):
        """Initialize state with reference to bot."""
        self.bot = bot
        self.next_state = next_state
        self.next_state_kwargs = next_state_kwargs
        self.max_elapsed = max_elapsed

    @abstractmethod
    def run(self, elapsed: float) -> StateResult:
        """Execute state logic.

        Returns:
            Next state to transition to, or None to stop the bot
        """
        pass


class ThreadToCathedralState(State):
    def run(self, elapsed: float) -> StateResult:
        self.bot.execute_commands(HotkeyCommand("esc"))
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": self.next_state or ApproachCathedralMasterState,
                "next_state_kwargs": self.next_state_kwargs,
                "sequence": (
                    "thread",
                    "thread_cathedral",
                    "thread_yes",
                ),
                "sequence_complete": lambda: not self.bot.execute_commands(
                    LocateTemplateCommand("thread_yes")
                ),
                "click_count": 2,
            },
        )


class ResetUIState(State):
    def run(self, elapsed: float) -> StateResult:
        self.bot.execute_commands(
            HotkeyCommand("esc"),
            HotkeyCommand("shift", "c"),
            HotkeyCommand("shift", "h"),
        )

        if not self.bot.execute_commands(LocateTemplateCommand("show_ui")):
            return StateResult(
                status=StateStatus.FAILURE,
                next_state=ResetUIState,
                next_state_kwargs={"next_state": self.next_state},
            )

        self.bot.execute_commands(
            HotkeyCommand("shift", "h"),
            LocateTemplateCommand("show_bar"),
            ClickCommand(),
        )
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": self.next_state,
                "sequence": ("inventory",),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("inventory_window")
                ),
            },
        )


class ResetCameraState(State):
    def run(self, elapsed: float) -> StateResult:
        self.bot.execute_commands(
            HotkeyCommand("shift", "h"),
            DragCommand(400, 0, drag_count=3),
            HotkeyCommand("shift", "h"),
        )
        return StateResult(
            status=StateStatus.SUCCESS, next_state=ThreadToCathedralState
        )


class RelogState(State):
    def run(self, elapsed: float) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ReloggedState,
                "sequence": (
                    "system",
                    "system_select_character",
                    "start_game",
                ),
                "sequence_complete": lambda: not self.bot.execute_commands(
                    LocateTemplateCommand("select_character")
                ),
            },
        )


class ReloggedState(State):
    def run(self, elapsed: float) -> StateResult:
        if not self.bot.execute_commands(LocateTemplateCommand("minimize")):
            return StateResult(status=StateStatus.FAILURE, next_state=ReloggedState)

        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=ResetUIState,
            next_state_kwargs={"next_state": ResetCameraState},
        )


class ApproachCathedralMasterState(State):
    def __init__(self, bot: Bot):
        super().__init__(bot, max_elapsed=5.0)

    def run(self, elapsed: float) -> StateResult:
        if elapsed == 0:
            self.bot.execute_commands(DragCommand(90, 0))
        elif elapsed >= self.max_elapsed:
            return StateResult(StateStatus.FAILURE, next_state=ThreadToCathedralState)

        if not self.bot.execute_commands(
            LocateTemplateCommand("cathedral_perform_rebirth_1")
        ):
            self.bot.execute_commands(ClickCommand())
            return StateResult(
                StateStatus.FAILURE, next_state=ApproachCathedralMasterState
            )

        return StateResult(StateStatus.SUCCESS, next_state=CathedralMasterState)


class CathedralMasterState(State):
    def run(self, elapsed: float) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ThreadToCathedralState,
                "next_state_kwargs": {"next_state": ApproachVivianState},
                "sequence": (
                    "cathedral_perform_rebirth_1",
                    "cathedral_perform_rebirth_2",
                    "cathedral_close",
                    "cathedral_stop",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("minimize")
                ),
            },
        )


class ApproachVivianState(State):
    def __init__(self, bot: Bot):
        super().__init__(bot, max_elapsed=7.5)

    def run(self, elapsed: float) -> StateResult:
        if elapsed == 0:
            self.bot.execute_commands(
                DragCommand(110, 0, drag_count=2),
                ClickCommand(),
            )
        elif elapsed >= self.max_elapsed:
            return StateResult(
                StateStatus.FAILURE,
                next_state=ThreadToCathedralState,
                next_state_kwargs={"next_state": ApproachVivianState},
            )

        if not self.bot.execute_commands(LocateTemplateCommand("vivian_demon_level")):
            self.bot.execute_commands(ClickCommand())
            return StateResult(StateStatus.FAILURE, next_state=ApproachVivianState)

        return StateResult(StateStatus.SUCCESS, next_state=VivianState)


class VivianState(State):
    def run(self, elapsed: float) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ThreadToCathedralState,
                "sequence": (
                    "vivian_demon_level",
                    "dialogue_end_conversation",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("minimize")
                ),
            },
        )


class SequenceState(State):
    def __init__(
        self,
        bot: Bot,
        sequence: tuple[str],
        sequence_complete: Callable[[], bool] = lambda: False,
        index: int = 0,
        click_count: int = 1,
        next_state: State | None = None,
        next_state_kwargs: dict | None = None,
    ):
        super().__init__(bot, next_state, next_state_kwargs)
        self.sequence = sequence
        self.sequence_complete = sequence_complete
        self.index = index
        self.click_count = click_count

    def run(self, elapsed: float) -> StateResult:
        try:
            next_template = self.sequence[self.index + 1]
        except IndexError:
            next_template = None

        self.bot.execute_commands(
            LocateTemplateCommand(self.sequence[self.index]),
            ClickCommand(click_count=self.click_count),
        )

        if next_template is not None and self.bot.execute_commands(
            LocateTemplateCommand(next_template)
        ):
            self.index += 1
        elif next_template is None and self.sequence_complete():
            return StateResult(
                status=StateStatus.SUCCESS,
                next_state=self.next_state,
                next_state_kwargs=self.next_state_kwargs,
            )

        return StateResult(
            status=StateStatus.FAILURE,
            next_state=SequenceState,
            next_state_kwargs={
                "sequence": self.sequence,
                "sequence_complete": self.sequence_complete,
                "index": self.index,
                "click_count": self.click_count,
                # Destination state (passed to final StateResult on sequence success)
                "next_state": self.next_state,
                "next_state_kwargs": self.next_state_kwargs,
            },
        )


if __name__ == "__main__":
    bot = RebirthBot()
    bot.state = ReloggedState(bot)
    bot.run()
