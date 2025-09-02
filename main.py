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

    def __init__(self, max_attempts: int = 1):
        self.max_attempts = max_attempts

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
        cx, cy = context.center
        for _ in range(self.drag_count):
            pydirectinput.moveTo(cx, cy, attempt_pixel_perfect=True)
            pydirectinput.mouseDown(button=pydirectinput.MOUSE_SECONDARY)
            pydirectinput.moveTo(
                cx + self.x,
                cy + self.y,
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
        color: bool = False,
    ):
        """Initialize template matching command.

        Args:
            template_id: Name of template file (without .png extension)
            confidence: Minimum confidence threshold for template matching
            color: Whether to perform matching in color
        """
        super().__init__()
        self.template_id = template_id
        self.confidence = confidence
        self.color = color

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

        if not self.color:
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
    def cx(self) -> int:
        """Horizontal center coordinate of the client frame."""
        return (self.x + self.right) // 2

    @property
    def cy(self) -> int:
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
        # Convert MSS capture to OpenCV format (BGR numpy array)
        capture_array = numpy.array(capture)
        return cv2.cvtColor(capture_array, cv2.COLOR_BGRA2BGR)

    @property
    def focused(self) -> bool:
        """Check if the client window is currently focused."""
        return self.IMAGINE_CLIENT_IDENTIFIER in pywinctl.getActiveWindowTitle()

    def focus(self) -> None:
        """Bring the client window to the foreground."""
        self._window.activate()


class ImagineBot:
    """Bot class that manages states and executes automation commands."""

    def __init__(self):
        """Initialize bot with IMAGINE client reference and starting state."""
        self.client = ImagineClient()
        self.state = ResetUIState(self, next_state=ResetCameraState)

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
            for attempt in range(command.max_attempts):
                time.sleep(0.1)

                if not self.client.focused:
                    sys.exit()

                context.capture = self.client.capture
                context.origin = (self.client.x, self.client.y)
                context.center = (self.client.cx, self.client.cy)
                result = command.execute(context)

                if result.status == CommandStatus.SUCCESS:
                    break

                print(
                    f"Command failed: {command.__class__.__name__}{f' - {result.message}' if result.message else ''}"
                )

                if attempt + 1 == command.max_attempts:
                    return False
        return True

    def run(self) -> None:
        """Run states until None is encountered."""
        self.client.focus()
        attempt = 1
        while self.state:
            result = self.state.run(attempt)

            if isinstance(self.state, result.next_state):
                if result.status == StateStatus.FAILURE:
                    attempt += 1
            else:
                attempt = 1

            if attempt > self.state.max_attempts:
                # TODO: Fallback state?
                self.state = ThreadToCathedralState(self)
            elif result.next_state is not None:
                self.state = result.next_state(self, **(result.next_state_kwargs or {}))
            else:
                self.state = None


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
        bot: ImagineBot,
        next_state: "State | None" = None,
        next_state_kwargs: dict | None = None,
        max_attempts: int = numpy.inf,
    ):
        """Initialize state with reference to bot."""
        self.bot = bot
        self.next_state = next_state
        self.next_state_kwargs = next_state_kwargs
        self.max_attempts = max_attempts

    @abstractmethod
    def run(self, attempt: int) -> StateResult:
        """Execute state logic.

        Returns:
            Next state to transition to, or None to stop the bot
        """
        pass


class ThreadToCathedralState(State):
    def run(self, attempt: int) -> StateResult:
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
    def run(self, attempt: int) -> StateResult:
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
    def run(self, attempt: int) -> StateResult:
        self.bot.execute_commands(
            HotkeyCommand("shift", "h"),
            DragCommand(400, 0, drag_count=3),
            HotkeyCommand("shift", "h"),
        )
        return StateResult(
            status=StateStatus.SUCCESS, next_state=ThreadToCathedralState
        )


class RelogState(State):
    def run(self, attempt: int) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": FreshState,
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


class FreshState(State):
    def run(self, attempt: int) -> StateResult:
        if not self.bot.execute_commands(LocateTemplateCommand("minimize")):
            return StateResult(status=StateStatus.FAILURE, next_state=FreshState)

        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=ResetUIState,
            next_state_kwargs={
                "next_state": ResetCameraState,
            },
        )


class ApproachCathedralMasterState(State):
    def __init__(self, bot: ImagineBot):
        super().__init__(bot, max_attempts=25)

    def run(self, attempt: int) -> StateResult:
        if attempt == 1:
            self.bot.execute_commands(
                DragCommand(90, 0),
                ClickCommand(),
            )

        if self.bot.execute_commands(LocateTemplateCommand("minimize")):
            return StateResult(
                StateStatus.FAILURE, next_state=ApproachCathedralMasterState
            )

        return StateResult(StateStatus.SUCCESS, next_state=CathedralMasterState)


class CathedralMasterState(State):
    def run(self, attempt: int) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ApproachVivianState,
                "sequence": (
                    "dialogue_arrow",
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
    def __init__(self, bot: ImagineBot, approach_directly: bool = False):
        super().__init__(bot, max_attempts=25)
        self.approach_directly = approach_directly

    def run(self, attempt: int) -> StateResult:
        if attempt == 1:
            if self.approach_directly:
                self.bot.execute_commands(
                    DragCommand(110, 0, drag_count=2),
                    ClickCommand(),
                )
            else:
                self.bot.execute_commands(
                    DragCommand(310, 0),
                    ClickCommand(),
                )
        elif attempt == self.max_attempts:
            return StateResult(
                StateStatus.FAILURE,
                next_state=ThreadToCathedralState,
                next_state_kwargs={
                    "next_state": ApproachVivianState,
                    "next_state_kwargs": {"approach_directly": True},
                },
            )

        if self.bot.execute_commands(LocateTemplateCommand("minimize")):
            return StateResult(StateStatus.FAILURE, next_state=ApproachVivianState)

        return StateResult(StateStatus.SUCCESS, next_state=VivianState)


class VivianState(State):
    def run(self, attempt: int) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ThreadToCathedralState,
                "sequence": (
                    "dialogue_arrow",
                    "dialogue_arrow",
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
        bot: ImagineBot,
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

    def run(self, attempt: int) -> StateResult:
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
    bot = ImagineBot()
    bot.run()
