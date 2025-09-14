"""Automation framework for the IMAGINE game client.

Provides tools for matching templates, dispatching commands in isolation and in sequence,
and managing state. The framework is built to have moderate resilience against network
instability and aberrant client responses. Created strictly for rebirthing, it may be extended
to fulfill other automation tasks (especially UI tasks).
"""

import abc
import collections.abc
import dataclasses
import enum
import logging
import os
import re
import sys
import time

import cv2
import dotenv
import mss
import numpy
import pydirectinput
import pywinctl

# Global delay between each pydirectinput call
pydirectinput.PAUSE = 0.06

logger = logging.getLogger(__name__)

# Load and cache the grayscale templates ahead of time to improve performance of LocateTemplateCommand
templates = {
    os.path.splitext(filename)[0]: cv2.cvtColor(
        cv2.imread(os.path.join("templates", filename)), cv2.COLOR_BGR2GRAY
    )
    for filename in os.listdir("templates")
    if filename.endswith(".png")
}


class CommandStatus(enum.Enum):
    """Status enumeration for command execution results."""

    SUCCESS = enum.auto()
    FAILURE = enum.auto()


@dataclasses.dataclass
class CommandResult:
    """Result of command execution with status and optional message."""

    status: CommandStatus
    message: str = ""


@dataclasses.dataclass
class CommandContext:
    """Execution context containing client data and state for commands."""

    capture: numpy.ndarray | None = None
    origin: tuple[int, int] | None = None
    center: tuple[int, int] | None = None
    last_template_location: tuple[int, int] | None = None
    last_template_index: int | None = None


class Command(abc.ABC):
    """Abstract base class for automation commands."""

    @abc.abstractmethod
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
        self.seconds = seconds

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute wait operation."""
        time.sleep(self.seconds)
        return CommandResult(CommandStatus.SUCCESS)


class HotkeyCommand(Command):
    """Command to send keyboard hotkey combinations."""

    def __init__(self, *keys: str, presses: int = 1):
        """Initialize hotkey command with key combination."""
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
        x: int = 0,
        y: int = 0,
        button: str = pydirectinput.MOUSE_PRIMARY,
        click_count: int = 1,
    ):
        """Initialize click command.

        Args:
            x: X offset relative to template/center location
            y: Y offset relative to template/center location
            button: Mouse button to click (default: left button)
            click_count: Number of clicks to perform
        """
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
        x, y = numpy.array(
            context.last_template_location or context.center
        ) + numpy.array((self.x, self.y))

        for _ in range(self.click_count):
            pydirectinput.moveTo(x, y, attempt_pixel_perfect=True)
            pydirectinput.mouseDown(x, y, button=self.button)
            pydirectinput.mouseUp(button=self.button)
        return CommandResult(CommandStatus.SUCCESS)


class DragCommand(Command):
    """Command to perform mouse drag operations."""

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        button: str = pydirectinput.MOUSE_SECONDARY,
        drag_count: int = 1,
    ):
        """Initialize drag command.

        Args:
            x: X offset to drag relative to current position
            y: Y offset to drag relative to current position
            button: Mouse button to use for dragging (default: secondary button)
            drag_count: Number of drag operations to perform
        """
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
        templates: list[str] | str,
        confidence: float = 0.8,
        region: tuple[int, int, int, int] | None = None,
    ):
        """Initialize template matching command.

        Args:
            templates: Template name(s) - single string or list of strings (without .png extension)
            confidence: Minimum confidence threshold for template matching
            region: Optional region to search within as (x, y, width, height)
        """
        self.templates = [templates] if isinstance(templates, str) else templates
        self.confidence = confidence
        self.region = region

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
        capture = cv2.cvtColor(context.capture, cv2.COLOR_BGR2GRAY)
        x, y = context.origin

        if self.region is not None:
            _x, _y = numpy.array(
                context.last_template_location or context.origin
            ) + numpy.array(self.region[:2])

            start_x = max(0, _x - x)
            start_y = max(0, _y - y)
            end_x = min(capture.shape[1], start_x + self.region[2])
            end_y = min(capture.shape[0], start_y + self.region[3])

            capture = capture[start_y:end_y, start_x:end_x]
            x += start_x
            y += start_y

        template_index = None
        for template in numpy.random.permutation(self.templates):
            result = cv2.matchTemplate(
                capture, templates[template], cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > self.confidence:
                template_index = self.templates.index(template)
                break

        if template_index is None:
            return CommandResult(
                CommandStatus.FAILURE,
                f"No match for '{', '.join(self.templates)}' with confidence {self.confidence}",
            )

        template_height, template_width = templates[
            self.templates[template_index]
        ].shape[:2]
        match_x, match_y = max_loc
        screen_x = x + match_x + template_width // 2
        screen_y = y + match_y + template_height // 2

        context.last_template_index = template_index
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
        try:
            self._window = windows[0]
        except IndexError:
            logger.error("No IMAGINE client windows available.")

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


class BotSelection(enum.StrEnum):
    """Bot type selection enumeration."""

    REBIRTH = enum.auto()


@dataclasses.dataclass
class BotConfig:
    """Base configuration for bot instances."""

    relog: bool
    sleep_amount: float


@dataclasses.dataclass
class BotContext:
    """Base context for bot instances."""

    pass


class Bot[BotConfigType: BotConfig, BotContextType: BotContext](abc.ABC):
    """Base bot class that manages state and executes automation commands."""

    def __init__(
        self,
        config: BotConfigType,
        context: BotContextType,
        state: "State | None" = None,
    ):
        """Initialize bot with starting state and IMAGINE client reference."""
        self.config = config
        self.context = context
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
        success, _ = self.execute_commands_with_context(*commands)
        return success

    def execute_commands_with_context(
        self, *commands: Command
    ) -> tuple[bool, CommandContext]:
        """Execute commands and return both success status and command context."""
        context = CommandContext()
        for command in commands:
            time.sleep(self.config.sleep_amount)

            if not self.client.focused:
                logger.error("IMAGINE client window lost focus.")
                sys.exit()

            context.capture = self.client.capture
            context.origin = (self.client.x, self.client.y)
            context.center = (self.client.center_x, self.client.center_y)
            result = command.execute(context)

            if result.status == CommandStatus.FAILURE:
                logger.debug(f"Failed: {command.__class__.__name__} - {result.message}")
                return (False, context)
        return (True, context)

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

            if result.message == "":
                continue

            match result.status:
                case StateStatus.FAILURE:
                    logger.error(result.message)
                case StateStatus.SUCCESS:
                    logger.info(result.message)


class RebirthPath(enum.StrEnum):
    """Enumeration of available rebirth paths in IMAGINE."""

    TIWAZ = enum.auto()
    PEORTH = enum.auto()
    FEHU = enum.auto()
    EIHWAZ = enum.auto()
    URUZ = enum.auto()
    HAGALAZ = enum.auto()
    LAGUZ = enum.auto()
    ANSUZ = enum.auto()
    NAUTHIZ = enum.auto()
    INGWAZ = enum.auto()
    SOWILO = enum.auto()
    WYRD = enum.auto()


@dataclasses.dataclass
class RebirthBotConfig(BotConfig):
    """Configuration for rebirth automation bot."""

    end_counts: list[int]
    end_path: RebirthPath


@dataclasses.dataclass
class RebirthBotContext(BotContext):
    """Context for rebirth automation bot."""

    counts: list[int] | None = None
    path_changing: bool = False


class RebirthBot(Bot[RebirthBotConfig, RebirthBotContext]):
    """Bot for automating rebirths."""

    def __init__(self, config: RebirthBotConfig):
        """Initialize bot with starting state."""
        super().__init__(config, RebirthBotContext())

    def next_path_index(self, counts: list[int]) -> int | None:
        """Determine the next rebirth path index to work on."""
        try:
            end_path_index = list(RebirthPath).index(self.config.end_path)
        except ValueError:
            end_path_index = None

        indices = [
            i
            for i, (current, target) in enumerate(zip(counts, self.config.end_counts))
            if current < target and i != end_path_index
        ]

        if (
            self.config.end_path is not None
            and counts[end_path_index] < self.config.end_counts[end_path_index]
        ):
            indices.append(end_path_index)

        try:
            return indices[0]
        except IndexError:
            return None

    def count_paths(self) -> list[int] | None:
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

        counts = [8] * 12
        roi_width = 162
        roi_height = 19
        for i in range(len(counts)):
            roi_y = y + (i * roi_height)
            roi_mask = empty_slot_mask[roi_y : roi_y + roi_height, x : x + roi_width]

            contours, _ = cv2.findContours(
                roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            empty_slots = sum(1 for contour in contours if cv2.contourArea(contour) > 0)

            counts[i] -= empty_slots
        return counts


class StateStatus(enum.Enum):
    """Status enumeration for state execution results."""

    SUCCESS = enum.auto()
    FAILURE = enum.auto()


@dataclasses.dataclass
class StateResult:
    """Result of state execution with status and next state transition information."""

    status: StateStatus
    message: str = ""
    next_state: "State | None" = None
    next_state_kwargs: dict | None = None


class State[BotType: Bot](abc.ABC):
    """Abstract base class for bot states."""

    def __init__(
        self,
        bot: BotType,
        next_state: "State | None" = None,
        next_state_kwargs: dict | None = None,
        max_elapsed: float = numpy.inf,
    ):
        """Initialize state with reference to bot."""
        self.bot = bot
        self.next_state = next_state
        self.next_state_kwargs = next_state_kwargs
        self.max_elapsed = max_elapsed

    @abc.abstractmethod
    def run(self, elapsed: float) -> StateResult:
        """Execute state logic.

        Returns:
            Next state to transition to, or None to stop the bot
        """
        pass


class ThreadToCathedralState(State[RebirthBot]):
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
                    "yes",
                ),
                "sequence_complete": lambda: not self.bot.execute_commands(
                    LocateTemplateCommand("yes")
                ),
                "click_count": 2,
            },
        )


class ResetUIState(State[Bot]):
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


class ResetCameraState(State[Bot]):
    def run(self, elapsed: float) -> StateResult:
        self.bot.execute_commands(
            HotkeyCommand("shift", "h"),
            DragCommand(400, 0, drag_count=3),
            HotkeyCommand("shift", "h"),
        )
        return StateResult(
            status=StateStatus.SUCCESS, next_state=ThreadToCathedralState
        )


class RelogState(State[Bot]):
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


class ReloggedState(State[Bot]):
    def run(self, elapsed: float) -> StateResult:
        if not self.bot.execute_commands(LocateTemplateCommand("player")):
            return StateResult(status=StateStatus.FAILURE, next_state=ReloggedState)

        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=ResetUIState,
            next_state_kwargs={"next_state": ResetCameraState},
        )


class ApproachCathedralMasterState(State[RebirthBot]):
    def __init__(self, bot: Bot, skipped_thread: bool = False):
        super().__init__(bot, max_elapsed=5.0)
        self.skipped_thread = skipped_thread

    def run(self, elapsed: float) -> StateResult:
        if elapsed == 0 and not self.skipped_thread:
            self.bot.execute_commands(DragCommand(90, 0))
        elif elapsed >= self.max_elapsed:
            return StateResult(
                StateStatus.FAILURE,
                message="Timed out approaching cathedral master. Retrying...",
                next_state=ThreadToCathedralState,
            )

        if not self.bot.execute_commands(LocateTemplateCommand("perform_rebirth_1")):
            self.bot.execute_commands(ClickCommand())
            return StateResult(
                StateStatus.FAILURE, next_state=ApproachCathedralMasterState
            )

        return StateResult(StateStatus.SUCCESS, next_state=CathedralMasterState)


class CathedralMasterState(State[RebirthBot]):
    def run(self, elapsed: float) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": RebirthCountState,
                "sequence": (
                    "perform_rebirth_1",
                    "perform_rebirth_2",
                    "view_demon_information",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("close_window")
                ),
            },
        )


class RebirthCountState(State[RebirthBot]):
    def run(self, elapsed: float):
        self.bot.execute_commands(
            LocateTemplateCommand("rebirth_tab"),
            ClickCommand(),
        )
        self.bot.context.counts = self.bot.count_paths()

        if self.bot.context.counts is None:
            return StateResult(
                StateStatus.FAILURE,
                message="Failed to count rebirth paths. Verify the window is in view.",
                next_state=RebirthCountState,
            )

        next_path_index = self.bot.next_path_index(self.bot.context.counts)

        if next_path_index is None:
            return StateResult(
                StateStatus.SUCCESS,
                message=f"Rebirths finished for paths: {','.join(map(str, self.bot.config.end_counts))} end path: {self.bot.config.end_path}",
            )

        path_index = None
        success, context = self.bot.execute_commands_with_context(
            LocateTemplateCommand("g_type"),
            LocateTemplateCommand(
                [f"path_{i}" for i in range(12)],
                region=(32, -10, 120, 20),
            ),
        )

        if success:
            path_index = context.last_template_index
        else:
            # Demon with a weird custom growth type, get the nonzero path
            path_index = self.bot.context.counts.index(1)

        self.bot.context.path_changing = next_path_index != path_index
        return StateResult(
            StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": RebirthState,
                "sequence": (
                    "close_window",
                    "rebirth_payment",
                    f"path_{next_path_index}",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand(f"path_icon_{next_path_index}")
                ),
            },
        )


class RebirthState(State[RebirthBot]):
    def run(self, elapsed: float):
        self.bot.execute_commands(
            LocateTemplateCommand("rebirth_payment"),
            ClickCommand(),
        )
        next_path_index = self.bot.next_path_index(self.bot.context.counts)

        # Matching templates for all possible items is unrealistic; attempt from left to right
        for payment_item_x in (190, 150, 110, 70):
            sequence = None
            projected_path_index = next_path_index

            if self.bot.execute_commands(
                LocateTemplateCommand("rebirth_level_warning")
            ):
                sequence = ("close", "stop")
            elif self.bot.execute_commands(
                LocateTemplateCommand("execute", confidence=0.95)
            ):
                sequence = ("execute", "yes", "rebirthing")

                if self.bot.context.path_changing:
                    self.bot.context.counts[next_path_index] = max(
                        1, self.bot.context.counts[next_path_index]
                    )
                else:
                    self.bot.context.counts[next_path_index] += 1

                projected_path_index = self.bot.next_path_index(self.bot.context.counts)

                if projected_path_index is not None:
                    self.bot.context.path_changing = (
                        next_path_index != projected_path_index
                    )

            if sequence is not None:
                result = StateResult(
                    status=StateStatus.SUCCESS,
                    next_state=SequenceState,
                    next_state_kwargs={
                        "sequence": sequence,
                        "next_state": ThreadToCathedralState,
                        "next_state_kwargs": {"next_state": ApproachVivianState},
                        "sequence_complete": lambda: self.bot.execute_commands(
                            LocateTemplateCommand("player")
                        ),
                    },
                )

                if projected_path_index is None or (
                    self.bot.context.path_changing
                    and self.bot.context.counts[projected_path_index] <= 1
                ):
                    result.next_state_kwargs |= {
                        "next_state": ApproachCathedralMasterState,
                        "next_state_kwargs": {"skipped_thread": True},
                    }

                return result

            self.bot.execute_commands(
                LocateTemplateCommand("rebirth_payment"),
                ClickCommand(-payment_item_x, 60),
                ClickCommand(),
            )
        return StateResult(
            StateStatus.FAILURE, message="Insufficient macca and/or rebirth items."
        )


class ApproachVivianState(State[RebirthBot]):
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
                message="Timed out approaching Vivian. Retrying...",
                next_state=ThreadToCathedralState,
                next_state_kwargs={"next_state": ApproachVivianState},
            )

        if not self.bot.execute_commands(LocateTemplateCommand("vivian_demon_level")):
            self.bot.execute_commands(ClickCommand())
            return StateResult(StateStatus.FAILURE, next_state=ApproachVivianState)

        return StateResult(StateStatus.SUCCESS, next_state=VivianState)


class VivianState(State[RebirthBot]):
    def run(self, elapsed: float) -> StateResult:
        next_count = self.bot.context.counts[
            self.bot.next_path_index(self.bot.context.counts)
        ]

        if self.bot.context.path_changing:
            next_count -= 1

        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ThreadToCathedralState,
                "sequence": (
                    "vivian_demon_level",
                    f"vivian_demon_level_{next_count - 1}",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("player")
                ),
            },
        )


class SequenceState(State[Bot]):
    def __init__(
        self,
        bot: Bot,
        sequence: tuple[str],
        sequence_complete: collections.abc.Callable[[], bool] = lambda: False,
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


def parse_dotenv_value(value: str) -> str | float | bool | list[int]:
    """Parse environment value to appropriate type."""
    try:
        return float(value)
    except ValueError:
        try:
            return [int(delimited) for delimited in value.split(",")]
        except ValueError:
            if value.casefold() in ("true", "false"):
                return value.casefold() == "true"

            return value if value != "" else None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = {}
    for key, value in dotenv.dotenv_values().items():
        match = re.match(r"(?i)IMAGINATION_(\w*BOT)(?:_(\w+))?$", key)

        if match is None:
            continue

        bot_key, bot_subkey = match.groups()
        parsed_value = parse_dotenv_value(value)

        if bot_subkey is None:
            selection = BotSelection(parsed_value)
        elif re.match(r"(?i)(%s_)?BOT" % selection, bot_key):
            config[bot_subkey.casefold()] = parsed_value

    match selection:
        case BotSelection.REBIRTH:
            bot = RebirthBot(RebirthBotConfig(**config))

    if bot.config.relog:
        bot.state = RelogState(bot)
    else:
        bot.state = ResetUIState(bot, next_state=ResetCameraState)

    bot.run()
