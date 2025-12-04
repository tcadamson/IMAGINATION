"""Automation framework for the SMT: IMAGINE client.

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
import pathlib
import re
import sys
import time

import cv2
import dotenv
import mss
import numpy
import pydirectinput
import pywinctl

logger = logging.getLogger(__name__)

# The program already uses client window focus state as its failsafe
pydirectinput.FAILSAFE = False

if getattr(sys, "frozen", False):
    root = pathlib.Path(sys.executable).parent
else:
    root = pathlib.Path(__file__).resolve().parent

# Load and cache templates in grayscale ahead of time to improve performance of LocateTemplateCommand
template_region_cache = {}
template_cache = {
    os.path.splitext(filename)[0]: cv2.cvtColor(
        cv2.imread(str(root / "templates" / filename)), cv2.COLOR_BGR2GRAY
    )
    for filename in os.listdir(root / "templates")
    if filename.endswith(".png")
}

demon_force_items = ("sands_0", "sands_1", "scabbard", "loop")


# Templates that may appear in different contexts/locations need to be parameterized
class CacheParameter(enum.StrEnum):
    PATH_G_TYPE = enum.auto()
    PATH_REBIRTH = enum.auto()
    YES_FUSION = enum.auto()
    YES_REBIRTH = enum.auto()
    YES_THREAD = enum.auto()
    CLOSE_DEMON_FORCE = enum.auto()
    CLOSE_REBIRTH = enum.auto()
    CLOSE_WINDOW_FUSION = enum.auto()
    CLOSE_WINDOW_REBIRTH = enum.auto()


class CommandStatus(enum.Enum):
    """Status enumeration for command execution results."""

    SUCCESS = enum.auto()
    FAILURE = enum.auto()


@dataclasses.dataclass
class CommandResult:
    """Result of command execution with status and optional message."""

    status: CommandStatus
    message: str | None = None


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


def move_to_wrapper(x, y, **kwargs) -> None:
    while pydirectinput.position() != (x, y):
        pydirectinput.moveTo(x, y, _pause=False, **kwargs)


class ClickCommand(Command):
    """Command to perform mouse click operations."""

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        button: str = pydirectinput.MOUSE_PRIMARY,
        pause: bool = True,
        reset_cursor: bool = False,
        click_count: int = 1,
    ):
        """Initialize click command.

        Args:
            x: X offset relative to template/center location
            y: Y offset relative to template/center location
            button: Mouse button to click
            pause: Apply pydirectinput.PAUSE
            reset_cursor: Move cursor to center of client after click to remove unwanted tooltips
            click_count: Number of clicks to perform
        """
        self.x = x
        self.y = y
        self.button = button
        self.pause = pause
        self.reset_cursor = reset_cursor
        self.click_count = click_count

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute click command.

        Args:
            context: Shared context containing last_template_location if coordinates not provided

        Returns:
            CommandResult with SUCCESS status
        """

        if context.last_template_location is not None:
            self.x, self.y = numpy.array(context.last_template_location) + numpy.array(
                (self.x, self.y)
            )

        # Absolutely do not get stuck clicking out of the client in a while loop, can you IMAGINE?
        right, bottom = numpy.array(context.origin) + numpy.array(
            context.capture.shape[:2][::-1]
        )
        if not (
            context.origin[0] <= self.x <= right
            and context.origin[1] <= self.y <= bottom
        ):
            return CommandResult(
                CommandStatus.FAILURE,
                message=f"Attempted to click out of client bounds: ({self.x}, {self.y})",
            )

        move_to_wrapper(self.x, self.y, attempt_pixel_perfect=True)
        for _ in range(self.click_count):
            pydirectinput.mouseDown(
                self.x, self.y, button=self.button, _pause=self.pause
            )
            pydirectinput.mouseUp(button=self.button, _pause=self.pause)

        if self.reset_cursor:
            move_to_wrapper(
                context.center[0],
                context.center[1],
                attempt_pixel_perfect=True,
            )

        return CommandResult(CommandStatus.SUCCESS)


class DragCommand(Command):
    """Command to perform mouse drag operations."""

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        button: str = pydirectinput.MOUSE_SECONDARY,
        drag_count: int = 1,
        drag_sleep: float = 0.0,
    ):
        """Initialize drag command.

        Args:
            x: X offset to drag relative to current position
            y: Y offset to drag relative to current position
            button: Mouse button to use for dragging
            drag_count: Number of drag operations to perform
            drag_sleep: Amount of time to sleep before performing a drag
        """
        self.x = x
        self.y = y
        self.button = button
        self.drag_count = drag_count
        self.drag_sleep = drag_sleep

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute drag command.

        Returns:
            CommandResult with SUCCESS status
        """
        center_x, center_y = context.center
        for _ in range(self.drag_count):
            time.sleep(self.drag_sleep)
            move_to_wrapper(center_x, center_y, attempt_pixel_perfect=True)
            pydirectinput.mouseDown(button=pydirectinput.MOUSE_SECONDARY)
            move_to_wrapper(
                center_x + self.x,
                center_y + self.y,
                attempt_pixel_perfect=True,
            )
            pydirectinput.mouseUp(button=pydirectinput.MOUSE_SECONDARY)
        return CommandResult(CommandStatus.SUCCESS)


def capture_subset(
    context: CommandContext,
    region: tuple[int, int, int, int],
    offset: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y = context.origin
    offset_x, offset_y = numpy.array(offset) + numpy.array(region[:2])

    x1 = max(0, offset_x - x)
    y1 = max(0, offset_y - y)
    x2 = min(context.capture.shape[1], x1 + region[2])
    y2 = min(context.capture.shape[0], y1 + region[3])

    return y1, y2, x1, x2


class LocateTemplateCommand(Command):
    """Command to locate template images within the client capture."""

    def __init__(
        self,
        templates: tuple[str] | str,
        confidence: float = 0.95,
        region: tuple[int, int, int, int] | None = None,
        mask: numpy.ndarray | None = None,
        debug: bool = False,
        permutate: bool = False,
    ):
        """Initialize template matching command.

        Args:
            templates: Collection of templates to attempt to match (or a single template string)
            confidence: Minimum confidence threshold for template matching
            region: Optional region to search (x, y, width, height)
            mask: Optional mask applied to the client capture
            debug: Log location and confidence data for all possible matches of the template
            permutate: Randomize the order of the input templates
        """
        self.templates = (templates,) if isinstance(templates, str) else templates
        self.confidence = confidence
        self.region = region
        self.mask = mask
        self.debug = debug
        self.permutate = permutate

    def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute template matching command.

        Args:
            context: Shared context containing client capture and coordinates

        Returns:
            CommandResult with SUCCESS if template found above confidence threshold, FAILURE
            otherwise. On success, sets context.last_template_location to the absolute screen
            coordinates of the center of the template.
        """
        x, y = context.origin

        template_index = None
        template_cache_key = None
        for template in (
            numpy.random.permutation(self.templates)
            if self.permutate
            else self.templates
        ):
            if self.region is not None:
                y1, y2, x1, x2 = capture_subset(
                    context, self.region, context.last_template_location
                )
            elif template in template_region_cache:
                y1, y2, x1, x2 = capture_subset(
                    context, template_region_cache[template], (0, 0)
                )
            else:
                y1, y2, x1, x2 = (
                    0,
                    context.capture.shape[0],
                    0,
                    context.capture.shape[1],
                )

            template_cache_key = template.split("?")[0]
            capture = cv2.bitwise_and(context.capture, context.capture, mask=self.mask)
            result = cv2.matchTemplate(
                cv2.cvtColor(capture[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY),
                template_cache[template_cache_key],
                cv2.TM_CCOEFF_NORMED,
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if self.debug:
                matches = []
                for location in zip(*numpy.where(result >= self.confidence)[::-1]):
                    confidence = result[location[1], location[0]]
                    matches.append((location, confidence))

                matches.sort(key=lambda location: location[1], reverse=True)

                for i, (location, confidence) in enumerate(matches):
                    logger.debug(
                        f"{self.__class__.__name__} - '{template}' location {i + 1}: {location} confidence: {confidence:.6f}"
                    )

            if max_val >= self.confidence:
                template_index = self.templates.index(template)
                break

        if template_index is None:
            return CommandResult(CommandStatus.FAILURE)

        template = self.templates[template_index]
        template_height, template_width = template_cache[template_cache_key].shape[:2]
        match_x, match_y = max_loc
        x += x1 + match_x + template_width // 2
        y += y1 + match_y + template_height // 2

        context.last_template_location = (x, y)
        context.last_template_index = template_index

        if template not in (
            template_region_cache.keys() | (("incense",) + demon_force_items)
        ):
            region = (
                x - template_width // 2,
                y - template_height // 2,
                template_width,
                template_height,
            )

            # These dialogue items are networked and may shift position for a split second on load (exactly 12 pixels up)
            if template.startswith("dialogue_"):
                region = numpy.array(region) + numpy.array((0, -12, 0, 24))

            template_region_cache[template] = region
            logger.info(f"Cached '{template}' with region: {region}")

        return CommandResult(
            CommandStatus.SUCCESS,
            message=None
            if self.debug
            else f"Matched '{template}' with confidence: {max_val:.6f}",
        )


class ImagineClient:
    """Wrapper class for IMAGINE client data and interaction."""

    IMAGINE_CLIENT_IDENTIFIER: str = "IMAGINE Version 1."

    def __init__(self):
        """Initialize the IMAGINE client window reference."""
        windows = pywinctl.getWindowsWithTitle(
            self.IMAGINE_CLIENT_IDENTIFIER, condition=pywinctl.Re.STARTSWITH
        )
        try:
            self._window = windows[0]
        except IndexError:
            logger.error("No IMAGINE client windows available.")
            sys.exit()

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
        self._window.activate(wait=True)


class BotSelection(enum.StrEnum):
    """Bot type selection enumeration."""

    REBIRTH = enum.auto()
    DEMON_FORCE = enum.auto()


@dataclasses.dataclass
class BotConfig:
    """Base configuration for bot instances."""

    relog: bool
    sleep_amount: float
    drag_sleep_amount: float


@dataclasses.dataclass
class BotContext:
    """Base context for bot instances."""

    pass


class Bot[BotConfigType: BotConfig, BotContextType: BotContext](abc.ABC):
    """Base bot class that manages state and executes automation commands."""

    DEMON_LIST_REGION: tuple[int] = (-60, 0, 50, 450)
    BAR_REGION: tuple[int] = (-150, 0, 300, 50)

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
        self.start_time = time.time()

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
            if not self.client.focused:
                logger.error("IMAGINE client window lost focus.")
                sys.exit()

            if self.config.drag_sleep_amount is not None and isinstance(
                command, DragCommand
            ):
                command.drag_sleep = self.config.drag_sleep_amount

            context.capture = self.client.capture
            context.origin = (self.client.x, self.client.y)
            context.center = (self.client.center_x, self.client.center_y)
            result = command.execute(context)

            if result.message is not None:
                logger.debug(
                    f"{result.status.name}: {command.__class__.__name__} - {result.message}"
                )

            if result.status == CommandStatus.FAILURE:
                return (False, context)

        return (True, context)

    def run(self) -> None:
        """Run states until None is encountered."""
        while not self.client.focused:
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

            if result.message is None:
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


class Mitama(enum.StrEnum):
    """Enumeration of available mitamas in IMAGINE."""

    ARA = enum.auto()
    NIGI = enum.auto()
    KUSI = enum.auto()
    SAKI = enum.auto()


class CathedralLocation(enum.StrEnum):
    """Enumeration of available cathedral locations to thread to."""

    HOME_III = enum.auto()
    BABEL = enum.auto()
    ARCADIA = enum.auto()
    SOUHONZAN = enum.auto()


@dataclasses.dataclass
class RebirthBotConfig(BotConfig):
    """Configuration for rebirth automation bot."""

    end_counts: list[int]
    mitama: Mitama
    mitama_end_counts: list[int]
    end_path: RebirthPath
    cathedral_location: CathedralLocation


@dataclasses.dataclass
class RebirthBotContext(BotContext):
    """Context for rebirth automation bot."""

    counts: list[int] | None = None
    path_changing: bool = False
    has_mitama: bool = False


class RebirthBot(Bot[RebirthBotConfig, RebirthBotContext]):
    """Bot for automating rebirths."""

    TYPE_REGION: tuple[int] = (32, -10, 120, 20)

    # Refresh every 27:30, i.e. the user has 2:30 to start the bot after using incense
    REFRESH_INCENSE_INTERVAL: int = 1650

    def next_path_index(self, counts: list[int]) -> int | None:
        """Determine the next rebirth path index to work on."""
        try:
            end_path_index = list(RebirthPath).index(self.config.end_path)
        except ValueError:
            end_path_index = None

        end_counts = (
            self.config.mitama_end_counts
            if self.context.has_mitama
            else self.config.end_counts
        )
        indices = [
            i
            for i, (current, target) in enumerate(zip(counts, end_counts))
            if current < target and i != end_path_index
        ]

        if (
            self.config.end_path is not None
            and counts[end_path_index] < end_counts[end_path_index]
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
            y, x = numpy.argwhere(border_mask > 0)[0]
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


@dataclasses.dataclass
class DemonForceBotContext(BotContext):
    """Context for demon force automation bot."""

    mask: numpy.ndarray | None = None


class DemonForceBot(Bot[BotConfig, DemonForceBotContext]):
    """Bot for automating demon force."""

    pass


class StateStatus(enum.Enum):
    """Status enumeration for state execution results."""

    SUCCESS = enum.auto()
    FAILURE = enum.auto()


@dataclasses.dataclass
class StateResult:
    """Result of state execution with status and next state transition information."""

    status: StateStatus
    message: str | None = None
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


class RelogState(State[Bot]):
    def run(self, elapsed: float) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ReloggedState,
                "sequence": (
                    "system",
                    "select_character",
                    "start_game",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand(("info", "hotbar"))
                ),
            },
        )


class ReloggedState(State[Bot]):
    def run(self, elapsed: float) -> StateResult:
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=PrepareUIState,
            next_state_kwargs={"next_state": ViewDemonListState},
        )


class PrepareUIState(State[Bot]):
    def run(self, elapsed: float) -> StateResult:
        for template in ("devil", "inventory", "system"):
            if not self.bot.execute_commands(
                LocateTemplateCommand("bar"),
                LocateTemplateCommand(template, region=self.bot.BAR_REGION),
            ):
                return StateResult(
                    StateStatus.FAILURE,
                    message="UI bar (PC, Devil, Item, etc.) obstructed and/or outside the screen.",
                )

        if self.bot.execute_commands(LocateTemplateCommand("inventory_window")):
            return StateResult(StateStatus.SUCCESS, next_state=self.next_state)

        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": ViewDemonListState,
                "sequence": ("inventory",),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("inventory_window")
                ),
            },
        )


class PrepareCameraState(State[Bot]):
    def run(self, elapsed: float) -> StateResult:
        self.bot.execute_commands(DragCommand(400, 0, drag_count=4))
        return StateResult(status=StateStatus.SUCCESS, next_state=self.next_state)


class ViewDemonListState(State[Bot]):
    def run(self, elapsed):
        # If the window is already open, reopen it to reset the current demon selection
        return StateResult(
            StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": CheckSummonedDemonState,
                "sequence": (
                    "devil",
                    "demon_list",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("demon_list_window")
                ),
                "loop": True,
                "reset_cursor": True,
            },
        )


class InitiateDemonForceState(State[DemonForceBot]):
    def run(self, elapsed):
        self.bot.execute_commands(
            LocateTemplateCommand("demon_list_window"),
            LocateTemplateCommand("summoned_demon", region=self.bot.DEMON_LIST_REGION),
            ClickCommand(),
            ClickCommand(button=pydirectinput.MOUSE_SECONDARY),
            LocateTemplateCommand("demon_force_tab"),
            ClickCommand(),
        )

        if not self.bot.execute_commands(LocateTemplateCommand("demon_force")):
            return StateResult(
                StateStatus.FAILURE, message="Please unlock demon force."
            )
        else:
            self.bot.context.mask = (
                numpy.ones(self.bot.client.capture.shape[:2], dtype=numpy.uint8) * 255
            )
            return StateResult(StateStatus.SUCCESS, next_state=DemonForceState)


class DemonForceState(State[DemonForceBot]):
    def run(self, elapsed):
        self.bot.execute_commands(
            LocateTemplateCommand("demon_force"),
            ClickCommand(),
        )
        while not self.bot.execute_commands(
            LocateTemplateCommand("demon_force_window"),
        ):
            continue
        command = LocateTemplateCommand(tuple(f"empty_slot_{i}" for i in range(2)))
        success, context = self.bot.execute_commands_with_context(command)

        if not success:
            self.bot.execute_commands(
                LocateTemplateCommand("demon_force_window"),
                ClickCommand(-160, 277, button=pydirectinput.MOUSE_SECONDARY),
            )
            while not success:
                success, context = self.bot.execute_commands_with_context(command)

        slot_location = context.last_template_location
        success, context = self.bot.execute_commands_with_context(
            LocateTemplateCommand("demon_force_window"),
            LocateTemplateCommand(
                demon_force_items,
                confidence=0.9,
                region=(-92, 75, 370, 200),
                mask=self.bot.context.mask,
            ),
        )
        x, y = numpy.array(context.last_template_location) - numpy.array(context.origin)

        if not success:
            return StateResult(
                StateStatus.FAILURE,
                message="Demon force items exhausted or not present.",
            )

        command = LocateTemplateCommand(
            tuple(f"effect_{i}" for i in range(3)), confidence=0.99
        )
        self.bot.execute_commands(
            ClickCommand(*context.last_template_location),
            ClickCommand(*slot_location, click_count=0),
            command,
            ClickCommand(),
            LocateTemplateCommand("disable_confirmation"),
            ClickCommand(),
            LocateTemplateCommand("use"),
            ClickCommand(click_count=100, pause=False),
        )
        while not self.bot.execute_commands(LocateTemplateCommand("pending_slot")):
            # The item hasn't been exhausted, but its uses have been exhausted and it should not be matched again
            # Re-evaluate loop condition in case of TOCTOU mismatch
            if not self.bot.execute_commands(command) and not self.bot.execute_commands(
                LocateTemplateCommand("pending_slot")
            ):
                cv2.circle(self.bot.context.mask, (x, y), 10, 0, -1)
                break
        while not self.bot.execute_commands(LocateTemplateCommand(("info", "hotbar"))):
            self.bot.execute_commands(
                LocateTemplateCommand("demon_force_window"),
                LocateTemplateCommand(
                    f"close?{CacheParameter.CLOSE_DEMON_FORCE}",
                    region=(220, 335, 60, 30),
                ),
                ClickCommand(),
            )
        return StateResult(StateStatus.SUCCESS, next_state=DemonForceState)


class CheckSummonedDemonState(State[RebirthBot | DemonForceBot]):
    def run(self, elapsed):
        success, context = self.bot.execute_commands_with_context(
            LocateTemplateCommand("demon_list_window"),
            LocateTemplateCommand(
                "summoned", confidence=0.9, region=self.bot.DEMON_LIST_REGION
            ),
        )

        if not success:
            return StateResult(
                StateStatus.FAILURE,
                message="Failed to determine the summoned demon. (Is it summoned?)",
            )

        x, y = (
            numpy.array(context.last_template_location)
            + numpy.array((-4, 5))
            - numpy.array(context.origin)
        )
        template_cache["summoned_demon"] = cv2.cvtColor(
            context.capture[y : y + 27, x : x + 28], cv2.COLOR_BGR2GRAY
        )

        if isinstance(bot, DemonForceBot):
            return StateResult(StateStatus.SUCCESS, next_state=InitiateDemonForceState)

        return StateResult(
            StateStatus.SUCCESS,
            next_state=PrepareCameraState,
            next_state_kwargs={"next_state": ThreadToCathedralState},
        )


class RebirthDialogueState(State[RebirthBot]):
    def run(self, elapsed):
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": RebirthCountState,
                "sequence": (
                    "dialogue_perform_rebirth_1",
                    "dialogue_perform_rebirth_2",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("view_demon_information")
                ),
            },
        )


class RebirthCountState(State[RebirthBot]):
    def run(self, elapsed: float):
        self.bot.execute_commands(
            LocateTemplateCommand("view_demon_information"),
            ClickCommand(),
            LocateTemplateCommand("rebirth_tab"),
            ClickCommand(),
        )
        self.bot.context.counts = self.bot.count_paths()

        if self.bot.context.counts is None:
            return StateResult(
                StateStatus.FAILURE,
                message="Failed to count rebirth paths. Verify the window is in view.",
            )

        success = self.bot.execute_commands(
            LocateTemplateCommand("m_type"),
            LocateTemplateCommand(
                tuple(f"mitama_{i}" for i, _ in enumerate(Mitama)),
                region=self.bot.TYPE_REGION,
                permutate=True,
            ),
        )
        self.bot.context.has_mitama = success
        next_path_index = self.bot.next_path_index(self.bot.context.counts)

        if next_path_index is None:
            if self.bot.config.mitama is None or self.bot.context.has_mitama:
                return StateResult(StateStatus.SUCCESS, message="Rebirths complete.")

            self.bot.execute_commands(
                LocateTemplateCommand(
                    f"close_window?{CacheParameter.CLOSE_WINDOW_REBIRTH}"
                ),
                ClickCommand(),
                LocateTemplateCommand(f"close?{CacheParameter.CLOSE_REBIRTH}"),
                ClickCommand(),
            )
            return StateResult(
                StateStatus.SUCCESS,
                next_state=SequenceState,
                next_state_kwargs={
                    "next_state": ApproachCathedralMasterState,
                    "next_state_kwargs": {
                        "next_state": FusionDialogueState,
                        "skipped_thread": True,
                    },
                    "sequence": ("dialogue_stop",),
                    "sequence_complete": lambda: self.bot.execute_commands(
                        LocateTemplateCommand(("info", "hotbar"))
                    ),
                },
            )

        return StateResult(StateStatus.SUCCESS, next_state=RebirthPathState)


class RebirthPathState(State[RebirthBot]):
    def run(self, elapsed: float):
        next_path_index = self.bot.next_path_index(self.bot.context.counts)
        path_index = None
        success, context = self.bot.execute_commands_with_context(
            LocateTemplateCommand("g_type"),
            LocateTemplateCommand(
                tuple(
                    f"path_{i}?{CacheParameter.PATH_G_TYPE}"
                    for i, _ in enumerate(RebirthPath)
                ),
                region=self.bot.TYPE_REGION,
                permutate=True,
            ),
        )

        if success:
            path_index = context.last_template_index
        else:
            # Demon with a weird custom growth type, get the nonzero path
            try:
                path_index = self.bot.context.counts.index(1)
            except ValueError:
                return StateResult(StateStatus.FAILURE, next_state=RebirthPathState)

        self.bot.context.path_changing = next_path_index != path_index
        self.bot.execute_commands(
            LocateTemplateCommand(
                f"close_window?{CacheParameter.CLOSE_WINDOW_REBIRTH}"
            ),
            ClickCommand(),
        )
        while not self.bot.execute_commands(
            LocateTemplateCommand("rebirth_window"),
            LocateTemplateCommand(
                f"path_{next_path_index}?{CacheParameter.PATH_REBIRTH}",
                region=(-60, 0, 60, 265),
            ),
            ClickCommand(),
            LocateTemplateCommand(f"path_icon_{next_path_index}"),
            LocateTemplateCommand("rebirth_payment"),
            ClickCommand(click_count=0),
        ):
            continue
        return StateResult(StateStatus.SUCCESS, next_state=RebirthState)


class RebirthState(State[RebirthBot]):
    def run(self, elapsed: float):
        next_path_index = self.bot.next_path_index(self.bot.context.counts)

        # Matching templates for all possible items is unrealistic; attempt from left to right
        for payment_item_x in (190, 150, 110, 70):
            sequence = None
            projected_path_index = next_path_index

            if self.bot.execute_commands(
                LocateTemplateCommand("rebirth_level_warning")
            ):
                self.bot.execute_commands(
                    LocateTemplateCommand(f"close?{CacheParameter.CLOSE_REBIRTH}"),
                    ClickCommand(),
                )
                sequence = ("dialogue_stop",)
            elif self.bot.execute_commands(LocateTemplateCommand("execute")):
                self.bot.execute_commands(
                    LocateTemplateCommand("execute"),
                    ClickCommand(),
                )
                sequence = (f"yes?{CacheParameter.YES_REBIRTH}", "rebirthing")

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
                            LocateTemplateCommand(("info", "hotbar"))
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
                ClickCommand(click_count=0),
            )
        return StateResult(
            StateStatus.FAILURE, message="Insufficient macca and/or rebirth items."
        )


class FusionDialogueState(State[RebirthBot]):
    def run(self, elapsed):
        return StateResult(
            status=StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": FusionState,
                "sequence": ("dialogue_double_fusion",),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand(
                        f"close_window?{CacheParameter.CLOSE_WINDOW_FUSION}"
                    )
                ),
            },
        )


class FusionState(State[RebirthBot]):
    def run(self, elapsed):
        if not self.bot.execute_commands(LocateTemplateCommand("summoned_demon")):
            self.bot.execute_commands(
                LocateTemplateCommand(
                    f"close_window?{CacheParameter.CLOSE_WINDOW_FUSION}"
                ),
                ClickCommand(),
            )
            while not self.bot.execute_commands(
                LocateTemplateCommand("demon_list_window"),
                ClickCommand(),
            ):
                continue
            return StateResult(StateStatus.SUCCESS, next_state=PostFusionState)

        material_locations = []
        for region, template in zip(
            ((0, 0, 195, 505), (0, 0, 565, 130)),
            (
                "summoned_demon",
                f"mitama_icon_{list(Mitama).index(self.bot.config.mitama)}",
            ),
        ):
            success, context = self.bot.execute_commands_with_context(
                LocateTemplateCommand("double_fusion_window"),
                LocateTemplateCommand(template, region=region),
            )

            if not success:
                return StateResult(
                    StateStatus.FAILURE,
                    message=f"Mitama fusion failed. Material '{template}' not found.",
                )

            material_locations.append(context.last_template_location)
        self.bot.execute_commands(
            ClickCommand(material_locations[1][0], material_locations[0][1])
        )
        return StateResult(
            StateStatus.SUCCESS,
            next_state=SequenceState,
            next_state_kwargs={
                "next_state": FusionState,
                "sequence": (
                    "rebirthing",
                    f"yes?{CacheParameter.YES_FUSION}",
                    "rebirthing",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand("double_fusion_window")
                ),
            },
        )


class PostFusionState(State[RebirthBot]):
    def run(self, elapsed):
        self.bot.execute_commands(
            LocateTemplateCommand("demon_list_window"),
            LocateTemplateCommand("summoned_demon", region=self.bot.DEMON_LIST_REGION),
            ClickCommand(click_count=2),
        )
        while not self.bot.execute_commands(
            LocateTemplateCommand("demon_list_window"),
            ClickCommand(),
            LocateTemplateCommand(
                "summoned", confidence=0.9, region=self.bot.DEMON_LIST_REGION
            ),
        ):
            logger.info("Waiting for demon to be summoned...")
        return StateResult(
            StateStatus.SUCCESS,
            next_state=ApproachCathedralMasterState,
            next_state_kwargs={"skipped_thread": True},
        )


class ApproachCathedralMasterState(State[RebirthBot]):
    def __init__(
        self,
        bot: Bot,
        next_state: State | None = RebirthDialogueState,
        skipped_thread: bool = False,
    ):
        super().__init__(bot, next_state, max_elapsed=5.5)
        self.skipped_thread = skipped_thread

    def run(self, elapsed: float) -> StateResult:
        if elapsed == 0 and not self.skipped_thread:
            self.bot.execute_commands(DragCommand(90, 0))
        elif elapsed >= self.max_elapsed:
            return StateResult(
                StateStatus.FAILURE,
                message="Timed out approaching cathedral master. Retrying...",
                next_state=ThreadToCathedralState,
                next_state_kwargs={
                    "next_state_kwargs": {"next_state": self.next_state},
                },
            )

        if not self.bot.execute_commands(
            LocateTemplateCommand("dialogue_cathedral_master")
        ):
            self.bot.execute_commands(
                ClickCommand(self.bot.client.center_x, self.bot.client.center_y)
            )
            return StateResult(
                StateStatus.FAILURE,
                next_state=ApproachCathedralMasterState,
                next_state_kwargs={"next_state": self.next_state},
            )

        return StateResult(StateStatus.SUCCESS, next_state=self.next_state)


class ApproachVivianState(State[RebirthBot]):
    def __init__(self, bot: Bot):
        super().__init__(bot, max_elapsed=7.5)

    def run(self, elapsed: float) -> StateResult:
        if elapsed == 0:
            self.bot.execute_commands(DragCommand(110, 0, drag_count=2))

            if (time.time() - self.bot.start_time) >= self.bot.REFRESH_INCENSE_INTERVAL:
                self.bot.start_time = time.time()

                if self.bot.execute_commands(
                    LocateTemplateCommand("incense"),
                    ClickCommand(click_count=0),
                    WaitCommand(0.5),
                    ClickCommand(click_count=2),
                ):
                    logger.info("Refreshed x10 demon incense.")
                    # Invalidate the cached thread region to ensure it's clicked again if it's the item version
                    del template_region_cache["thread"]
                else:
                    logger.info(f"Couldn't locate x10 demon incense to refresh.")
        elif elapsed >= self.max_elapsed:
            return StateResult(
                StateStatus.FAILURE,
                message="Timed out approaching Vivian. Retrying...",
                next_state=ThreadToCathedralState,
                next_state_kwargs={"next_state": ApproachVivianState},
            )

        if not self.bot.execute_commands(LocateTemplateCommand("dialogue_demon_level")):
            self.bot.execute_commands(
                ClickCommand(self.bot.client.center_x, self.bot.client.center_y)
            )
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
                    "dialogue_demon_level",
                    f"dialogue_demon_level_{next_count - 1}",
                ),
                "sequence_complete": lambda: self.bot.execute_commands(
                    LocateTemplateCommand(("info", "hotbar"))
                ),
            },
        )


class ThreadToCathedralState(State[RebirthBot]):
    def __init__(
        self,
        bot: Bot,
        next_state: State | None = ApproachCathedralMasterState,
        next_state_kwargs: dict | None = None,
    ):
        super().__init__(bot, next_state, next_state_kwargs)

    def run(self, elapsed: float) -> StateResult:
        cached = "thread" in template_region_cache
        success, context = self.bot.execute_commands_with_context(
            LocateTemplateCommand("thread", confidence=0.999)
        )

        if not success:
            return StateResult(
                StateStatus.FAILURE, message="Thread obstructed or not present."
            )

        x, y = context.last_template_location

        if not cached:
            last_template_location = tuple(context.last_template_location)
            success, context = self.bot.execute_commands_with_context(
                # TODO: Replace hotbar template (as seen in babel cathedral, it can potentially be affected by shaders)
                LocateTemplateCommand("hotbar", confidence=0.8),
                LocateTemplateCommand("thread", region=(-460, -30, 445, 75)),
            )

            if not success or (
                success and context.last_template_location != last_template_location
            ):
                self.bot.execute_commands(ClickCommand(x, y))

        self.bot.execute_commands(ClickCommand(x, y))
        while not self.bot.execute_commands(LocateTemplateCommand("warp")):
            continue
        commands = (
            LocateTemplateCommand("thread_cathedral"),
            ClickCommand(click_count=2),
        )

        if self.bot.config.cathedral_location is not None:
            commands[0].region = (-177, 0, 354, 100)
            self.bot.execute_commands(
                LocateTemplateCommand(
                    f"thread_cathedral_{list(CathedralLocation).index(self.bot.config.cathedral_location)}"
                ),
                *commands,
            )
        else:
            self.bot.execute_commands(*commands)

        while not self.bot.execute_commands(
            LocateTemplateCommand(f"yes?{CacheParameter.YES_THREAD}"),
            ClickCommand(reset_cursor=True),
        ):
            continue
        command = LocateTemplateCommand(("info", "hotbar"), confidence=0.9999)
        while self.bot.execute_commands(command):
            continue
        while not self.bot.execute_commands(command):
            continue
        return StateResult(
            StateStatus.SUCCESS,
            next_state=self.next_state,
            next_state_kwargs=self.next_state_kwargs,
        )


class SequenceState(State[Bot]):
    def __init__(
        self,
        bot: Bot,
        sequence: tuple[str],
        sequence_complete: collections.abc.Callable[[], bool] = lambda: False,
        loop: bool = False,
        reset_cursor: bool = False,
        index: int = 0,
        next_state: State | None = None,
        next_state_kwargs: dict | None = None,
    ):
        super().__init__(bot, next_state, next_state_kwargs)
        self.sequence = sequence
        self.sequence_complete = sequence_complete
        self.loop = loop
        self.reset_cursor = reset_cursor
        self.index = index

    def run(self, elapsed: float) -> StateResult:
        try:
            next_template = self.sequence[self.index + 1]
        except IndexError:
            next_template = None

        commands = (
            LocateTemplateCommand(self.sequence[self.index]),
            ClickCommand(reset_cursor=self.reset_cursor),
        )

        if self.reset_cursor:
            self.bot.execute_commands(
                *commands,
                WaitCommand(0.05),
            )
        else:
            self.bot.execute_commands(*commands)

        if next_template is not None and self.bot.execute_commands(
            LocateTemplateCommand(next_template)
        ):
            self.index += 1
        elif next_template is None:
            if self.sequence_complete():
                return StateResult(
                    status=StateStatus.SUCCESS,
                    next_state=self.next_state,
                    next_state_kwargs=self.next_state_kwargs,
                )
            elif self.loop:
                self.index = 0

        return StateResult(
            status=StateStatus.FAILURE,
            next_state=SequenceState,
            next_state_kwargs={
                "sequence": self.sequence,
                "sequence_complete": self.sequence_complete,
                "loop": self.loop,
                "reset_cursor": self.reset_cursor,
                "index": self.index,
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
            value = value.casefold()

            if value in ("true", "false"):
                return value == "true"

            return value if value != "" else None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(root / "debug.log"), logging.StreamHandler()],
    )

    try:
        config = {}
        for key, value in dotenv.dotenv_values(root / ".env").items():
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
                bot = RebirthBot(RebirthBotConfig(**config), RebirthBotContext())
            case BotSelection.DEMON_FORCE:
                bot = DemonForceBot(BotConfig(**config), DemonForceBotContext())

        pydirectinput.PAUSE = bot.config.sleep_amount

        if bot.config.relog:
            bot.state = RelogState(bot)
        else:
            bot.state = ReloggedState(bot)

        bot.run()
    except Exception:
        logger.exception(f"Fatal exception:", exc_info=True)
