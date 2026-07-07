"""Automation framework for the SMT: IMAGINE client."""

import abc
import collections.abc
import ctypes
import ctypes.wintypes
import dataclasses
import json
import logging
import pathlib
import sys
import time
import typing

import cv2
import mss
import numpy
import pydirectinput
import pywinctl

type _Padding = int | tuple[int, int]

type Workflow = collections.abc.Iterator[Handoff]

DEFAULT_CONFIDENCE: typing.Final = 0.85
DEFAULT_SLEEP: typing.Final = 0.06

ROOT_DIRECTORY: typing.Final = (
    pathlib.Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else pathlib.Path(__file__).resolve().parents[1]
)

pydirectinput.PAUSE = DEFAULT_SLEEP
pydirectinput.FAILSAFE = (
    False  # Use client window focus as the failsafe (see Session._guard)
)

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except AttributeError, OSError:
    ctypes.windll.user32.SetProcessDPIAware()

_logger: logging.Logger = logging.getLogger(__name__)


def _is_unit_scale(scale: float) -> bool:
    """Whether `scale` is close enough to 1.0 to skip resampling."""
    return abs(scale - 1.0) <= 1e-3


@dataclasses.dataclass(frozen=True)
class Point:
    """Immutable, two-dimensional integer coordinate."""

    x: int
    y: int

    def offset(self, dx: int = 0, dy: int = 0) -> Point:
        """Return a new point offset from this one by `dx` and `dy`."""
        return Point(self.x + dx, self.y + dy)


@dataclasses.dataclass(frozen=True)
class Rect:
    """Immutable rectangle containing origin, width, and height."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_bounds(cls, x1: int, y1: int, x2: int, y2: int) -> Rect:
        """Create rectangle from `x1`, `y1`, `x2`, and `y2` bounds."""
        return cls(x1, y1, x2 - x1, y2 - y1)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Rectangle as x1, y1, x2, and y2 bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def origin(self) -> Point:
        """Origin point of the rectangle."""
        return Point(self.x, self.y)

    @property
    def center(self) -> Point:
        """Center point of the rectangle."""
        return Point(self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, point: Point) -> bool:
        """Return whether `point` falls within this rectangle."""
        return (
            self.x <= point.x < self.x + self.width
            and self.y <= point.y < self.y + self.height
        )

    def inflate(self, padding: _Padding) -> Rect:
        """Return a rectangle grown outward by the `padding` amount on each side.

        A single int expands both axes equally; a (dx, dy) pair expands them
        independently. Negative amounts are clamped to zero, so the result is
        never smaller than the original.
        """
        dx, dy = (padding, padding) if isinstance(padding, int) else padding
        dx, dy = max(0, dx), max(0, dy)

        if dx == 0 and dy == 0:
            return self

        return Rect(self.x - dx, self.y - dy, self.width + 2 * dx, self.height + 2 * dy)

    def relative(self, dx: int, dy: int, width: int, height: int) -> Rect:
        """Return a rectangle of the given size, positioned relative to this one.

        Calculated relative to origin, not center.
        """
        return Rect(self.origin.x + dx, self.origin.y + dy, width, height)

    def clamp(self, width: int, height: int) -> Rect:
        """Return this rectangle clipped to (`width`, `height`) bounds at the origin.

        A rectangle lying fully outside the bounds clamps to zero extent rather than a
        negative one.
        """
        x1, y1, x2, y2 = self.bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        return Rect.from_bounds(x1, y1, max(x1, x2), max(y1, y2))


@dataclasses.dataclass(frozen=True)
class Handoff:
    """Cooperative yield point a workflow returns to release the scheduler."""

    reason: str = ""


@dataclasses.dataclass(frozen=True)
class BotSpec:
    """Bot definition; the config type and workflow are derived from `bot_type`."""

    bot_type: type[Bot]
    bot_id: str = dataclasses.field(default="", kw_only=True)
    help: str = dataclasses.field(default="", kw_only=True)

    @property
    def workflow(self) -> collections.abc.Callable[[Session, BotConfig], Workflow]:
        """Workflow factory, inherited from `Bot` unless overridden."""
        return self.bot_type.workflow

    @property
    def bot_config_type(self) -> type[BotConfig]:
        """Config type from the `bot_config` annotation, or base `BotConfig`."""
        return typing.get_type_hints(self.bot_type).get("bot_config", BotConfig)


@dataclasses.dataclass(frozen=True)
class BotConfig:
    """Base configuration for a bot workflow."""

    cycles_limit: int = dataclasses.field(
        default=0, kw_only=True
    )  # 0 runs indefinitely


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Run configuration for a bot workflow."""

    confidence: float = DEFAULT_CONFIDENCE
    sleep: float = DEFAULT_SLEEP
    scale: float | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class Template:
    """Immutable template data with frame and associated spec."""

    frame: numpy.ndarray
    spec: TemplateSpec


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    """Immutable template spec."""

    confidence: float | None = None
    grayscale: bool = True


def _load_template_specs(template_directory: pathlib.Path) -> dict[str, TemplateSpec]:
    """Parse specs colocated in `template_directory` into specs keyed by template.

    Absent file means no overrides. Tracked from the repo, so a template's spec can be
    updated without a client release.
    """
    path = template_directory / "specs.json"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exception:
        raise RuntimeError(f"Malformed JSON at {path}") from exception

    try:
        return {template_id: TemplateSpec(**spec) for template_id, spec in data.items()}
    except TypeError as exception:
        raise RuntimeError(f"Invalid spec at {path}") from exception


@dataclasses.dataclass(frozen=True)
class TemplateMatch:
    """Immutable template match data in logical client-space coordinates."""

    template_id: str
    rect: Rect
    confidence: float


@dataclasses.dataclass(frozen=True, eq=False)
class LocateParams:
    """Immutable locate parameters, in logical client-space pixels."""

    region: Rect | None = None
    region_padding: _Padding = 0
    region_cache_id: str | None = None
    masks: tuple[Rect, ...] = ()
    confidence: float | None = None


_DIALOGUE_ARROW_PARAMS: typing.Final[LocateParams] = LocateParams(
    region_padding=(400, 50)
)


@dataclasses.dataclass(frozen=True)
class ClickParams:
    """Immutable click parameters."""

    button: str = pydirectinput.MOUSE_PRIMARY
    count: int = 1
    pause: bool = True


class Aborted(Exception):
    """The user intentionally halted a workflow while paused."""


ABORT_MESSAGE: typing.Final = "Aborted by user."


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


class Bot(abc.ABC):
    """Abstract base for bots; subclasses implement `cycle` and override `setup`."""

    def __init__(self, session: Session, bot_config: BotConfig):
        self.session = session
        self.bot_config = bot_config

    def setup(self) -> None:
        """One-time blocking setup before cycling.

        Override if needed.
        """

    @abc.abstractmethod
    def cycle_logic(self) -> Workflow | None:
        """Perform one unit of work, yielding at safe intra/inter-cycle boundaries."""

    def cycle(self) -> Workflow:
        """Yield a handoff at each boundary."""
        cycles_completed = 0
        while True:
            result = self.cycle_logic()

            if result is not None:
                yield from result

            cycles_completed += 1
            yield Handoff(f"cycle {cycles_completed}")

            if cycles_completed >= self.bot_config.cycles_limit > 0:
                return

    @classmethod
    def workflow(cls, session: Session, bot_config: BotConfig) -> Workflow:
        """One-time blocking setup, then yield from the bot's cycle loop."""
        bot = cls(session, bot_config)
        bot.setup()
        yield from bot.cycle()


class Actions:
    """Mouse and keyboard action layer."""

    @staticmethod
    def move(point: Point) -> None:
        """Move the cursor to the requested screen-space `point`."""
        while pydirectinput.position() != (point.x, point.y):
            pydirectinput.moveTo(
                point.x, point.y, _pause=False, attempt_pixel_perfect=True
            )

    @staticmethod
    def click(point: Point, *, click_params: ClickParams = ClickParams()) -> None:
        """Move to a screen-space `point` and click."""
        Actions.move(point)
        for _ in range(click_params.count):
            pydirectinput.mouseDown(
                point.x, point.y, button=click_params.button, _pause=click_params.pause
            )
            pydirectinput.mouseUp(button=click_params.button, _pause=click_params.pause)

    @staticmethod
    def drag(
        point: Point,
        dx: int,
        dy: int,
        *,
        button: str = pydirectinput.MOUSE_SECONDARY,
        count: int = 1,
    ) -> None:
        """Drag from a screen-space `point` by `dx` and `dy`."""
        for _ in range(count):
            Actions.move(point)
            pydirectinput.mouseDown(button=button)
            Actions.move(point.offset(dx, dy))
            pydirectinput.mouseUp(button=button)

    @staticmethod
    def hotkey(*keys: str, count: int = 1) -> None:
        """Press the `keys` together as a single combination, `count` times."""
        for _ in range(count):
            pydirectinput.hotkey(*keys, wait=0.05)


class Client:
    """Handle and associated functions for a single IMAGINE client window."""

    CLIENT_IDENTIFIER: typing.Final = "IMAGINE Version 1."

    def __init__(self, window: pywinctl.Window):
        self._window = window
        self._mss = None  # Single reused mss instance, lazily loaded in capture call

    @classmethod
    def locate_all(cls, identifier: str = CLIENT_IDENTIFIER) -> tuple[Client, ...]:
        """Return all client windows whose titles begin with the client identifier."""
        windows = pywinctl.getWindowsWithTitle(
            identifier, condition=pywinctl.Re.STARTSWITH
        )

        if not windows:
            raise RuntimeError("No IMAGINE client window(s) available.")

        return tuple(cls(window) for window in windows)

    @property
    def rect(self) -> Rect:
        """Client rectangle, in screen space."""
        return Rect.from_bounds(*self._window.getClientFrame())

    @property
    def handle(self) -> int:
        """Client window handle."""
        return self._window.getHandle()

    @property
    def is_focused(self) -> bool:
        """Client window is the active window."""
        window = pywinctl.getActiveWindow()
        return window is not None and self._window.getHandle() == window.getHandle()

    def focus(self) -> None:
        """Activate the client window and wait until focus is confirmed."""
        while not self.is_focused:
            self._window.activate()
            # Wait parameter doesn't actually do anything :/
            # https://github.com/Kalmat/PyWinCtl/blob/9d06c4d5d5fa90ad54d56c36d12fd83eda5fb5d0/src/pywinctl/_pywinctl_win.py#L636
            time.sleep(0.025)

    def calculate_scale(self) -> float:
        """Client window scale factor, as determined by containing monitor DPI.

        This value is snapshotted when creating a session instance and shouldn't be
        queried otherwise.
        """
        try:
            dpi = ctypes.wintypes.UINT()

            # On x64 systems, widen the arg and return types to prevent truncation
            ctypes.windll.shcore.GetDpiForMonitor.argtypes = [
                ctypes.wintypes.HMONITOR,
                ctypes.c_int,
                ctypes.POINTER(ctypes.wintypes.UINT),
                ctypes.POINTER(ctypes.wintypes.UINT),
            ]
            ctypes.windll.user32.MonitorFromWindow.restype = ctypes.wintypes.HMONITOR

            ctypes.windll.shcore.GetDpiForMonitor(
                ctypes.windll.user32.MonitorFromWindow(self.handle, 2),
                0,
                ctypes.byref(dpi),
                ctypes.byref(dpi),
            )  # MONITOR_DEFAULTTONEAREST, MDT_EFFECTIVE_DPI
            dpi = dpi.value
        except AttributeError, OSError:
            _logger.exception("Failed to get monitor DPI")
            dpi = 0

        return dpi / 96.0 if dpi > 0 else 1.0

    def capture(self, scale: float = 1.0) -> numpy.ndarray:
        """Capture the client frame in BGR, downscaled to logical space."""
        if self._mss is None:
            self._mss = mss.MSS()

        frame = cv2.cvtColor(
            numpy.array(self._mss.grab(self.rect.bounds)), cv2.COLOR_BGRA2BGR
        )

        if _is_unit_scale(scale):
            return frame

        frame_height, frame_width = frame.shape[:2]
        return cv2.resize(
            frame,
            (round(frame_width / scale), round(frame_height / scale)),
            interpolation=cv2.INTER_AREA,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class _ScoreMap:
    """Correlation scores for one template, with frame-space coordinate context.

    scores[y, x] is the correlation of the template placed with its origin point at
    frame coordinates (x + x1, y + y1).
    """

    scores: numpy.ndarray
    template_id: str
    template_width: int
    template_height: int
    confidence: float
    x1: int
    y1: int

    def suppress(self, mask: Rect) -> None:
        """Invalidate every placement whose match region would intersect `mask`.

        Bounds are clamped to zero on both edges; a negative stop would otherwise be
        interpreted as end-relative and suppress valid placements.
        """
        mask_x1, mask_y1, mask_x2, mask_y2 = mask.bounds
        self.scores[
            max(mask_y1 - self.y1 - self.template_height + 1, 0) : max(
                mask_y2 - self.y1, 0
            ),
            max(mask_x1 - self.x1 - self.template_width + 1, 0) : max(
                mask_x2 - self.x1, 0
            ),
        ] = -1.0

    def best(self) -> TemplateMatch | None:
        """Return the highest-scoring placement remaining in the map."""
        _, max_val, _, max_loc = cv2.minMaxLoc(self.scores)

        if (
            max_val <= -1.0 or max_val < self.confidence
        ):  # Suppressed placements (-1.0) must never be returned, or locate_all will hang
            return None

        return TemplateMatch(
            self.template_id,
            Rect(
                self.x1 + max_loc[0],
                self.y1 + max_loc[1],
                self.template_width,
                self.template_height,
            ),
            max_val,
        )


class TemplateMatcher:
    """Template matching utility with template registry and region cache."""

    def __init__(self, scale: float, confidence: float):
        self._templates: dict[str, Template] = {}
        self._region_cache: dict[tuple[str, str | None], Rect] = {}
        self._scale = scale
        self._confidence = confidence

    @classmethod
    def from_template_directory(
        cls,
        template_directory: pathlib.Path,
        *,
        bot_id: str | None = None,
        scale: float,
        confidence: float,
    ) -> TemplateMatcher:
        """Create and seed matcher with PNGs from `template_directory`.

        If `bot_id` is specified, PNGs from `template_directory/<bot_id>` will seed the
        matcher in a second pass.

        `scale` values >1.0 will modify template contents to correlate better against a
        scaled client capture; dimensions are unchanged.
        """
        template_matcher = cls(scale, confidence)
        template_matcher._register_template_directory(template_directory)

        if bot_id is not None:
            template_matcher._register_template_directory(template_directory / bot_id)

        return template_matcher

    def register_template(
        self,
        template_id: str,
        frame: numpy.ndarray,
        *,
        spec: TemplateSpec | None = None,
    ) -> None:
        """Register a template frame and clear any stale cached regions.

        `frame` is expected to already be in logical space and registered as-is.
        """
        spec = spec if spec is not None else TemplateSpec()

        if spec.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._templates[template_id] = Template(frame.copy(), spec)
        stale = [
            region_cache_key
            for region_cache_key in self._region_cache
            if region_cache_key[0] == template_id  # Key: (template_id, region_cache_id)
        ]
        for region_cache_key in stale:
            del self._region_cache[region_cache_key]

    def _condition_template(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Round-trip a template to mimic transformations done to the client capture.

        First scale up the template bilinearly (approximate DWM), then downscale it
        using the same method applied to the client capture. The result will correlate
        better when matching.
        """
        if _is_unit_scale(self._scale):
            return frame

        template_height, template_width = frame.shape[:2]
        upscaled = cv2.resize(
            frame,
            (
                round(template_width * self._scale),
                round(template_height * self._scale),
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(
            upscaled,
            (template_width, template_height),
            interpolation=cv2.INTER_AREA,
        )

    def _correlate_template(
        self,
        frame: numpy.ndarray,
        template_id: str,
        region: Rect | None,
        masks: tuple[Rect, ...],
        confidence: float | None = None,
    ) -> _ScoreMap | None:
        """Correlate `template_id` against `frame`, returning a score map.

        Placements whose match region would overlap a mask are invalidated.
        """
        frame_height, frame_width = frame.shape[:2]

        if region is None:
            x1, y1, x2, y2 = 0, 0, frame_width, frame_height
        else:
            x1, y1, x2, y2 = region.clamp(frame_width, frame_height).bounds

        template = self._templates[template_id]
        template_height, template_width = template.frame.shape[:2]

        if x2 - x1 < template_width or y2 - y1 < template_height:
            return None

        frame_slice = frame[y1:y2, x1:x2]

        if template.spec.grayscale:
            frame_slice = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY)

        score_map = _ScoreMap(
            cv2.matchTemplate(frame_slice, template.frame, cv2.TM_CCOEFF_NORMED),
            template_id,
            template_width,
            template_height,
            confidence or template.spec.confidence or self._confidence,
            x1,
            y1,
        )
        for mask in masks:
            score_map.suppress(mask)
        return score_map

    def _register_template_directory(self, template_directory: pathlib.Path) -> None:
        """Register every template PNG in `template_directory`, skipping if it is absent."""
        if not template_directory.is_dir():
            return

        specs = _load_template_specs(template_directory)
        for filename in template_directory.glob("*.png"):
            frame = cv2.imread(str(filename))

            if frame is None:
                raise RuntimeError(f"Template missing or corrupt: {filename}")

            self.register_template(
                filename.stem,
                self._condition_template(frame),
                spec=specs.get(filename.stem),
            )

    def get_region_cached(
        self, template_id: str, *, region_cache_id: str | None = None
    ) -> Rect | None:
        """Return the cached region for an associated template, if it exists.

        Parameterize the computed cache key with `region_cache_id` for templates
        appearing in multiple locations.
        """
        return self._region_cache.get((template_id, region_cache_id))

    def locate(
        self,
        frame: numpy.ndarray,
        template_id: str,
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> TemplateMatch | None:
        """Attempt to match a single template on the given `frame`.

        An explicit region passed in `locate_params` will override any cached region
        for `template_id`.
        """
        region = locate_params.region
        region_cache_key = None

        if locate_params.region is None:
            region_cached = self.get_region_cached(
                template_id, region_cache_id=locate_params.region_cache_id
            )

            if region_cached is not None:
                region = region_cached.inflate(locate_params.region_padding)
            else:
                region_cache_key = (template_id, locate_params.region_cache_id)

        score_map = self._correlate_template(
            frame,
            template_id,
            region,
            locate_params.masks,
            locate_params.confidence,
        )

        if score_map is None:
            return None

        template_match = score_map.best()

        if template_match is None:
            return None

        if region_cache_key is not None:
            self._region_cache[region_cache_key] = template_match.rect

        _logger.debug(
            "Matched %s@%.6f", template_match.template_id, template_match.confidence
        )
        return template_match

    def locate_all(
        self,
        frame: numpy.ndarray,
        template_ids: tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
        group: bool = True,
    ) -> list[TemplateMatch]:
        """Match every instance of every template in `template_ids` on `frame`.

        Note that the region cache is bypassed, as multiple instances of a template
        cannot have a single canonical region. Matches are returned in descending
        confidence order and grouped by template_id if the `group` flag is set.
        """
        score_maps: list[_ScoreMap] = []
        for template_id in template_ids:
            score_map = self._correlate_template(
                frame,
                template_id,
                locate_params.region,
                locate_params.masks,
                locate_params.confidence,
            )

            if score_map is not None:
                score_maps.append(score_map)

        template_matches: list[TemplateMatch] = []
        while True:
            candidates = []
            for score_map in score_maps:
                template_match = score_map.best()

                if template_match is not None:
                    candidates.append(template_match)

            if not candidates:
                break

            best = max(candidates, key=lambda template_match: template_match.confidence)
            template_matches.append(best)
            _logger.debug("Matched %s@%.6f", best.template_id, best.confidence)

            for score_map in score_maps:
                score_map.suppress(best.rect)

        if group:
            return sorted(
                template_matches,
                key=lambda template_match: template_ids.index(
                    template_match.template_id
                ),
            )

        return template_matches


class Observation:
    """Single client capture abstraction with coordinate conversion utilities."""

    def __init__(
        self,
        frame: numpy.ndarray,
        template_matcher: TemplateMatcher,
    ):
        self.frame = frame
        self._template_matcher = template_matcher

    def locate(
        self, template_id: str, *, locate_params: LocateParams = LocateParams()
    ) -> TemplateMatch | None:
        """Match `template_id` on this frame.

        See `TemplateMatcher.locate`.
        """
        return self._template_matcher.locate(
            self.frame, template_id, locate_params=locate_params
        )

    def locate_any(
        self,
        template_ids: tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> TemplateMatch | None:
        """Match any template in `template_ids` on this frame.

        Return first match or None.
        """
        for template_id in template_ids:
            template_match = self.locate(template_id, locate_params=locate_params)

            if template_match is not None:
                return template_match
        return None

    def locate_all(
        self,
        template_ids: str | tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
        group: bool = True,
    ) -> list[TemplateMatch]:
        """Match every instance of every template in `template_ids` on this frame.

        See `TemplateMatcher.locate_all`.
        """
        if isinstance(template_ids, str):
            template_ids = (template_ids,)

        return self._template_matcher.locate_all(
            self.frame, template_ids, locate_params=locate_params, group=group
        )

    def register_frame_slice(
        self, template_id: str, region: Rect, *, spec: TemplateSpec | None = None
    ) -> None:
        """Register a slice of this observation's frame as a matchable template.

        `region` is in logical-space coordinates and is clamped to the frame bounds.
        The slice is already in logical space, so it is registered without
        conditioning.
        """
        frame_height, frame_width = self.frame.shape[:2]
        x1, y1, x2, y2 = region.clamp(frame_width, frame_height).bounds

        self._template_matcher.register_template(
            template_id, self.frame[y1:y2, x1:x2], spec=spec
        )


class Session:
    """Automation session associated with a specific client window.

    All coordinates are in logical space; points passed to `move`, `click`, and `drag`
    are converted to screen space internally.
    """

    def __init__(self, client: Client, template_matcher: TemplateMatcher, scale: float):
        self._template_matcher = template_matcher

        self.client = client
        self.scale = scale

    def _to_screen_space(self, point: Point) -> Point:
        """Convert a logical-space `point` to screen space."""
        client_rect = self.client.rect
        return Point(
            client_rect.x + round(point.x * self.scale),
            client_rect.y + round(point.y * self.scale),
        )

    def _guard(self, *points: Point) -> None:
        """Assert `points` are inside the client; pause while out of focus."""
        if not self.client.is_focused:
            _logger.info(
                "Paused. Refocus client to resume, or Ctrl+C in this console to quit."
            )

            try:
                while not self.client.is_focused:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                safe_abort()
                raise Aborted from None

            pydirectinput.mouseUp()  # Don't move window if user clicks title bar to regain focus
            _logger.info("Resumed.")

        client_rect = self.client.rect
        for point in points:
            if not client_rect.contains(point):
                raise RuntimeError(f"Point outside client: {point} {client_rect}")

    @property
    def _logical_center(self) -> Point:
        """Center of the client, in logical space."""
        client_rect = self.client.rect
        return Point(
            round(client_rect.width / self.scale) // 2,
            round(client_rect.height / self.scale) // 2,
        )

    def observe(self) -> Observation:
        """Observe a fresh client capture."""
        self._guard()
        return Observation(self.client.capture(self.scale), self._template_matcher)

    def observe_until[T](
        self,
        condition: collections.abc.Callable[[Observation], T | None],
        *,
        on_condition_failed: collections.abc.Callable[[Observation], None]
        | None = None,
        timeout: float | None = None,
        sleep: float = 0.0,
    ) -> tuple[Observation, T]:
        """Poll observations until `condition` succeeds or the `timeout` expires.

        On each failed poll, `on_condition_failed` (if given) runs against the
        observation, then `sleep` seconds elapse before the next capture.

        Raises on timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            observation = self.observe()
            result = condition(observation)

            if result:
                return observation, result

            if on_condition_failed is not None:
                on_condition_failed(observation)

            if sleep > 0.0:
                time.sleep(sleep)
        raise TimeoutError("Timed out waiting on observation condition.")

    @staticmethod
    def present(
        template_ids: str | tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> collections.abc.Callable[[Observation], TemplateMatch | None]:
        """Predicate that holds while any of `template_ids` are on screen."""
        if isinstance(template_ids, str):
            template_ids = (template_ids,)

        return lambda observation: observation.locate_any(
            template_ids, locate_params=locate_params
        )

    @staticmethod
    def absent(
        template_ids: str | tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> collections.abc.Callable[[Observation], bool]:
        """Predicate that holds while none of `template_ids` are on screen."""
        return lambda observation: (
            Session.present(template_ids, locate_params=locate_params)(observation)
            is None
        )

    def move(self, point: Point) -> None:
        """Guard client focus and bounds, then move the cursor to the `point`."""
        screen_point = self._to_screen_space(point)
        self._guard(screen_point)
        Actions.move(screen_point)

    def move_center(self) -> None:
        """Convenience method to move the cursor to the client center."""
        self.move(self._logical_center)

    def click(self, point: Point, click_params: ClickParams = ClickParams()) -> None:
        """Guard client focus and bounds, then click at the `point`."""
        for _ in range(click_params.count):
            screen_point = self._to_screen_space(point)
            self._guard(screen_point)
            Actions.click(
                screen_point, click_params=dataclasses.replace(click_params, count=1)
            )

    def click_center(self) -> None:
        """Convenience method to click the client center."""
        self.click(self._logical_center)

    def _click_template_attempt(
        self,
        observation: Observation,
        template_id: str,
        *,
        click_params: ClickParams = ClickParams(),
        locate_params: LocateParams = LocateParams(),
    ) -> TemplateMatch | None:
        """Locate the template in the `observation` and click it if found."""
        template_match = observation.locate(template_id, locate_params=locate_params)

        if template_match is not None:
            self.click(template_match.rect.center, click_params)

        return template_match

    def click_template(
        self,
        template_id: str,
        *,
        click_params: ClickParams = ClickParams(),
        locate_params: LocateParams = LocateParams(),
    ) -> None:
        """Poll observations until the template is located and clicked."""
        self.observe_until(
            lambda observation: self._click_template_attempt(
                observation,
                template_id,
                click_params=click_params,
                locate_params=locate_params,
            )
        )

    def click_template_until[T](
        self,
        template_id: str,
        condition: collections.abc.Callable[[Observation], T | None],
        *,
        interval: float = 0.25,
        click_params: ClickParams = ClickParams(),
        locate_params: LocateParams = LocateParams(),
    ) -> tuple[Observation, T]:
        """Click the template on an `interval` until the `condition` succeeds.

        The template is clicked at most once per `interval` seconds while polling, and
        must be clicked at least once. Returns once `condition` returns a truthy
        result.

        Raises on timeout.
        """
        last_click_time = 0.0
        clicked = False

        def click_attempt(observation: Observation) -> None:
            nonlocal last_click_time, clicked

            if time.monotonic() - last_click_time < interval:
                return

            if (
                self._click_template_attempt(
                    observation,
                    template_id,
                    click_params=click_params,
                    locate_params=locate_params,
                )
                is not None
            ):
                last_click_time = time.monotonic()
                clicked = True

        def gated_condition(observation: Observation) -> T | None:
            return condition(observation) if clicked else None

        return self.observe_until(gated_condition, on_condition_failed=click_attempt)

    def click_through(
        self,
        template_id: str,
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> None:
        """Click `template_id` until it leaves the frame."""
        self.click_template_until(
            template_id,
            self.absent(template_id, locate_params=locate_params),
            locate_params=locate_params,
        )

    def click_through_dialogue_until(
        self,
        template_ids: str | tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> tuple[Observation, TemplateMatch]:
        """Click the frame center until one of `template_ids` appears."""
        return self.observe_until(
            self.present(template_ids, locate_params=locate_params),
            on_condition_failed=lambda observation: (
                self.click_center()
                if self.present("dialogue_arrow", locate_params=_DIALOGUE_ARROW_PARAMS)(
                    observation
                )
                else None
            ),
        )

    def drag(self, point: Point, dx: int, dy: int, **drag_kwargs) -> None:
        """Guard client focus and bounds, then drag from `point` by `dx` and `dy`."""
        screen_point = self._to_screen_space(point)
        self._guard(screen_point, screen_point.offset(dx, dy))
        Actions.drag(screen_point, dx, dy, **drag_kwargs)

    def hotkey(self, *keys: str, **hotkey_kwargs) -> None:
        """Guard client focus, then press the key combination."""
        self._guard()
        Actions.hotkey(*keys, **hotkey_kwargs)
