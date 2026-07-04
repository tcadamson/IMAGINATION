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
DEFAULT_SLEEP: typing.Final = 0.08

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


@dataclasses.dataclass(frozen=True)
class Point:
    """Immutable, two-dimensional integer coordinate."""

    x: int
    y: int

    def offset(self, dx: int = 0, dy: int = 0) -> Point:
        """Return a new point shifted by the given deltas."""
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
    """Immutable template match data in client-space coordinates."""

    template_id: str
    rect: Rect
    confidence: float


@dataclasses.dataclass(frozen=True, eq=False)
class LocateParams:
    """Immutable locate parameters.

    Note that `region_padding` is scaled at locate time and should not be scaled
    manually.
    """

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
        """Drag from a screen-space `point` by the given deltas."""
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

    def capture(self) -> numpy.ndarray:
        """Capture the client frame in BGR."""
        if self._mss is None:
            self._mss = mss.MSS()

        frame = cv2.cvtColor(
            numpy.array(self._mss.grab(self.rect.bounds)), cv2.COLOR_BGRA2BGR
        )
        return frame


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

        `scale` resizes every template to match the client scale factor.
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

        `frame` is expected to already be at client scale.
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

    def _scale_to_client(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Resize a frame to match the client scale factor."""
        if abs(self._scale - 1.0) <= 1e-3:
            return frame

        template_height, template_width = frame.shape[:2]
        return cv2.resize(
            frame,
            (
                round(template_width * self._scale),
                round(template_height * self._scale),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

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
                self._scale_to_client(frame),
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

    def _scaled_padding(self, padding: _Padding) -> _Padding:
        """Scale hand-authored padding to the client's scale factor."""
        if isinstance(padding, int):
            return round(padding * self._scale)

        dx, dy = padding
        return round(dx * self._scale), round(dy * self._scale)

    def _locate(
        self,
        frame: numpy.ndarray,
        template_id: str,
        region: Rect | None,
        masks: tuple[Rect, ...],
        confidence: float | None = None,
    ) -> TemplateMatch | None:
        """Attempt to match a single template on the given `frame`.

        The public-facing method seeds this internal method with regions from
        the region cache when applicable.
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

        if masks:
            frame_mask = numpy.full((frame_height, frame_width), 255, dtype=numpy.uint8)
            for mask in masks:
                x1_mask, y1_mask, x2_mask, y2_mask = mask.clamp(
                    frame_width, frame_height
                ).bounds
                frame_mask[y1_mask:y2_mask, x1_mask:x2_mask] = 0
            frame_slice = cv2.bitwise_and(
                frame_slice, frame_slice, mask=frame_mask[y1:y2, x1:x2]
            )

        if template.spec.grayscale:
            frame_slice = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(frame_slice, template.frame, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < (confidence or template.spec.confidence or self._confidence):
            return None

        _logger.debug("Matched %s@%.6f", template_id, max_val)
        return TemplateMatch(
            template_id,
            Rect(x1 + max_loc[0], y1 + max_loc[1], template_width, template_height),
            max_val,
        )

    def locate(
        self,
        frame: numpy.ndarray,
        template_id: str,
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> TemplateMatch | None:
        """Attempt to match a single template on the given `frame`."""
        region = locate_params.region
        cache = False

        if locate_params.region is None:
            region_cached = self.get_region_cached(
                template_id, region_cache_id=locate_params.region_cache_id
            )

            if region_cached is not None:
                region = region_cached.inflate(
                    self._scaled_padding(locate_params.region_padding)
                )
            else:
                cache = True

        template_match = self._locate(
            frame,
            template_id,
            region,
            locate_params.masks,
            locate_params.confidence,
        )

        if cache and template_match is not None:
            self._region_cache[(template_id, locate_params.region_cache_id)] = (
                template_match.rect
            )

        return template_match


class Observation:
    """Single client capture abstraction with coordinate conversion utilities."""

    def __init__(
        self,
        frame: numpy.ndarray,
        rect: Rect,
        template_matcher: TemplateMatcher,
    ):
        self.frame = frame
        self.rect = rect
        self._template_matcher = template_matcher

    def locate(
        self, template_id: str, *, locate_params: LocateParams = LocateParams()
    ) -> TemplateMatch | None:
        """Attempt to match `template_id` on the observation frame."""
        return self._template_matcher.locate(
            self.frame, template_id, locate_params=locate_params
        )

    def locate_any(
        self,
        template_ids: tuple[str, ...],
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> TemplateMatch | None:
        """Attempt to match any template in `template_ids` on the observation frame.

        Return first match or None.
        """
        for template_id in template_ids:
            template_match = self.locate(template_id, locate_params=locate_params)

            if template_match is not None:
                return template_match
        return None

    def register_frame_slice(
        self, template_id: str, region: Rect, *, spec: TemplateSpec | None = None
    ) -> None:
        """Register a slice of this observation's frame as a matchable template.

        `region` is in client-space coordinates and is clamped to the frame bounds. The
        slice is already at client scale, so it is registered without scaling.
        """
        frame_height, frame_width = self.frame.shape[:2]
        x1, y1, x2, y2 = region.clamp(frame_width, frame_height).bounds

        self._template_matcher.register_template(
            template_id, self.frame[y1:y2, x1:x2], spec=spec
        )

    def to_screen_space(self, point: Point) -> Point:
        """Convert a client-space `point` to screen space."""
        return point.offset(self.rect.x, self.rect.y)


class Session:
    """Automation session associated with a specific client window.

    `scale` exposes the client's scale factor for converting hand-authored pixel
    offsets.
    """

    def __init__(self, client: Client, template_matcher: TemplateMatcher, scale: float):
        self._template_matcher = template_matcher

        self.client = client
        self.scale = scale

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
            _logger.info("Resuming.")

        client_rect = self.client.rect
        for point in points:
            if not client_rect.contains(point):
                raise RuntimeError(f"Point outside client: {point} {client_rect}")

    def scaled(self, *values: float) -> tuple[int, ...]:
        """Scale hand-authored pixel offsets by the client's scale factor."""
        return tuple(round(value * self.scale) for value in values)

    def observe(self) -> Observation:
        """Observe a fresh client capture."""
        self._guard()
        return Observation(
            self.client.capture(), self.client.rect, self._template_matcher
        )

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
        self._guard(point)
        Actions.move(point)

    def move_center(self) -> None:
        """Convenience method to move the cursor to the client center."""
        self.move(self.client.rect.center)

    def click(self, point: Point, click_params: ClickParams = ClickParams()) -> None:
        """Guard client focus and bounds, then click at the `point`."""
        for _ in range(click_params.count):
            self._guard(point)
            Actions.click(
                point, click_params=dataclasses.replace(click_params, count=1)
            )

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
            self.click(
                observation.to_screen_space(template_match.rect.center), click_params
            )

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
                self.click(observation.rect.center)
                if self.present("dialogue_arrow", locate_params=_DIALOGUE_ARROW_PARAMS)(
                    observation
                )
                else None
            ),
        )

    def drag(self, point: Point, dx: int, dy: int, **drag_kwargs) -> None:
        """Guard client focus and bounds, then drag from `point` by given deltas."""
        self._guard(point, point.offset(dx, dy))
        Actions.drag(point, dx, dy, **drag_kwargs)

    def hotkey(self, *keys: str, **hotkey_kwargs) -> None:
        """Guard client focus, then press the key combination."""
        self._guard()
        Actions.hotkey(*keys, **hotkey_kwargs)
