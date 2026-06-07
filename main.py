"""Automation framework for the SMT: IMAGINE client."""

import collections.abc
import dataclasses
import pathlib
import sys
import time

import cv2
import mss
import numpy
import pydirectinput
import pywinctl

Padding = int | tuple[int, int]

DEFAULT_CONFIDENCE: float = 0.85
DEFAULT_SLEEP: float = 0.08

if getattr(sys, "frozen", False):
    ROOT = pathlib.Path(sys.executable).parent
else:
    ROOT = pathlib.Path(__file__).resolve().parent

TEMPLATE_OVERRIDES: dict[str, TemplateMetadata] = {}


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
    def from_bounds(cls, left: int, top: int, right: int, bottom: int) -> Rect:
        """Create rectangle from `left`, `top`, `right`, and `bottom` bounds."""
        return cls(left, top, right - left, bottom - top)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Rectangle as left, top, right, and bottom bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def origin(self) -> Point:
        """Origin (top-left) point of the rectangle."""
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

    def inflate(self, padding: Padding) -> Rect:
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


@dataclasses.dataclass(frozen=True, eq=False)
class Template:
    """Immutable template data with frame and associated metadata."""

    frame: numpy.ndarray
    metadata: TemplateMetadata


@dataclasses.dataclass(frozen=True)
class TemplateMetadata:
    """Immutable template metadata."""

    confidence: float = DEFAULT_CONFIDENCE
    grayscale: bool = True


@dataclasses.dataclass(frozen=True)
class TemplateMatch:
    """Immutable template match data in client-space coordinates."""

    template_id: str
    rect: Rect
    confidence: float


@dataclasses.dataclass(frozen=True, eq=False)
class LocateParams:
    """Immutable locate parameters."""

    region: Rect | None = None
    region_padding: Padding = 0
    region_cache_id: str | None = None
    mask: numpy.ndarray | None = None
    confidence: float | None = None


@dataclasses.dataclass(frozen=True)
class ClickParams:
    """Immutable click parameters."""

    button: str = pydirectinput.MOUSE_PRIMARY
    count: int = 1
    pause: bool = True


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
            pydirectinput.hotkey(*keys, wait=pydirectinput.PAUSE)


class Client:
    """Handle and associated functions for a single IMAGINE client window."""

    CLIENT_IDENTIFIER: str = "IMAGINE Version 1."

    def __init__(self, window: pywinctl.Window):
        self._window = window
        self._mss = None  # Single reused mss instance, lazily loaded in capture call

    @classmethod
    def locate_all(cls, client_identifier: str = CLIENT_IDENTIFIER) -> list[Client]:
        """Return all client windows whose titles match the client identifier."""
        windows = pywinctl.getWindowsWithTitle(
            client_identifier, condition=pywinctl.Re.STARTSWITH
        )

        if not windows:
            raise RuntimeError("No IMAGINE client window(s) available.")

        return [cls(window) for window in windows]

    @property
    def rect(self) -> Rect:
        """Client rectangle, in screen space."""
        return Rect.from_bounds(*self._window.getClientFrame())

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

    def __init__(self):
        self._templates: dict[str, Template] = {}
        self._region_cache: dict[tuple[str, str | None], Rect] = {}

    @classmethod
    def load(cls, root: pathlib.Path = ROOT) -> TemplateMatcher:
        """Create and seed matcher with PNGs from the templates directory."""
        template_matcher = cls()
        for filename in root.glob("templates/*.png"):
            frame = cv2.imread(str(filename))

            if frame is None:
                raise RuntimeError(f"Template missing or corrupt: {filename}")

            template_matcher.register_template(filename.stem, frame)
        return template_matcher

    def register_template(
        self,
        template_id: str,
        frame: numpy.ndarray,
        metadata: TemplateMetadata | None = None,
        overrides: dict[str, TemplateMetadata] | None = None,
    ) -> None:
        """Register a template image and clear any stale cached regions."""
        if overrides is None:
            overrides = TEMPLATE_OVERRIDES

        if metadata is None:
            metadata = overrides.get(template_id, TemplateMetadata())

        if metadata.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._templates[template_id] = Template(frame.copy(), metadata)
        stale = [
            region_cache_key
            for region_cache_key in self._region_cache
            if region_cache_key[0] == template_id  # Key: (template_id, region_cache_id)
        ]
        for region_cache_key in stale:
            del self._region_cache[region_cache_key]

    def region_cached(
        self, template_id: str, *, region_cache_id: str | None = None
    ) -> Rect | None:
        """Return the cached region for an associated template, if it exists.

        Parameterize the computed cache key with `region_cache_id` for templates
        appearing in multiple locations.
        """
        return self._region_cache.get((template_id, region_cache_id))

    def _locate(
        self,
        frame: numpy.ndarray,
        template_id: str,
        region: Rect | None,
        mask: numpy.ndarray | None,
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
            x1, y1 = max(0, region.x), max(0, region.y)
            x2, y2 = (
                min(frame_width, region.x + region.width),
                min(frame_height, region.y + region.height),
            )

        template = self._templates[template_id]
        template_height, template_width = template.frame.shape[:2]

        if x2 - x1 < template_width or y2 - y1 < template_height:
            return None

        frame_slice = frame[y1:y2, x1:x2]

        if mask is not None:
            frame_slice = cv2.bitwise_and(
                frame_slice, frame_slice, mask=mask[y1:y2, x1:x2]
            )

        if template.metadata.grayscale:
            frame_slice = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(frame_slice, template.frame, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < (
            confidence if confidence is not None else template.metadata.confidence
        ):
            return None

        rect_in_client_space = Rect(
            x1 + max_loc[0], y1 + max_loc[1], template_width, template_height
        )
        return TemplateMatch(template_id, rect_in_client_space, max_val)

    def locate(
        self,
        frame: numpy.ndarray,
        template_id: str,
        *,
        locate_params: LocateParams = LocateParams(),
    ) -> TemplateMatch | None:
        """Attempt to match a single template on the given `frame`."""
        if locate_params.region is not None:
            return self._locate(
                frame,
                template_id,
                locate_params.region,
                locate_params.mask,
                locate_params.confidence,
            )

        region_cache_key = (template_id, locate_params.region_cache_id)
        region_cached = self.region_cached(
            template_id, region_cache_id=locate_params.region_cache_id
        )

        if region_cached is not None:
            return self._locate(
                frame,
                template_id,
                region_cached.inflate(locate_params.region_padding),
                locate_params.mask,
                locate_params.confidence,
            )

        template_match = self._locate(
            frame, template_id, None, locate_params.mask, locate_params.confidence
        )

        if template_match is not None:
            self._region_cache[region_cache_key] = template_match.rect

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
        template_ids: collections.abc.Iterable[str],
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

    def to_screen_space(self, point: Point) -> Point:
        """Convert a client-space `point` to screen space."""
        return point.offset(self.rect.x, self.rect.y)


class Session:
    """Automation session associated with a specific client window."""

    def __init__(
        self,
        client: Client,
        template_matcher: TemplateMatcher,
    ):
        self.client = client
        self.template_matcher = template_matcher

    def _guard(self, *points: Point) -> None:
        """Assert the client has focus and all `points` are inside of it."""
        if not self.client.is_focused:
            raise RuntimeError("IMAGINE client window lost focus.")

        client_rect = self.client.rect
        for point in points:
            if not client_rect.contains(point):
                raise RuntimeError(f"Point outside client: {point} {client_rect}")

    def observe(self) -> Observation:
        """Observe a fresh client capture."""
        self._guard()
        return Observation(
            self.client.capture(), self.client.rect, self.template_matcher
        )

    def observe_until[T](
        self,
        condition: collections.abc.Callable[[Observation], T | None],
        condition_failed: collections.abc.Callable[[Observation], None] | None = None,
        *,
        timeout: float = 7.5,
        sleep: float = 0.0,
    ) -> tuple[Observation, T]:
        """Poll observations until `condition` succeeds or the `timeout` expires.

        On each failed poll, `condition_failed` (if given) runs against the
        observation, then `sleep` seconds elapse before the next capture.

        Raises on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            observation = self.observe()
            result = condition(observation)

            if result:
                return observation, result

            if condition_failed is not None:
                condition_failed(observation)

            if sleep > 0.0:
                time.sleep(sleep)
        raise TimeoutError("Timed out waiting on observation condition.")

    def move(self, point: Point) -> None:
        """Guard client focus and bounds, then move the cursor to the `point`."""
        self._guard(point)
        Actions.move(point)

    def click(self, point: Point, click_params: ClickParams = ClickParams()) -> None:
        """Guard client focus and bounds, then click at the `point`."""
        self._guard(point)
        Actions.click(point, click_params=click_params)

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

        The template is clicked at most once per `interval` seconds while
        polling. Returns once `condition` returns a truthy result.

        Raises on timeout.
        """
        last_click_time = 0.0

        def click_attempt(observation: Observation) -> None:
            nonlocal last_click_time

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

        return self.observe_until(condition, click_attempt)

    def drag(self, point: Point, dx: int, dy: int, **drag_kwargs) -> None:
        """Guard client focus and bounds, then drag from `point` by given deltas."""
        self._guard(point, point.offset(dx, dy))
        Actions.drag(point, dx, dy, **drag_kwargs)

    def hotkey(self, *keys: str, **hotkey_kwargs) -> None:
        """Guard client focus, then press the key combination."""
        self._guard()
        Actions.hotkey(*keys, **hotkey_kwargs)


if __name__ == "__main__":
    pydirectinput.PAUSE = DEFAULT_SLEEP
    sessions: list[Session] = []
    for client in Client.locate_all():
        sessions.append(Session(client, TemplateMatcher.load()))
    for session in sessions:
        session.client.focus()
        session.click_template("quit", click_params=ClickParams(count=0))
