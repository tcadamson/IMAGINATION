"""Orchestrate cooperative scheduling and workflows for IMAGINATION bots."""

import collections.abc
import dataclasses
import logging
import threading

import pydirectinput

import core.api
import core.config

type OnEvent = collections.abc.Callable[[str, Event], None]

_logger: logging.Logger = logging.getLogger(__name__)


class Event:
    """Base for scheduler lifecycle events."""


@dataclasses.dataclass(frozen=True)
class HandoffEvent(Event):
    """A workflow parked at a cycle boundary and yielded control."""

    reason: str


@dataclasses.dataclass(frozen=True)
class FinishedEvent(Event):
    """A workflow raised `StopIteration` and was dropped."""


@dataclasses.dataclass(frozen=True)
class CrashedEvent(Event):
    """A workflow raised an unhandled exception."""

    exception: BaseException


@dataclasses.dataclass(frozen=True)
class BotAssignment:
    """A labeled bot bound to its session and workflow factory."""

    label: str
    session: core.api.Session
    workflow: collections.abc.Callable[[], core.api.Workflow]


@dataclasses.dataclass(frozen=True)
class Bind:
    """A bot spec paired with the client and config it should run against."""

    client: core.api.Client
    spec: core.api.BotSpec
    bot_config: core.api.BotConfig


class Scheduler:
    """Round-robin scheduler advancing bot workflows one handoff at a time."""

    def __init__(
        self,
        assignments: tuple[BotAssignment, ...],
        on_event: OnEvent | None = None,
    ):
        self._on_event = on_event if on_event is not None else log_event
        self._stop = threading.Event()
        self._queue = collections.deque(
            (assignment, assignment.workflow()) for assignment in assignments
        )

    @classmethod
    def from_binds(
        cls,
        binds: collections.abc.Iterable[Bind],
        run_config: core.api.RunConfig,
        *,
        on_event: OnEvent | None = None,
    ):
        """Build a scheduler from `binds` (specs against clients)."""
        assignments = tuple(_assign(bind, run_config) for bind in binds)
        pydirectinput.PAUSE = (
            run_config.sleep
        )  # Sits naturally here, but could also expose a setter method in api.py
        return cls(assignments, on_event)

    def stop(self) -> None:
        """Signal the scheduler to halt before advancing the next workflow."""
        self._stop.set()

    def _emit(self, label: str, event: Event) -> None:
        """Dispatch an `event` for `label` to the callback, logging any error it raises."""
        try:
            self._on_event(label, event)
        except Exception:
            _logger.exception("_on_event callback raised for %s", label)

    def run(self) -> None:
        """Advance each queued workflow one handoff at a time until all finish."""
        while self._queue and not self._stop.is_set():
            assignment, workflow = self._queue.popleft()
            assignment.session.client.focus()

            try:
                handoff = next(workflow)
            except StopIteration:
                self._emit(assignment.label, FinishedEvent())
                continue
            except Exception as exception:
                self._emit(assignment.label, CrashedEvent(exception))
                continue

            self._emit(assignment.label, HandoffEvent(handoff.reason))
            self._queue.append((assignment, workflow))


def _assign(bind: Bind, run_config: core.api.RunConfig) -> BotAssignment:
    """Bind a bot `spec` to a `client`, returning a `BotAssignment`.

    Isolated scope ensures the workflow lambda closes over its own variables.
    """
    scale = bind.client.calculate_scale()
    session = core.api.Session(
        bind.client,
        core.api.TemplateMatcher.from_template_directory(
            core.config.TEMPLATE_DIRECTORY,
            bot_id=bind.spec.bot_id,
            scale=scale,
            confidence=run_config.confidence,
        ),
        scale,
    )
    return BotAssignment(
        f"{bind.spec.bot_id}@{bind.client.handle}",
        session,
        lambda: bind.spec.workflow(session, bind.bot_config),
    )


def log_event(label: str, event: Event) -> None:
    """Log a scheduler `event` for `label` at a severity matching its type."""
    match event:
        case HandoffEvent(reason=reason):
            _logger.debug("%s handoff: %s", label, reason)
        case FinishedEvent():
            _logger.info("%s finished", label)
        case CrashedEvent(exception=exception):
            _logger.error("%s crashed", label, exc_info=exception)
