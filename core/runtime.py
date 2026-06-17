"""Orchestrate cooperative scheduling and workflows for IMAGINATION bots."""

import collections.abc
import dataclasses
import logging
import threading

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
    def from_client_binds(
        cls,
        client_binds: collections.abc.Iterable[tuple[str, core.api.Client]],
        bots: collections.abc.Mapping[str, core.api.BotSpec],
        *,
        on_event: OnEvent | None = None,
    ):
        """Build a scheduler from `client_binds` against the `bots` registry."""
        assignments = tuple(
            _assign(bot_id, bots[bot_id], client) for bot_id, client in client_binds
        )
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


def _assign(
    bot_id: str, spec: core.api.BotSpec, client: core.api.Client
) -> BotAssignment:
    """Bind a bot `spec` to a `client`, returning a `BotAssignment`.

    Isolated scope ensures the workflow lambda closes over its own variables.
    """
    session = core.api.Session.from_client(
        client, core.config.TEMPLATE_DIRECTORY, bot_id=bot_id
    )
    bot_config = spec.bot_config_type()
    return BotAssignment(
        f"{bot_id}@{client.handle}",
        session,
        lambda: spec.workflow(session, bot_config),
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
