"""Demon force bot implementation."""

import collections.abc
import dataclasses
import enum
import time
import typing

import cv2
import numpy
import pydirectinput

import core.api

_FLATNESS_RATIO: typing.Final = 0.8

_DIMMED_EFFECT_RATIO: typing.Final = 0.92
_DIMMED_SLOT_RATIO: typing.Final = 0.85

_EXHAUSTED_DEBOUNCE_SECONDS: typing.Final = 0.3

_DEMON_FORCE_ITEM_EFFECTS: typing.Final[collections.abc.Mapping[str, int]] = {
    "sands_0": 1,
    "sands_1": 1,
    "sands_2": 1,
    "scabbard": 3,
    "loop": 1,
}


def _frame_slice(frame: numpy.ndarray, rect: core.api.Rect) -> numpy.ndarray:
    """Return `frame` sliced by `rect`.

    Raises if `rect` is empty or out of frame.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = rect.bounds

    if (
        x1 < 0
        or y1 < 0
        or x2 > frame_width
        or y2 > frame_height
        or x1 == x2
        or y1 == y2
    ):
        raise ValueError(f"Region outside frame: {rect}")

    return frame[y1:y2, x1:x2]


def _mean_grayscale_intensity(
    frame: numpy.ndarray, rect: core.api.Rect | None = None
) -> float:
    """Grayscale intensity of `frame`, optionally sliced by logical-space `rect`.

    Used to determine if the demon force slot has dimmed, indicating the roll went
    through successfully and is pending.
    """
    frame_slice = frame if rect is None else _frame_slice(frame, rect)

    if frame_slice.ndim == 3:
        frame_slice = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY)

    return float(frame_slice.mean())


def _flatness(frame: numpy.ndarray, rect: core.api.Rect, tol: float = 18.0) -> float:
    """Fraction of `rect` within `tol` color distance of the region's median color.

    Used to determine if the demon force slot is occupied by a roll.
    """
    frame_pixels = _frame_slice(frame, rect).reshape(-1, 3).astype(numpy.float32)
    distances = numpy.linalg.norm(
        frame_pixels - numpy.median(frame_pixels, axis=0), axis=1
    )
    return float(numpy.mean(distances < tol))


@dataclasses.dataclass(frozen=True)
class DemonForceBotConfig(core.api.BotConfig):
    """Demon force bot whitelist configuration."""

    nra_magic: bool = dataclasses.field(
        default=False, metadata={"help": "Magic reflect/null."}
    )
    nra_phys: bool = dataclasses.field(
        default=False, metadata={"help": "Physical reflect/null."}
    )
    attack: bool = dataclasses.field(
        default=False, metadata={"help": "Attack action boost."}
    )
    rapid: bool = dataclasses.field(
        default=False, metadata={"help": "Rapid action boost."}
    )
    rush: bool = dataclasses.field(
        default=False, metadata={"help": "Rush action boost."}
    )
    shot: bool = dataclasses.field(
        default=False, metadata={"help": "Shot action boost."}
    )
    spin: bool = dataclasses.field(
        default=False, metadata={"help": "Spin action boost."}
    )
    lbp: bool = dataclasses.field(
        default=False, metadata={"help": "Limit break power."}
    )
    lbc: bool = dataclasses.field(
        default=False, metadata={"help": "Limit break chance."}
    )
    fcc: bool = dataclasses.field(
        default=False, metadata={"help": "Final critical correction."}
    )
    tac: bool = dataclasses.field(
        default=False, metadata={"help": "Technical attack chance."}
    )
    tap: bool = dataclasses.field(
        default=False, metadata={"help": "Technical attack power."}
    )
    puc: bool = dataclasses.field(default=False, metadata={"help": "Pursuit chance."})
    pup: bool = dataclasses.field(default=False, metadata={"help": "Pursuit power."})

    @property
    def whitelist(self) -> tuple[str, ...]:
        """Collection of template_ids corresponding to the whitelist flags passed."""
        return tuple(
            template_id
            for flag, template_ids in _WHITELIST_TEMPLATE_IDS.items()
            if getattr(self, flag)
            for template_id in template_ids
        )


_WHITELIST_TEMPLATE_IDS: typing.Final[
    collections.abc.Mapping[str, str | tuple[str, ...]]
] = {
    flag: (
        (f"whitelist_{flag}_0", f"whitelist_{flag}_1")
        if flag in ("nra_magic", "nra_phys")
        else (f"whitelist_{flag}",)
    )
    for flag in DemonForceBotConfig.__annotations__
}


class DemonForceOutcome(enum.StrEnum):
    """Possible outcomes of a demon force cycle."""

    ROLLED = enum.auto()
    EXHAUSTED = enum.auto()


class DemonForceBot(core.api.Bot):
    bot_config: DemonForceBotConfig

    def __init__(self, session: core.api.Session, bot_config: core.api.BotConfig):
        super().__init__(session, bot_config)
        self._queue: collections.deque[tuple[str, core.api.Rect]] = collections.deque()

    def setup(self) -> None:
        """Initiate demon force on the currently summoned demon.

        The demon list window is intentionally reopened if it's already open. Doing so
        clears any selected demon, allowing it to be clicked without desummoning.
        """
        observation = self.session.observe()
        devil_sentinel = observation.locate("devil_sentinel")

        if devil_sentinel is None:
            raise RuntimeError("Could not locate DEVIL on UI bar.")

        demon_1 = devil_sentinel.rect.relative(0, 0, 32, 60)
        demon_2 = devil_sentinel.rect.relative(0, -210, 50, 420)
        while True:
            self.session.click_template("devil_sentinel")
            demon_sentinel_1 = self.session.observe().locate(
                "demon_sentinel", locate_params=core.api.LocateParams(demon_1)
            )

            if demon_sentinel_1 is None:
                raise RuntimeError("Ensure the UI bar is fully in view.")

            masks = (demon_sentinel_1.rect,)
            observation, demon_sentinel_2 = self.session.click_template_until(
                "demon_sentinel",
                self.session.present(
                    "demon_sentinel",
                    locate_params=core.api.LocateParams(demon_2, masks=masks),
                ),
                locate_params=core.api.LocateParams(demon_1),
            )
            self.session.click_through(
                "demon_sentinel",
                locate_params=core.api.LocateParams(demon_2, masks=masks),
            )

            masks += (demon_sentinel_2.rect,)
            observation = self.session.observe()
            demon_sentinel_3 = observation.locate(
                "demon_sentinel", locate_params=core.api.LocateParams(masks=masks)
            )

            if demon_sentinel_3 is not None:
                summoned_sentinel = observation.locate_any(
                    ("summoned_sentinel_0", "summoned_sentinel_1"),
                    locate_params=core.api.LocateParams(
                        demon_sentinel_3.rect.relative(0, 100, 20, 475)
                    ),
                )

                if summoned_sentinel is None:
                    raise RuntimeError(
                        "Failed to determine the summoned demon. (Is it summoned?)"
                    )

                observation.register_frame_slice(
                    "summoned",
                    summoned_sentinel.rect.relative(32, -7, 30, 13),
                )
                break
        self.session.click_template("summoned")

        if self.session.observe().locate("demon_information_sentinel") is None:
            self.session.click_template(
                "summoned",
                click_params=core.api.ClickParams(button=pydirectinput.MOUSE_SECONDARY),
            )

        observation, demon_information_sentinel = self.session.observe_until(
            self.session.present("demon_information_sentinel")
        )
        demon_force_params = core.api.LocateParams(
            demon_information_sentinel.rect.relative(250, 35, 100, 20)
        )

        if observation.locate("demon_force", locate_params=demon_force_params):
            self.session.click_template("demon_force", locate_params=demon_force_params)

        if self.session.observe().locate("perform_demon_force") is None:
            raise RuntimeError(
                "Could not open demon force tab. (Is demon force unlocked?)"
            )

        self.session.click_template("perform_demon_force")

    def cycle_logic(self) -> None:
        """Roll demon force items until they are all exhausted.

        In whitelist mode (any CLI option passed), ignore all rolls except for the
        roll(s) or categories of rolls represented by the set of options.

        Item priority: green sands > red sands > blue sands > scabbards > loops.
        """
        observation, demon_force_sentinel = self.session.observe_until(
            self.session.present("demon_force_sentinel")
        )
        slot_rect = demon_force_sentinel.rect.relative(-82, 274, 24, 24)
        yes_button_params = core.api.LocateParams(slot_rect.relative(-40, 0, 40, 25))

        if _flatness(observation.frame, slot_rect) <= _FLATNESS_RATIO:
            self.session.click(
                slot_rect.center,
                click_params=core.api.ClickParams(button=pydirectinput.MOUSE_SECONDARY),
            )
            whitelist = self.bot_config.whitelist

            if whitelist:
                yes_button_observation, yes_button = self.session.observe_until(
                    self.session.present("yes_button", locate_params=yes_button_params)
                )
                roll_rect = yes_button.rect.relative(-63, -188, 36, 36)

                if (
                    yes_button_observation.locate_any(
                        whitelist, locate_params=core.api.LocateParams(roll_rect)
                    )
                    is None
                ):
                    self.session.click(yes_button.rect.center)

            self.session.observe_until(
                lambda observation: (
                    _flatness(observation.frame, slot_rect) > _FLATNESS_RATIO
                    and self.session.absent(
                        "yes_button", locate_params=yes_button_params
                    )(observation)
                )
            )

        self.session.click(demon_force_sentinel.rect.center.offset(-280, 380))

        if not self.queue:
            demon_force_items = observation.locate_all(
                tuple(_DEMON_FORCE_ITEM_EFFECTS),
                locate_params=core.api.LocateParams(
                    region=demon_force_sentinel.rect.relative(0, 100, 365, 180),
                ),
            )
            self.queue = collections.deque(
                (template_match.template_id, template_match.rect)
                for template_match in demon_force_items
            )

        def locate_effects(
            observation: core.api.Observation,
        ) -> list[core.api.TemplateMatch]:
            return observation.locate_all(
                ("effect_0", "effect_1", "effect_2"),
                locate_params=core.api.LocateParams(
                    region=demon_force_sentinel.rect.relative(0, 310, 100, 30)
                ),
            )

        def expected_effects_visible(
            template_id: str,
        ) -> collections.abc.Callable[
            [core.api.Observation], list[core.api.TemplateMatch]
        ]:
            def predicate(
                observation: core.api.Observation,
            ) -> list[core.api.TemplateMatch]:
                effects = locate_effects(observation)
                return (
                    effects
                    if len(effects) == _DEMON_FORCE_ITEM_EFFECTS[template_id]
                    else []
                )

            return predicate

        available_effect = None
        while available_effect is None and self.queue:
            template_id, rect = self.queue[0]

            if (
                self.session.observe().locate(
                    template_id, locate_params=core.api.LocateParams(rect)
                )
                is None
            ):
                self.queue.popleft()
                continue

            self.session.click(rect.center)
            self.session.move_center()
            observation, effects = self.session.observe_until(
                expected_effects_visible(template_id)
            )
            for effect in effects:
                if _mean_grayscale_intensity(
                    observation.frame, effect.rect
                ) >= _DIMMED_EFFECT_RATIO * _mean_grayscale_intensity(
                    observation.get_template(effect.template_id).frame
                ):
                    available_effect = effect
                    break

            if available_effect is None:
                self.queue.popleft()
                continue

            self.session.click(available_effect.rect.center)
            self.session.click(
                demon_force_sentinel.rect.center.offset(0, 340),
                click_params=core.api.ClickParams(count=100, pause=False),
            )
            slot_baseline = _mean_grayscale_intensity(observation.frame, slot_rect)
            effects_absent_since: float | None = None

            def possible_outcomes(
                observation: core.api.Observation,
            ) -> DemonForceOutcome | None:
                nonlocal effects_absent_since

                if (
                    _mean_grayscale_intensity(observation.frame, slot_rect)
                    < _DIMMED_SLOT_RATIO * slot_baseline
                ):
                    return DemonForceOutcome.ROLLED

                if locate_effects(observation):
                    effects_absent_since = None
                    return None

                if effects_absent_since is None:  # Debounce in case roll went through
                    effects_absent_since = time.monotonic()
                    return None

                if (
                    time.monotonic() - effects_absent_since
                    >= _EXHAUSTED_DEBOUNCE_SECONDS
                ):
                    return DemonForceOutcome.EXHAUSTED

                return None

            observation, outcome = self.session.observe_until(possible_outcomes)

            if outcome == DemonForceOutcome.ROLLED:
                break

            available_effect = None  # Demon force item exhausted; move on to the next
            self.queue.popleft()

        if not self.queue:
            raise RuntimeError(
                "Demon force items exhausted or missing."
            )  # TODO: Finished event

        self.session.click(demon_force_sentinel.rect.center.offset(330, 342))
        _, perform_demon_force = self.session.observe_until(
            self.session.present("perform_demon_force")
        )
        self.session.click(perform_demon_force.rect.center)

        # TODO: Use this block IFF the above block proves too error-prone for users
        # self.session.click_through("close")
        # self.session.click_through("perform_demon_force")


SPEC: typing.Final = core.api.BotSpec(
    DemonForceBot,
    help="""Roll demon force items on the currently summoned demon.
    
    Normally, the bot will wait for you to accept or discard a role before continuing.
    Including any of the below options will put the bot in whitelist mode and discard
    any roll or category of roll not included in the set of options passed.
    
    \b
    Example (whitelist mode):
    run demon_force --nra-magic --nra-phys --shot --lbp --tap
    """,
)
