"""Rebirth bot implementation for IMAGINATION."""

import dataclasses
import typing

import core.api


@dataclasses.dataclass(frozen=True)
class RebirthBotConfig(core.api.BotConfig):
    cycles_limit: int = dataclasses.field(default=1, kw_only=True)


class RebirthBot(core.api.Bot):
    bot_config: RebirthBotConfig

    def setup(self) -> None:
        self.session.click_template("quit", click_params=core.api.ClickParams(count=0))

    def cycle(self):
        cycles_completed = 0
        while True:
            # TODO: Rebirth blocking work performed here every cycle
            cycles_completed += 1
            yield core.api.Handoff(f"cycle {cycles_completed}")

            if cycles_completed >= self.bot_config.cycles_limit > 0:
                return


SPEC: typing.Final = core.api.BotSpec(
    "rebirth",
    RebirthBotConfig,
    RebirthBot.workflow,
    help="Perform rebirths on the currently summoned demon.",
)
