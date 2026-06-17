"""Rebirth bot implementation for IMAGINATION."""

import dataclasses
import typing

import api


@dataclasses.dataclass(frozen=True)
class RebirthBotConfig(api.BotConfig):
    cycles_limit: int = 1


class RebirthBot(api.Bot):
    bot_config: RebirthBotConfig

    def pre_cycle(self) -> None:
        self.session.click_template("quit", click_params=api.ClickParams(count=0))

    def cycle(self):
        cycles_completed = 0
        while True:
            # TODO: Rebirth blocking work performed here every cycle
            cycles_completed += 1
            yield api.Handoff(f"cycle {cycles_completed}")

            if cycles_completed >= self.bot_config.cycles_limit > 0:
                return


SPEC: typing.Final = api.BotSpec(RebirthBotConfig, RebirthBot.workflow)
