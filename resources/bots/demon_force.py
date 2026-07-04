"""Demon force bot implementation."""

import dataclasses
import typing

import pydirectinput

import core.api


@dataclasses.dataclass(frozen=True)
class DemonForceBotConfig(core.api.BotConfig):
    pass


class DemonForceBot(core.api.Bot):
    bot_config: DemonForceBotConfig

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

    def cycle_logic(self) -> None:
        self.session.click_through("perform_demon_force")


SPEC: typing.Final = core.api.BotSpec(
    DemonForceBot, help="Roll demon force items on the currently summoned demon."
)
