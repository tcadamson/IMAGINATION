"""TMG bot implementation."""

import dataclasses
import typing

import core.api


@dataclasses.dataclass(frozen=True)
class TMGBotConfig(core.api.BotConfig):
    false: bool = dataclasses.field(
        default=False,
        metadata={"help": "Go straight to the roof (two mag pressers per run)."},
    )


class TMGBot(core.api.Bot):
    bot_config: TMGBotConfig

    def _sentinel_option_params(
        self, sentinel: core.api.TemplateMatch, region_cache_id: str | None = None
    ) -> core.api.LocateParams:
        """Return locate params for the option below a matched sentinel."""
        return core.api.LocateParams(
            region=sentinel.rect.relative(-5, sentinel.rect.height, 150, 75),
            region_cache_id=region_cache_id,
        )

    def cycle_logic(self) -> None:
        """Travel through menuing hell to farm Lucifuge.

        Users may elect to do false runs for mag pressers via the CLI option.
        """
        # Go to top floor
        _, dungeon_mode_sentinel = self.session.observe_until(
            self.session.present("dungeon_mode_sentinel")
        )
        self.session.click_through(
            "normal", locate_params=self._sentinel_option_params(dungeon_mode_sentinel)
        )
        self.session.click_through("yes")
        self.session.move_center()
        self.session.observe_until(
            self.session.present(
                "dialogue_arrow", locate_params=core.api._DIALOGUE_ARROW_PARAMS
            )
        )
        self.session.click_through_dialogue_until("show_grimoire_1")
        self.session.click_through("show_grimoire_1")
        _, go_to_top_floor_sentinel = self.session.click_through_dialogue_until(
            "go_to_top_floor_sentinel"
        )
        self.session.click_through(
            "go_to_top_floor",
            locate_params=self._sentinel_option_params(go_to_top_floor_sentinel),
        )
        self.session.move_center()

        # Remove all ghosts
        self.session.observe_until(
            self.session.present(
                "dialogue_arrow", locate_params=core.api._DIALOGUE_ARROW_PARAMS
            )
        )

        if not self.bot_config.false:
            self.session.click_through_dialogue_until("show_grimoire_2")
            self.session.click_through("show_grimoire_2")
            self.session.click_through_dialogue_until("remove_all_ghosts")
            self.session.click_through("remove_all_ghosts")
            self.session.click_through_dialogue_until("info")
            self.session.observe_until(
                self.session.present(
                    "dialogue_arrow", locate_params=core.api._DIALOGUE_ARROW_PARAMS
                )
            )

        self.session.click_through_dialogue_until("go_to_roof")
        self.session.click_through("go_to_roof")
        _, go_to_roof_sentinel = self.session.click_through_dialogue_until(
            "go_to_roof_sentinel"
        )
        self.session.click_through(
            "yes",
            locate_params=self._sentinel_option_params(
                go_to_roof_sentinel, "yes_yagishima"
            ),
        )
        self.session.move_center()

        # Go to lucifuge
        _, go_to_lucifuge_sentinel = self.session.observe_until(
            self.session.present("go_to_lucifuge_sentinel")
        )
        self.session.click_through(
            "yes",
            locate_params=self._sentinel_option_params(
                go_to_lucifuge_sentinel, "yes_roof"
            ),
        )
        self.session.move_center()

        # Talk to lucifuge
        self.session.observe_until(
            self.session.present(
                "dialogue_arrow", locate_params=core.api._DIALOGUE_ARROW_PARAMS
            )
        )
        self.session.click_through_dialogue_until("info")

        # Loot from lucifuge and exit
        _, treasure_box = self.session.observe_until(
            self.session.present("treasure_box")
        )
        self.session.click_template(
            "take_all",
            locate_params=core.api.LocateParams(
                treasure_box.rect.relative(30, 150, 90, 30)
            ),
        )
        self.session.move_center()
        _, exit_lucifuge_sentinel = self.session.observe_until(
            self.session.present("exit_lucifuge_sentinel")
        )
        self.session.click_through(
            "yes",
            locate_params=self._sentinel_option_params(
                exit_lucifuge_sentinel, "yes_leave"
            ),
        )
        self.session.move_center()


SPEC: typing.Final = core.api.BotSpec(
    TMGBot, help="Farm hazel branches without wanting to mudo yourself."
)
