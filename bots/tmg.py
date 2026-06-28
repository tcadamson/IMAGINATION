"""TMG bot implementation."""

import dataclasses
import typing

import core.api


@dataclasses.dataclass(frozen=True)
class TMGBotConfig(core.api.BotConfig):
    pass  # TODO: --false option


class TMGBot(core.api.Bot):
    bot_config: TMGBotConfig

    def _banner_option_params(
        self, template_match: core.api.TemplateMatch, region_cache_id: str | None = None
    ) -> core.api.LocateParams:
        """Return locate params for the option below a matched banner."""
        return core.api.LocateParams(
            region=template_match.rect.relative(
                -5, template_match.rect.height, *self.session.scaled(150, 75)
            ),
            region_cache_id=region_cache_id,
        )

    def cycle_logic(self):
        while True:
            # Go to top floor
            _, template_match = self.session.observe_until(
                self.session.present("dungeon_mode_banner")
            )
            self.session.click_through(
                "normal", self._banner_option_params(template_match)
            )
            self.session.click_through("yes")
            self.session.move_center()
            self.session.observe_until(
                self.session.present("dialogue_arrow", core.api._DIALOGUE_ARROW_PARAMS)
            )
            self.session.click_through_dialogue_until("show_grimoire_1")
            self.session.click_through("show_grimoire_1")
            _, template_match = self.session.click_through_dialogue_until(
                "go_to_top_floor_banner"
            )
            self.session.click_through(
                "go_to_top_floor", self._banner_option_params(template_match)
            )
            self.session.move_center()

            # Remove all ghosts
            self.session.observe_until(
                self.session.present("dialogue_arrow", core.api._DIALOGUE_ARROW_PARAMS)
            )
            self.session.click_through_dialogue_until("show_grimoire_2")
            self.session.click_through("show_grimoire_2")
            self.session.click_through_dialogue_until("remove_all_ghosts")
            self.session.click_through("remove_all_ghosts")
            self.session.click_through_dialogue_until("info")
            self.session.observe_until(
                self.session.present("dialogue_arrow", core.api._DIALOGUE_ARROW_PARAMS)
            )
            self.session.click_through_dialogue_until("go_to_roof")
            self.session.click_through("go_to_roof")
            _, template_match = self.session.click_through_dialogue_until(
                "go_to_roof_banner"
            )
            self.session.click_through(
                "yes", self._banner_option_params(template_match, "yes_yagishima")
            )
            self.session.move_center()

            # Go to lucifuge
            _, template_match = self.session.observe_until(
                self.session.present("go_to_lucifuge_banner")
            )
            self.session.click_through(
                "yes", self._banner_option_params(template_match, "yes_roof")
            )
            self.session.move_center()

            # Talk to lucifuge
            self.session.observe_until(
                self.session.present("dialogue_arrow", core.api._DIALOGUE_ARROW_PARAMS)
            )
            self.session.click_through_dialogue_until("info")

            # Loot from lucifuge and exit
            self.session.observe_until(self.session.present("take_all"))
            self.session.click_template("take_all")
            self.session.move_center()
            _, template_match = self.session.observe_until(
                self.session.present("exit_lucifuge_banner")
            )
            self.session.click_through(
                "yes", self._banner_option_params(template_match, "yes_leave")
            )
            self.session.move_center()


SPEC: typing.Final = core.api.BotSpec(
    TMGBot, help="Farm hazel branches without wanting to mudo yourself."
)
