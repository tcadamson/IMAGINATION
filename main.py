"""Program entry point for IMAGINATION."""

import logging
import sys

import core.api
import core.config
import core.registry
import core.runtime

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s: %(filename)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                core.api.ROOT_DIRECTORY / "debug.log", encoding="utf-8"
            ),
        ],
    )
    try:
        clients = core.api.Client.locate_all()
        scheduler = core.runtime.Scheduler.from_client_binds(
            (("rebirth", clients[0]),),
            core.registry.register_bot_directory(core.config.BOT_DIRECTORY),
        )
        scheduler.run()
    except Exception:
        logging.exception("Fatal exception:")
        sys.exit(1)
