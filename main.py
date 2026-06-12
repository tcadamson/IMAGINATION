"""Program entry point for IMAGINATION."""

import logging
import sys

import api
import config
import registry
import runtime

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s: %(filename)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(api.ROOT_DIRECTORY / "debug.log", encoding="utf-8"),
        ],
    )
    try:
        clients = api.Client.locate_all()
        scheduler = runtime.Scheduler.from_client_binds(
            (("rebirth", clients[0]),),
            registry.register_bot_directory(config.BOT_DIRECTORY),
        )
        scheduler.run()
    except Exception:
        logging.exception("Fatal exception:")
        sys.exit(1)
