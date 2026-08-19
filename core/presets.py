"""Load, parse, and persist the IMAGINATION console presets."""

import json
import typing

import core.paths

PRESETS_PATH: typing.Final = core.paths.USER_DIRECTORY / "resources" / "presets.json"


def load_presets() -> dict[str, str]:
    """Load command presets, which are simple (command name) : (command) pairings.

    User-authored presets.json shadows the tracked presets.default.json.
    """
    presets = {}
    for path in (PRESETS_PATH.with_suffix(".default.json"), PRESETS_PATH):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exception:
            raise RuntimeError(f"Malformed JSON at {path}") from exception

        if not isinstance(data, dict):
            raise RuntimeError(f"Expected object at {path}")

        for preset_id, preset in data.items():
            if not isinstance(preset, str):
                raise RuntimeError(f"Invalid preset {preset_id!r} at {path}")
        presets |= data
    return presets


def ensure_presets() -> None:
    """Create an empty user presets file next to the defaults if it is absent."""
    if not PRESETS_PATH.exists():
        PRESETS_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )  # On a completely fresh launch, resources/ does not exist
        PRESETS_PATH.write_text("{}\n", encoding="utf-8")
