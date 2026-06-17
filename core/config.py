"""Load, parse, and persist the IMAGINATION program configuration."""

import collections.abc
import dataclasses
import enum
import json
import typing

import core.api

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance

USER_DIRECTORY: typing.Final = core.api.ROOT_DIRECTORY  # TODO: Derive from platformdirs
BOT_DIRECTORY: typing.Final = USER_DIRECTORY / "bots"
TEMPLATE_DIRECTORY: typing.Final = USER_DIRECTORY / "templates"

_CONFIG_PATH: typing.Final = USER_DIRECTORY / "config.json"


@dataclasses.dataclass(frozen=True)
class AppConfig:
    """Immutable program configuration."""

    confidence: float = core.api.DEFAULT_CONFIDENCE
    sleep: float = core.api.DEFAULT_SLEEP


@dataclasses.dataclass(frozen=True)
class PresetConfig:
    """Immutable configuration for a CLI preset."""  # TODO: CLI framework will inform the preset construction


class Config(typing.TypedDict):
    """Type specification for the program configuration."""

    app: AppConfig
    bots: collections.abc.Mapping[str, core.api.BotConfig]
    presets: collections.abc.Mapping[str, PresetConfig]


def _enum_type(hint: object) -> type[enum.Enum] | None:
    """Return the `Enum` subclass in `hint`, unwrapping optional `Enum | None` unions."""
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint

    for arg in typing.get_args(hint):
        if isinstance(arg, type) and issubclass(arg, enum.Enum):
            return arg
    return None


def _from_mapping[T: DataclassInstance](
    config_type: type[T], data: collections.abc.Mapping[str, typing.Any]
) -> T:
    """Instantiate a configuration dataclass from `data`, ignoring unknown keys.

    String values for enum-typed fields are promoted to the matching member (e.g.
    `StrEnum`), including optional fields.
    """
    hints = typing.get_type_hints(config_type)
    kwargs: dict[str, typing.Any] = {}
    for field in dataclasses.fields(config_type):
        if field.name not in data:
            continue

        value = data[field.name]
        enum_type = _enum_type(hints.get(field.name))

        if (
            enum_type is not None
            and value is not None
            and not isinstance(value, enum_type)
        ):
            value = enum_type(value)

        kwargs[field.name] = value
    return config_type(**kwargs)


def load(specs: collections.abc.Mapping[str, core.api.BotSpec] | None = None) -> Config:
    """Load and parse config.json into a typed `Config`.

    `specs` supplies each bot's concrete config type so per-bot configurations parse
    into the right dataclass; bots absent from it are skipped. A missing file and
    missing sections alike fall back to dataclass defaults.
    """
    if specs is None:
        specs = {}

    data: dict[str, typing.Any] = {}

    try:
        with _CONFIG_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exception:
        raise RuntimeError(f"Malformed JSON at {_CONFIG_PATH}") from exception

    return Config(
        app=_from_mapping(AppConfig, data.get("app", {})),
        bots={
            bot_id: _from_mapping(
                spec.bot_config_type, data.get("bots", {}).get(bot_id, {})
            )
            for bot_id, spec in specs.items()
        },
        presets={
            preset_id: _from_mapping(PresetConfig, preset)
            for preset_id, preset in data.get("presets", {}).items()
        },
    )


def _to_json(data: object) -> typing.Any:
    """Serialize a dataclass or enum `data` value for json.dump."""
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        return dataclasses.asdict(data)

    if isinstance(data, enum.Enum):
        return data.value

    raise TypeError(f"Cannot serialize {type(data).__name__}")


def save(config: Config) -> None:
    """Serialize `config` to config.json on disk."""
    with _CONFIG_PATH.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(config, fp, default=_to_json, indent=4)
        fp.write("\n")
