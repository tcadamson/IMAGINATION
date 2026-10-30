"""Program entry point for IMAGINATION."""

import collections.abc
import ctypes
import logging
import shlex
import traceback
import typing

import typer
import typer.core

import core.output
import core.paths
import core.presets
import core.registry

_BANNER: typing.Final = r"""

6666666666666666666666666
6666666666666666666666666
666666   666666   6666666   IMAGINATION
666666   666666   6666666
666666   666666   6666666   Automation framework for the SMT: IMAGINE client.
6666666666666666666666666   Bots for rebirth, demon force, and more.
666   66        66   6666
666    6        6    6666   Version: v2.0.0-beta.5
6666                66666   Repo: https://github.com/tcadamson/IMAGINATION
6666666666666666666666666
6666666666666666666666666

"""
_CONTEXT_SETTINGS: typing.Final[collections.abc.Mapping[str, typing.Any]] = {
    "max_content_width": 120,
    "token_normalize_func": lambda token: token.lower(),  # Case-insensitive commands
}


class _UsageLineOverride(typer.core.TyperGroup):
    """Custom --help formatting to omit the leading empty program name."""

    def format_usage(self, ctx, formatter):
        pieces = self.collect_usage_pieces(ctx)

        if ctx.command_path:
            formatter.write_usage(ctx.command_path, " ".join(pieces))
        else:
            formatter.write_usage("", " ".join(pieces), prefix="Usage:")


_cli = typer.Typer(
    cls=_UsageLineOverride,
    context_settings=_CONTEXT_SETTINGS,
    rich_markup_mode=None,
    no_args_is_help=True,
)
_cli_run = typer.Typer(no_args_is_help=True)
_cli_presets = typer.Typer(no_args_is_help=True)

_preset_names: set[str] = set()

_group: typer.core.TyperGroup | None = None

_logger: logging.Logger = logging.getLogger(__name__)


@_cli_run.callback()
def _cli_run_callback() -> None:
    """Run one of the installed bots."""


@_cli_presets.callback()
def _cli_presets_callback() -> None:
    """Manage command presets.

    To edit existing presets and/or add your own, edit presets.json at
    %LOCALAPPDATA%\\IMAGINATION\\resources
    """


@_cli_presets.command(name="list")
def _cli_presets_list() -> None:
    """List available presets."""
    presets = core.presets.load_presets()

    if not presets:
        print(f"No presets defined in {core.presets.PRESETS_PATH}")
        return

    for preset_id, line in presets.items():
        print(f"{preset_id.lower()}: {line}")


@_cli_presets.command(name="load")
def _cli_presets_load() -> None:
    """Load any presets on disk and register them as commands."""
    presets = core.presets.load_presets()

    # Drop previously registered presets to reflect recent edits
    _cli.registered_commands = [
        command
        for command in _cli.registered_commands
        if command.name not in _preset_names
    ]
    _preset_names.clear()

    reserved = set(name.lower() for name in typer.main.get_group(_cli).commands)
    for preset_id, line in presets.items():
        preset_id = preset_id.lower()

        if preset_id in reserved:
            print(f"Skipping preset with reserved name: {preset_id}")
            continue

        _cli.command(name=preset_id, help=line, hidden=True)(
            _generate_preset_command(line)
        )
        _preset_names.add(preset_id)
        reserved.add(preset_id)

    global _group
    _group = typer.main.get_group(_cli)

    _logger.info(f"Loaded {len(_preset_names)} preset(s)")


@_cli.callback(help=None)
def _cli_callback() -> None:
    """IMAGINATION interactive console."""


@_cli.command()
def update() -> None:
    """Apply available updates to installed bots, and install any new bots.

    Automatically runs once on program launch.
    """
    print("Checking for updates...")

    if not core.paths.USER_DIRECTORY_OVERRIDE_PASSED:
        core.paths.migrate()
        core.presets.ensure_presets()

        try:
            core.registry.sync()
        except Exception as exception:  # Update attempt is best-effort
            _logger.exception("Update failed:")
            _error_dialog(exception)

    print("Done.")

    global _group
    _group = typer.main.get_group(_cli)


def _error_dialog(exception: BaseException) -> None:
    """Surface `exception` as a dialog above the active client."""
    ctypes.windll.user32.MessageBoxW(
        None,
        "".join(traceback.format_exception_only(type(exception), exception)),
        "IMAGINATION",
        0x10 | 0x00001000,  # MB_ICONERROR | MB_SYSTEMMODAL
    )


def _dispatch(args: collections.abc.Sequence[str]) -> None:
    """Dispatch console `args` through the active command group."""
    typing.cast(typer.core.TyperGroup, _group).main(args=args, prog_name="")


def _generate_preset_command(line: str):
    """Build a typer command that populates the command line with a saved preset."""

    def command() -> None:
        _dispatch(shlex.split(line))

    return command


def _guard(callback: collections.abc.Callable[[], None]) -> None:
    """Run `callback`, surfacing any crash as an error dialog."""
    try:
        callback()
    except SystemExit:  # Raised by typer internals
        pass
    except Exception as exception:
        _logger.exception("Fatal exception:")
        _error_dialog(exception)


if __name__ == "__main__":
    core.output.setup()
    _cli.add_typer(_cli_presets, name="presets")
    _cli.add_typer(_cli_run, name="run")
    _guard(update)
    _guard(_cli_presets_load)
    print(_BANNER)
    _guard(lambda: _dispatch(("--help",)))

    while True:
        print()

        try:
            response = input(">>> ").strip()
        except EOFError, KeyboardInterrupt:
            break

        if response == "":
            continue

        _guard(lambda: _dispatch(shlex.split(response)))
