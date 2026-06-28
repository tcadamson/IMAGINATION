"""Program entry point for IMAGINATION."""

import collections.abc
import ctypes
import dataclasses
import inspect
import logging
import shlex
import traceback
import typing

import typer
import typer.core

import core.api
import core.config
import core.registry
import core.runtime

_BANNER: typing.Final = r"""

6666666666666666666666666
6666666666666666666666666
666666   666666   6666666   IMAGINATION
666666   666666   6666666
666666   666666   6666666   Automation framework for the SMT: IMAGINE client.
6666666666666666666666666   Bots for rebirth, demon force, and more.
666   66        66   6666
666    6        6    6666   Version: v2.0.0-beta
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

_group: typer.core.TyperGroup | None = None


@_cli_run.callback()
def _cli_run_callback(
    ctx: typer.Context,
    confidence: typing.Annotated[
        float,
        typer.Option(
            help="Required confidence for a template to match.", min=0.7, max=1.0
        ),
    ] = core.api.DEFAULT_CONFIDENCE,
    sleep: typing.Annotated[
        float,
        typer.Option(
            help="Amount of time to sleep after each mouse action, in seconds.",
            min=0.04,
        ),
    ] = core.api.DEFAULT_SLEEP,
) -> None:
    """Run one of the installed bots."""
    ctx.obj = core.api.RunConfig(confidence=confidence, sleep=sleep)


@_cli.callback(help=None)
def _cli_callback() -> None:
    """IMAGINATION interactive console."""


@_cli.command()
def update() -> None:
    """Apply available updates to installed bots, and install any new bots.

    Automatically runs once on program launch.
    """
    _cli_run.registered_commands = []

    print("Checking for updates...")
    stale_bot_ids = core.registry.sync()
    print("Done.")

    for spec in core.registry.register_bot_directory(
        core.config.BOT_DIRECTORY, stale_bot_ids
    ).values():
        _cli_run.command(name=spec.bot_id, help=spec.help)(_generate_command(spec))

    global _group
    _group = typer.main.get_group(_cli)


def _error_dialog(exception: BaseException) -> None:
    """Surface `exception` as a dialog above the active client."""
    ctypes.windll.user32.MessageBoxW(
        None,
        "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        ),
        "IMAGINATION",
        0x10 | 0x00001000,  # MB_ICONERROR | MB_SYSTEMMODAL
    )


def _on_event_override(label: str, event: core.runtime.Event) -> None:
    """Log every event, and alert the user via dialog on a crash."""
    core.runtime.log_event(label, event)

    if isinstance(event, core.runtime.CrashedEvent):
        _error_dialog(event.exception)


def _launch(
    spec: core.api.BotSpec,
    bot_config: core.api.BotConfig,
    run_config: core.api.RunConfig,
) -> None:
    """Bind `spec` to the located client and run its workflow to completion."""
    clients = core.api.Client.locate_all()
    scheduler = core.runtime.Scheduler.from_binds(
        (core.runtime.Bind(clients[0], spec, bot_config),),
        run_config,
        on_event=_on_event_override,
    )
    scheduler.run()


def _generate_command(spec: core.api.BotSpec):
    """Build a typer command exposing config fields in `spec` as CLI options."""
    params = [
        inspect.Parameter(
            "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context
        )
    ]
    for field in dataclasses.fields(spec.bot_config_type):
        default = field.default if field.default is not dataclasses.MISSING else ...
        params.append(
            inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.type,
            )
        )

    def command(ctx: typer.Context, **kwargs) -> None:
        _launch(spec, spec.bot_config_type(**kwargs), ctx.obj)

    command.__signature__ = inspect.Signature(params)
    return command


def _guard(callback: typing.Callable[[], None]) -> None:
    """Run `callback`, surfacing any crash as an error dialog."""
    try:
        callback()
    except SystemExit:  # Raised by typer internals
        pass
    except Exception as exception:
        logging.exception("Fatal exception:")
        _error_dialog(exception)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                core.api.ROOT_DIRECTORY / "debug.log", encoding="utf-8"
            ),
        ],
    )
    _cli.add_typer(_cli_run, name="run")
    _guard(update)
    group = typing.cast(
        typer.core.TyperGroup, _group
    )  # Make type narrowing visible to lambda scopes by assigning to new local
    print(_BANNER)
    _guard(
        lambda: group.main(args=("--help",), prog_name="")
    )  # Print help message on initial launch

    while True:
        print()

        try:
            response = input(">>> ").strip()
        except EOFError, KeyboardInterrupt:
            break

        if response == "":
            continue

        _guard(lambda: group.main(args=shlex.split(response), prog_name=""))
