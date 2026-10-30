"""Core of IMAGINATION: the client window, its updates, and its output."""

import importlib
import re
import typing

VERSION: typing.Final = "2.0.0b5"  # Equal to pyproject.toml's, which the build checks

_PRE_RELEASE_NAMES: typing.Final = {"a": "alpha", "b": "beta", "rc": "rc"}


def _branch() -> str:
    """The branch updates come from: set by the build, stable when run from source."""
    try:
        build_constants = importlib.import_module("BUILD_CONSTANTS")  # By cx_Freeze
    except ModuleNotFoundError:
        return "stable"

    return build_constants.BRANCH


BRANCH: typing.Final = _branch()


def version_text(version: str = VERSION) -> str:
    """Return `version` as shown to the user: 2.0.0b5 is v2.0.0-beta.5."""
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?", version)

    if match is None:
        raise ValueError(f"not a version: {version}")

    release, phase, number = match.groups()
    return f"v{release}" + (f"-{_PRE_RELEASE_NAMES[phase]}.{number}" if phase else "")
