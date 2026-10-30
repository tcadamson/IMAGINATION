"""Build and export module for IMAGINATION.

Configuration for cx_Freeze is done here instead of `pyproject.toml` because cx_Freeze
struggles to locate the python DLL for uv python installations. The include path needs
to be determined via `sys.base_prefix`.

As added convenience, the project name and version may be retrieved dynamically.

--branch names the branch the build takes its updates from: beta for builds given to
testers, stable for releases.

Run with `uv run python -m scripts.build build --branch stable`.
"""

import argparse
import collections.abc
import pathlib
import shutil
import sys
import tomllib
import typing

import cx_Freeze

import core

# Specifying bin_path_includes voids the default bin_excludes; restore them
# https://github.com/marcelotduarte/cx_Freeze/blob/29f21da72e64e21cacd47f47206edc82f3e49e94/cx_Freeze/freezer.py#L1094
_DEFAULT_BIN_EXCLUDES: typing.Final = (
    "comctl32.dll",
    "oci.dll",
    "concrt140.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcamp140.dll",
    "vccorlib140.dll",
    "vcomp140.dll",
    "vcruntime140.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
    "api-ms-win-*.dll",
    "ucrtbase.dll",
)

# The frozen program holds only the modules its own code imports. The bot bundle, loaded
# by a worker at run time, imports these; another needs a new release
_BUNDLE_MODULES: typing.Final = (
    "base64",
    "collections",
    "contextlib",
    "ctypes",
    "ctypes.wintypes",
    "dataclasses",
    "functools",
    "hashlib",
    "json",
    "logging",
    "math",
    "os",
    "re",
    "struct",
    "time",
    "zlib",
)

_parser = argparse.ArgumentParser(allow_abbrev=False)
_parser.add_argument("--branch", choices=("stable", "beta"), required=True)
_arguments, sys.argv[1:] = _parser.parse_known_args()  # The rest are cx_Freeze's

with open(pathlib.Path(__file__).parents[1] / "pyproject.toml", "rb") as fp:
    PROJECT: typing.Final[collections.abc.Mapping[str, typing.Any]] = tomllib.load(fp)[
        "project"
    ]

PROJECT_NAME: typing.Final = PROJECT["name"]
PROJECT_VERSION: typing.Final = PROJECT["version"]

if PROJECT_VERSION != core.VERSION:
    sys.exit(
        f"Can't build: pyproject.toml's version {PROJECT_VERSION} differs from "
        f"core.VERSION {core.VERSION}; bump both"
    )

BUILD_DIRECTORY: typing.Final = (
    pathlib.Path("build") / f"{PROJECT_NAME}_{core.version_text(PROJECT_VERSION)}"
)

cx_Freeze.setup(
    executables=[
        cx_Freeze.Executable("main.py", target_name=PROJECT_NAME, icon=PROJECT_NAME)
    ],
    options={
        "build_exe": {
            "build_exe": str(BUILD_DIRECTORY),
            "include_files": ["LICENSE"],
            "includes": [*_BUNDLE_MODULES],
            "constants": [f"BRANCH={_arguments.branch!r}"],
            "bin_excludes": [*_DEFAULT_BIN_EXCLUDES],
            "bin_path_includes": [sys.base_prefix],
        }
    },
)

for egg_info in pathlib.Path(__file__).parents[1].glob("*.egg-info"):
    shutil.rmtree(egg_info, ignore_errors=True)

frozen_application_license = BUILD_DIRECTORY / "frozen_application_license.txt"

if frozen_application_license.exists():
    frozen_application_license.unlink()
