"""Build and export module for IMAGINATION.

Configuration for cx_Freeze is done here instead of `pyproject.toml` because cx_Freeze
struggles to locate the python DLL for uv python installations. The include path needs
to be determined via `sys.base_prefix`.

As added convenience, the project name and version may be retrieved dynamically.

Run with `uv run python build.py build`.
"""

import pathlib
import shutil
import sys
import tomllib

import cx_Freeze

# Specifying bin_path_includes voids the default bin_excludes; restore them
# https://github.com/marcelotduarte/cx_Freeze/blob/29f21da72e64e21cacd47f47206edc82f3e49e94/cx_Freeze/freezer.py#L1094
DEFAULT_BIN_EXCLUDES: tuple[str] = (
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

with open(pathlib.Path(__file__).parent / "pyproject.toml", "rb") as bytes:
    project = tomllib.load(bytes)["project"]

PROJECT_NAME: str = project["name"]
PROJECT_VERSION: str = project["version"]

BUILD_DIR: pathlib.Path = pathlib.Path("build") / f"{PROJECT_NAME}_v{PROJECT_VERSION}"

cx_Freeze.setup(
    executables=[
        cx_Freeze.Executable("main.py", target_name=PROJECT_NAME, icon=PROJECT_NAME)
    ],
    options={
        "build_exe": {
            "build_exe": str(BUILD_DIR),
            "bin_excludes": [*DEFAULT_BIN_EXCLUDES],
            "bin_path_includes": [sys.base_prefix],
            "include_files": ["templates", "bots"],
        }
    },
)

for egg_info in pathlib.Path(__file__).parent.glob("*.egg-info"):
    shutil.rmtree(egg_info, ignore_errors=True)

frozen_application_license = BUILD_DIR / "frozen_application_license.txt"

if frozen_application_license.exists():
    frozen_application_license.unlink()
