"""Zip the built program as IMAGINATION.zip for a release, and point release.json at it.

Lives in the repository, not the client. Run after `scripts.build`, naming the branch the
build was made for, which is checked: `uv run python -m scripts.release --branch beta`.
Then upload build\\IMAGINATION.zip to the release tagged with the version, commit
release.json, and fast-forward the update branch: programs taking updates from it update
themselves on their next start.

The folder in the zip is IMAGINATION, not named by version: after updates, it holds
whatever version is installed.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import typing
import zipfile
import zipimport

import core
import core.api

_NAME: typing.Final = "IMAGINATION"
_RELEASE_PATH: typing.Final = core.api.ROOT_DIRECTORY / "resources" / "release.json"
_DOWNLOAD_URL: typing.Final = (
    "https://github.com/tcadamson/IMAGINATION/releases/download/{tag}/{name}"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--branch", choices=("stable", "beta"), required=True)
    arguments = parser.parse_args()

    build_name = f"{_NAME}_{core.version_text()}"
    build_directory = core.api.ROOT_DIRECTORY / "build" / build_name

    if not (build_directory / f"{_NAME}.exe").is_file():
        sys.exit(f"Can't release: no build at {build_directory}; run scripts.build")

    spec = zipimport.zipimporter(
        str(build_directory / "lib" / "library.zip")
    ).find_spec("BUILD_CONSTANTS")
    build_constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_constants)

    if build_constants.BRANCH != arguments.branch:
        sys.exit(
            f"Can't release: {build_name} was built for {build_constants.BRANCH}, "
            f"not {arguments.branch}"
        )

    zip_path = build_directory.parent / f"{_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(build_directory.rglob("*")):
            if path.is_file():
                zip_file.write(
                    path, f"{_NAME}/{path.relative_to(build_directory).as_posix()}"
                )

    release = {
        "version": core.VERSION,
        "url": _DOWNLOAD_URL.format(tag=core.version_text(), name=zip_path.name),
        "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }
    with _RELEASE_PATH.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(release, fp, indent=4)
        fp.write("\n")
    print(
        f"Wrote build\\{zip_path.name} and {_RELEASE_PATH.name}; upload the zip to "
        f"release {core.version_text()}"
    )
