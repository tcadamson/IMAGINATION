"""Regenerate manifest.json from the repository tree.

Lives in the repository, not the client. Run after editing any bot or template, commit,
and fast-forward the stable branch to go live.

Run with `uv run python -m scripts.manifest`.
"""

import hashlib
import json
import typing

import core.api

_MANIFEST_PATH: typing.Final = core.api.ROOT_DIRECTORY / "resources" / "manifest.json"

if __name__ == "__main__":
    paths = [
        core.api.ROOT_DIRECTORY / "resources" / "presets.default.json",
        *sorted((core.api.ROOT_DIRECTORY / "resources" / "bots").glob("*.py")),
        *(
            path
            for path in sorted(
                (core.api.ROOT_DIRECTORY / "resources" / "templates").rglob("*")
            )
            if path.is_file()
        ),
    ]
    manifest = {
        path.relative_to(core.api.ROOT_DIRECTORY).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    with _MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(manifest, fp, indent=4, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {_MANIFEST_PATH.name}: {len(manifest)} files")
