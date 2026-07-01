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
    relative_paths = [
        # core.api.ROOT_DIRECTORY / "presets.json",
        *sorted((core.api.ROOT_DIRECTORY / "resources" / "bots").glob("*.py")),
        *(
            relative_path
            for relative_path in sorted(
                (core.api.ROOT_DIRECTORY / "resources" / "templates").rglob("*")
            )
            if relative_path.is_file()
        ),
    ]
    manifest = {
        relative_path.relative_to(core.api.ROOT_DIRECTORY).as_posix(): hashlib.sha256(
            relative_path.read_bytes()
        ).hexdigest()
        for relative_path in relative_paths
    }
    with _MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(manifest, fp, indent=4, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {_MANIFEST_PATH.name}: {len(manifest)} files")
