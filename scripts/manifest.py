"""Regenerate manifest.json from the repository tree.

Lives in the repository, not the client. Run after editing any resource, commit,
and fast-forward the stable branch to go live.

Entries downloaded from a URL (the bot bundle) are not files of the tree; they are kept
as they are, and changed by the release that uploads them.

Run with `uv run python -m scripts.manifest`.
"""

import hashlib
import json
import typing

import core.api
import core.output

_MANIFEST_PATH: typing.Final = core.api.ROOT_DIRECTORY / "resources" / "manifest.json"

if __name__ == "__main__":
    paths = [
        core.api.ROOT_DIRECTORY / "resources" / "presets.default.json",
        *(
            path
            for path in (core.api.ROOT_DIRECTORY / "resources" / "bots.json",)
            if path.is_file()
        ),
    ]
    manifest: dict[str, object] = {
        path.relative_to(core.api.ROOT_DIRECTORY).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    manifest |= {
        relative_path: entry
        for relative_path, entry in json.loads(
            _MANIFEST_PATH.read_text(encoding="utf-8")
        ).items()
        if isinstance(entry, dict)
    }
    with _MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(manifest, fp, indent=4, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {_MANIFEST_PATH.name}: {core.output.count(len(manifest), 'file')}")
