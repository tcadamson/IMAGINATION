"""Regenerate manifest.json from the repository tree.

Lives in the repository, not the client. Run after editing any bot or template, commit,
and fast-forward the stable branch to go live.
"""

import hashlib
import json
import typing

import api
import config

MANIFEST_PATH: typing.Final = api.ROOT_DIRECTORY / "manifest.json"

if __name__ == "__main__":
    paths = [
        *sorted((config.BOT_DIRECTORY).glob("*.py")),
        *(
            path
            for path in sorted((config.TEMPLATE_DIRECTORY).rglob("*"))
            if path.is_file()
        ),
    ]
    manifest = {
        path.relative_to(api.ROOT_DIRECTORY).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=4, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {MANIFEST_PATH.name}: {len(manifest)} files")
