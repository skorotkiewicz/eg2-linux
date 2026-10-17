#!/usr/bin/env python3
"""Download the app's v5 model set using curl; verify size and SHA-256."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

BASE = "https://haku-models.tslfiles.org/eg2demo/"
ROOT = Path(__file__).resolve().parent / "models"


def checked_path(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or "\\" in value
    ):
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return path


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]


def fetch(url, target):
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            "10",
            "--retry",
            "3",
            "--output",
            str(target),
            url,
        ],
        check=True,
    )


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = ROOT / "manifest.json.part"
    if manifest.is_symlink():
        raise ValueError(f"Refusing symlink: {manifest}")
    fetch(BASE + "v5/manifest.json", manifest)
    data = json.loads(manifest.read_text())
    if data.get("name") != "eg2demo" or data.get("version") != 5:
        raise ValueError("Unexpected model manifest")
    entries = data["files"]
    seen = set()
    for entry in entries:
        path = checked_path(entry["path"])
        url = checked_path(entry["url"])
        if path in seen or not re.fullmatch(r"v[1-5]/files/.+", str(url)):
            raise ValueError("Duplicate path or unexpected download URL")
        if type(entry["size"]) is not int or entry["size"] < 0:
            raise ValueError("Invalid file size")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid SHA-256")
        seen.add(path)
        target = ROOT / path
        if not target.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Model path escapes download directory")
    manifest.replace(ROOT / "manifest.json")
    for entry in entries:
        target = ROOT / entry["path"]
        if valid(target, entry):
            print(f"Verified {entry['path']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        if partial.is_symlink():
            raise ValueError(f"Refusing symlink: {partial}")
        print(f"Downloading {entry['path']} ({entry['size']:,} bytes)", flush=True)
        fetch(BASE + entry["url"], partial)
        if not valid(partial, entry):
            raise ValueError(f"Size or SHA-256 mismatch: {entry['path']}")
        partial.replace(target)


if __name__ == "__main__":
    main()
