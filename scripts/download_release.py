#!/usr/bin/env python3
"""Download a complete public GitHub release with Python stdlib and curl only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from urllib.parse import quote

DEFAULT_REPO = "EndlessChasing/mamba2-8b-e8w5"
MANIFEST_NAME = "release_manifest.json"
FORMAT = "MAMBA2_E8W5_RELEASE_V1"
BUFFER_BYTES = 8 << 20
MAX_METADATA_BYTES = 32 << 20


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(data)
    return digest.hexdigest()


def valid_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def valid_size(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer byte count")
    return value


def portable_path(name):
    """Require a canonical relative path; reserve .part names for this tool."""
    if not isinstance(name, str) or not name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError(f"Invalid release asset path: {name!r}")
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or ".." in path.parts
            or "\\" in name or ":" in name or name == "."
            or any(part.endswith((".part", ".", " ")) for part in path.parts)):
        raise ValueError(f"Unsafe or noncanonical release asset path: {name!r}")
    for part in path.parts:
        if part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
            raise ValueError(f"Nonportable release asset path: {name!r}")
    return path


def validate_manifest(raw, expected_sha256=None):
    """Validate paths, flat upload names, size ledger and container-part ledger."""
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Release manifest is too large")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != valid_digest(expected_sha256):
        raise ValueError("Release manifest differs from the independently trusted SHA-256")
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT or manifest.get("complete") is not True:
        raise ValueError("Incomplete or unsupported release manifest")
    entries, basenames, folded_paths = {}, {MANIFEST_NAME.casefold()}, set()
    parent_spellings = {}
    for entry in manifest["assets"]:
        name = entry["path"]
        path = portable_path(name)
        if name.casefold() in folded_paths or path.name.casefold() in basenames:
            raise ValueError(f"Duplicate path or flat GitHub asset basename: {name}")
        for parent in path.parents:
            spelling = parent.as_posix()
            previous = parent_spellings.setdefault(spelling.casefold(), spelling)
            if previous != spelling:
                raise ValueError(f"Case-ambiguous directory spelling: {name}")
        valid_size(entry["bytes"])
        valid_digest(entry["sha256"])
        entries[name] = entry
        folded_paths.add(name.casefold())
        basenames.add(path.name.casefold())
    if not entries:
        raise ValueError("Release has no assets")
    for name in entries:
        for parent in PurePosixPath(name).parents:
            if parent.as_posix().casefold() in folded_paths | {MANIFEST_NAME.casefold()}:
                raise ValueError(f"Asset is both a file and a parent directory: {name}")
    sizes = manifest["sizes"]
    total = sum(entry["bytes"] for entry in entries.values())
    if (valid_size(sizes["all_asset_bytes"]) != total
            or valid_size(sizes["manifest_bytes"]) != len(raw)
            or valid_size(sizes["total_release_bytes"]) != total + len(raw)):
        raise ValueError("Release byte ledger mismatch")
    container, position, seen = manifest["container"], 0, set()
    valid_digest(container["sha256"])
    if container.get("format") != "E8HUF001" or not container["parts"]:
        raise ValueError("Unsupported or empty weight container")
    for part in container["parts"]:
        name = part["path"]
        if name not in entries or name in seen or part["offset"] != position:
            raise ValueError("Container part is missing, repeated, or out of order")
        entry = entries[name]
        if part["bytes"] != entry["bytes"] or part["sha256"] != entry["sha256"]:
            raise ValueError("Container part differs from asset ledger")
        position += part["bytes"]
        seen.add(name)
    if position != valid_size(container["bytes"]):
        raise ValueError("Container size differs from parts")
    return manifest, entries, digest


def assert_no_symlink(path):
    """Reject existing symlinks in the requested destination and its parents."""
    path = Path(path).absolute()
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError(f"Refusing symlink traversal: {part}")
    return path


def prepare_directory(directory, names):
    directory = assert_no_symlink(directory)
    allowed_files = set(names) | {name + ".part" for name in names}
    allowed_dirs = {parent.as_posix() for name in allowed_files
                    for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
    if directory.exists():
        if not directory.is_dir():
            raise ValueError("Output is not a directory")
        for path in directory.rglob("*"):
            name = path.relative_to(directory).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Refusing release symlink: {name}")
            if stat.S_ISREG(mode) and name in allowed_files:
                if path.stat().st_nlink != 1:
                    raise ValueError(f"Refusing multiply linked release file: {name}")
                continue
            if stat.S_ISDIR(mode) and name in allowed_dirs:
                continue
            raise ValueError(f"Output contains an unexpected entry: {name}")
    else:
        directory.mkdir(parents=True)
    return directory


def curl_bytes(url, limit=MAX_METADATA_BYTES):
    """No curl config files, credentials, shell interpolation, or auth tokens."""
    with tempfile.TemporaryDirectory(prefix="mamba-release-metadata-") as tmp:
        target = Path(tmp) / "metadata"
        subprocess.run(["curl", "--disable", "--fail", "--location", "--silent", "--show-error",
                        "--proto", "=https", "--proto-redir", "=https", "--retry", "3",
                        "--connect-timeout", "20", "--max-time", "120", "--max-filesize", str(limit),
                        "--header", "Accept: application/vnd.github+json",
                        "--user-agent", "mamba2-e8w5-release-downloader", "--output", str(target),
                        "--url", url], check=True)
        if target.stat().st_size > limit:
            raise ValueError("Metadata response exceeds the size limit")
        return target.read_bytes()


def curl_payload(url, target, maximum_bytes):
    command = ["curl", "--disable", "--fail", "--location", "--silent", "--show-error",
               "--proto", "=https", "--proto-redir", "=https", "--retry", "3",
               "--connect-timeout", "20", "--max-filesize", str(maximum_bytes),
               "--continue-at", "-", "--output", str(target), "--url", url]
    subprocess.run(command, check=True)


def public_release_assets(repo, tag, fetch_bytes=curl_bytes):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Repository must be owner/name")
    if not tag or any(ord(c) < 32 for c in tag) or any(p in (".", "..") for p in tag.split("/")):
        raise ValueError("Invalid release tag")
    endpoint = f"https://api.github.com/repos/{repo}/releases"
    release = json.loads(fetch_bytes(endpoint + "/tags/" + quote(tag, safe="")))
    if release.get("tag_name") != tag or release.get("draft") or type(release.get("id")) is not int:
        raise ValueError("GitHub response is not the requested published release")
    assets = {}
    for page in range(1, 101):
        rows = json.loads(fetch_bytes(f"{endpoint}/{release['id']}/assets?per_page=100&page={page}"))
        if not isinstance(rows, list):
            raise ValueError("Invalid GitHub asset listing")
        for asset in rows:
            name = asset["name"]
            if name in assets:
                raise ValueError(f"Duplicate GitHub asset name: {name}")
            if asset.get("state") != "uploaded":
                raise ValueError(f"GitHub asset is not fully uploaded: {name}")
            valid_size(asset["size"])
            assets[name] = asset
        if len(rows) < 100:
            return assets
    raise ValueError("GitHub asset pagination exceeds the supported limit")


def download_url(repo, tag, basename):
    return f"https://github.com/{repo}/releases/download/{quote(tag, safe='')}/{quote(basename, safe='')}"


def download_asset(url, target, entry, fetch_payload=curl_payload):
    target = assert_no_symlink(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    assert_no_symlink(partial)
    if target.exists():
        if target.stat().st_size != entry["bytes"] or digest_file(target) != entry["sha256"]:
            raise ValueError(f"Existing final file does not match this release: {target}; move it out before retrying")
        if partial.exists():
            partial.unlink()
        return "already verified"
    for attempt in range(2):
        if partial.exists() and partial.stat().st_size > entry["bytes"]:
            partial.unlink()
        if not partial.exists() or partial.stat().st_size < entry["bytes"]:
            if entry["bytes"] == 0:
                partial.touch()
            else:
                try:
                    fetch_payload(url, partial, entry["bytes"])
                except subprocess.CalledProcessError as error:
                    # A server refusing a Range request permits one fresh retry.
                    if error.returncode == 33 and partial.exists() and attempt == 0:
                        partial.unlink()
                        continue
                    raise
        assert_no_symlink(partial)
        if partial.stat().st_size == entry["bytes"] and digest_file(partial) == entry["sha256"]:
            os.replace(partial, target)
            return "downloaded and verified"
        partial.unlink()  # Only this tool's incomplete download is discarded.
    raise ValueError(f"Downloaded bytes fail SHA/size verification: {target.name}")


def verify_directory(directory, manifest, entries, raw_manifest):
    """Recheck exact file inventory, all bytes, and concatenated container SHA."""
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink appeared during download")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(entries) | {MANIFEST_NAME}:
        raise ValueError("Download directory contains missing, extra, or partial files")
    if (directory / MANIFEST_NAME).read_bytes() != raw_manifest:
        raise ValueError("Manifest changed during download")
    for name, entry in entries.items():
        path = directory / name
        if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
            raise ValueError(f"Final asset identity mismatch: {name}")
    digest = hashlib.sha256()
    for part in manifest["container"]["parts"]:
        with (directory / part["path"]).open("rb") as stream:
            for data in iter(lambda: stream.read(BUFFER_BYTES), b""):
                digest.update(data)
    if digest.hexdigest() != manifest["container"]["sha256"]:
        raise ValueError("Concatenated weight container SHA mismatch")


def download_release(repo, tag, output, manifest_sha256=None, fetch_bytes=curl_bytes,
                     fetch_payload=curl_payload):
    assets = public_release_assets(repo, tag, fetch_bytes)
    if MANIFEST_NAME not in assets:
        raise ValueError("Release has no release_manifest.json asset")
    size = assets[MANIFEST_NAME]["size"]
    if not 0 < size <= MAX_METADATA_BYTES:
        raise ValueError("Release manifest size exceeds the metadata limit")
    raw = fetch_bytes(download_url(repo, tag, MANIFEST_NAME))
    if len(raw) != size:
        raise ValueError("Manifest size differs from GitHub asset metadata")
    manifest, entries, digest = validate_manifest(raw, manifest_sha256)
    for name, entry in entries.items():
        basename = PurePosixPath(name).name
        if basename not in assets or assets[basename]["size"] != entry["bytes"]:
            raise ValueError(f"Missing GitHub asset or published size mismatch: {basename}")
    directory = prepare_directory(output, set(entries) | {MANIFEST_NAME})
    target = directory / MANIFEST_NAME
    if target.exists() and target.read_bytes() != raw:
        raise ValueError("Output already belongs to a different release manifest; choose another directory")
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(raw)
    os.replace(partial, target)
    for number, (name, entry) in enumerate(entries.items(), 1):
        print(f"[{number}/{len(entries)}] {name}", flush=True)
        result = download_asset(download_url(repo, tag, PurePosixPath(name).name),
                                directory / name, entry, fetch_payload)
        print("  " + result, flush=True)
    verify_directory(directory, manifest, entries, raw)
    return {"complete": True, "repository": repo, "tag": tag,
            "output": str(directory), "manifest_sha256": digest,
            "manifest_authenticity_checked": bool(manifest_sha256),
            "assets_verified": len(entries), "total_release_bytes": manifest["sizes"]["total_release_bytes"],
            "container_sha256": manifest["container"]["sha256"],
            "note": "Verified archive bytes; no Torch installation or model-quality claim. No receipt file was added to the release directory."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", help="Optional independently trusted release manifest SHA-256")
    args = parser.parse_args()
    try:
        receipt = download_release(args.repo, args.tag, args.output, args.manifest_sha256)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Download failed: {error}\nResume with the same command after resolving the reported issue.\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
