#!/usr/bin/env python3
"""Build, verify, and restore a complete E8/W5 release without changing raw files.

Build performs a trusted raw-package Huffman readback. Release verification is
bounded in RAM; split restoration temporarily assembles one seekable container.
This tool records supplied quality measurements and never infers a quality pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.huffman import Reader, pack_directory, sha256_file, verify_container
from mamba_e8w5.runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256

FORMAT = "MAMBA2_E8W5_RELEASE_V1"
MANIFEST_NAME = "release_manifest.json"
CONTAINER_NAME = "mamba2-8b-e8w5.huff"
DEFAULT_PART_BYTES = 1_500_000_000
BUFFER_BYTES = 8 << 20
TOKENIZER_NAME = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
TOKENIZER_SHA256 = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"


def serialize(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path, value):
    Path(path).write_bytes(serialize(value))


def safe_path(directory, name):
    """Manifest names are portable relative paths, never absolute/symlink paths."""
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe release asset path: {name!r}")
    path = Path(directory).joinpath(*relative.parts)
    if path.resolve() != Path(directory).resolve().joinpath(*relative.parts):
        raise ValueError(f"Symlink is not a release asset: {name}")
    return path


def record_asset(directory, path, role):
    path = Path(path)
    return {"path": path.relative_to(directory).as_posix(), "role": role,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git_value(*arguments):
    result = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def software_snapshot(path):
    """Ship matching small source files, including dirty changes, not Git history."""
    candidates = set()
    for name in ("mamba_e8w5", "scripts", "tests", "docs", "licenses", "third_party"):
        directory = ROOT / name
        if directory.exists():
            candidates.update(p for p in directory.rglob("*") if p.is_file())
    candidates.update(p for p in ROOT.iterdir() if p.is_file())
    suffixes = {".py", ".cpp", ".c", ".h", ".hpp", ".md", ".toml", ".txt", ".json", ".sh", ".yml", ".yaml"}
    files = []
    for source in sorted(candidates):
        relative = source.relative_to(ROOT)
        if source.is_symlink() or any(p.startswith(".") or p == "__pycache__" for p in relative.parts):
            continue
        if source.suffix not in suffixes and source.name not in ("LICENSE", "COPYING", "NOTICE"):
            continue
        if source.stat().st_size > 16 << 20:
            raise ValueError(f"Unexpected large source snapshot file: {relative}")
        files.append((relative.as_posix(), source.read_bytes()))
    index = {"git_commit": git_value("rev-parse", "HEAD"),
             "git_worktree_dirty": bool(git_value("status", "--porcelain")),
             "scope": "Exact corresponding source bytes, including uncommitted working files.",
             "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                       for name, data in files]}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in [*files, ("SOURCE_FILES.json", serialize(index))]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return index


def validate_raw(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("complete") or not manifest.get("files"):
        raise ValueError("Raw package must be complete and have a file SHA ledger")
    if (manifest.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256 or
            manifest.get("tokenizer_sha256") != TOKENIZER_SHA256 or
            manifest.get("model_config") != MODEL_CONFIG or
            manifest.get("total_parameter_count") != 8236999680):
        raise ValueError("Raw package identity/configuration differs from the pinned 8B model")
    wanted = set(manifest["files"]) | {"manifest.json"}
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if actual != wanted or any(p.is_dir() or p.is_symlink() for p in directory.iterdir()):
        raise ValueError(f"Raw package inventory differs from its completed manifest: {actual ^ wanted}")
    for name, entry in manifest["files"].items():
        path = safe_path(directory, name)
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Raw package file identity mismatch: {name}")
    if json.loads((directory / "config.json").read_text()) != manifest["model_config"]:
        raise ValueError("Raw config.json differs from the quantization manifest")
    return manifest


def split_container(path, part_bytes):
    """Create bounded parts, remove only this tool's verified generated container."""
    if not 0 < part_bytes < 2 ** 31:
        raise ValueError("Part size must be positive and below 2 GiB")
    count = math.ceil(path.stat().st_size / part_bytes)
    parts = []
    total_digest = hashlib.sha256()
    position = 0
    with path.open("rb") as source:
        for index in range(count):
            target = path.with_name(f"{path.name}.part-{index + 1:05d}-of-{count:05d}")
            written = 0
            with target.open("xb") as destination:
                while written < part_bytes:
                    data = source.read(min(BUFFER_BYTES, part_bytes - written))
                    if not data:
                        break
                    destination.write(data)
                    total_digest.update(data)
                    written += len(data)
            if not written:
                raise ValueError("Unexpected empty container part")
            parts.append({"path": target.name, "offset": position, "bytes": written})
            position += written
        if source.read(1):
            raise ValueError("Split did not consume the full container")
    if position != path.stat().st_size or total_digest.hexdigest() != sha256_file(path):
        raise ValueError("Split identity mismatch")
    path.unlink()
    return parts


def finalize_manifest(directory, manifest):
    """Count the manifest itself with a convergent byte-length fixed point."""
    asset_bytes = sum(entry["bytes"] for entry in manifest["assets"])
    manifest["sizes"]["all_asset_bytes"] = asset_bytes
    manifest["sizes"]["manifest_bytes"] = 0
    manifest["sizes"]["total_release_bytes"] = asset_bytes
    for _ in range(16):
        raw = serialize(manifest)
        if len(raw) == manifest["sizes"]["manifest_bytes"]:
            (directory / MANIFEST_NAME).write_bytes(raw)
            return
        manifest["sizes"]["manifest_bytes"] = len(raw)
        manifest["sizes"]["total_release_bytes"] = asset_bytes + len(raw)
    raise RuntimeError("Manifest byte accounting failed to converge")


def make_model_card(raw, quality_paths):
    quality = ("Supplied measurement reports: " + ", ".join(f"[{p}]({p})" for p in quality_paths)
               if quality_paths else "No quality report was supplied. PPL and recall remain unmeasured in this release.")
    return f"""# Mamba-2 8B E8/W5 — experimental quantized release

This is an independent modified version of NVIDIA's pure Mamba-2 8B base model.
Projection weights use E8P12 with calibrated LDLQ and rotations; the independent
input embedding and output head use group-128 W5. Smaller tensors are FP16.
Huffman coding is lossless relative to these quantized files. Quantization is
lossy relative to the original floating-point checkpoint.

## Quality and scope

{quality}

The packaging tool does not assign a quality pass, claim equal-quality accuracy,
or certify a globally smallest model. Inspect the supplied protocol, exact
measurements, source audit, and all file counts. No quality value is inferred
from codec integrity tests. Raw manifest parameter count:
`{raw.get('total_parameter_count', 'not recorded')}`.

## Complete storage accounting

`release_manifest.json` records the real container bytes, every required part,
tokenizer, configuration, code, license, provenance, and quality-report byte.
It also includes its own metadata length in total release bytes. Both codebooks
and all weight scales/signs/offsets are inside the container. No external weight
base or adapter is required. The source checkpoint is needed to reproduce
quantization, not to restore or load this release.

## Loading

Extract `source.zip` into a software checkout. Follow `docs/PACKAGING.md` there.
Run the `restore` subcommand to verify all assets and reconstruct exact raw files.
The public reference runtime decodes weights into FP16; its memory use is
different from the compressed archive size. An online Huffman GPU inference
kernel is not part of this release.

## Attribution and licenses

NVIDIA model/tokenizer provenance and Apache-2.0 declaration are retained in
`licenses/NVIDIA_MAMBA2_MODEL_CARD.md` and `licenses/APACHE-2.0.txt`.
The E8/LDLQ implementation is derived from GPL-3.0 QuIP#, whose license and
attribution are included. `source.zip` contains the matching public source,
vendored primitive pin, and a SHA ledger. See `licenses/THIRD_PARTY.md` for the
separate model and software scopes. There is no NVIDIA or QuIP# endorsement.
"""


def build_release(args):
    raw_dir, destination = args.raw_dir.resolve(), args.output.resolve()
    if destination == raw_dir or raw_dir in destination.parents:
        raise ValueError("Release output must be outside the immutable raw package")
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Inspect previous incomplete build before retrying: {staging}")
    raw = validate_raw(raw_dir)
    raw_manifest_sha = sha256_file(raw_dir / "manifest.json")
    tokenizer = args.tokenizer or (args.source_dir / TOKENIZER_NAME if args.source_dir else None)
    if tokenizer is None:
        raise ValueError("Pass --source-dir or --tokenizer")
    tokenizer = tokenizer.resolve()
    if sha256_file(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the pinned NVIDIA tokenizer")
    if not (ROOT / "licenses").is_dir():
        raise FileNotFoundError("Repository license snapshots are missing")
    defaults = [ROOT / "reports/source_model.json", ROOT / "reports/source_download.json"]
    supplied = [(path, "quality") for path in args.quality_report]
    supplied += [(path, "provenance") for path in (args.provenance_report or [p for p in defaults if p.exists()])]
    for path, role in supplied:
        report = json.loads(path.read_text())
        if role == "quality":
            if report.get("complete") is False:
                raise ValueError(f"Incomplete quality report: {path}")
            for key in ("package_receipt", "candidate_package"):
                measured_sha = report.get(key, {}).get("manifest_sha256")
                if measured_sha and measured_sha != raw_manifest_sha:
                    raise ValueError(f"Quality report belongs to a different package: {path}")
            measured_source = report.get("source_checkpoint_sha256")
            if measured_source and measured_source != SOURCE_CHECKPOINT_SHA256:
                raise ValueError(f"Quality report uses a different source model: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    container = staging / CONTAINER_NAME
    pack_directory(raw_dir, container)
    receipt = verify_container(container, trusted_directory=raw_dir)
    if sha256_file(raw_dir / "manifest.json") != raw_manifest_sha:
        raise ValueError("Raw manifest changed during packaging")
    receipt["container"] = CONTAINER_NAME
    write_json(staging / "container_verification.json", receipt)
    weight_parts = (split_container(container, args.split_bytes) if args.split_bytes
                    else [{"path": CONTAINER_NAME, "offset": 0, "bytes": container.stat().st_size}])
    shutil.copyfile(tokenizer, staging / TOKENIZER_NAME)
    shutil.copyfile(raw_dir / "config.json", staging / "config.json")
    shutil.copytree(ROOT / "licenses", staging / "licenses")
    if (ROOT / "LICENSE").is_file():
        shutil.copyfile(ROOT / "LICENSE", staging / "licenses" / "PROJECT-LICENSE")
    reports = []
    quality_paths = []
    for number, (path, role) in enumerate(supplied, 1):
        path = path.resolve()
        json.loads(path.read_text())  # Reports are copied verbatim, never executed.
        name = f"reports/{role}-{number:02d}-{path.name}"
        target = staging / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        reports.append({"path": name, "kind": role})
        if role == "quality":
            quality_paths.append(name)
    write_json(staging / "raw_manifest.json", raw)
    (staging / "MODEL_CARD.md").write_text(make_model_card(raw, quality_paths))
    source_index = software_snapshot(staging / "source.zip")
    roles = {part["path"]: "weight-container-part" for part in weight_parts}
    roles.update({TOKENIZER_NAME: "tokenizer", "config.json": "configuration", "source.zip": "corresponding-source",
                  "raw_manifest.json": "quantization-manifest", "container_verification.json": "verification",
                  "MODEL_CARD.md": "model-card"})
    roles.update({report["path"]: report["kind"] + "-report" for report in reports})
    assets = [record_asset(staging, p, roles.get(p.relative_to(staging).as_posix(), "license"))
              for p in sorted(staging.rglob("*")) if p.is_file()]
    for part in weight_parts:
        part["sha256"] = next(e["sha256"] for e in assets if e["path"] == part["path"])
    manifest = {"format": FORMAT, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "container": {"format": "E8HUF001", "bytes": receipt["actual_file_bytes"], "sha256": receipt["sha256"],
                      "parts": weight_parts, "raw_manifest_sha256": raw_manifest_sha,
                      "raw_package_bytes": receipt["original_package_bytes"], "raw_members": receipt["files_verified"]},
        "tokenizer": {"path": TOKENIZER_NAME, "sha256": TOKENIZER_SHA256},
        "quality": {"status": "reports-attached-no-automatic-pass" if quality_paths else "unmeasured",
                    "reports": quality_paths, "smallest_claim": "not certified by packaging"},
        "software": {"source_archive": "source.zip", "git_commit": source_index["git_commit"],
                     "git_worktree_dirty": source_index["git_worktree_dirty"]},
        "assets": assets, "sizes": {"weight_container_bytes": receipt["actual_file_bytes"],
            "tokenizer_bytes": (staging / TOKENIZER_NAME).stat().st_size,
            "configuration_bytes": (staging / "config.json").stat().st_size,
            "scope": "Every shipped file, including licenses, source, reports and this manifest; no external weight base."},
        "integrity_note": "Asset SHA values are inside this manifest. Its own SHA is printed in the build/verify receipt and must be trusted separately for authenticity."}
    finalize_manifest(staging, manifest)
    verified = verify_release(staging)
    os.replace(staging, destination)
    verified["release_dir"] = str(destination)
    return verified


def verify_release(directory, expected_manifest_sha256=None):
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise ValueError("Release manifest differs from independently trusted SHA")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or not manifest.get("complete"):
        raise ValueError("Incomplete or unsupported release manifest")
    entries = {}
    for entry in manifest["assets"]:
        name = entry["path"]
        if name in entries or name == MANIFEST_NAME:
            raise ValueError("Duplicate or recursive manifest asset")
        path = safe_path(directory, name)
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Release asset identity mismatch: {name}")
        entries[name] = entry
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(entries) | {MANIFEST_NAME}:
        raise ValueError("Release has missing or uncounted assets")
    total = sum(e["bytes"] for e in entries.values())
    sizes = manifest["sizes"]
    if (sizes["all_asset_bytes"] != total or sizes["manifest_bytes"] != manifest_path.stat().st_size
            or sizes["total_release_bytes"] != total + manifest_path.stat().st_size):
        raise ValueError("Release byte ledger mismatch")
    digest, position = hashlib.sha256(), 0
    parts = manifest["container"]["parts"]
    if not parts:
        raise ValueError("Missing weight container parts")
    seen = set()
    for part in parts:
        if part["path"] in seen or part["offset"] != position:
            raise ValueError("Container parts overlap, repeat, or have gaps")
        seen.add(part["path"])
        asset = entries[part["path"]]
        if asset["bytes"] != part["bytes"] or asset["sha256"] != part["sha256"]:
            raise ValueError("Container part descriptor differs from asset ledger")
        with safe_path(directory, part["path"]).open("rb") as stream:
            for data in iter(lambda: stream.read(BUFFER_BYTES), b""):
                digest.update(data)
                position += len(data)
    if position != manifest["container"]["bytes"] or digest.hexdigest() != manifest["container"]["sha256"]:
        raise ValueError("Concatenated container SHA or byte count mismatch")
    raw = json.loads((directory / "raw_manifest.json").read_text())
    if not raw.get("complete"):
        raise ValueError("Release references an incomplete quantized model")
    if sha256_file(directory / manifest["tokenizer"]["path"]) != TOKENIZER_SHA256:
        raise ValueError("Release tokenizer is not the pinned NVIDIA tokenizer")
    return {"complete": True, "release_dir": str(directory), "manifest_sha256": manifest_sha,
            "manifest_authenticity_checked": bool(expected_manifest_sha256),
            "assets_verified": len(entries), "container_sha256": digest.hexdigest(),
            "container_bytes": position, "total_release_bytes": sizes["total_release_bytes"],
            "quality_status": manifest["quality"]["status"]}


def restore_release(args):
    release = args.release_dir.resolve()
    receipt = verify_release(release, args.expected_manifest_sha256)
    manifest = json.loads((release / MANIFEST_NAME).read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    parts = manifest["container"]["parts"]
    with tempfile.TemporaryDirectory(prefix="container-", dir=output) as temporary:
        if len(parts) == 1:
            container = safe_path(release, parts[0]["path"])
        else:
            container = Path(temporary) / CONTAINER_NAME
            digest = hashlib.sha256()
            with container.open("xb") as target:
                for part in parts:
                    with safe_path(release, part["path"]).open("rb") as source:
                        for data in iter(lambda: source.read(BUFFER_BYTES), b""):
                            target.write(data)
                            digest.update(data)
            if digest.hexdigest() != manifest["container"]["sha256"]:
                raise ValueError("Restoration assembly SHA mismatch")
        reader = Reader(container)
        trusted_raw = json.loads((release / "raw_manifest.json").read_text())
        expected_files = {name: (e["bytes"], e["sha256"]) for name, e in trusted_raw["files"].items()}
        # The copied outer raw manifest may be JSON-normalized; the container
        # retains the exact original manifest bytes and its separately bound SHA.
        original_manifest = reader.files.get("manifest.json")
        if original_manifest is None or original_manifest["sha256"] != manifest["container"]["raw_manifest_sha256"]:
            raise ValueError("Container raw manifest binding mismatch")
        expected_files["manifest.json"] = (original_manifest["original_bytes"], original_manifest["sha256"])
        actual_files = {name: (e["original_bytes"], e["sha256"]) for name, e in reader.files.items()}
        if expected_files != actual_files:
            raise ValueError("Container members differ from the release quantization manifest")
        reader.restore(output / "raw")
    validate_raw(output / "raw")
    shutil.copyfile(release / TOKENIZER_NAME, output / TOKENIZER_NAME)
    shutil.copyfile(release / "config.json", output / "config.json")
    receipt.update(restored_raw_dir=str(output / "raw"), restored_raw_files=len(actual_files),
                   raw_manifest_sha256=sha256_file(output / "raw/manifest.json"),
                   quality_reports=manifest["quality"]["reports"],
                   note="Every raw file SHA matches the original quantized package. This is not a model-quality evaluation.")
    write_json(output / "restore_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Pack and verify a complete immutable raw package")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--source-dir", type=Path)
    build.add_argument("--tokenizer", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--split-bytes", type=int, default=0,
                       help=f"0 for one file; {DEFAULT_PART_BYTES} for GitHub-sized parts below 2 GiB")
    build.add_argument("--quality-report", type=Path, action="append", default=[])
    build.add_argument("--provenance-report", type=Path, action="append", default=[])
    verify = commands.add_parser("verify", help="Verify every asset, container part, hash and byte ledger")
    verify.add_argument("--release-dir", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256")
    restore = commands.add_parser("restore", help="Verify and recover exact raw files for the reference runtime")
    restore.add_argument("--release-dir", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    restore.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    result = (build_release(args) if args.command == "build" else
              restore_release(args) if args.command == "restore" else
              verify_release(args.release_dir, args.expected_manifest_sha256))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
