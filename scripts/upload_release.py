#!/usr/bin/env python3
"""Stream verified remote assets through this Mac to an existing GitHub draft.

Uses only Python's standard library, SSH and local gh authentication. The token
stays in local memory. No large local temporary file, release creation,
publication, asset replacement, or deletion is performed.

GitHub REST contract: https://docs.github.com/en/rest/releases/assets
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import urlencode, urlsplit

BUFFER_BYTES = 8 << 20
API_VERSION = "2026-03-10"
MANIFEST_NAME = "release_manifest.json"
USER_AGENT = "mamba2-e8w5-verified-draft-uploader"


def validate_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name.endswith("."):
        raise ValueError(f"Asset basename is not portable to GitHub: {name!r}")


def upload_plan(path):
    path = Path(path)
    if path.stat().st_size > 16 << 20:
        raise ValueError("Unexpectedly large local release manifest")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != "MAMBA2_E8W5_RELEASE_V1" or manifest.get("complete") is not True:
        raise ValueError("Expected a complete packaged release manifest")
    names, paths, assets = set(), set(), []
    for entry in manifest["assets"]:
        relative = PurePosixPath(entry["path"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]
                or relative.as_posix() != entry["path"] or any(ord(c) < 32 for c in entry["path"])):
            raise ValueError("Unsafe or noncanonical release asset path")
        name = relative.name
        validate_name(name)
        if name.casefold() in names or entry["path"] in paths or name.casefold() == MANIFEST_NAME:
            raise ValueError(f"Release assets do not have unique flat basenames: {name}")
        if type(entry["bytes"]) is not int or not 0 < entry["bytes"] < 2 ** 31:
            raise ValueError(f"Asset must be positive and below 2 GiB; split the container first: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid asset SHA: {name}")
        names.add(name.casefold())
        paths.add(entry["path"])
        assets.append({"name": name, "path": entry["path"], "bytes": entry["bytes"],
                       "sha256": entry["sha256"], "source": "remote"})
    sizes = manifest["sizes"]
    if (sizes["all_asset_bytes"] != sum(e["bytes"] for e in assets)
            or sizes["manifest_bytes"] != len(raw)
            or sizes["total_release_bytes"] != len(raw) + sum(e["bytes"] for e in assets)):
        raise ValueError("Local release byte ledger is inconsistent")
    # Upload the manifest last. GitHub uses flat asset names; the original path
    # remains in its label and in the unchanged manifest for download recovery.
    assets.append({"name": MANIFEST_NAME, "path": MANIFEST_NAME, "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest(), "source": "local-manifest"})
    return manifest, raw, assets


def auth_token(gh_path):
    command = gh_path or shutil.which("gh")
    if not command:
        raise RuntimeError("gh is unavailable; provide --gh-path")
    result = subprocess.run([command, "auth", "token", "--hostname", "github.com"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Local gh authentication failed; authenticate github.com before uploading")
    token = result.stdout.strip()
    if not token or any(c.isspace() for c in token):
        raise RuntimeError("gh returned an invalid authentication token")
    return token


def headers(token):
    return {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT, "X-GitHub-Api-Version": API_VERSION}


def api_get(path, token):
    if not path.startswith("/repos/") or any(c in path for c in "\r\n"):
        raise ValueError("Unexpected GitHub API path")
    connection = http.client.HTTPSConnection("api.github.com", timeout=60)
    try:
        connection.request("GET", path, headers=headers(token))
        response = connection.getresponse()
        raw = response.read((8 << 20) + 1)
        if response.status != 200:
            raise RuntimeError(f"GitHub read request failed with HTTP {response.status}")
        if len(raw) > 8 << 20:
            raise RuntimeError("Oversized GitHub metadata response")
        return json.loads(raw)
    finally:
        connection.close()


def require_draft(repo, release_id, token):
    release = api_get(f"/repos/{repo}/releases/{release_id}", token)
    if release.get("id") != release_id or release.get("draft") is not True:
        raise RuntimeError("Refusing upload: the selected release is not a draft")
    parsed = urlsplit(release["upload_url"].split("{", 1)[0])
    expected = f"/repos/{repo}/releases/{release_id}/assets"
    if (parsed.scheme != "https" or parsed.hostname != "uploads.github.com" or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.casefold() != expected.casefold()):
        raise ValueError("Release returned an unexpected upload endpoint")
    return release, parsed.path


def release_assets(repo, release_id, token):
    result = {}
    page = 1
    while True:
        entries = api_get(f"/repos/{repo}/releases/{release_id}/assets?per_page=100&page={page}", token)
        if not isinstance(entries, list):
            raise ValueError("Invalid asset-list response")
        for entry in entries:
            if entry["name"] in result:
                raise ValueError("GitHub release contains duplicate asset names")
            result[entry["name"]] = entry
        if len(entries) < 100:
            return result
        page += 1


def verified_server_asset(asset, expected, require_digest):
    if (asset.get("name") != expected["name"] or asset.get("size") != expected["bytes"]
            or asset.get("state") != "uploaded"):
        raise RuntimeError(f"GitHub asset name/size/state mismatch: {expected['name']}")
    digest = asset.get("digest")
    if digest is None or digest == "":
        if require_digest:
            raise RuntimeError(f"Cannot safely skip existing asset without a server SHA: {expected['name']}")
        return False
    if digest != "sha256:" + expected["sha256"]:
        raise RuntimeError(f"GitHub server digest mismatch: {expected['name']}")
    return True


def remote_stream(host, directory, relative):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", host):
        raise ValueError("Invalid SSH host alias")
    base = PurePosixPath(directory)
    if not base.is_absolute() or ".." in base.parts or any(ord(c) < 32 for c in directory):
        raise ValueError("--remote-directory must be an absolute safe POSIX path")
    filename = str(base / relative)
    # The token is never stored in this environment. Remove any pre-existing
    # GitHub token environment variables as an additional SSH boundary.
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"}}
    process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "--", host,
                                "cat -- " + shlex.quote(filename)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env=environment)
    return process


def stop_process(process):
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()


def stream_exact(source, send, expected_bytes, expected_sha256, progress=None):
    """Transfer at most 8 MiB at once, verify exact EOF and the streamed SHA."""
    digest, sent = hashlib.sha256(), 0
    while sent < expected_bytes:
        data = source.read(min(BUFFER_BYTES, expected_bytes - sent))
        if not data:
            raise RuntimeError(f"Asset stream truncated after {sent}/{expected_bytes} bytes")
        if len(data) > BUFFER_BYTES or len(data) > expected_bytes - sent:
            raise RuntimeError("Source violated bounded read contract")
        digest.update(data)
        send(data)
        sent += len(data)
        if progress:
            progress(sent)
    if source.read(1):
        raise RuntimeError("Asset stream has bytes beyond manifest length")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Streamed asset SHA differs from the release manifest")
    return digest.hexdigest(), sent


def upload_stream(source, expected, endpoint, token, timeout=120):
    connection = http.client.HTTPSConnection("uploads.github.com", timeout=timeout)
    last_update = [time.monotonic()]
    def progress(sent):
        if time.monotonic() - last_update[0] >= 30:
            print(f"[stream] {expected['name']}: {sent}/{expected['bytes']} bytes", flush=True)
            last_update[0] = time.monotonic()
    try:
        query = urlencode({"name": expected["name"], "label": expected["path"]})
        connection.putrequest("POST", endpoint + "?" + query)
        request_headers = headers(token)
        request_headers.update({"Content-Length": str(expected["bytes"]),
            "Content-Type": mimetypes.guess_type(expected["name"])[0] or "application/octet-stream"})
        for key, value in request_headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        digest, sent = stream_exact(source, connection.send, expected["bytes"], expected["sha256"], progress)
        response = connection.getresponse()
        raw = response.read((2 << 20) + 1)
        if response.status != 201:
            raise RuntimeError(f"GitHub upload failed with HTTP {response.status}; no asset is replaced or deleted automatically")
        if len(raw) > 2 << 20:
            raise RuntimeError("Oversized GitHub upload response")
        asset = json.loads(raw)
        server_verified = verified_server_asset(asset, expected, require_digest=False)
        return {"asset": asset, "stream_sha256": digest, "stream_bytes": sent,
                "server_sha256_verified": server_verified}
    finally:
        connection.close()


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def perform_upload(args):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo) or args.release_id <= 0:
        raise ValueError("Invalid repository or release ID")
    manifest, manifest_bytes, planned = upload_plan(args.manifest)
    report = {"complete": False, "repo": args.repo, "release_id": args.release_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "expected_assets": len(planned), "expected_total_bytes": manifest["sizes"]["total_release_bytes"],
        "stream_buffer_bytes": BUFFER_BYTES, "large_local_files_created": False,
        "credentials_sent_to_remote": False, "release_created_or_published": False, "assets": []}
    save_report(args.report, report)
    token = None
    started = time.monotonic()
    try:
        token = auth_token(args.gh_path)
        release, endpoint = require_draft(args.repo, args.release_id, token)
        report["release_html_url"] = release.get("html_url")
        report["release_tag"] = release.get("tag_name")
        existing = release_assets(args.repo, args.release_id, token)
        unexpected = set(existing) - {entry["name"] for entry in planned}
        if unexpected:
            raise RuntimeError("Draft contains assets absent from this release manifest: " + ", ".join(sorted(unexpected)))
        # Check all conflicts before uploading the first asset.
        for entry in planned:
            if entry["name"] in existing:
                verified_server_asset(existing[entry["name"]], entry, require_digest=True)
        process = remote_stream(args.ssh_host, args.remote_directory, MANIFEST_NAME)
        try:
            stream_exact(process.stdout, lambda _data: None, len(manifest_bytes), report["manifest_sha256"])
            if process.wait(timeout=30):
                raise RuntimeError("Remote release manifest read failed")
        finally:
            stop_process(process)
        for entry in planned:
            if entry["name"] in existing:
                asset = existing[entry["name"]]
                record = {**entry, "action": "skipped-server-size-and-sha-match",
                    "github_asset_id": asset["id"], "server_sha256_verified": True,
                    "server_digest": asset["digest"]}
            else:
                _, endpoint = require_draft(args.repo, args.release_id, token)
                process = None
                try:
                    if entry["source"] == "local-manifest":
                        source = io.BytesIO(manifest_bytes)
                    else:
                        process = remote_stream(args.ssh_host, args.remote_directory, entry["path"])
                        source = process.stdout
                    uploaded = upload_stream(source, entry, endpoint, token, args.timeout)
                    if process is not None and process.wait(timeout=30):
                        raise RuntimeError("SSH asset stream exited unsuccessfully after upload")
                    asset = uploaded["asset"]
                    if not uploaded["server_sha256_verified"]:
                        asset = api_get(f"/repos/{args.repo}/releases/assets/{asset['id']}", token)
                        uploaded["server_sha256_verified"] = verified_server_asset(asset, entry, require_digest=False)
                    record = {**entry, "action": "uploaded", "github_asset_id": asset["id"],
                        "stream_sha256": uploaded["stream_sha256"], "stream_bytes": uploaded["stream_bytes"],
                        "server_sha256_verified": uploaded["server_sha256_verified"],
                        "server_digest": asset.get("digest")}
                finally:
                    stop_process(process)
            report["assets"].append(record)
            report["elapsed_seconds"] = time.monotonic() - started
            save_report(args.report, report)
            print(f"[draft asset] {entry['name']}: {record['action']}; server SHA verified={record['server_sha256_verified']}", flush=True)
        require_draft(args.repo, args.release_id, token)
        final_assets = release_assets(args.repo, args.release_id, token)
        if set(final_assets) != {entry["name"] for entry in planned}:
            raise RuntimeError("Final draft asset inventory differs from the plan")
        for record, entry in zip(report["assets"], planned):
            verified = verified_server_asset(final_assets[entry["name"]], entry, require_digest=False)
            record["server_sha256_verified"] = verified
            record["server_digest"] = final_assets[entry["name"]].get("digest")
        report.update(complete=True, final_release_is_draft=True,
            all_server_sha256_verified=all(r["server_sha256_verified"] for r in report["assets"]),
            elapsed_seconds=time.monotonic()-started,
            note="All uploads target an existing draft. Missing server digests are explicitly unverified. GitHub flat names map back to manifest paths via the path fields and asset labels.")
        save_report(args.report, report)
        return report
    except Exception as error:
        message = str(error)
        if token:
            message = message.replace(token, "[REDACTED]")
        report.update(complete=False, error_type=type(error).__name__, error=message,
                      elapsed_seconds=time.monotonic()-started,
                      note="No overwrite, deletion, or automatic retry was performed. An interrupted HTTP request may have left a draft asset; inspect it before retrying.")
        save_report(args.report, report)
        raise RuntimeError(message) from None


def self_test():
    """CPU memory-only mocks. No network, SSH, gh auth, or real release access."""
    payload = bytes(range(256)) * ((BUFFER_BYTES // 256) + 29)
    expected = hashlib.sha256(payload).hexdigest()
    chunks = []
    observed_hash = hashlib.sha256()
    def sink(data):
        chunks.append(len(data))
        observed_hash.update(data)
    digest, count = stream_exact(io.BytesIO(payload), sink, len(payload), expected)
    if digest != expected or count != len(payload) or observed_hash.hexdigest() != expected or max(chunks) > BUFFER_BYTES:
        raise AssertionError("Bounded stream failed")
    for data, size, sha in ((payload[:-1], len(payload), expected),
                            (payload + b"x", len(payload), expected),
                            (payload, len(payload), "0" * 64)):
        try:
            stream_exact(io.BytesIO(data), lambda _x: None, size, sha)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid input stream was not rejected")
    entry = {"name": "part.bin", "bytes": len(payload), "sha256": expected}
    asset = {"name": "part.bin", "size": len(payload), "state": "uploaded", "digest": "sha256:" + expected}
    if not verified_server_asset(asset, entry, True):
        raise AssertionError("Matching server digest rejected")
    del asset["digest"]
    if verified_server_asset(asset, entry, False):
        raise AssertionError("Missing digest incorrectly verified")
    try:
        verified_server_asset(asset, entry, True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing server digest incorrectly accepted for skip")
    asset["digest"] = "sha256:" + "0" * 64
    try:
        verified_server_asset(asset, entry, False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Mismatched server digest not rejected")
    from unittest import mock
    class FakeResponse:
        status = 201
        def read(self, _limit):
            return json.dumps({"id": 1, "name": "part.bin", "size": len(payload),
                               "state": "uploaded", "digest": "sha256:" + expected}).encode()
    class FakeConnection:
        def __init__(self):
            self.request_headers, self.sent, self.body_hash = {}, 0, hashlib.sha256()
            self.closed = False
        def putrequest(self, method, path):
            self.method, self.path = method, path
        def putheader(self, name, value):
            self.request_headers[name] = value
        def endheaders(self):
            pass
        def send(self, data):
            if len(data) > BUFFER_BYTES:
                raise AssertionError("HTTP mock received oversized chunk")
            self.sent += len(data)
            self.body_hash.update(data)
        def getresponse(self):
            return FakeResponse()
        def close(self):
            self.closed = True
    connection = FakeConnection()
    entry["path"] = "weights/part.bin"
    with mock.patch("http.client.HTTPSConnection", return_value=connection) as factory:
        result = upload_stream(io.BytesIO(payload), entry, "/repos/test/repo/releases/1/assets", "mock-token")
        if not result["server_sha256_verified"] or connection.sent != len(payload) or not connection.closed:
            raise AssertionError("Mock HTTP upload failed")
        if connection.request_headers["Content-Length"] != str(len(payload)) or connection.body_hash.hexdigest() != expected:
            raise AssertionError("Content-Length or streamed HTTP body mismatch")
        if factory.call_args.args[0] != "uploads.github.com" or connection.method != "POST":
            raise AssertionError("Unexpected HTTP upload target")
    with mock.patch.dict(os.environ, {"GH_TOKEN": "mock-token", "GITHUB_TOKEN": "mock-token"}):
        with mock.patch("subprocess.Popen") as process_factory:
            remote_stream("horde-gpu", "/remote/a'b", "weights/part.bin")
            call = process_factory.call_args
            if "GH_TOKEN" in call.kwargs["env"] or "GITHUB_TOKEN" in call.kwargs["env"]:
                raise AssertionError("Authentication leaked to SSH environment")
            if shlex.split(call.args[0][-1]) != ["cat", "--", "/remote/a'b/weights/part.bin"]:
                raise AssertionError("Remote path was not shell quoted correctly")
    with mock.patch.dict(require_draft.__globals__, {"api_get": lambda *_: {"id": 1, "draft": False}}):
        try:
            require_draft("test/repo", 1, "mock-token")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Published release was not rejected")
    return {"complete": True, "self_test_only": True, "network_calls": 0, "credential_reads": 0,
            "stream_bytes": count, "max_chunk_bytes": max(chunks), "stream_chunks": len(chunks),
            "checks": ["bounded stream", "exact SHA", "truncation", "extra bytes", "wrong input SHA",
                       "verified skip only", "missing digest reported unverified", "server SHA mismatch",
                       "mock HTTPS POST with Content-Length", "SSH path quoting", "SSH token environment removal",
                       "published release rejected"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="EndlessChasing/mamba2-8b-e8w5")
    parser.add_argument("--release-id", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ssh-host", default="horde-gpu")
    parser.add_argument("--remote-directory")
    parser.add_argument("--gh-path", help="Explicit gh executable, otherwise discovered on PATH")
    parser.add_argument("--report", type=Path, default=Path("upload_receipt.json"))
    parser.add_argument("--timeout", type=int, default=120, help="Per-socket operation timeout, seconds")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2), flush=True)
        return
    if not args.release_id or not args.manifest or not args.remote_directory:
        parser.error("--release-id, --manifest, and --remote-directory are required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = perform_upload(args)
    print(json.dumps({key: result[key] for key in ("complete", "repo", "release_id", "expected_assets",
        "expected_total_bytes", "all_server_sha256_verified", "final_release_is_draft", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
