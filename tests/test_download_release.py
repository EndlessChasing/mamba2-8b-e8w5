"""Offline public-release download tests; Python stdlib only, no Torch/GPU."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_release.py"
SPEC = importlib.util.spec_from_file_location("download_release", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture():
    files = {"weights.huff.part-00001-of-00002": b"E8HUF001-first-part",
             "weights.huff.part-00002-of-00002": b"second-weight-part",
             "licenses/APACHE-2.0.txt": b"license fixture\n",
             "reports/quality-01-test.json": b'{"complete": true}\n',
             "empty.txt": b""}
    entries = [{"path": name, "bytes": len(data), "sha256": sha(data), "role": "fixture"}
               for name, data in files.items()]
    parts, offset = [], 0
    for entry in entries[:2]:
        parts.append({**entry, "offset": offset})
        offset += entry["bytes"]
    manifest = {"format": download.FORMAT, "complete": True, "assets": entries,
                "container": {"format": "E8HUF001", "parts": parts, "bytes": offset,
                              "sha256": sha(b"".join(files[e["path"]] for e in parts))},
                "sizes": {"all_asset_bytes": sum(map(len, files.values())),
                          "manifest_bytes": 0, "total_release_bytes": 0}}
    for _ in range(16):
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        if len(raw) == manifest["sizes"]["manifest_bytes"]:
            break
        manifest["sizes"]["manifest_bytes"] = len(raw)
        manifest["sizes"]["total_release_bytes"] = sum(map(len, files.values())) + len(raw)
    else:
        raise AssertionError("Fixture byte ledger did not converge")
    return files, raw


class FakeGitHub:
    def __init__(self, files, raw):
        self.files = {Path(name).name: data for name, data in files.items()}
        self.files[download.MANIFEST_NAME] = raw
        self.calls = []

    def bytes(self, url):
        if "/tags/" in url:
            return json.dumps({"id": 7, "tag_name": "test-v1", "draft": False}).encode()
        if "/assets?" in url:
            return json.dumps([{"name": name, "size": len(data), "state": "uploaded"}
                               for name, data in self.files.items()]).encode()
        return self.files[url.rsplit("/", 1)[1]]

    def payload(self, url, path, maximum):
        name = url.rsplit("/", 1)[1]
        data = self.files[name]
        offset = path.stat().st_size if path.exists() else 0
        self.calls.append((name, offset))
        with path.open("ab") as stream:
            stream.write(data[offset:])


class ReleaseDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.files, self.raw = fixture()
        self.fake = FakeGitHub(self.files, self.raw)

    def tearDown(self):
        self.temp.cleanup()

    def run_download(self, target=None, expected=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return download.download_release(download.DEFAULT_REPO, "test-v1",
                target or self.root / "release", expected, self.fake.bytes, self.fake.payload)

    def test_complete_download_nested_inventory_and_idempotent_skip(self):
        report = self.run_download(expected=sha(self.raw))
        target = self.root / "release"
        self.assertTrue(report["complete"])
        self.assertTrue(report["manifest_authenticity_checked"])
        self.assertEqual({p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()},
                         set(self.files) | {download.MANIFEST_NAME})
        for name, data in self.files.items():
            self.assertEqual((target / name).read_bytes(), data)
        calls = len(self.fake.calls)
        (target / "empty.txt.part").write_bytes(b"stale interrupted download")
        self.run_download(expected=sha(self.raw))
        self.assertEqual(len(self.fake.calls), calls)
        self.assertFalse((target / "empty.txt.part").exists())

    def test_resumes_partial_file_and_cleans_it(self):
        target = self.root / "release"
        target.mkdir()
        name = next(iter(self.files))
        (target / (name + ".part")).write_bytes(self.files[name][:5])
        self.run_download(target)
        self.assertIn((name, 5), self.fake.calls)
        self.assertFalse((target / (name + ".part")).exists())

    def test_bad_partial_is_retried_fresh_after_full_hash_mismatch(self):
        target = self.root / "release"
        target.mkdir()
        name = next(iter(self.files))
        (target / (name + ".part")).write_bytes(b"WRONG")
        self.run_download(target)
        self.assertEqual([offset for n, offset in self.fake.calls if n == name], [5, 0])

    def test_manifest_trust_mismatch_does_not_create_output(self):
        with self.assertRaisesRegex(ValueError, "trusted SHA"):
            self.run_download(expected="0" * 64)
        self.assertFalse((self.root / "release").exists())

    def test_unsafe_paths_and_duplicate_flat_basename_rejected(self):
        for name in ("../outside", "/outside", "x/../outside", "x//a", "x/./a", "C:/a",
                     "a\\b", "x.part", "x/CON", "x/NUL.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                download.portable_path(name)
        for name in ("nested/release_manifest.json", "other/APACHE-2.0.txt", "other/apache-2.0.TXT"):
            manifest = json.loads(self.raw)
            manifest["assets"].append({"path": name, "bytes": 1, "sha256": sha(b"a")})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "basename"):
                download.validate_manifest(json.dumps(manifest).encode())

    def test_case_ambiguous_parent_directories_rejected(self):
        manifest = json.loads(self.raw)
        manifest["assets"].extend([
            {"path": "Reports/a.json", "bytes": 1, "sha256": sha(b"a")},
            {"path": "reports/b.json", "bytes": 1, "sha256": sha(b"b")},
        ])
        with self.assertRaisesRegex(ValueError, "Case-ambiguous"):
            download.validate_manifest(json.dumps(manifest).encode())

    def test_symlink_traversal_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        target = self.root / "release"
        target.mkdir()
        (target / "licenses").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_download(target)
        self.assertEqual(list(outside.iterdir()), [])

    def test_unexpected_and_invalid_existing_files_are_preserved(self):
        target = self.root / "release"
        target.mkdir()
        extra = target / "my-notes.txt"
        extra.write_text("keep")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.run_download(target)
        self.assertEqual(extra.read_text(), "keep")
        extra.unlink()
        name = next(iter(self.files))
        final = target / name
        final.write_bytes(b"keep wrong final file")
        with self.assertRaisesRegex(ValueError, "Existing final"):
            self.run_download(target)
        self.assertEqual(final.read_bytes(), b"keep wrong final file")

    def test_missing_asset_and_api_size_mismatch_rejected_before_writes(self):
        key = next(iter(self.files))
        del self.fake.files[key]
        with self.assertRaisesRegex(ValueError, "Missing GitHub asset"):
            self.run_download()
        self.assertFalse((self.root / "release").exists())

    def test_invalid_combined_container_hash_fails_final_verification(self):
        target = self.root / "release"
        self.run_download(target)
        manifest, entries, _ = download.validate_manifest(self.raw)
        manifest["container"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Concatenated"):
            download.verify_directory(target, manifest, entries, self.raw)


if __name__ == "__main__":
    unittest.main()
