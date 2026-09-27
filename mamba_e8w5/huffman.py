"""Bounded-memory canonical Huffman container for E8/W5 raw packages.

E8HUF001 retains independently indexed 1024-symbol E8 / 4096-symbol W5
chunks. Absolute file offsets are uint64; each member's payload-relative
offsets are uint32 and are explicitly range checked. No complete vocabulary
is expanded during packing or verification.
"""
import ctypes
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

from .codec import pack_uniform_codes, unpack_uniform_codes

MAGIC = b"E8HUF001"
BATCH_SYMBOLS = 1 << 20
IO_BYTES = 8 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(IO_BYTES), b""):
            h.update(data)
    return h.hexdigest()


def huffman_lengths(hist):
    heap = [(int(c), i, i) for i, c in enumerate(hist) if c]
    heapq.heapify(heap)
    if not heap:
        raise ValueError("Empty symbol histogram")
    lengths = np.zeros(len(hist), dtype=np.uint8)
    if len(heap) == 1:
        lengths[heap[0][2]] = 1
        return lengths
    serial = len(hist)
    while len(heap) > 1:
        a, _, x = heapq.heappop(heap)
        b, _, y = heapq.heappop(heap)
        heapq.heappush(heap, (a + b, serial, (x, y)))
        serial += 1
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            if depth > 32:
                raise ValueError("Huffman code exceeds supported 32-bit width")
            lengths[node] = depth
        else:
            stack.extend((child, depth + 1) for child in node)
    return lengths


def table(lengths):
    lengths = np.asarray(lengths, dtype=np.uint8)
    if lengths.shape != (65536,) or lengths.max() > 32 or not lengths.any():
        raise ValueError("Invalid Huffman lengths")
    codes = np.zeros(65536, dtype="<u4")
    value = prior = 0
    for symbol in sorted(np.flatnonzero(lengths), key=lambda s: (lengths[s], s)):
        length = int(lengths[symbol])
        value <<= length - prior
        if value >= 1 << length:
            raise ValueError("Oversubscribed Huffman table")
        codes[symbol] = value
        value += 1
        prior = length
    return codes, lengths


def _ptr(array, kind):
    return array.ctypes.data_as(ctypes.POINTER(kind))


def load_codec(path=None):
    """Compile the small portable C++ codec once into the user cache."""
    if path is None:
        source = Path(__file__).with_name("huffman_codec.cpp")
        digest = sha256_file(source)[:20]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mamba-e8w5"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / ("huffman_" + digest + ".so")
        if not path.exists():
            tmp = path.with_name(path.name + "." + str(os.getpid()) + ".partial")
            subprocess.run([os.environ.get("CXX", "c++"), "-O3", "-std=c++17",
                            "-shared", "-fPIC", str(source), "-o", str(tmp)], check=True)
            os.replace(tmp, path)
    library = ctypes.CDLL(str(Path(path).resolve()))
    p8, p16, p32 = [ctypes.POINTER(t) for t in (ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32)]
    library.pack_verify_scale_u16.argtypes = [p16, ctypes.c_size_t, p32, p8, p8,
        ctypes.c_size_t, p32, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    library.pack_verify_scale_u16.restype = ctypes.c_int
    library.unpack_huffman_u16.argtypes = [p8, ctypes.c_size_t, p32, ctypes.c_size_t,
        ctypes.c_size_t, p32, p8, p16]
    library.unpack_huffman_u16.restype = ctypes.c_int
    return library


def _descriptor(path):
    size = path.stat().st_size
    entry = {"name": path.name, "original_bytes": size, "sha256": sha256_file(path),
             "family": None, "prefix_bytes": size}
    if path.suffix not in (".e8", ".uniform"):
        return entry
    with path.open("rb") as f:
        magic, length = struct.unpack("<8sI", f.read(12))
        if length > 65536 or 12 + length > size:
            raise ValueError("Invalid raw header")
        header = json.loads(f.read(length))
    count = math.prod(header["shape"])
    if path.suffix == ".e8":
        if magic != b"ME8HD001" or header.get("axis_residual_amplitude", 0):
            raise ValueError("Expected pure 16-bit E8 indices")
        rows, cols = header["shape"]
        if cols % 8:
            raise ValueError("E8 width must be divisible by 8")
        count //= 8
        bits, chunk, family = 16, 1024, "e8"
        tail_bytes = 2 * cols + (cols + 7) // 8 + (rows + 7) // 8 + 4
    else:
        if magic != b"MEQG0128" or header["bits"] not in range(2, 9) or header["group"] != 128:
            raise ValueError("Invalid uniform format")
        bits, chunk, family = header["bits"], 4096, "embedding"
        tail_bytes = count // 128 * 2
    if count <= 0 or count % chunk:
        raise ValueError(f"{path.name}: symbol count must be divisible by {chunk}")
    prefix = 12 + length
    symbol_bytes = count * bits // 8
    if prefix + symbol_bytes + tail_bytes != size:
        raise ValueError("Raw member size mismatch")
    entry.update(family=family, bits=bits, chunk=chunk, count=count,
                 prefix_bytes=prefix, symbol_bytes=symbol_bytes, tail_bytes=tail_bytes)
    return entry


def _raw_symbols(path, entry):
    with path.open("rb") as f:
        f.seek(entry["prefix_bytes"])
        for first in range(0, entry["count"], BATCH_SYMBOLS):
            count = min(BATCH_SYMBOLS, entry["count"] - first)
            raw = f.read(count * entry["bits"] // 8)
            if len(raw) != count * entry["bits"] // 8:
                raise ValueError("Truncated raw symbols")
            if entry["bits"] == 16:
                yield np.frombuffer(raw, dtype="<u2").copy()
            else:
                yield unpack_uniform_codes(raw, entry["bits"]).astype("<u2")


def _copy_range(source, destination, offset, count):
    source.seek(offset)
    while count:
        data = source.read(min(IO_BYTES, count))
        if not data:
            raise ValueError("Truncated member")
        destination.write(data)
        count -= len(data)


def pack_directory(source, target, codec=None):
    """Write an actual complete container. Call verify_container afterwards."""
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    paths = sorted(p for p in source.iterdir() if p.is_file() and not p.name.endswith(".partial"))
    if target.parent.resolve() == source.resolve():
        raise ValueError("Container must be outside the raw package directory")
    entries = [_descriptor(p) for p in paths]
    histograms = {family: np.zeros(65536, dtype=np.int64)
                  for family in {e["family"] for e in entries if e["family"]}}
    for path, entry in zip(paths, entries):
        if entry["family"]:
            for symbols in _raw_symbols(path, entry):
                histograms[entry["family"]] += np.bincount(symbols, minlength=65536)
    tables = {family: table(huffman_lengths(hist)) for family, hist in histograms.items()}
    report = {"format": MAGIC.decode(), "files": [], "codebooks": {}, "entropy": {},
              "batch_symbols": BATCH_SYMBOLS, "source_manifest_sha256":
              sha256_file(source / "manifest.json") if (source / "manifest.json").exists() else None}
    library = load_codec(codec)
    tmp = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w+b") as out:
        out.write(bytes(24))
        for family, (codes, lengths) in sorted(tables.items()):
            n = 65536 if family == "e8" else 256
            if np.any(lengths[n:]):
                raise ValueError("Symbol outside table domain")
            report["codebooks"][family] = {"offset": out.tell(), "bytes": n}
            out.write(lengths[:n].tobytes())
            hist = histograms[family]
            probabilities = hist[hist > 0] / hist.sum()
            report["entropy"][family] = {"symbols": int(hist.sum()),
                "shannon_bits": float(-(probabilities * np.log2(probabilities)).sum()),
                "huffman_average_bits": float(hist @ lengths / hist.sum())}
        for number, (path, original) in enumerate(zip(paths, entries), 1):
            entry = dict(original)
            entry["prefix_offset"] = out.tell()
            with path.open("rb") as raw:
                _copy_range(raw, out, 0, entry["prefix_bytes"])
                if entry["family"]:
                    entry["tail_offset"] = out.tell()
                    _copy_range(raw, out, entry["prefix_bytes"] + entry["symbol_bytes"], entry["tail_bytes"])
            if entry["family"]:
                chunks = entry["count"] // entry["chunk"]
                offsets = np.empty(chunks, dtype="<u4")
                entry["offsets_offset"] = out.tell()
                entry["offsets_bytes"] = chunks * 4
                out.seek(chunks * 4, 1)
                entry["payload_offset"] = out.tell()
                payload_bytes = first_chunk = 0
                codes, lengths = tables[entry["family"]]
                for symbols in _raw_symbols(path, entry):
                    local_offsets = np.empty(len(symbols) // entry["chunk"], dtype="<u4")
                    output = np.empty(len(symbols) * 4, dtype=np.uint8)
                    used = ctypes.c_size_t()
                    status = library.pack_verify_scale_u16(_ptr(symbols, ctypes.c_uint16), len(symbols),
                        _ptr(codes, ctypes.c_uint32), _ptr(lengths, ctypes.c_uint8),
                        _ptr(output, ctypes.c_uint8), len(output), _ptr(local_offsets, ctypes.c_uint32),
                        entry["chunk"], ctypes.byref(used))
                    if status:
                        raise ValueError(f"Huffman encode/verify failed: {path.name}, {status}")
                    if payload_bytes + used.value >= 1 << 32:
                        raise ValueError("Member payload exceeds uint32 offset format")
                    offsets[first_chunk:first_chunk + len(local_offsets)] = local_offsets.astype(np.uint64) + payload_bytes
                    first_chunk += len(local_offsets)
                    payload_bytes += used.value
                    out.write(memoryview(output[:used.value]))
                end = out.tell()
                entry["payload_bytes"] = payload_bytes
                out.seek(entry["offsets_offset"])
                out.write(offsets.tobytes())
                out.seek(end)
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("Source member changed during packing")
            report["files"].append(entry)
            print(f"[pack] {number}/{len(paths)} {path.name}", flush=True)
        position = out.tell()
        raw = json.dumps(report, separators=(",", ":")).encode()
        out.write(raw)
        out.seek(0)
        out.write(struct.pack("<8sQQ", MAGIC, position, len(raw)))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, target)
    return report


class Reader:
    """Seekable container. iter_member streams exact original raw-file bytes."""
    def __init__(self, path, codec=None):
        self.path = Path(path)
        self.library = load_codec(codec)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as f:
            raw = f.read(24)
            if len(raw) != 24:
                raise ValueError("Truncated container")
            magic, at, length = struct.unpack("<8sQQ", raw)
            if magic != MAGIC or length > 64 << 20 or at < 24 or at + length != self.size:
                raise ValueError("Invalid container framing")
            self.data_end = at
            f.seek(at)
            self.manifest = json.loads(f.read(length))
        self.files = {}
        self.tables = {}
        intervals = []
        for family, entry in self.manifest["codebooks"].items():
            if family not in ("e8", "embedding") or entry["bytes"] != (65536 if family == "e8" else 256):
                raise ValueError("Invalid codebook")
            lengths = np.zeros(65536, dtype=np.uint8)
            lengths[:entry["bytes"]] = np.frombuffer(self._read(entry["offset"], entry["bytes"]), dtype=np.uint8)
            self.tables[family] = table(lengths)
            intervals.append((entry["offset"], entry["offset"] + entry["bytes"]))
        for entry in self.manifest["files"]:
            name = entry["name"]
            if name in self.files or Path(name).name != name or name in (".", ".."):
                raise ValueError("Unsafe or duplicate member name")
            self.files[name] = entry
            sections = ["prefix"]
            if entry["family"]:
                if entry["family"] not in self.tables or entry["bits"] not in range(2, 17):
                    raise ValueError("Invalid coded family")
                chunk = 1024 if entry["family"] == "e8" else 4096
                if entry["chunk"] != chunk or entry["count"] <= 0 or entry["count"] % chunk:
                    raise ValueError("Invalid chunk geometry")
                if entry["offsets_bytes"] != entry["count"] // chunk * 4 or entry["payload_bytes"] >= 1 << 32:
                    raise ValueError("Invalid index array")
                if entry["original_bytes"] != entry["prefix_bytes"] + entry["count"] * entry["bits"] // 8 + entry["tail_bytes"]:
                    raise ValueError("Invalid restored size")
                sections += ["tail", "offsets", "payload"]
            elif entry["original_bytes"] != entry["prefix_bytes"]:
                raise ValueError("Invalid raw member size")
            for section in sections:
                offset, count = entry[section + "_offset"], entry[section + "_bytes"]
                self._bounds(offset, count)
                intervals.append((offset, offset + count))
        cursor = 24
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError("Container data overlap or unaccounted bytes")
            cursor = end
        if cursor != self.data_end:
            raise ValueError("Container data ledger mismatch")

    def _bounds(self, offset, count):
        if not isinstance(offset, int) or not isinstance(count, int) or offset < 24 or count < 0 or offset + count > self.data_end:
            raise ValueError("Container range outside payload")

    def _read(self, offset, count):
        self._bounds(offset, count)
        with self.path.open("rb") as f:
            f.seek(offset)
            raw = f.read(count)
        if len(raw) != count:
            raise ValueError("Truncated container data")
        return raw

    def _range(self, offset, count):
        self._bounds(offset, count)
        for first in range(0, count, IO_BYTES):
            yield self._read(offset + first, min(IO_BYTES, count - first))

    def _offsets(self, entry):
        offsets = np.frombuffer(self._read(entry["offsets_offset"], entry["offsets_bytes"]), dtype="<u4")
        if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]) or int(offsets[-1]) >= entry["payload_bytes"]:
            raise ValueError("Invalid or unsorted chunk offsets")
        return offsets

    def _decode(self, entry, offsets, first, stop):
        start = int(offsets[first])
        end = int(offsets[stop]) if stop < len(offsets) else entry["payload_bytes"]
        raw = np.frombuffer(self._read(entry["payload_offset"] + start, end - start), dtype=np.uint8)
        local = (offsets[first:stop].astype(np.uint64) - start).astype("<u4")
        symbols = np.empty((stop - first) * entry["chunk"], dtype="<u2")
        codes, lengths = self.tables[entry["family"]]
        status = self.library.unpack_huffman_u16(_ptr(raw, ctypes.c_uint8), len(raw),
            _ptr(local, ctypes.c_uint32), len(symbols), entry["chunk"],
            _ptr(codes, ctypes.c_uint32), _ptr(lengths, ctypes.c_uint8), _ptr(symbols, ctypes.c_uint16))
        if status:
            raise ValueError(f"Huffman decode failed: {entry['name']}, {status}")
        return symbols

    def get_chunk(self, name, index):
        """Decode an independent symbol chunk with no prior decoder state."""
        entry = self.files[name]
        if not entry["family"]:
            raise ValueError("Member has no coded chunks")
        offsets = self._offsets(entry)
        if not 0 <= index < len(offsets):
            raise IndexError(index)
        return self._decode(entry, offsets, index, index + 1)

    def iter_member(self, name):
        entry = self.files[name]
        yield from self._range(entry["prefix_offset"], entry["prefix_bytes"])
        if entry["family"]:
            offsets = self._offsets(entry)
            batch_chunks = BATCH_SYMBOLS // entry["chunk"]
            for first in range(0, len(offsets), batch_chunks):
                symbols = self._decode(entry, offsets, first, min(first + batch_chunks, len(offsets)))
                yield (symbols.astype("<u2", copy=False).tobytes() if entry["bits"] == 16
                       else pack_uniform_codes(symbols, entry["bits"]))
            yield from self._range(entry["tail_offset"], entry["tail_bytes"])

    def get(self, name):
        """Return full member bytes; use iter_member/restore for large members."""
        return b"".join(self.iter_member(name))

    def restore(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for name, entry in self.files.items():
            digest, count = hashlib.sha256(), 0
            path = directory / name
            tmp = path.with_name(path.name + ".partial")
            with tmp.open("wb") as f:
                for data in self.iter_member(name):
                    f.write(data)
                    digest.update(data)
                    count += len(data)
            if count != entry["original_bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("Restored member integrity failure: " + name)
            os.replace(tmp, path)


def verify_container(path, codec=None, trusted_directory=None):
    """Full disk readback with SHA checks and three fresh chunks per matrix."""
    reader = Reader(path, codec)
    if trusted_directory is not None:
        directory = Path(trusted_directory)
        trusted = {p.name: (p.stat().st_size, sha256_file(p)) for p in directory.iterdir()
                   if p.is_file() and not p.name.endswith(".partial")}
        supplied = {name: (e["original_bytes"], e["sha256"]) for name, e in reader.files.items()}
        if trusted != supplied:
            raise ValueError("Container is not bound to trusted source package")
    independent = 0
    for number, (name, entry) in enumerate(reader.files.items(), 1):
        digest, count = hashlib.sha256(), 0
        for data in reader.iter_member(name):
            digest.update(data)
            count += len(data)
        if digest.hexdigest() != entry["sha256"] or count != entry["original_bytes"]:
            raise ValueError("Restored member SHA/size mismatch: " + name)
        if entry["family"]:
            indices = sorted({0, min(1, entry["count"] // entry["chunk"] - 1), entry["count"] // entry["chunk"] - 1})
            for index in indices:
                symbols = reader.get_chunk(name, index)
                if trusted_directory is not None:
                    with (Path(trusted_directory) / name).open("rb") as source:
                        source.seek(entry["prefix_bytes"] + index * entry["chunk"] * entry["bits"] // 8)
                        raw = source.read(entry["chunk"] * entry["bits"] // 8)
                    wanted = (np.frombuffer(raw, dtype="<u2") if entry["bits"] == 16
                              else unpack_uniform_codes(raw, entry["bits"]))
                    if not np.array_equal(symbols, wanted):
                        raise ValueError("Independent source chunk mismatch")
                independent += 1
        print(f"[verify] {number}/{len(reader.files)} {name}", flush=True)
    return {"complete": True, "container": str(path), "actual_file_bytes": reader.size,
            "sha256": sha256_file(path), "original_package_bytes": sum(e["original_bytes"] for e in reader.files.values()),
            "files_verified": len(reader.files), "independent_chunks_verified": independent,
            "trusted_source_bound": trusted_directory is not None, "entropy": reader.manifest["entropy"],
            "note": "Exact to quantized raw files; no claim of lossless recovery of source floating-point model."}
