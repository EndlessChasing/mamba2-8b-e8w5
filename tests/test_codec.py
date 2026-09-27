"""Meaningful raw codec / streaming container integrity checks.

Run: python -m unittest discover -s tests -p test_codec.py -v
Optional real E8 GPU gate: E8_TEST_GPU=1 <same command>.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from mamba_e8w5.codec import (write_uniform, read_uniform, read_uniform_rows,
    iter_uniform, write_e8, read_e8, decode_e8, load_reference_primitives,
    vector_quantize, rotate_last)
from mamba_e8w5.huffman import pack_directory, verify_container, Reader


def legacy_uniform_bytes(weight, bits):
    """Original whole-matrix writer oracle; only small fixtures call this."""
    limit = 2 ** (bits - 1) - 1
    x = weight.float().reshape(-1, 128)
    scale = (x.abs().amax(-1, keepdim=True) / limit).half()
    scale = torch.where(scale == 0, 1, scale)
    codes = (torch.round(x / scale.float()).clamp(-limit, limit) + limit).to(torch.uint8).numpy().flatten()
    groups = codes.reshape(-1, 8)
    values = np.zeros(len(groups), dtype=np.uint64)
    for i in range(8):
        values |= groups[:, i].astype(np.uint64) << (i * bits)
    packed = np.empty((len(groups), bits), dtype=np.uint8)
    for i in range(bits):
        packed[:, i] = (values >> (i * 8)).astype(np.uint8)
    header = json.dumps({"shape": list(weight.shape), "bits": bits, "group": 128, "scale": "fp16"}).encode()
    return (b"MEQG0128" + struct.pack("<I", len(header)) + header + packed.tobytes()
            + scale.numpy().astype("<f2").tobytes())


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(826)

    def test_streamed_uniform_matches_legacy_every_byte(self):
        weight = torch.randn(67, 256).half()
        weight[0] = 0
        weight[1] *= 1e-7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.uniform"
            for bits in (2, 3, 4, 5, 6, 8):
                report = write_uniform(path, weight, bits, device="cpu", rows_per_chunk=13)
                self.assertEqual(path.read_bytes(), legacy_uniform_bytes(weight, bits))
                self.assertTrue(report["disk_roundtrip_fp16_equal"])
                self.assertEqual(report["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                decoded = read_uniform(path)
                rows = list(iter_uniform(path, chunk_rows=11))
                self.assertTrue(torch.equal(decoded, torch.cat([v for _, v in rows])))
                self.assertTrue(torch.equal(decoded[9:41], read_uniform_rows(path, 9, 41)))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_uniform(path)

    def test_streamed_huffman_exact_restore_and_random_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            shape = (512, 128)
            indices = torch.randint(0, 65536, (shape[0], shape[1] // 8), dtype=torch.int32)
            indices[0] = 0
            payload = {"indices": indices, "balance": torch.ones(shape[1]).half(),
                "input_sign": torch.ones(shape[1], dtype=torch.int8),
                "output_sign": -torch.ones(shape[0], dtype=torch.int8), "scale": torch.tensor(.25)}
            write_e8(raw / "layer0.in_proj.e8", payload, {"shape": shape})
            self.assertTrue(torch.equal(read_e8(raw / "layer0.in_proj.e8")["indices"], indices))
            write_uniform(raw / "embedding.uniform", torch.randn(64, 128), device="cpu", rows_per_chunk=7)
            write_uniform(raw / "lm_head.uniform", torch.randn(64, 128) * .1, device="cpu", rows_per_chunk=9)
            (raw / "manifest.json").write_text('{"complete":true}\n')
            (raw / "other_fp16.pt").write_bytes(b"fixture-only\0\1\2")
            container = root / "model.bin"
            with mock.patch("mamba_e8w5.huffman.BATCH_SYMBOLS", 4096):
                pack_directory(raw, container)
                report = verify_container(container, trusted_directory=raw)
                reader = Reader(container)
                self.assertEqual(report["files_verified"], 5)
                self.assertEqual(report["independent_chunks_verified"], 7)
                self.assertTrue(report["trusted_source_bound"])
                for path in raw.iterdir():
                    self.assertEqual(reader.get(path.name), path.read_bytes())
                reader.restore(root / "restored")
                self.assertTrue(np.array_equal(reader.get_chunk("layer0.in_proj.e8", 1), indices.numpy().reshape(-1)[1024:2048]))
                for path in raw.iterdir():
                    self.assertEqual((root / "restored" / path.name).read_bytes(), path.read_bytes())
                position = reader.files["layer0.in_proj.e8"]["payload_offset"]
                with container.open("r+b") as f:
                    f.seek(position)
                    first = f.read(1)
                    f.seek(position)
                    f.write(bytes([first[0] ^ 128]))
                with self.assertRaises(ValueError):
                    verify_container(container, trusted_directory=raw)

    @unittest.skipUnless(os.environ.get("E8_TEST_GPU") == "1" and torch.cuda.is_available(), "Explicit small GPU gate")
    def test_e8_gpu_quantize_disk_decode(self):
        cb, ldlq = load_reference_primitives()
        cb = cb.cuda()
        weight = torch.randn(128, 128, device="cuda").half()
        samples = torch.randn(128, 512, device="cuda")
        hessian = samples @ samples.T / samples.shape[1]
        for width in (128, 4096, 8192, 18560):
            sample = torch.randn(4, width, device="cuda")
            error = (rotate_last(rotate_last(sample), inverse=True) - sample).norm() / sample.norm()
            self.assertLess(float(error), 3e-6, f"width={width}")
        restored, info, payload = vector_quantize(weight, hessian, cb, ldlq, 1000, tune_iters=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.e8"
            write_e8(path, payload, info)
            self.assertTrue(torch.equal(restored, decode_e8(read_e8(path), cb)))
            raw = Path(directory) / "vocab.uniform"
            report = write_uniform(raw, torch.randn(33, 128).half(), device="cuda", rows_per_chunk=7)
            self.assertTrue(report["disk_roundtrip_fp16_equal"])


if __name__ == "__main__":
    unittest.main()
