"""Public Mamba-2 runtime, exact NVIDIA tokenizer, and E8/W5 reference loading.

Execution uses state-spaces/mamba (Apache-2.0). There is no Resurface dependency.
The Megatron key mapping follows NVIDIA's public checkpoint layout and Quamba's
public conversion recipe. Quantized reference loading expands weights to FP16;
it is a quality reference, not a claim about compressed runtime residency.
"""
from __future__ import annotations

import argparse
import enum
import gc
import hashlib
import importlib
import json
import platform
import re
from pathlib import Path

import torch
from torch import nn

SOURCE_CHECKPOINT_SHA256 = "47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb"
TOKENIZER_SHA256 = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
TOKENIZER_FILENAME = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
MODEL_CONFIG = {
    "d_model": 4096, "d_intermediate": 0, "n_layer": 56, "vocab_size": 256000,
    "ssm_cfg": {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2,
                "headdim": 64, "ngroups": 8, "chunk_size": 128,
                "rmsnorm": True, "norm_before_gate": False,
                "use_mem_eff_path": False},
    "rms_norm": True, "residual_in_fp32": False, "fused_add_norm": False,
    "pad_vocab_size_multiple": 128, "tie_embeddings": False,
}
_VERIFIED_FILES = {}


class _MegatronModelType(enum.Enum):
    # Passive metadata enum from NVIDIA Megatron-LM core_r0.10.0/core/enums.py.
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def environment_receipt():
    import importlib.metadata
    packages = {}
    for name in ("torch", "mamba-ssm", "numpy", "triton", "datasets", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}


def gpu_memory_receipt():
    if not torch.cuda.is_available():
        return None
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "current_allocated_bytes": torch.cuda.memory_allocated(),
            "note": "Decoded FP16 quality reference, not compressed model residency"}


def checkpoint_path(source_dir):
    directory = Path(source_dir)
    if directory.is_file():
        return directory
    for relative in ("model_optim_rng.pt", "release/mp_rank_00/model_optim_rng.pt"):
        candidate = directory / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"NVIDIA checkpoint missing under {directory}")


def normalize_source_key(key):
    """Anchored mapping: decoder.layers.1 must never match layer 10 or 11."""
    special = {"embedding.word_embeddings.weight": "backbone.embedding.weight",
               "decoder.final_norm.weight": "backbone.norm_f.weight",
               "output_layer.weight": "lm_head.weight"}
    if key in special:
        return special[key]
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", key)
    if match:
        return f"backbone.layers.{int(match[1])}.{match[2]}"
    # The loader can inspect an already mapped state without changing its keys.
    if key.startswith("backbone.") or key == "lm_head.weight":
        return key.replace("backbone.embeddings.", "backbone.embedding.", 1)
    raise ValueError(f"Unrecognized model tensor key: {key}")


def normalize_source_state(state):
    result = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor model entry {name}: {type(value).__name__}")
        target = normalize_source_key(name)
        if target in result:
            raise ValueError(f"Duplicate normalized tensor: {target}")
        result[target] = value
    return result


def load_source_state(source_dir, expected_sha256=SOURCE_CHECKPOINT_SHA256):
    """Load a pinned official CPU checkpoint without an unrestricted unpickler.

    Megatron stores an argparse namespace, a model-kind enum, and NumPy RNG state.
    Only those inspected passive metadata types are additionally allowed.
    Unexpected pickle globals fail closed. mmap avoids a second disk copy.
    """
    path = checkpoint_path(source_dir)
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    digest = _VERIFIED_FILES.get(cache_key)
    if digest is None:
        digest = sha256_file(path)
        _VERIFIED_FILES[cache_key] = digest
    if digest != expected_sha256:
        raise ValueError(f"Source checkpoint SHA-256 mismatch: {digest}")
    import numpy as np
    reconstruct = importlib.import_module("numpy.core.multiarray")._reconstruct
    allowed = [argparse.Namespace, np.ndarray, np.dtype, type(np.dtype("uint32")),
               (reconstruct, "numpy.core.multiarray._reconstruct"),
               (_MegatronModelType, "megatron.core.enums.ModelType")]
    try:
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        raise RuntimeError(f"Safe checkpoint loading failed; inspect globals {unsafe}") from error
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    result = normalize_source_state(state)
    del checkpoint
    return result


def make_model(config=None, device="meta", dtype=torch.float16):
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    config = json.loads(json.dumps(MODEL_CONFIG if config is None else config))
    model = MambaLMHeadModel(MambaConfig(**config), device=device, dtype=dtype)
    for block in model.backbone.layers:
        block.mixer.use_mem_eff_path = False
        expected_group_size = block.mixer.d_ssm // block.mixer.ngroups
        if block.mixer.norm.group_size != expected_group_size:
            raise RuntimeError("Runtime does not implement grouped gated RMSNorm")
    return model.eval().requires_grad_(False)


def load_source_model(source_dir, device="cuda", dtype=torch.float16,
                      expected_sha256=SOURCE_CHECKPOINT_SHA256):
    model = make_model()
    state = load_source_state(source_dir, expected_sha256=expected_sha256)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device=device, dtype=dtype)
    if model.backbone.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise RuntimeError("8B source requires independent embedding and output head")
    model._package_receipt = {"format": "original BF16 checkpoint cast to requested dtype",
                              "source_checkpoint_sha256": expected_sha256,
                              "parameter_count": sum(p.numel() for p in model.parameters())}
    return model.eval().requires_grad_(False)


def _set_parameter(model, name, tensor, device, dtype):
    parent, leaf = name.rsplit(".", 1)
    module = model.get_submodule(parent)
    original = getattr(module, leaf)
    if original.shape != tensor.shape:
        raise ValueError(f"Wrong tensor shape for {name}: {tensor.shape} != {original.shape}")
    setattr(module, leaf, nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False))


def load_quantized_model(raw_dir, device="cuda", dtype=torch.float16, quip_source=None):
    """Load an E8/W5 raw package into ordinary FP16 Linear/Embedding modules."""
    from .codec import (load_reference_primitives, read_e8, decode_e8, iter_uniform)
    directory = Path(raw_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Refusing incomplete quantized package")
    config = manifest.get("model_config", MODEL_CONFIG)
    if config != MODEL_CONFIG:
        raise ValueError("Package configuration differs from audited NVIDIA Mamba-2 8B")
    source_binding = manifest.get("binding", {}).get("source", {})
    source_sha = manifest.get("source_checkpoint_sha256", source_binding.get("checkpoint_sha256"))
    tokenizer_sha = manifest.get("tokenizer_sha256", source_binding.get("tokenizer_sha256"))
    if source_sha != SOURCE_CHECKPOINT_SHA256 or tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError("Package source or tokenizer does not match the pinned original")
    files = manifest.get("files", {})
    required = {f"layer{i}.{which}.e8" for i in range(config["n_layer"])
                for which in ("in_proj", "out_proj")}
    required.update(("embedding.uniform", "lm_head.uniform", "other_fp16.pt", "e8_codebook.bin", "config.json"))
    if not required.issubset(files):
        raise ValueError(f"Missing raw package file ledger entries: {required-set(files)}")
    for filename, receipt in files.items():
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe package filename: {filename}")
        path = directory / filename
        if path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"Package file verification failed: {filename}")
    if json.loads((directory / "config.json").read_text()) != config:
        raise ValueError("Stored configuration does not match manifest")
    if config.get("tie_embeddings", False):
        raise ValueError("Mamba-2 8B package must retain independent vocabulary matrices")
    model = make_model(config=config)
    codebook, _ = load_reference_primitives(quip_source)
    expected_book = codebook.grid_packed_abs.cpu().numpy().astype("<i4").tobytes()
    if (directory / "e8_codebook.bin").read_bytes() != expected_book:
        raise ValueError("Stored E8 codebook differs from pinned decoder codebook")
    codebook = codebook.to(device)
    covered = set()
    for index in range(config["n_layer"]):
        for which in ("in_proj", "out_proj"):
            label = f"layer{index}.{which}"
            name = f"backbone.layers.{index}.mixer.{which}.weight"
            payload = read_e8(directory / f"{label}.e8")
            decoded = decode_e8(payload, codebook, device=device)
            _set_parameter(model, name, decoded, device, dtype)
            covered.add(name)
            del payload, decoded
        if (index + 1) % 8 == 0:
            print(f"[FP16 reference load] {index+1}/{config['n_layer']} layers", flush=True)
    for filename, name in (("embedding.uniform", "backbone.embedding.weight"),
                           ("lm_head.uniform", "lm_head.weight")):
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        shape = getattr(module, leaf).shape
        output = torch.empty(shape, device=device, dtype=dtype)
        next_row = 0
        for start, rows in iter_uniform(directory / filename, device=device):
            if start != next_row:
                raise ValueError(f"Noncontiguous W5 rows in {filename}")
            output[start:start + len(rows)].copy_(rows)
            next_row += len(rows)
        if next_row != shape[0]:
            raise ValueError(f"Incomplete W5 rows in {filename}")
        _set_parameter(model, name, output, device, dtype)
        covered.add(name)
        del output, rows
    other = torch.load(directory / "other_fp16.pt", map_location="cpu", weights_only=True)
    for name, value in other.items():
        if name in covered:
            raise ValueError(f"Overlapping quantized and FP16 tensor: {name}")
        _set_parameter(model, name, value, device, dtype)
        covered.add(name)
    expected = set(dict(model.named_parameters()))
    if covered != expected:
        raise ValueError(f"Tensor coverage mismatch: missing={expected-covered}; extra={covered-expected}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("Unmaterialized parameter in decoded runtime")
    del other, codebook
    gc.collect()
    model._package_receipt = {"format": "E8/W5 raw package decoded to FP16",
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "source_checkpoint_sha256": source_sha, "tokenizer_sha256": tokenizer_sha,
        "verified_files": files, "verified_file_bytes": sum(item["bytes"] for item in files.values()),
        "parameter_count": sum(p.numel() for p in model.parameters())}
    return model.eval().requires_grad_(False)


class SentencePieceTokenizer:
    """NVIDIA GPTSentencePiece token IDs without automatic BOS/EOS insertion."""
    def __init__(self, path):
        import sentencepiece as spm
        path = Path(path)
        if path.is_dir():
            path = path / TOKENIZER_FILENAME
        self.path = path
        self.processor = spm.SentencePieceProcessor(model_file=str(path))
        self.eos_token_id = self.processor.eos_id()
        self.bos_token_id = self.processor.bos_id()
        self.pad_token_id = self.processor.pad_id()
        self.vocab_size = self.processor.vocab_size()
        self.sha256 = sha256_file(path)
        if self.sha256 != TOKENIZER_SHA256:
            raise ValueError("Tokenizer does not match the pinned original NVIDIA SentencePiece model")

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("This protocol never inserts BOS/EOS automatically")
        return self.processor.encode_as_ids(text)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.eos_token_id, self.bos_token_id, self.pad_token_id)]
        return self.processor.decode_ids(ids)


def make_cache(model, max_seqlen, batch_size=1, state_dtype=torch.float16):
    """Native one-token cache: state is rounded to the requested dtype each step."""
    from mamba_ssm.utils.generation import InferenceParams
    cache = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
    cache.key_value_memory_dict = model.allocate_inference_cache(
        batch_size, max_seqlen, dtype=state_dtype)
    return cache


def cache_bytes(cache):
    return sum(tensor.numel() * tensor.element_size()
               for pair in cache.key_value_memory_dict.values() for tensor in pair)


@torch.inference_mode()
def backbone_tokenwise(model, ids, state_dtype=torch.float16):
    cache = make_cache(model, ids.shape[1], ids.shape[0], state_dtype)
    result = []
    for position in range(ids.shape[1]):
        cache.seqlen_offset = position
        result.append(model.backbone(ids[:, position:position+1], inference_params=cache))
    return torch.cat(result, dim=1), cache


load_model = load_source_model
load_quantized = load_quantized_model
