"""Strict original-parent overlay replacing only the existing FP16 norm values.

Inference needs the original quantized parent and this two-file overlay. Source
weights, training checkpoints and calibration files are optional audit inputs,
never hidden inference dependencies. No frozen runtime/codec code is modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "MAMBA2_E8W5_NORM_OVERLAY_V1"
PARENT_MANIFEST_SHA256 = "ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed"
PROTOCOL_SHA256 = "125cf066981e0b014c6cb1eb3cfd01608d63a9b7271185038b4dea30938308a3"
CALIBRATION_TOKENS_SHA256 = "ebeb62135d074ba41fcb44cc64f8fe8aa09f452dfd28183c1b28f84645e6f577"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def selected_norms():
    result = {}
    for layer in range(56):
        result[f"backbone.layers.{layer}.norm.weight"] = {"shape": [4096], "numel": 4096}
        result[f"backbone.layers.{layer}.mixer.norm.weight"] = {"shape": [8192], "numel": 8192}
    result["backbone.norm_f.weight"] = {"shape": [4096], "numel": 4096}
    return result


def other_shapes():
    result = {name: entry["shape"] for name, entry in selected_norms().items()}
    for layer in range(56):
        prefix = f"backbone.layers.{layer}.mixer."
        for suffix in ("dt_bias", "A_log", "D"):
            result[prefix + suffix] = [128]
        result[prefix + "conv1d.weight"] = [10240, 1, 4]
        result[prefix + "conv1d.bias"] = [10240]
    return result


def tensor_sha(value):
    if value.dtype != torch.float16:
        raise ValueError("Norm overlay requires FP16 tensor bytes")
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().astype("<f2", copy=False).tobytes()).hexdigest()


def bitwise_equal(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.detach().cpu().contiguous().view(torch.uint8),
                        right.detach().cpu().contiguous().view(torch.uint8)))


def compare_other_tensors(parent, candidate):
    """Validate the entire393-key payload, including signed-zero identities."""
    expected, selected = other_shapes(), selected_norms()
    if set(parent) != set(expected) or set(candidate) != set(expected):
        raise ValueError("FP16 payload must contain exactly the original393 architectural keys")
    changed, unchanged, hashes = [], [], {}
    for name, shape in expected.items():
        before, after = parent[name], candidate[name]
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise ValueError(f"Expected plain tensor values: {name}")
        if before.dtype != torch.float16 or after.dtype != torch.float16 or list(before.shape) != shape or list(after.shape) != shape:
            raise ValueError(f"FP16 dtype/shape changed: {name}")
        if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all()):
            raise ValueError(f"Nonfinite FP16 payload: {name}")
        equal = bitwise_equal(before, after)
        if not equal and name not in selected:
            raise ValueError(f"Forbidden non-norm tensor changed: {name}")
        (unchanged if equal else changed).append(name)
        hashes[name] = tensor_sha(after)
    return {"tensor_count": len(expected), "selected_norm_tensor_count": len(selected),
            "selected_norm_parameter_count": sum(entry["numel"] for entry in selected.values()),
            "protected_non_norm_tensor_count": len(expected) - len(selected),
            "actual_changed_norm_keys": sorted(changed), "actual_changed_norm_tensor_count": len(changed),
            "unchanged_tensor_count": len(unchanged), "candidate_fp16_tensor_sha256": hashes}


def checked_file(directory, name, entry):
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"Unsafe filename: {name}")
    path = Path(directory) / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
        raise ValueError(f"File integrity mismatch: {name}")
    return path


def verify_final_checkpoint(checkpoint, exported, binding, smoke_sha256):
    """Bind the actual FP16 export to final-step FP32 masters and accounting."""
    generator = torch.Generator(device="cpu").manual_seed(20260927)
    schedule = [index for _ in range(4) for index in torch.randperm(32, generator=generator).tolist()]
    overflows, attempts = checkpoint.get("overflow_retries"), checkpoint.get("attempts")
    if (checkpoint.get("format") != "MAMBA2_NORM_TRAINING_CHECKPOINT_V1"
            or checkpoint.get("binding") != binding
            or checkpoint.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
            or checkpoint.get("smoke_report_sha256") != smoke_sha256
            or checkpoint.get("schedule") != schedule
            or checkpoint.get("successful_updates") != 128
            or type(overflows) is not int or not 0 <= overflows <= 8
            or attempts != 128 + overflows):
        raise ValueError("Final checkpoint provenance or update accounting differs")
    masters = checkpoint.get("masters", {})
    if set(masters) != set(selected_norms()):
        raise ValueError("Final checkpoint norm master inventory differs")
    for name, entry in selected_norms().items():
        master = masters[name]
        if (not isinstance(master, torch.Tensor) or master.dtype != torch.float32
                or list(master.shape) != entry["shape"] or not torch.isfinite(master).all()
                or not bitwise_equal(master.half(), exported[name])):
            raise ValueError(f"Export differs from rounded final FP32 master: {name}")
    return {"final_step": 128, "rounded_master_tensors_verified": 113,
            "rounded_master_parameters_verified": 692224, "overflow_retries": overflows,
            "attempts": attempts}


def verify_overlay(parent_dir, overlay_dir, *, calibration_manifest=None, calibration_tokens=None,
                   protocol=None, training_report=None, training_checkpoint=None, smoke_report=None,
                   expected_hyperparameters=None):
    """CPU-only inventory/content verification, with optional training audit."""
    parent_dir, overlay_dir = Path(parent_dir), Path(overlay_dir)
    pp, op = parent_dir / "manifest.json", overlay_dir / "manifest.json"
    if sha(pp) != PARENT_MANIFEST_SHA256:
        raise ValueError("Norm repair must start from the original two-sweep parent")
    parent, overlay = json.loads(pp.read_text()), json.loads(op.read_text())
    if not parent.get("complete") or not overlay.get("complete") or overlay.get("format") != FORMAT:
        raise ValueError("Requires a completed original parent and norm overlay")
    if overlay.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256:
        raise ValueError("Overlay parent identity differs")
    if parent["model_config"] != MODEL_CONFIG or overlay.get("model_config") != MODEL_CONFIG:
        raise ValueError("Model configuration differs")
    if parent["binding"]["tune_iters"] != 2:
        raise ValueError("Parent projection recipe differs")
    if parent.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256 or parent.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("Parent source/tokenizer differs")
    for name, digest in parent["binding"]["code_sha256"].items():
        if sha(ROOT / "mamba_e8w5" / name) != digest:
            raise ValueError(f"Frozen pipeline changed: {name}")
    for name, digest in parent["binding"]["quip_sha256"].items():
        if sha(ROOT / "third_party/quip-sharp" / name) != digest:
            raise ValueError(f"Frozen QuIP source changed: {name}")
    binding = overlay["binding"]
    required_binding = {"source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "hessian_manifest_sha256": parent["binding"]["hessian_manifest_sha256"],
        "norm_overlay_source_sha256": sha(__file__),
        "protocol_sha256": PROTOCOL_SHA256, "calibration_tokens_sha256": CALIBRATION_TOKENS_SHA256}
    for name, digest in required_binding.items():
        if binding.get(name) != digest:
            raise ValueError(f"Overlay binding differs: {name}")
    for key in ("calibration_tokens_sha256", "protocol_sha256", "trainer_source_sha256"):
        value = binding.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing provenance hash: {key}")
    for key, path in (("trainer_source_sha256", ROOT / "scripts/train_norm_compensation.py"),
                      ("training_helper_source_sha256", ROOT / "mamba_e8w5/norm_training.py")):
        if binding.get(key) != sha(path):
            raise ValueError(f"Training implementation differs: {key}")
    if not isinstance(binding.get("hyperparameters"), dict) or not binding["hyperparameters"]:
        raise ValueError("Missing fixed hyperparameters")
    from .norm_training import HYPERPARAMETERS
    if binding["hyperparameters"] != HYPERPARAMETERS:
        raise ValueError("Training hyperparameters differ from the frozen training implementation")
    if expected_hyperparameters is not None and binding["hyperparameters"] != expected_hyperparameters:
        raise ValueError("Training hyperparameters differ from the declared protocol")
    if overlay.get("selected_norms") != selected_norms():
        raise ValueError("Selected norm inventory must be exactly113 tensors/692224 parameters")
    if overlay.get("parameter_mapping") != parent["parameter_mapping"]:
        raise ValueError("Full model parameter mapping differs")
    inherited = {name: entry for name, entry in parent["files"].items() if name != "other_fp16.pt"}
    expected_files = {f"layer{i}.{part}.e8" for i in range(56) for part in ("in_proj", "out_proj")}
    expected_files.update(("embedding.uniform", "lm_head.uniform", "e8_codebook.bin", "config.json"))
    if set(inherited) != expected_files or overlay.get("inherited_files") != inherited:
        raise ValueError("Inherited116-file ledger differs")
    if set(overlay.get("files", {})) != {"other_fp16.pt"}:
        raise ValueError("Norm overlay replaces only other_fp16.pt")
    if {p.name for p in overlay_dir.iterdir()} != {"manifest.json", "other_fp16.pt"}:
        raise ValueError("Overlay must contain only manifest.json and other_fp16.pt")
    if {p.name for p in parent_dir.iterdir()} != set(parent["files"]) | {"manifest.json"}:
        raise ValueError("Parent has unlisted files")
    for name, entry in parent["files"].items():
        checked_file(parent_dir, name, entry)
    candidate_path = checked_file(overlay_dir, "other_fp16.pt", overlay["files"]["other_fp16.pt"])
    before = torch.load(parent_dir / "other_fp16.pt", map_location="cpu", weights_only=True)
    after = torch.load(candidate_path, map_location="cpu", weights_only=True)
    tensor_receipt = compare_other_tensors(before, after)
    if len(parent["parameter_mapping"]) != 507 or parent["total_parameter_count"] != 8236999680:
        raise ValueError("Parent full parameter coverage differs")
    if {key for key, value in parent["parameter_mapping"].items() if value == "other_fp16.pt"} != set(before):
        raise ValueError("Parent FP16 parameter mapping differs")
    training = overlay.get("training_receipt", {})
    if training.get("final_step") != 128:
        raise ValueError("Only the final128-update checkpoint may be evaluated")
    for key in ("checkpoint_sha256", "report_sha256", "smoke_report_sha256"):
        value = training.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing final training receipt: {key}")
    audit = {}
    if calibration_manifest is not None:
        calibration_manifest = Path(calibration_manifest)
        if sha(calibration_manifest) != binding["hessian_manifest_sha256"]:
            raise ValueError("Calibration manifest differs")
        calibration = json.loads(calibration_manifest.read_text())
        if (not calibration.get("complete") or calibration.get("calibration_split") != "train"
                or calibration.get("evaluation_data_used") is not False
                or calibration["source_checkpoint_sha256"] != SOURCE_CHECKPOINT_SHA256
                or calibration["dataset"]["tokenizer_sha256"] != TOKENIZER_SHA256
                or calibration["token_windows_file_sha256"] != binding["calibration_tokens_sha256"]):
            raise ValueError("Calibration is not the original train-only token input")
        audit["calibration_manifest_verified"] = True
    if calibration_tokens is not None:
        if sha(Path(calibration_tokens)) != binding["calibration_tokens_sha256"]:
            raise ValueError("Training token file differs")
        audit["calibration_tokens_verified"] = True
    if protocol is not None:
        if sha(Path(protocol)) != binding["protocol_sha256"]:
            raise ValueError("Training protocol differs")
        audit["protocol_verified"] = True
    if training_report is not None:
        training_report = Path(training_report)
        if sha(training_report) != training["report_sha256"]:
            raise ValueError("Training report differs")
        tr = json.loads(training_report.read_text())
        if (not tr.get("complete") or tr.get("final_step") != 128 or tr.get("binding") != binding
                or tr.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
                or tr.get("selected_norms") != selected_norms()
                or tr.get("final_checkpoint", {}).get("sha256") != training["checkpoint_sha256"]):
            raise ValueError("Final training report provenance differs")
        generator = torch.Generator(device="cpu").manual_seed(20260927)
        schedule = [index for _ in range(4) for index in torch.randperm(32, generator=generator).tolist()]
        overflows, attempts = tr.get("overflow_retries"), tr.get("attempts")
        if (tr.get("schedule") != schedule or tr.get("successful_updates") != 128
                or type(overflows) is not int or not 0 <= overflows <= 8
                or attempts != 128 + overflows
                or tr.get("successful_target_exposures") != 262016
                or tr.get("attempted_target_exposures") != attempts * 2047):
            raise ValueError("Final training schedule or exposure/update accounting differs")
        audit["training_report_verified"] = True
    if training_checkpoint is not None:
        if sha(Path(training_checkpoint)) != training["checkpoint_sha256"]:
            raise ValueError("Final training checkpoint differs")
        checkpoint = torch.load(training_checkpoint, map_location="cpu", weights_only=True)
        audit["checkpoint_export_receipt"] = verify_final_checkpoint(checkpoint, after, binding, training["smoke_report_sha256"])
        audit["training_checkpoint_verified"] = True
    if smoke_report is not None:
        smoke_report = Path(smoke_report)
        if sha(smoke_report) != training["smoke_report_sha256"]:
            raise ValueError("Successful smoke receipt differs")
        smoke = json.loads(smoke_report.read_text())
        if (smoke.get("complete") is not True or smoke.get("passed") is not True
                or smoke.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
                or smoke.get("binding") != binding):
            raise ValueError("Successful smoke input/code binding differs")
        audit["successful_smoke_verified"] = True
    resolved = {name: {"origin": "parent", **entry} for name, entry in inherited.items()}
    resolved["other_fp16.pt"] = {"origin": "overlay", **overlay["files"]["other_fp16.pt"]}
    logical = sum(entry["bytes"] for entry in resolved.values())
    return parent, overlay, {"parent_manifest_sha256": sha(pp), "overlay_manifest_sha256": sha(op),
        "binding": binding, "training_receipt": training, "provenance_audit": audit,
        "tensor_receipt": tensor_receipt, "inherited_files": inherited, "resolved_files": resolved,
        "resolved_file_ledger_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "resolved_parameter_tensors": 507, "resolved_parameter_count": 8236999680,
        "logical_candidate_data_bytes": logical, "parent_raw_payload_bytes": parent["data_file_bytes"],
        "overlay_data_file_bytes": candidate_path.stat().st_size,
        "other_fp16_tensor_payload_bytes": sum(value.numel() * value.element_size() for value in after.values()),
        "serialized_replacement_byte_delta": candidate_path.stat().st_size - parent["files"]["other_fp16.pt"]["bytes"],
        "manifest_bytes": {"parent": pp.stat().st_size, "overlay": op.stat().st_size},
        "physical_parent_plus_overlay_directory_bytes": parent["data_file_bytes"] + pp.stat().st_size + candidate_path.stat().st_size + op.stat().st_size,
        "storage_note": "Logical raw bytes include all116 inherited files and replacement393-tensor FP16 payload. Physical inference experiment retains parent plus overlay. Tokenizer, software/licenses, reports and training/source/calibration files are separate."}


@torch.no_grad()
def apply_overlay(model, overlay_dir, overlay):
    """Reload exported FP16 values, replace113 norms, verify all393 small tensors."""
    if overlay.get("format") != FORMAT or not overlay.get("complete"):
        raise ValueError("Incomplete norm overlay")
    if model._package_receipt["manifest_sha256"] != PARENT_MANIFEST_SHA256:
        raise ValueError("Loaded model is not the original two-sweep parent")
    parameters = dict(model.named_parameters())
    if len(parameters) != 507 or sum(p.numel() for p in parameters.values()) != 8236999680:
        raise ValueError("Loaded model parameter coverage differs")
    path = checked_file(overlay_dir, "other_fp16.pt", overlay["files"]["other_fp16.pt"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    before = {name: parameters[name].detach().cpu().half() for name in other_shapes()}
    receipt = compare_other_tensors(before, payload)
    identities = {name: id(value) for name, value in parameters.items()}
    for name in selected_norms():
        owner, leaf = name.rsplit(".", 1)
        value = payload[name].to(device=parameters[name].device, dtype=torch.float16)
        setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(value, requires_grad=False))
    hashes = {}
    for name in other_shapes():
        value = model.get_parameter(name)
        if not bitwise_equal(value, payload[name]):
            raise ValueError(f"Reloaded FP16 model tensor differs: {name}")
        hashes[name] = tensor_sha(value)
    if any(id(model.get_parameter(name)) != old for name, old in identities.items() if name not in selected_norms()):
        raise ValueError("Overlay replaced an unrelated parameter object")
    if any(p.dtype != torch.float16 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Candidate contains non-FP16 or nonfinite parameters")
    return {**receipt, "reloaded_fp16_tensor_sha256": hashes, "unchanged_parameter_objects": 394,
            "resolved_parameter_tensors": len(parameters),
            "resolved_parameter_count": sum(p.numel() for p in model.parameters()),
            "coverage": "113 norm tensors reloaded from exportedFP16; 280 protected small tensors checked bitwise;112 E8 and2 W5 parameter objects retained"}


def load_norm_model(parent_dir, overlay_dir, device="cuda"):
    """Load the resolved FP16 candidate using no teacher or training checkpoint."""
    from .runtime import load_quantized_model
    _, overlay, receipt = verify_overlay(parent_dir, overlay_dir)
    model = load_quantized_model(parent_dir, device=device, dtype=torch.float16)
    application = apply_overlay(model, overlay_dir, overlay)
    model._package_receipt = {"format": "Original E8/W5 parent plus FP16 norm overlay",
        "parent_manifest_sha256": receipt["parent_manifest_sha256"],
        "overlay_manifest_sha256": receipt["overlay_manifest_sha256"],
        "resolved_file_ledger_sha256": receipt["resolved_file_ledger_sha256"],
        "parameter_count": receipt["resolved_parameter_count"],
        "verified_file_bytes": receipt["logical_candidate_data_bytes"]}
    model._norm_overlay_application = application
    return model.eval().requires_grad_(False)
