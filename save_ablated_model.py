"""Bake the refusal direction ablation into the weights and save the model.

inference.py removes the refusal direction at runtime with forward hooks. That
only lives as long as the process does. This script instead rewrites every
weight matrix that writes into the residual stream, so the resulting directory
is a normal model that any runtime (Transformers, vLLM, llama.cpp converters,
...) can load without extra code.

For a matrix W whose output lands in the residual stream, the edit is the
projection W <- (I - r r^T) W. The layer can then no longer emit any component
along r, which is exactly what the runtime hook enforced.

The conversion streams the safetensors shards one at a time and never
instantiates the model, so peak memory is the size of a single shard rather than
the size of the checkpoint. That is what makes it usable on models far larger
than RAM + VRAM.

compressed-tensors checkpoints (nvfp4/mxfp4-pack-quantized) are handled, but a
residual writer that is stored in 4 bits cannot stay in 4 bits: MXFP4 has a
round-trip error around 11%, which is far larger than the refusal component
being removed, so re-packing the ablated matrix puts most of that component
straight back. Measured on Kimi K3, re-quantizing restores ~70% of it and
iterating the projection does not converge. Those matrices are therefore
decompressed to dense and added to the quantization config's `ignore` list,
which every compressed-tensors runtime already understands. In the usual layout
this affects only the routed expert output projection, because attention output
projections, shared experts, dense MLPs and the embeddings are normally on the
`ignore` list to begin with.
"""

import json
import math
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig

# Shared with the loading path: model_loading.py has to repair the same
# over-broad `targets` regex before `from_pretrained`, or the checkpoint cannot
# be loaded at all.
from model_loading import (format_duration, format_memory,
                           ignore_dense_targets, leaf_pattern,
                           module_paths, narrow_targets, select_modules)
MODEL_ID = "../Kimi-K3"
OUTPUT_DIR = "../Kimi-K3-Abliterated"

REFUSAL_DIR_PATH = MODEL_ID.replace("/", "_") + "_refusal_dir.pt"

# Where the projection itself is computed. One tensor is resident at a time, so
# even a 100k x 100k matrix is nothing for a single GPU. The job is bound by
# disk throughput, not by arithmetic, which is why a second GPU would not help.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Print the list of tensors that would be rewritten and stop. Only the
# safetensors headers are read, so this is instant even on a multi-terabyte
# checkpoint. Worth doing once before committing to a full rewrite.
DRY_RUN = False

# Numeric format the ablated residual writers are written back in. The ablation
# itself is always computed on the *dequantized* weight in float32, whatever the
# checkpoint stores; this only decides how the result is encoded again.
#
#   None / "checkpoint"          dense, in the checkpoint's own dtype (recommended)
#   "bf16" / "fp16" / "fp32"     dense, in that dtype
#   "keep"                       back into the checkpoint's own quantized format
#   "mxfp4" / "nvfp4" / "mxfp8"  back into that quantized format
#
# Keeping a residual writer quantized costs most of the ablation, and it gets
# worse as models get wider, not better. The edit is only
# `alignment / sqrt(hidden_size)` of the matrix in relative terms, and the
# alignment measures ~1.0 on every model tried (Qwen3 0.6B-8B, Kimi K3), so the
# edit is essentially 1/sqrt(hidden_size): 3.3% at hidden 1024, 1.6% at 4096,
# ~1.2% at Kimi K3's 7168. A format whose round-trip error exceeds that simply
# rounds the edit away, and the 4-bit formats are an order of magnitude over it:
#
#   format   round-trip error   of the refusal component it puts back
#   MXFP4              11.3%    73-92%
#   NVFP4               9.5%    57-85%
#   MXFP8               2.7%    16-34%
#   dense bf16             -    ~0.3%   (the runtime's own rounding)
#
# Picking a quantized format other than the checkpoint's own also produces a
# `mixed-precision` checkpoint, which transformers cannot load with CPU
# offloading; see split_group. The residual report printed after every
# conversion measures the real cost for the format that was picked.
WRITER_PRECISION = None

# Repair a quantization group whose `targets` regex also selects non-Linear
# modules. Such a checkpoint cannot be loaded at all, ablated or not.
FIX_QUANTIZATION_TARGETS = True

# Last name component of the linear layers whose output is added to the
# residual stream, across the architectures this repo has been used with
# (llama/qwen/gemma `o_proj`+`down_proj`, falcon/neox `dense`+`dense_4h_to_h`,
# qwen1 `c_proj`, kimi `routed_expert_up_proj`, ...).
WRITER_SUFFIXES = (
    "o_proj",
    "out_proj",
    "down_proj",
    "wo",
    "c_proj",
    "dense",
    "dense_4h_to_h",
    "routed_expert_up_proj",
)

EMBEDDING_SUFFIXES = ("embed_tokens", "wte", "word_embeddings", "embed_in")

PROJECTOR_HINTS = ("mm_projector", "multi_modal_projector", "projector")


def resolve_source(model_id):
    """Local directory holding the checkpoint, downloading it first if needed."""
    path = Path(model_id)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download
    print(f"{model_id} is not a local directory, fetching it from the Hub")
    return Path(snapshot_download(model_id))


def hidden_size(config):
    for cfg in (getattr(config, "text_config", None), config):
        size = getattr(cfg, "hidden_size", None) if cfg is not None else None
        if size is not None:
            return size
    raise AttributeError("Could not determine hidden_size from the config")


def tied_embeddings(config):
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is not None and getattr(cfg, "tie_word_embeddings", None):
            return True
    return False


def natural_key(name):
    """Sort key that orders `proj.2` before `proj.10`."""
    return [int(part) if part.isdigit() else part for part in name.split(".")]


def projector_root(name):
    """Return the enclosing multimodal projector of `name`, if there is one."""
    parts = name.split(".")
    for index, part in enumerate(parts):
        if any(hint in part for hint in PROJECTOR_HINTS):
            return ".".join(parts[:index + 1])
    return None


# --- compressed-tensors ------------------------------------------------------
# A quantized Linear does not store a usable `<module>.weight`. The packed
# formats put the values in `<module>.weight_packed`; the fp8 ones do keep a
# `<module>.weight`, but it holds the values *divided by* `<module>.weight_scale`
# and in a dtype no matmul accepts. Either way the real weight only exists after
# the scale has been applied, which is what `decompress_weight` is for: ablating
# the stored tensor directly would project the scaled values instead of the real
# ones, and the per-group scale varies along the output axis, so the two are not
# the same operation.

COMPANION_SUFFIXES = ("weight", "weight_packed", "weight_scale",
                      "weight_global_scale", "weight_zero_point", "weight_shape")

# Quantized output formats this script can write. The arguments have to match
# the format: MX formats are E8M0 (`scale_dtype=uint8`) block scales over 32
# values, and the element type follows `num_bits`.
QUANTIZED_PRECISIONS = {
    "mxfp4": ("mxfp4-pack-quantized",
              dict(num_bits=4, type="float", strategy="group", group_size=32,
                   symmetric=True, scale_dtype=torch.uint8,
                   observer="memoryless_minmax")),
    "mxfp8": ("mxfp8-quantized",
              dict(num_bits=8, type="float", strategy="group", group_size=32,
                   symmetric=True, scale_dtype=torch.uint8,
                   observer="memoryless_minmax")),
    "nvfp4": ("nvfp4-pack-quantized",
              dict(num_bits=4, type="float", strategy="tensor_group",
                   group_size=16, symmetric=True,
                   scale_dtype=torch.float8_e4m3fn,
                   observer="memoryless_minmax")),
}

DENSE_PRECISIONS = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                    "fp16": torch.float16, "float16": torch.float16,
                    "fp32": torch.float32, "float32": torch.float32}


def load_compressor(config):
    """Return (compressor, scheme, format, config dict) if the checkpoint is compressed."""
    quantization = None
    for cfg in (config, getattr(config, "text_config", None)):
        quantization = getattr(cfg, "quantization_config", None) if cfg else None
        if quantization is not None:
            break
    if quantization is None:
        return None, None, None, None
    if not isinstance(quantization, dict):
        quantization = quantization.to_dict()
    if quantization.get("quant_method") != "compressed-tensors":
        return None, None, None, None
    if quantization.get("quantization_status") != "compressed":
        return None, None, None, quantization

    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    groups = quantization.get("config_groups") or {}
    if len(groups) != 1:
        raise NotImplementedError(
            f"the checkpoint has {len(groups)} quantization groups; this script "
            "assumes a single one, extend it to pick the right scheme per module")
    group = next(iter(groups.values()))

    scheme = QuantizationScheme(
        targets=group.get("targets", ["Linear"]),
        weights=QuantizationArgs(**group["weights"]))
    fmt = group.get("format") or quantization.get("format")
    compressor = type(BaseCompressor.load_from_registry(fmt))
    print(f"Checkpoint is compressed-tensors / {fmt}")
    return compressor, scheme, fmt, quantization


def decompress_weight(compressor, scheme, parts):
    """Dense weight of a quantized module, from its stored parts."""
    local = {name.rsplit(".", 1)[-1]: value for name, value in parts.items()}
    return compressor.decompress(local, scheme)["weight"]


def build_writer(precision, config, compressor, scheme, checkpoint_format):
    """How the ablated residual writers get encoded again.

    Returns (encode, decode, label, target_format, target_args). `target_format`
    is None for a dense output, in which case no `weight_scale` is produced at
    all and the module has to leave the quantization config; otherwise the scale
    is recomputed from the ablated weight, because the old one describes the old
    values. `target_args` is the serialized QuantizationArgs the new format
    needs, for the config group that has to describe it.
    """
    key = precision.lower() if isinstance(precision, str) else precision

    if key in (None, "checkpoint", "dense") or key in DENSE_PRECISIONS:
        if key in DENSE_PRECISIONS:
            dtype = DENSE_PRECISIONS[key]
        else:
            dtype = getattr(config, "dtype", None) or torch.bfloat16
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)

        def encode(weight):
            return {"weight": weight.to(dtype)}

        def decode(local):
            return local["weight"].to(torch.float32)

        return (encode, decode, f"dense {str(dtype).removeprefix('torch.')}",
                None, None)

    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.utils.helpers import (
        compute_dynamic_scales_and_zp, generate_gparam)

    if key == "keep":
        if compressor is None:
            raise ValueError('WRITER_PRECISION="keep" but this checkpoint is not '
                             "quantized; pick a dense dtype instead")
        target_format, target_compressor, target_scheme = (
            checkpoint_format, compressor, scheme)
    elif key in QUANTIZED_PRECISIONS:
        target_format, kwargs = QUANTIZED_PRECISIONS[key]
        target_compressor = type(BaseCompressor.load_from_registry(target_format))
        target_scheme = QuantizationScheme(
            targets=["Linear"], weights=QuantizationArgs(**kwargs))
    else:
        raise ValueError(
            f"unknown WRITER_PRECISION {precision!r}; use None, "
            + ", ".join(sorted(DENSE_PRECISIONS)) + ", 'keep' or "
            + ", ".join(sorted(QUANTIZED_PRECISIONS)))

    args = target_scheme.weights
    # NVFP4 puts a per-tensor fp32 scale on top of its fp8 block scales, so that
    # the block scales can use the whole fp8 range.
    tensor_group = str(args.strategy).endswith("tensor_group")

    def encode(weight):
        # Every scale here is derived from the ablated weight. The stored ones
        # describe the values before the projection, and this is the only place
        # a `weight_scale` is produced at all: a dense output never gets one.
        state = {"weight": weight}
        global_scale = None
        if tensor_group:
            low, high = torch.aminmax(weight)
            global_scale = generate_gparam(low, high).to(weight.device)
            state["weight_global_scale"] = global_scale
        scale, zero_point = compute_dynamic_scales_and_zp(
            weight, args, None, global_scale=global_scale)
        state["weight_scale"] = scale
        if not args.symmetric:
            state["weight_zero_point"] = zero_point
        return target_compressor.compress(state, target_scheme)

    def decode(local):
        return target_compressor.decompress(local, target_scheme)["weight"].to(
            torch.float32)

    return (encode, decode, target_format, target_format,
            args.model_dump(mode="json"))


def add_to_ignore(quantization, modules, all_modules):
    """Extend the quantization `ignore` list to cover the decompressed modules.

    Prefers one regex per distinct layer name, but only if that regex matches
    exactly the modules that were decompressed; otherwise the names are listed
    verbatim so nothing else is accidentally excluded from quantization.
    """
    ignore = list(quantization.get("ignore") or [])
    targeted = set(modules)

    suffixes = {name.rsplit(".", 1)[-1] for name in targeted}
    patterns = [f"re:.*{re.escape(suffix)}.*" for suffix in sorted(suffixes)]
    covered = {name for name in all_modules
               if any(re.search(p[3:], name) for p in patterns)}

    ignore.extend(patterns if covered == targeted else sorted(targeted))
    quantization["ignore"] = ignore
    return patterns if covered == targeted else sorted(targeted)


def split_group(quantization, requantized, others, paths, target_format,
                target_args):
    """Give the re-encoded writers a config group of their own.

    They no longer share a format with the rest of the checkpoint, and
    compressed-tensors resolves the format per group, so the single group has to
    become two and the top-level format becomes `mixed-precision`. Both
    selections are verified against the module names; a regex that cannot be
    made exact is an error rather than a checkpoint that loads wrong.
    """
    import copy

    groups = quantization["config_groups"]
    if len(groups) != 1:
        raise NotImplementedError(
            "cannot split a quantization config that already has "
            f"{len(groups)} groups")
    name, group = next(iter(groups.items()))
    ignore = quantization.get("ignore") or []

    moved = leaf_pattern(requantized)
    kept = leaf_pattern(others)
    if select_modules([moved], paths, ignore) != set(requantized):
        raise NotImplementedError(
            f"the re-encoded writers cannot be selected by their leaf names "
            f"({moved}); they share a name with a module that keeps the old "
            "format, so the two groups would overlap")
    if select_modules([kept], paths, ignore) != set(others):
        raise NotImplementedError(
            f"the modules that keep {quantization.get('format')} cannot be "
            f"selected by their leaf names ({kept})")

    new_group = copy.deepcopy(group)
    new_group["targets"] = [moved]
    new_group["format"] = target_format
    # The scheme has to describe the new format, not the old one: a group that
    # says 4-bit while its format is mxfp8 cannot even be compressed again.
    new_group["weights"] = target_args
    activations = new_group.get("input_activations")
    if activations:
        activations["num_bits"] = target_args["num_bits"]
        activations["type"] = target_args["type"]
    group["targets"] = [kept]
    group.setdefault("format", quantization.get("format"))

    groups[f"{name}_ablated"] = new_group
    quantization["format"] = "mixed-precision"
    return (f"{len(requantized)} re-encoded writer(s) moved into a second "
            f"quantization group ({target_format}); the checkpoint format is "
            "now mixed-precision.\n"
            "  NOTE: transformers builds a mixed-precision checkpoint densely "
            "and compresses it afterwards, and the tensor names that produces "
            "do not line up with accelerate's offload index. Such a checkpoint "
            "loads only when it fits in VRAM without CPU offloading. Since "
            "keeping a residual writer quantized costs most of the ablation "
            "anyway, this is an experiment rather than something to ship.")


def writer_axis(shape, size):
    """Which axis of a residual writer's weight lives in the residual stream.

    2D is the `nn.Linear` case, `[out_features, in_features]`. 3D shows up in
    fused mixture-of-experts checkpoints that stack the experts into one tensor;
    there the layout differs between implementations (`[E, in, out]` vs
    `[E, out, in]`), so the hidden dimension is identified by its size instead
    of by position. `None` means "leave this tensor alone".
    """
    if len(shape) == 1:
        return 0 if shape[0] == size else None

    if len(shape) == 2:
        return 0 if shape[0] == size else None

    if len(shape) == 3:
        matches = [axis for axis in (1, 2) if shape[axis] == size]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 2:
            print(f"  ambiguous fused expert shape {tuple(shape)}: both axes "
                  f"are {size} wide, skipping it. Pick the output axis by hand.")
        return None

    return None


def plan_ablation(shapes, size, tied, packed_modules):
    """Map every tensor that writes to the residual stream to its residual axis.

    Everything not in the returned plan is copied through untouched. Names of
    compressed modules appear as `<module>.weight_packed`; their logical shape
    is restored first so the axis is picked from the real matrix.
    """
    plan = {}
    projector_candidates = {}

    for name, shape in shapes.items():
        parts = name.split(".")
        if len(parts) < 2:
            continue
        kind, owner = parts[-1], parts[-2]

        if kind == "weight_packed":
            # two 4-bit values live in every byte of the last axis
            shape = list(shape[:-1]) + [shape[-1] * 2]
            kind = "weight"
        elif kind not in ("weight", "bias"):
            continue

        if kind == "weight" and owner in EMBEDDING_SUFFIXES:
            if len(shape) == 2 and shape[1] == size:
                if tied:
                    # The same tensor is the unembedding; ablating it would
                    # delete the direction from the logits instead of from the
                    # residual stream.
                    print(f"  skipping tied embedding {name}")
                    continue
                plan[name] = 1
            continue

        axis = writer_axis(shape, size)
        if axis is None:
            continue

        if owner in WRITER_SUFFIXES:
            plan[name] = axis
        elif projector_root(name) is not None and kind == "weight":
            # Multimodal projectors feed the text residual stream too, but their
            # output layer is usually just an index inside an nn.Sequential, so
            # it cannot be recognised by name. Only the *last* linear of such a
            # module writes to the residual stream; the earlier ones are
            # internal and must be left alone.
            root = projector_root(name)
            previous = projector_candidates.get(root)
            if previous is None or natural_key(name) > natural_key(previous):
                projector_candidates[root] = name

    for name in projector_candidates.values():
        plan[name] = 0

    return plan


def residual_component(value, axis, direction):
    """The part of `value` that still points along `direction`."""
    if value.dim() == 1:
        return value @ direction
    if value.dim() == 3:
        return value.movedim(axis, -1) @ direction
    if axis == 0:
        return direction @ value
    return value @ direction


def project_out(value, axis, direction):
    """W <- (I - r r^T) W along `axis`, in float32."""
    converted = value.to(device=direction.device, dtype=torch.float32)
    # `.to()` has already copied unless it had nothing to do, and only then does
    # the caller's tensor need protecting from the in-place update below.
    value = converted.clone() if converted is value else converted
    component = residual_component(value, axis, direction)

    if value.dim() == 1:
        # bias of a residual writer
        value -= component * direction
    elif value.dim() == 3:
        # fused experts: same projection, applied inside every expert at once
        value -= torch.tensordot(component, direction, dims=0).movedim(-1, axis)
    elif axis == 0:
        # [out=hidden, in]: drop the component of every column along r. `addr_`
        # is the same rank-1 update as `-= torch.outer(...)`, but it writes
        # straight into the matrix instead of building a second one the same
        # size first, which is what sets the peak of this whole script.
        value.addr_(direction, component, alpha=-1)
    else:
        # [vocab, hidden]: drop the component of every row along r
        value.addr_(component, direction, alpha=-1)
    return value


def measure_residual(value, axis, direction):
    """What is left of the direction: absolute, and relative to an ordinary matrix.

    The relative number is the useful one. A matrix with no relationship to the
    direction scores 1.0 (its component is ||W||_F / sqrt(hidden) by chance
    alone), a perfectly ablated one scores 0.0, and anything near 1.0 means the
    ablation did not survive whatever the weight was encoded into.
    """
    value = value.to(device=direction.device, dtype=torch.float32)
    component = residual_component(value, axis, direction)
    scale = value.norm() / math.sqrt(direction.numel())
    relative = (component.norm() / scale).item() if scale > 0 else 0.0
    return component.abs().max().item(), relative


def orthogonalize(tensor, axis, direction, dtype=None):
    """Project the direction out of a plain dense tensor and round it back.

    Measured on what actually gets stored: rounding the projected matrix back to
    fp16/bf16 puts a little of the direction back, and the honest number is the
    one after that rounding.
    """
    target = dtype or tensor.dtype
    result = project_out(tensor, axis, direction).to(dtype=target, device="cpu")
    return result, measure_residual(result, axis, direction)


print(f"Reading {MODEL_ID}")
source = resolve_source(MODEL_ID)
destination = Path(OUTPUT_DIR)
destination.mkdir(parents=True, exist_ok=True)

shards = sorted(source.glob("*.safetensors"))
if not shards:
    raise FileNotFoundError(
        f"{source} holds no .safetensors shards. Convert the checkpoint to "
        "safetensors first; streaming a pickled .bin is not safe.")

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
size = hidden_size(config)

refusal_dir = torch.load(REFUSAL_DIR_PATH).view(-1).to(torch.float32)
refusal_dir /= refusal_dir.norm()
if refusal_dir.numel() != size:
    raise ValueError(
        f"refusal direction has {refusal_dir.numel()} dims but the model's "
        f"residual stream has {size}; was {REFUSAL_DIR_PATH} computed for "
        f"another model?")
refusal_dir = refusal_dir.to(DEVICE)

# Shapes only: safetensors serves them from the header, nothing is read yet.
shapes = {}
shard_of = {}
for shard in shards:
    with safe_open(shard, framework="pt") as reader:
        for name in reader.keys():
            shapes[name] = reader.get_slice(name).get_shape()
            shard_of[name] = shard

compressor, scheme, checkpoint_format, quantization = load_compressor(config)
# A module is stored quantized when it owns a `weight_scale`: the packed formats
# put the values in `weight_packed`, the fp8 ones keep a `weight` that is the
# values divided by that scale. Both need decoding before they can be ablated.
packed_modules = {name.rsplit(".", 1)[0] for name in shapes
                  if name.endswith((".weight_packed", ".weight_scale"))}
if packed_modules and compressor is None:
    raise RuntimeError(
        "the checkpoint holds quantized weights but the config does not "
        "describe a compressed-tensors quantization; cannot decode them")

plan = plan_ablation(shapes, size, tied_embeddings(config), packed_modules)
if not plan:
    raise RuntimeError(
        "Found no residual stream writers. This architecture names its output "
        "projections differently; add them to WRITER_SUFFIXES.")

(encode_writer, decode_writer, precision_label, target_format,
 target_args) = build_writer(WRITER_PRECISION, config, compressor, scheme,
                             checkpoint_format)

# Residual writers that are stored quantized: the ablation is computed on their
# decoded weight and the result is re-encoded per WRITER_PRECISION. Left in a
# format whose round-trip error exceeds the size of the edit, the ablation is
# simply rounded away, which is why the default takes them out of the quantized
# format entirely.
quantized_writers = sorted({name.rsplit(".", 1)[0] for name in plan
                            if name.rsplit(".", 1)[0] in packed_modules})
to_dequantize = quantized_writers if target_format is None else []
primary_of = {}
companions = {}
for module in quantized_writers:
    companions[module] = [f"{module}.{suffix}" for suffix in COMPANION_SUFFIXES
                          if f"{module}.{suffix}" in shapes]
    primary_of[module] = (f"{module}.weight_packed"
                          if f"{module}.weight_packed" in shapes
                          else f"{module}.weight")
    # The replacement is emitted into the shard holding the primary tensor, so
    # any companion sitting in an earlier shard would already be written out.
    primary_shard = shards.index(shard_of[primary_of[module]])
    for name in companions[module]:
        if shards.index(shard_of[name]) < primary_shard:
            raise NotImplementedError(
                f"{name} lives in an earlier shard than {primary_of[module]}; "
                "the single-pass writer cannot handle that interleaving")

counts = {}
for name, axis in plan.items():
    key = (name.split(".")[-2], axis, name.rsplit(".", 1)[0] in packed_modules)
    counts[key] = counts.get(key, 0) + 1
for (owner, axis, quantized), count in sorted(counts.items()):
    note = f" [{checkpoint_format} -> {precision_label}]" if quantized else ""
    print(f"  {count:5d} x *.{owner} (axis {axis}){note}")
print(f"{len(plan)} of {len(shapes)} tensors will be ablated, "
      f"across {len(shards)} shard(s), on {DEVICE}")

if DRY_RUN:
    print("DRY_RUN is set, stopping before writing anything.")
    raise SystemExit

residuals = []
emitted = {}
size_before = size_after = 0

# Reading and writing the whole checkpoint is what this job is, so the bar
# tracks bytes rather than shards: on a 1.5TB model a single shard is minutes,
# and GiB/s is the number that says whether the storage or something else is the
# limit.
total_bytes = sum(shard.stat().st_size for shard in shards)
bar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
           desc="Rewriting", smoothing=0.1)
started = time.perf_counter()

for shard in shards:
    bar.set_postfix_str(shard.name)
    with safe_open(shard, framework="pt") as reader:
        metadata = reader.metadata()
        tensors = {}
        for name in reader.keys():
            tensor = reader.get_tensor(name)
            size_before += tensor.numel() * tensor.element_size()

            module = name.rsplit(".", 1)[0]
            if name == primary_of.get(module):
                # Pull the whole quantized module together, wherever its parts
                # live, decode it, ablate the real weight and encode it again.
                parts = {}
                for part in companions[module]:
                    if shard_of[part] == shard:
                        parts[part] = reader.get_tensor(part)
                    else:
                        with safe_open(shard_of[part], framework="pt") as other:
                            parts[part] = other.get_tensor(part)
                axis = plan[name]

                dense = decompress_weight(compressor, scheme, parts).to(DEVICE)
                before = measure_residual(dense, axis, refusal_dir)[1]
                encoded = encode_writer(project_out(dense, axis, refusal_dir))
                after = measure_residual(decode_writer(encoded), axis, refusal_dir)

                residuals.append((module, axis, before, after))
                emitted[module] = [f"{module}.{suffix}" for suffix in encoded]
                for suffix, value in encoded.items():
                    tensors[f"{module}.{suffix}"] = value.cpu()
                continue

            if module in companions:
                continue  # folded into the re-encoded weight above

            if name in plan:
                before = measure_residual(tensor, plan[name], refusal_dir)[1]
                tensor, after = orthogonalize(tensor, plan[name], refusal_dir)
                residuals.append((module, plan[name], before, after))
            tensors[name] = tensor

    size_after += sum(t.numel() * t.element_size() for t in tensors.values())
    save_file(tensors, destination / shard.name, metadata=metadata)
    del tensors
    bar.update(shard.stat().st_size)

bar.close()
elapsed = time.perf_counter() - started
print(f"Rewrote {format_memory(total_bytes)} in {format_duration(elapsed)} "
      f"({format_memory((total_bytes + size_after) / elapsed)}/s of read+write)")

# --- residual report ---------------------------------------------------------
# Measured on the tensors that were actually written, so it includes whatever
# the chosen precision rounded back in. `relative` is the number to read: 1.0 is
# as much of the direction as a matrix that was never ablated, 0.0 is none.

groups_by_leaf = {}
for module, axis, before, (absolute, relative) in residuals:
    stored = precision_label if module in primary_of else "as stored"
    key = (module.rsplit(".", 1)[-1], axis, stored)
    groups_by_leaf.setdefault(key, []).append((before, relative))

print(f"\nResidual refusal component, measured on what was written "
      f"(1.0 = a matrix that was never ablated):")
print(f"  {'writer':<24} {'written as':<22} {'n':>3} {'before':>15} "
      f"{'after':>19} {'removed':>8}")
for (leaf, axis, stored), rows in sorted(groups_by_leaf.items()):
    before = [row[0] for row in rows]
    after = [row[1] for row in rows]
    removed = 1 - sum(after) / sum(before) if sum(before) else 1.0
    print(f"  *.{leaf + f' [{axis}]':<22} {stored:<22} {len(rows):>3} "
          f"{min(before):>6.3f}..{max(before):<7.3f} "
          f"{min(after):>8.2e}..{max(after):<8.2e} {removed:>7.2%}")

worst_relative = max((row[3][1] for row in residuals), default=0.0)
worst_leak = max((row[3][0] for row in residuals), default=0.0)
print(f"  worst remaining component: {worst_relative:.2e} relative, "
      f"{worst_leak:.2e} absolute")
if worst_relative > 0.05:
    if target_format is not None:
        print(f"  WARNING: {precision_label} rounds most of the ablation away, "
              "because its round-trip error is larger than the edit. Set "
              "WRITER_PRECISION to a dense dtype (None) to keep the ablation.")
    else:
        print("  WARNING: much of the refusal direction survived even though the "
              "writers were written densely. The projection did not take; check "
              "that the refusal direction belongs to this model.")

if to_dequantize:
    delta = (size_after - size_before) / 2 ** 30
    print(f"{len(to_dequantize)} module(s) left the quantized format, "
          f"checkpoint grew by {delta:+.2f} GiB")
elif quantized_writers:
    print(f"{len(quantized_writers)} module(s) re-encoded as {precision_label}")

# Everything that is not a weight shard: config, tokenizer, the shard index and
# the `trust_remote_code` modules.
for path in sorted(source.iterdir()):
    if path.is_dir() or path.suffix == ".safetensors":
        continue
    shutil.copy2(path, destination / path.name)

requantized = [module for module in quantized_writers
               if target_format is not None and target_format != checkpoint_format]

if quantization is not None and (to_dequantize or requantized
                                 or FIX_QUANTIZATION_TARGETS):
    messages = []

    if to_dequantize:
        # Tell the runtime not to expect those modules in the quantized format.
        # First, so that the two steps below see them as already excluded.
        added = add_to_ignore(quantization, to_dequantize, packed_modules)
        messages.append("Added to the quantization ignore list: " + ", ".join(added))

    if FIX_QUANTIZATION_TARGETS:
        # A `targets` of ["Linear"] selects Linears the exporter left dense, and
        # compressed-tensors then fills them with noise rather than failing. The
        # source checkpoint's own config is not rewritten by this script, so the
        # output has to carry the exclusion or it inherits the bug.
        dense_fix = ignore_dense_targets(quantization, set(shapes))
        if dense_fix:
            pattern, modules = dense_fix
            messages.append(
                f"`targets` also selected {len(modules)} module(s) that this "
                f"checkpoint stores dense (e.g. {sorted(modules)[0]}), whose "
                f"weights would have been replaced by noise. Excluded them "
                f"with: {pattern}")

        # Some exports lose the `Linear` restriction and leave a bare regex that
        # also selects containers and activations, which breaks loading. Before
        # any group is added, since it only handles a single group.
        fixed = narrow_targets(quantization, packed_modules - set(to_dequantize),
                               module_paths(shapes))
        if fixed:
            narrowed, over_matched = fixed
            messages.append(
                f"`targets` also selected {len(over_matched)} module(s) that are "
                f"not quantized (e.g. {sorted(over_matched)[0]}), which makes "
                f"the checkpoint fail to load. Narrowed it to: {narrowed}")

    if requantized:
        # The writers now use a different format from the rest of the
        # checkpoint, which compressed-tensors expresses as a second config
        # group and a `mixed-precision` top-level format.
        messages.append(split_group(quantization, requantized,
                                    packed_modules - set(requantized),
                                    module_paths(shapes), target_format,
                                    target_args))

    if messages:
        config_path = destination / "config.json"
        raw = json.loads(config_path.read_text())
        holder = raw
        if "quantization_config" in raw.get("text_config", {}):
            holder = raw["text_config"]
        holder["quantization_config"] = quantization
        config_path.write_text(json.dumps(raw, indent=2))
        for message in messages:
            print(message)

index_path = destination / "model.safetensors.index.json"
if index_path.exists() and quantized_writers:
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    for module in quantized_writers:
        shard_name = weight_map[primary_of[module]]
        for part in companions[module]:
            weight_map.pop(part, None)
        for part in emitted[module]:
            weight_map[part] = shard_name
    index.setdefault("metadata", {})["total_size"] = size_after
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Rewrote model.safetensors.index.json for the {precision_label} writers")

print(f"Saved to {OUTPUT_DIR}")
print("Load it like any other model:")
print(f'    AutoModelForCausalLM.from_pretrained("{OUTPUT_DIR}", trust_remote_code=True)')
