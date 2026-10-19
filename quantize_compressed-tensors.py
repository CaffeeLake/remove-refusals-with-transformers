"""Quantize a checkpoint into a compressed-tensors format.

This streams the safetensors shards one at a time, replaces every selected
projection with the tensors its format stores (`weight_packed` / `weight_scale`
for the packed formats, `weight` / `weight_scale` for MXFP8), and writes a
`quantization_config` that describes the result. The model is never
instantiated, so peak memory is one shard rather than the whole checkpoint --
the same property that makes save_ablated_model.py usable on checkpoints larger
than RAM.

`SELECTION` decides which modules are quantized, because the two architectures
this has been used on want opposite things:

* `"routed_experts"` -- a MoE checkpoint (Kimi K3, Hy-Memory-30B-A3B), where the
  experts are nearly all the parameters and everything else writes to the
  residual stream or decides which expert runs. The notes below are about this
  case.
* `"all_projections"` -- a dense checkpoint, where leaving the attention and MLP
  projections in bf16 would quantize nothing at all. This is what NVIDIA's own
  Dory1.0-4B MXFP8 and NVFP4 builds do, and `KEEP_WRITERS_DENSE` is the one
  place this departs from them; see its comment.

The scheme follows moonshotai/Kimi-K3: MXFP4 weights (E2M1 elements, one E8M0
power-of-two scale per 32 values along the input axis), `targets: ["Linear"]`,
and only the routed experts quantized. Attention, the dense MLP, the shared
expert, the router and `lm_head` stay bf16, because they are the modules whose
output lands in the residual stream or decides which expert runs at all, and
they are 5.6% of the parameters.

Note that Kimi-K3's own config has `input_activations: null`, i.e. it is
*weight-only* MXFP4 and not the "MXFP4 weights / MXFP8 activations" it is
sometimes described as. `ACTIVATION_PRECISION` below can request MXFP8 or MXFP4
activations anyway, and because those schemes are dynamic they store nothing in
the checkpoint: the safetensors are byte-for-byte the same either way and only
config.json changes. It can therefore be flipped afterwards without re-running
the conversion -- but a runtime that has no W4A8-MX kernel will refuse the
checkpoint, which is why the default is Kimi-K3's own `null`.

Three things had to be adjusted from the Kimi-K3 layout, and all three are
silent failures rather than errors if they are left alone:

1. transformers renames and fuses this architecture's MoE on load.
   `mlp.experts.{0..127}.{gate,up}_proj.weight` become one `nn.Parameter`
   `mlp.experts.gate_up_proj`, `mlp.shared_mlp.*` becomes `mlp.shared_experts.*`
   and `mlp.router.gate` becomes `mlp.gate`. The `ignore` list is matched
   against the *built module* names, so Kimi-K3's `re:.*shared_experts.*`
   matches nothing here (the checkpoint calls it `shared_mlp`), the shared
   expert is then selected by `targets: ["Linear"]`, and transformers
   initialises the `weight_packed` it expects -- but does not find -- with
   noise. Measured end to end, that produces `nan` logits on the first forward
   pass while still "loading successfully". `ignore` therefore has to cover both
   spellings, which is what `TRANSFORMERS_ALIASES` is for.

2. The routed experts are `nn.Parameter`s after that fusion, not `nn.Linear`s,
   so `targets: ["Linear"]` does not select them and compressed-tensors
   quantizes nothing at all. transformers still decodes the packed tensors while
   merging them, so the checkpoint loads and runs correctly -- but it is resident
   as dense bf16. Under transformers this format is a download- and disk-size
   win only, not a VRAM win; that needs a runtime with a fused-MoE MXFP4 kernel
   (vLLM, SGLang). The dense/packed sizes printed at the end are the honest
   numbers for each.

3. Hy-Memory-30B-A3B uses the leaf names `gate_proj` / `up_proj` / `down_proj`
   in three different places (the dense layer-0 MLP, the shared expert and the
   routed experts), so the leaf-name patterns that model_loading.py's
   `ignore_dense_targets` and Kimi-K3's own config rely on cannot separate them:
   ignoring the dense `down_proj` would also un-quantize every expert's
   `down_proj`. `exclusion_patterns` therefore grows the pattern from the leaf
   towards the root until it separates the two sets, and refuses to write a
   config it cannot prove is exact.

The round-trip error of MXFP4 is ~11.2% on these weights, which is the format's
own floor rather than anything about the scale policy: compressed-tensors picks
`2 ** floor(log2(amax / 6))` per group, which is the error-minimising power of
two for 96% of the groups here and within 0.03 points of a per-group search
overall. A model whose weights were edited by less than that -- an ablated
residual writer, say -- must not be packed, which is why save_ablated_model.py
dequantizes those instead; see its WRITER_PRECISION notes and
`KEEP_WRITERS_DENSE` below.

On matching NVIDIA's own Dory builds. Their MXFP8 is compressed-tensors
`mxfp8-quantized`, which is what this writes: decoding their checkpoint with the
compressor used here reproduces their bf16 weights to 2.66%, i.e. the formats
are the same down to the bytes. Their NVFP4 is *not* compressed-tensors but
modelopt (`quant_algo: W4A16_NVFP4`, plus an `hf_quant_config.json`), and the
difference is in the names and one convention rather than in the arithmetic:
`weight` holds what compressed-tensors calls `weight_packed`, `weight_scale_2`
holds the per-tensor scale as the *reciprocal* of `weight_global_scale`, and
with those two substitutions their checkpoint decodes at 8.15%, the expected
NVFP4 floor. This writes the compressed-tensors spelling because transformers
has no modelopt loader at all, so a modelopt-layout checkpoint could be served
by vLLM but could not be loaded -- or KL-measured -- by anything in this repo.
"""

import json
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig

from model_loading import (expert_width, format_duration, format_memory,
                           ignore_dense_targets, matches, module_paths,
                           narrow_targets, select_modules, tensor_sizes)

MODEL_ID = "../NVIDIA-Nemotron-Labs-Dory1.0-4B-Abliterated"
OUTPUT_DIR = "../NVIDIA-Nemotron-Labs-Dory1.0-4B-Abliterated-MXFP8"

# Format the selected weights are packed into. All three are compressed-tensors
# formats that a `quantization_config` can name; MXFP4 is Kimi-K3's, and MXFP8
# and NVFP4 are the two NVIDIA publishes Dory1.0-4B in.
#
#   format   bits/weight   round-trip error
#   mxfp4        4.25           11.2%
#   nvfp4        4.5             8.2%
#   mxfp8        8.25            2.7%
WEIGHT_PRECISION = "mxfp8"

# Which modules get quantized.
#
#   "routed_experts"   a MoE checkpoint's routed experts and nothing else. The
#                      experts are 94% of Kimi-K3's parameters, and everything
#                      left dense either writes to the residual stream or picks
#                      which expert runs.
#   "all_projections"  every attention and MLP projection, i.e. what NVIDIA's
#                      Dory builds quantize. A dense architecture has no experts
#                      to select, so the other setting would quantize nothing.
#
# Either way the embeddings and `lm_head` stay dense, which is also what both
# official Dory builds do (`ignore: ["backbone.embeddings", "lm_head"]`).
SELECTION = "all_projections"

# Leave the residual writers dense, and on the `ignore` list, rather than
# quantizing them with everything else.
#
# This is the one deliberate difference from NVIDIA's own builds, and it only
# matters for an *ablated* checkpoint. The ablation `W <- (I - r r^T)W` is a
# relative edit of about `1/sqrt(hidden_size)`, which is 1.98% at Dory's 2560,
# and a format's round-trip error lands isotropically -- so whatever that error
# is, that much of the refusal direction comes back. Measured on the
# Dory1.0-4B abliterated writers, as the share of an un-ablated matrix's
# component that is left (save_ablated_model.py's `relative`, where 1.0 is a
# matrix that was never touched):
#
#     dense/bf16   6.4e-03    the floor: bf16 rounding of the dense edit
#     MXFP8        2.8e-02    4x the floor, under this repo's 5% warning
#     NVFP4        9.7e-02    ~10% of the direction restored
#
# The cost is that the writers are 31% of the projection parameters, so the
# output is larger than the official build: 5.4GiB against 4.4GiB for MXFP8 and
# 4.4GiB against 3.0GiB for NVFP4. Set this to False to match the official
# layout exactly and accept the leak; the residual report at the end measures
# whichever choice was made, and measure_kl.py says whether it changed the
# model's behaviour.
KEEP_WRITERS_DENSE = False

# Last name component of the projections whose output is added to the residual
# stream -- the same list save_ablated_model.py ablates. Only consulted when
# KEEP_WRITERS_DENSE is set.
WRITER_SUFFIXES = ("o_proj", "out_proj", "down_proj", "wo", "c_proj", "dense",
                   "dense_4h_to_h", "routed_expert_up_proj")

# Activation scheme written into the config. None is Kimi-K3's own setting and
# means the activations stay in the model's compute dtype. "mxfp8" / "mxfp4" ask
# for dynamic per-group activation quantization, which stores nothing in the
# checkpoint -- it is a config-only change that any runtime without the matching
# kernel will reject, so it is off by default.
ACTIVATION_PRECISION = None

# Print the plan and the predicted sizes, then stop. Only the safetensors headers
# are read, so this is instant.
DRY_RUN = False

# Where the packing arithmetic runs. 57x faster on a GPU than on CPU here
# (5.8 vs 0.10 Gparam/s), but the job is bound by reading the checkpoint either
# way. One tensor is resident at a time.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Decode every packed tensor again and report the error per projection. Costs a
# second decompression per tensor -- noise next to the I/O -- and is the only
# thing that distinguishes "packed correctly" from "packed into garbage".
MEASURE_ERROR = True

# Correct a `moe_intermediate_size` that disagrees with the shapes in the
# checkpoint. Hy-Memory-30B-A3B ships `expert_hidden_dim: 768` next to a
# `moe_intermediate_size: 1536` that nothing produced, and transformers reads
# only the latter: without this, `from_pretrained` on the output fails with a
# wall of MISMATCH lines (or, with `ignore_mismatched_sizes`, silently
# reinitialises every expert). model_loading.py repairs this in memory for the
# scripts in this repo; writing it into the output config makes the result load
# anywhere without that help.
FIX_EXPERT_WIDTH = True

# A routed expert's projections, per expert or fused across them. `(?:^|\.)`
# before `experts` keeps `shared_experts` out, and nothing matches
# `shared_mlp.*` or the dense `mlp.*_proj` -- those are the modules that must
# stay dense.
ROUTED_EXPERT = re.compile(
    r"(?:^|\.)experts(?:\.\d+)?\."
    r"(?:gate_up_proj|gate_proj|up_proj|down_proj|w1|w2|w3)$")

# Every projection inside a decoder layer, whatever the architecture calls the
# block that holds them (`mixer` on Dory, `self_attn`/`mlp` elsewhere). Anchored
# on the layer index so that nothing outside the stack -- the embeddings, the
# head, a vision tower -- can be selected by accident.
LAYER_PROJECTION = re.compile(
    r"(?:^|\.)(?:layers|h|blocks|block)\.\d+\..*"
    r"(?:q_proj|k_proj|v_proj|g_proj|o_proj|out_proj|qkv_proj|gate_proj"
    r"|up_proj|gate_up_proj|down_proj|dense|dense_4h_to_h|c_proj|wo"
    r"|w1|w2|w3)$")

SELECTIONS = {"routed_experts": ROUTED_EXPERT,
              "all_projections": LAYER_PROJECTION}

# Module names that exist only after the architecture's load-time renaming, so
# they cannot be derived from the checkpoint. `ignore` is matched against the
# built model, so a config that lists only the checkpoint's spelling lets
# `targets: ["Linear"]` select these and fills them with noise.
#
#   checkpoint                  transformers (modeling_hy_v3.py)
#   mlp.shared_mlp.*_proj   ->  mlp.shared_experts.*_proj
#   mlp.router.gate         ->  mlp.gate
#
# Specific to the MoE architectures `SELECTION = "routed_experts"` is for: a
# dense checkpoint is built under the names it ships with, so adding these would
# only put patterns in `ignore` that match nothing.
MOE_TRANSFORMERS_ALIASES = (
    r"re:(?:.*\.)?shared_experts\.(?:gate|up|down|gate_up)_proj$",
    r"re:(?:.*\.)?mlp\.gate$",
)
TRANSFORMERS_ALIASES = (MOE_TRANSFORMERS_ALIASES
                        if SELECTION == "routed_experts" else ())

# Quantization arguments per format. The MX formats are E8M0 power-of-two block
# scales (`scale_dtype=uint8`) over 32 values with the element type following
# `num_bits`; NVFP4 is fp8 scales over 16 values plus a per-tensor fp32 scale,
# which is what `tensor_group` selects. `observer` only records how the scales
# were derived; the values themselves are in the checkpoint.
QUANTIZED_PRECISIONS = {
    "mxfp4": ("mxfp4-pack-quantized",
              dict(num_bits=4, type="float", strategy="group", group_size=32,
                   symmetric=True, scale_dtype=torch.uint8, observer="minmax")),
    "mxfp8": ("mxfp8-quantized",
              dict(num_bits=8, type="float", strategy="group", group_size=32,
                   symmetric=True, scale_dtype=torch.uint8, observer="minmax")),
    "nvfp4": ("nvfp4-pack-quantized",
              dict(num_bits=4, type="float", strategy="tensor_group",
                   group_size=16, symmetric=True,
                   scale_dtype=torch.float8_e4m3fn, observer="minmax")),
}

# Activation schemes, taken from compressed-tensors' own presets rather than
# written out here, so they cannot drift from what the library validates.
ACTIVATION_PRESETS = {"mxfp4": "MXFP4", "mxfp8": "MXFP8", "nvfp4": "NVFP4"}


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


def build_scheme(precision):
    """(format, encode, decode, serialized weight args) for `precision`.

    `encode` turns one dense weight into the tensors the format stores, and
    `decode` turns those back into a dense weight, which is what the error
    report measures. Every scale is derived from the weight being packed; there
    is nothing in the checkpoint to reuse.
    """
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.utils.helpers import (
        compute_dynamic_scales_and_zp, generate_gparam)

    key = str(precision).lower()
    if key not in QUANTIZED_PRECISIONS:
        raise ValueError(f"unknown WEIGHT_PRECISION {precision!r}; use one of "
                         + ", ".join(sorted(QUANTIZED_PRECISIONS)))

    target_format, kwargs = QUANTIZED_PRECISIONS[key]
    args = QuantizationArgs(**kwargs)
    # The scheme handed to the compressor deliberately carries no
    # `input_activations`: the dynamic activation schemes store nothing, so
    # leaving them out keeps the written tensors independent of
    # ACTIVATION_PRECISION.
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    compressor = type(BaseCompressor.load_from_registry(target_format))

    # NVFP4 puts a per-tensor fp32 scale on top of its fp8 block scales so that
    # the block scales can use the whole fp8 range; the MX formats do not.
    tensor_group = str(args.strategy).endswith("tensor_group")

    def encode(weight):
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
        return compressor.compress(state, scheme)

    def decode(encoded):
        return compressor.decompress(encoded, scheme)["weight"]

    return target_format, encode, decode, args.model_dump(mode="json")


def activation_args(precision):
    """Serialized `input_activations` for the config, or None."""
    if precision in (None, "none", "null"):
        return None

    from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES

    key = str(precision).lower()
    if key not in ACTIVATION_PRESETS:
        raise ValueError(f"unknown ACTIVATION_PRECISION {precision!r}; use None "
                         "or one of " + ", ".join(sorted(ACTIVATION_PRESETS)))
    args = PRESET_SCHEMES[ACTIVATION_PRESETS[key]].get("input_activations")
    if args is None:
        raise ValueError(f"compressed-tensors' {ACTIVATION_PRESETS[key]} preset "
                         "has no activation scheme")
    if not args.dynamic:
        # A static activation scale has to be calibrated on real data and
        # stored; this script only reads weights, so it cannot produce one.
        raise ValueError(
            f"{key} activations are static in this version of "
            "compressed-tensors, which needs calibration data and an "
            "`input_global_scale` in the checkpoint. Use llmcompressor for that.")
    return args.model_dump(mode="json")


def tail_pattern(name, depth):
    """A `re:` pattern matching the last `depth` components of `name`.

    Numeric components become `\\d+`, so one pattern covers every layer and every
    expert index. Anchored at the end, and at the start too when `name` is not
    deeper than `depth`, so it cannot be widened by accident.
    """
    parts = name.split(".")
    tail = parts[-depth:]
    body = r"\.".join(r"\d+" if part.isdigit() else re.escape(part)
                      for part in tail)
    prefix = "" if len(parts) <= depth else r"(?:.*\.)?"
    return f"re:{prefix}{body}$"


def exclusion_patterns(dense, packed, paths):
    """`ignore` entries that cover every module in `dense` and none in `packed`.

    The leaf name alone is enough on most architectures, and that is what
    model_loading.py's `ignore_dense_targets` and Kimi-K3's own config use. It is
    not enough here: `gate_proj` is the name of a routed expert's projection, of
    the shared expert's and of the dense layer-0 MLP's, so the shortest pattern
    that separates them needs the parent too. The pattern is grown one component
    at a time and the first depth that selects exactly `dense` wins.
    """
    for depth in range(1, 6):
        patterns = sorted({tail_pattern(name, depth) for name in dense})
        selected = select_modules(patterns, paths)
        if selected == dense:
            return patterns, depth
    raise RuntimeError(
        "could not build an `ignore` list that separates the modules this "
        "script packs from the ones it leaves dense; they share too much of "
        "their path (e.g. "
        f"{sorted(packed & select_modules(patterns, paths))[:1]}). "
        "Write `ignore` by hand.")


print(f"Reading {MODEL_ID}")
source = resolve_source(MODEL_ID)
destination = Path(OUTPUT_DIR)

shards = sorted(source.glob("*.safetensors"))
if not shards:
    raise FileNotFoundError(
        f"{source} holds no .safetensors shards. Convert the checkpoint to "
        "safetensors first; streaming a pickled .bin is not safe.")

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
size = hidden_size(config)

# Shapes and dtypes only: safetensors serves them from the header, so nothing is
# read off the disk yet and the plan below costs nothing even at a terabyte.
shapes = {}
dtypes = {}
for shard in shards:
    with safe_open(shard, framework="pt") as reader:
        for name in reader.keys():
            view = reader.get_slice(name)
            shapes[name] = tuple(view.get_shape())
            dtypes[name] = view.get_dtype()

if any(name.endswith((".weight_packed", ".weight_scale")) for name in shapes):
    raise RuntimeError(
        f"{MODEL_ID} already holds quantized weights. Re-packing an already "
        "quantized checkpoint compounds the round-trip error; quantize the "
        "dense model instead.")

target_format, encode_weight, decode_weight, weight_args = build_scheme(
    WEIGHT_PRECISION)
input_activations = activation_args(ACTIVATION_PRECISION)
group_size = weight_args["group_size"]

# --- plan -------------------------------------------------------------------
# Which tensors get packed, and why the rest do not. A selected projection
# whose input axis is not a multiple of the group size cannot be packed at all,
# and silently leaving it dense would contradict the config, so it is an error.

if SELECTION not in SELECTIONS:
    raise ValueError(f"unknown SELECTION {SELECTION!r}; use one of "
                     + ", ".join(sorted(SELECTIONS)))
selected = SELECTIONS[SELECTION]

to_pack = []
held_back = []
for name, shape in sorted(shapes.items()):
    module, _, leaf = name.rpartition(".")
    if leaf != "weight" or not selected.search(module):
        continue
    if len(shape) not in (2, 3):
        raise RuntimeError(f"{name} has shape {shape}; expected a 2D projection "
                           "or a 3D stack of them")
    if KEEP_WRITERS_DENSE and module.rpartition(".")[2] in WRITER_SUFFIXES:
        held_back.append(name)
        continue
    if shape[-1] % group_size:
        raise RuntimeError(
            f"{name} has {shape[-1]} input features, which is not a multiple of "
            f"the {group_size}-value group {WEIGHT_PRECISION} scales. This "
            "checkpoint cannot be packed in that format.")
    to_pack.append(name)

if not to_pack:
    raise RuntimeError(
        f"SELECTION = {SELECTION!r} matched nothing in this checkpoint. Its "
        f"modules are named differently; extend {SELECTION.upper()} or "
        "SELECTIONS.")

packed_modules = {name.rpartition(".")[0] for name in to_pack}
# Everything else that owns a real weight has to be on the `ignore` list, or
# `targets: ["Linear"]` selects it, compressed-tensors expects a `weight_packed`
# that is not in the checkpoint, and the module is initialised with noise.
dense_modules = {name.rpartition(".")[0] for name in shapes
                 if name.rpartition(".")[2] == "weight"} - packed_modules

def element_count(shape):
    count = 1
    for dimension in shape:
        count *= dimension
    return count


sizes = tensor_sizes({name: (shape, dtypes[name])
                      for name, shape in shapes.items()})

# Bits per weight: the element itself, plus one scale per group. The scale is a
# byte in every format here -- E8M0 for the MX ones, fp8 for NVFP4 -- so MXFP4
# is 4 + 8/32 = 4.25 bits, NVFP4 4 + 8/16 = 4.5 and MXFP8 8.25.
packed_bits = weight_args["num_bits"] + 8 / group_size
dense_total = sum(sizes.values())
packed_params = sum(element_count(shapes[name]) for name in to_pack)
packed_total = (dense_total
                - sum(sizes[name] for name in to_pack)
                + int(packed_params * packed_bits / 8))
all_params = sum(element_count(shapes[name]) for name in shapes)

counts = {}
for name in to_pack:
    counts[name.split(".")[-2]] = counts.get(name.split(".")[-2], 0) + 1
for leaf, count in sorted(counts.items()):
    print(f"  {count:5d} x *.{leaf}")
if held_back:
    kept = {}
    for name in held_back:
        kept[name.split(".")[-2]] = kept.get(name.split(".")[-2], 0) + 1
    print("  held back as residual writers, left dense (KEEP_WRITERS_DENSE): "
          + ", ".join(f"{count} x *.{leaf}" for leaf, count in sorted(kept.items()))
          + f" -- {sum(element_count(shapes[name]) for name in held_back) / 1e9:.2f}B "
            "parameters that would otherwise be quantized")
print(f"{len(to_pack)} of {len(shapes)} tensors will be packed as "
      f"{target_format}: {packed_params / 1e9:.2f}B of {all_params / 1e9:.2f}B "
      f"parameters ({packed_params / all_params:.1%}), "
      f"{format_memory(dense_total)} -> {format_memory(packed_total)} on disk, "
      f"across {len(shards)} shard(s), on {DEVICE}")
print(f"  {len(dense_modules)} module(s) stay dense and go on the `ignore` list")
if input_activations is not None:
    print(f"  activations: {input_activations['num_bits']}-bit "
          f"{input_activations['type']}, {input_activations['strategy']}/"
          f"{input_activations['group_size']}, dynamic -- nothing is stored for "
          "this, it only tells the runtime what to do")
else:
    print("  activations: not quantized (Kimi-K3's own setting)")

# --- quantization config ----------------------------------------------------
# Built before anything is written, so a config that cannot be proven exact
# fails the run instead of producing a checkpoint that loads wrong.

paths = module_paths(shapes)
ignore, depth = exclusion_patterns(dense_modules, packed_modules, paths)
ignore = ignore + [pattern for pattern in TRANSFORMERS_ALIASES
                   if pattern not in ignore]

uncovered = {module for module in dense_modules
             if not any(matches(pattern, module) for pattern in ignore)}
shadowed = {module for module in packed_modules
            if any(matches(pattern, module) for pattern in ignore)}
if uncovered:
    raise RuntimeError(
        f"{len(uncovered)} dense module(s) are not covered by `ignore` "
        f"(e.g. {sorted(uncovered)[0]}); `targets: [\"Linear\"]` would select "
        "them and their weights would be replaced by noise")
if shadowed:
    raise RuntimeError(
        f"`ignore` covers {len(shadowed)} module(s) this script packs "
        f"(e.g. {sorted(shadowed)[0]}); the runtime would read their packed "
        "weights as if they were dense")

quantization = {
    "config_groups": {
        "group_0": {
            "format": target_format,
            "input_activations": input_activations,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": weight_args,
        },
    },
    "format": target_format,
    "global_compression_ratio": round(dense_total / packed_total, 4),
    "ignore": ignore,
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}

print(f"\n`ignore` ({len(ignore)} entries, {depth} path component(s) deep):")
for pattern in ignore:
    print(f"  {pattern}")

if DRY_RUN:
    print("\nDRY_RUN is set, stopping before writing anything.")
    raise SystemExit

# --- rewrite ----------------------------------------------------------------

destination.mkdir(parents=True, exist_ok=True)
pack_set = set(to_pack)
errors = {}
weight_map = {}
written_bytes = 0

total_bytes = sum(shard.stat().st_size for shard in shards)
bar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
           desc="Packing", smoothing=0.1)
started = time.perf_counter()

for shard in shards:
    bar.set_postfix_str(shard.name)
    with safe_open(shard, framework="pt") as reader:
        metadata = reader.metadata()
        tensors = {}
        for name in reader.keys():
            tensor = reader.get_tensor(name)
            if name not in pack_set:
                tensors[name] = tensor
                continue

            module = name.rpartition(".")[0]
            weight = tensor.to(DEVICE)
            encoded = encode_weight(weight)

            if MEASURE_ERROR:
                reference = weight.to(torch.float32)
                norm = reference.norm()
                relative = ((decode_weight(encoded).to(torch.float32)
                             - reference).norm() / norm).item() if norm else 0.0
                errors.setdefault(name.split(".")[-2], []).append(relative)

            for suffix, value in encoded.items():
                tensors[f"{module}.{suffix}"] = value.cpu()

    for name, value in tensors.items():
        weight_map[name] = shard.name
        written_bytes += value.numel() * value.element_size()
    save_file(tensors, destination / shard.name, metadata=metadata)
    del tensors
    bar.update(shard.stat().st_size)

bar.close()
elapsed = time.perf_counter() - started
print(f"Read {format_memory(total_bytes)} and wrote "
      f"{format_memory(written_bytes)} in {format_duration(elapsed)} "
      f"({format_memory((total_bytes + written_bytes) / elapsed)}/s)")

# --- error report -----------------------------------------------------------
# Measured on the tensors that were actually written. MXFP4's floor is ~11% on
# weights of this kind; a number far below it means the plan matched something
# that was not a weight, and one far above it means the scales are wrong.

if errors:
    print(f"\nRelative round-trip error of the packed weights "
          f"(||dequantized - original|| / ||original||):")
    print(f"  {'projection':<24} {'n':>6} {'min':>8} {'median':>8} {'max':>8}")
    for leaf, values in sorted(errors.items()):
        values = sorted(values)
        median = values[len(values) // 2]
        print(f"  *.{leaf:<22} {len(values):>6} {values[0]:>8.4f} "
              f"{median:>8.4f} {values[-1]:>8.4f}")
    worst = max(max(values) for values in errors.values())
    if worst > 0.5:
        print(f"  WARNING: worst error {worst:.3f} -- that is not quantization "
              "noise, the scales or the packing are wrong. Do not ship this.")

# --- config and index -------------------------------------------------------

for path in sorted(source.iterdir()):
    if path.is_dir() or path.suffix == ".safetensors":
        continue
    shutil.copy2(path, destination / path.name)

config_path = destination / "config.json"
raw = json.loads(config_path.read_text())
holder = raw
if isinstance(raw.get("text_config"), dict) and "num_hidden_layers" in raw["text_config"]:
    holder = raw["text_config"]
holder["quantization_config"] = quantization

# A quantized checkpoint has to say what its *compute* dtype is. `dtype="auto"`
# infers it from the checkpoint, and after this conversion the tensor it infers
# from is a uint8 `weight_packed`, so the load dies with "cannot be instantiated
# under dtype=torch.uint8 as it's not a floating-point dtype" -- not on a
# plausible-looking wrong answer, but not on anything obviously to do with
# quantization either. Kimi-K3's config carries `"dtype": "bfloat16"`, which is
# why it never shows this; Hy-Memory-30B-A3B's carries no dtype at all.
if not holder.get("dtype") and not holder.get("torch_dtype"):
    weights = {name: dtypes[name] for name in shapes
               if name.rpartition(".")[2] == "weight"}
    tally = {}
    for name, dtype in weights.items():
        if dtype in ("BF16", "F16", "F32", "F64"):
            tally[dtype] = tally.get(dtype, 0) + sizes[name]
    if tally:
        dominant = max(tally, key=tally.get)
        holder["dtype"] = {"BF16": "bfloat16", "F16": "float16",
                           "F32": "float32", "F64": "float64"}[dominant]
        print(f"\nSet `dtype` to {holder['dtype']}: the source config declares "
              "none, and `dtype=\"auto\"` would infer it from a packed uint8 "
              "tensor and refuse to build the model.")

if FIX_EXPERT_WIDTH:
    declared = holder.get("moe_intermediate_size")
    measured = expert_width({name: (shape, dtypes[name])
                             for name, shape in shapes.items()},
                            size, holder.get("num_experts"))
    if isinstance(declared, int) and measured is not None and measured != declared:
        holder["moe_intermediate_size"] = measured
        print(f"\nSet `moe_intermediate_size` to {measured}: the config says "
              f"{declared} but the checkpoint's expert weights are {measured} "
              "wide. Left alone, loading the output would fail with a MISMATCH "
              "report, or reinitialise every expert.")

config_path.write_text(json.dumps(raw, indent=2) + "\n")

index_path = destination / "model.safetensors.index.json"
if index_path.exists():
    index = json.loads(index_path.read_text())
    index["weight_map"] = dict(sorted(weight_map.items()))
    index.setdefault("metadata", {})["total_size"] = written_bytes
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Rewrote model.safetensors.index.json ({len(weight_map)} tensors)")

# A last check against the repo's own loader-side repairs: if either of them
# would still want to change something, the config is not self-consistent.
probe = deepcopy(quantization)
if ignore_dense_targets(probe, set(weight_map)) is not None:
    raise RuntimeError("the written `ignore` list still leaves dense modules "
                       "selected by `targets`; this is a bug in this script")
if narrow_targets(probe, packed_modules, module_paths(weight_map)) is not None:
    raise RuntimeError("the written `targets` still select unquantized "
                       "modules; this is a bug in this script")

print(f"\nSaved to {OUTPUT_DIR}")
print(f"  on disk            {format_memory(written_bytes)} "
      f"(was {format_memory(dense_total)}, "
      f"{dense_total / written_bytes:.2f}x smaller)")
print(f"  resident, packed   {format_memory(written_bytes)}  "
      f"(a runtime with a {WEIGHT_PRECISION} kernel: vLLM, SGLang)")
if SELECTION == "routed_experts":
    print(f"  resident, unpacked {format_memory(dense_total)}  "
          "(transformers: the experts are merged into one dense nn.Parameter "
          "per layer, so they cannot stay packed)")
elif target_format.endswith("pack-quantized"):
    print(f"  resident, unpacked {format_memory(dense_total)}  "
          "(transformers without model_loading.py's KEEP_PACKED, which "
          "compressed-tensors would otherwise decompress on the first forward)")
else:
    # MXFP8 keeps a real `weight` of a dtype no matmul accepts, so
    # compressed-tensors has to decompress it eagerly whatever the caller wants;
    # `weights_stay_packed` in model_loading.py is what knows this.
    print(f"  resident, unpacked {format_memory(dense_total)}  "
          f"(transformers: {target_format} is not a packed format, so it is "
          "decompressed on load whatever KEEP_PACKED says)")
print("Load it like any other model:")
print(f'    AutoModelForCausalLM.from_pretrained("{OUTPUT_DIR}")')
