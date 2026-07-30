"""Load a model that does not fit in VRAM, and keep it runnable afterwards.

`device_map="auto"` on its own is not enough once a checkpoint stops fitting in
VRAM. Four things go wrong, and all four only show up *after* the weights have
loaded, which is the most expensive moment to fail at:

1. accelerate budgets the GPUs for weights only, and it uses every free byte to
   do so. The forward pass then has nowhere to put activations, the KV cache or
   the buffers a quantized weight is unpacked through, and it dies with a few
   megabytes left free on a 141GB card. `build_max_memory` leaves a per-GPU
   reserve instead, which pushes the overflow to CPU RAM (and to
   `offload_folder` if that is set too).

2. A compressed-tensors checkpoint (`nvfp4-pack-quantized`,
   `mxfp4-pack-quantized`) is not run compressed: compressed-tensors installs a
   forward pre-hook that decompresses the whole model in place on the first
   forward pass. 4-bit weights become bfloat16, so the footprint quadruples,
   *after* accelerate has already filled the GPUs to the brim with packed
   weights. `keep_weights_packed` drops that hook and unpacks each weight on
   demand instead, which keeps the resident size equal to the checkpoint size.

3. An offloaded module's weights are only onloaded while its own forward runs.
   Architectures that read `submodule.weight` from a parent's forward (Kimi K3
   does this for its attention-residual projections) therefore see a meta
   tensor and abort. `pin_tiny_offloaded_modules` keeps such modules resident;
   they are always small.

4. `model.get_input_embeddings().weight.device` is the meta device when the
   embeddings are offloaded, and moving the input ids there fails with
   "Cannot copy out of meta tensor". `get_input_device` returns the device the
   embeddings actually execute on.

A fifth problem is not about size at all, but has to be solved in the same place
because it happens even earlier: an llmcompressor export can leave a `targets`
regex that selects containers and norms as well as the Linears it packed, and
compressed-tensors then refuses the checkpoint outright with "Quantization of
module type ... is not supported". `repair_quantization_targets` narrows the
regex before `from_pretrained` sees it.
"""

import faulthandler
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

__all__ = [
    "load_model",
    "get_decoder_layers",
    "get_input_device",
    "build_max_memory",
    "keep_weights_packed",
    "pin_tiny_offloaded_modules",
    "layer_count",
    "truncate_layers",
    "kv_cache_bytes",
    "activation_bytes",
    "attention_workspace_bytes",
    "decompression_workspace_bytes",
    "decompressed_weight_growth",
    "weights_stay_packed",
    "matches",
    "module_paths",
    "quantized_modules",
    "select_modules",
    "leaf_pattern",
    "narrow_targets",
    "ignore_dense_targets",
    "checkpoint_tensor_names",
    "checkpoint_tensor_sizes",
    "checkpoint_footprint",
    "repair_quantization_targets",
    "parse_memory",
    "format_memory",
    "format_duration",
    "suppress_per_tensor_cache_clears",
]

# Everything this module and its callers print goes to stdout, which Python
# block-buffers as soon as it is redirected to a file or a pipe. The progress
# that matters most is exactly the progress of a run long enough to be logged,
# and it would sit in an 8KB buffer for minutes while transformers' own output
# -- which goes to stderr -- appeared immediately, making the run look wedged.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # not a regular stream
    pass

# A load that takes an hour and prints nothing is indistinguishable from a wedged
# one, and most of that hour is spent inside transformers and accelerate where
# this module cannot put a progress bar. `kill -USR1 <pid>` dumps the stack of
# every thread to stderr without disturbing the process, which says exactly which
# function is running. SIGUSR1 does not exist on Windows; nothing is lost there.
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

_UNITS = {"": 1, "B": 1, "KIB": 2 ** 10, "MIB": 2 ** 20, "GIB": 2 ** 30,
          "TIB": 2 ** 40, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9,
          "TB": 10 ** 12}


def parse_memory(value):
    """Turn "40GiB" (or 42, or None) into a number of bytes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", value)
    if match is None or match.group(2).upper() not in _UNITS:
        raise ValueError(f"cannot parse a memory size out of {value!r}")
    return int(float(match.group(1)) * _UNITS[match.group(2).upper()])


def format_memory(size):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{size:.0f}B"
        size /= 1024


def format_duration(seconds):
    """A wall-clock span, in the units a person would use to read it."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def get_decoder_layers(model):
    """Locate the ModuleList of decoder layers, whatever the architecture.

    Handles plain causal LMs (model.model.layers), multimodal wrappers such as
    KimiK3ForConditionalGeneration (model.language_model.model.layers) and
    older naming schemes (transformer.h, gpt_neox.layers, ...).
    """
    candidates = []

    # Preferred: let the model tell us where its decoder lives.
    getter = getattr(model, "get_decoder", None)
    if callable(getter):
        try:
            decoder = getter()
        except (AttributeError, NotImplementedError):
            decoder = None
        if decoder is not None:
            candidates.append(decoder)

    candidates.append(model)

    seen = set()
    while candidates:
        module = candidates.pop(0)
        if id(module) in seen:
            continue
        seen.add(id(module))

        for name in ("layers", "h", "blocks", "block"):
            layers = getattr(module, name, None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers

        for name in ("model", "language_model", "transformer", "gpt_neox",
                     "decoder", "base_model"):
            child = getattr(module, name, None)
            if isinstance(child, nn.Module):
                candidates.append(child)

    raise AttributeError(
        f"Could not locate decoder layers on {type(model).__name__}")


def _text_config(config):
    """The sub-config that describes the language model."""
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            text_config = getter()
        except (AttributeError, TypeError):
            text_config = None
        if text_config is not None:
            return text_config
    return getattr(config, "text_config", None) or config


def _dtype_bytes(config):
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return dtype.itemsize
    return 2


def _find_quantization_config(config):
    """The object that owns the checkpoint's quantization config, and the config.

    The owner matters for `repair_quantization_targets`: the config may be a dict
    (mutable in place) or a `QuantizationConfigMixin` (only reachable through a
    `to_dict()` copy, which then has to be written back).
    """
    for candidate in (config, _text_config(config)):
        quantization_config = getattr(candidate, "quantization_config", None)
        if quantization_config is None:
            continue
        if not isinstance(quantization_config, dict):
            to_dict = getattr(quantization_config, "to_dict", None)
            quantization_config = to_dict() if callable(to_dict) else {}
        if quantization_config:
            return candidate, quantization_config
    return None, {}


def _quantization_config(config):
    return _find_quantization_config(config)[1]


_LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")


def layer_count(config):
    """How many decoder layers the config asks for, whatever it calls them."""
    text_config = _text_config(config)
    for key in _LAYER_COUNT_KEYS:
        value = getattr(text_config, key, None)
        if value:
            return int(value)
    return 0


def truncate_layers(config, keep):
    """Build only the first `keep` decoder layers.

    Everything that reads activations out of the middle of a model -- which is
    what a refusal direction is -- never runs the layers past the one it reads
    from. Those layers are still allocated and their weights still loaded, which
    on a model that does not fit in VRAM is the difference between fitting and
    offloading. Dropping them changes nothing about the captured activation: a
    layer cannot influence its own input.

    Returns the number of layers that were dropped.
    """
    total = layer_count(config)
    if not total or keep >= total or keep < 1:
        return 0
    # Both the text config and the top level, since which one the model reads
    # depends on the architecture and setting the unused one is harmless.
    for candidate in {id(_text_config(config)): _text_config(config),
                      id(config): config}.values():
        for key in _LAYER_COUNT_KEYS:
            if getattr(candidate, key, None):
                setattr(candidate, key, keep)
    return total - keep


def kv_cache_bytes(config, tokens):
    """Size of the KV cache for `tokens` tokens (batch size times length).

    An over-estimate on purpose: too much headroom only means a few more layers
    are offloaded, too little means the run dies after the weights are loaded.
    """
    config = _text_config(config)
    layers = getattr(config, "num_hidden_layers", 0) or 0
    heads = getattr(config, "num_attention_heads", 0) or 0
    hidden = getattr(config, "hidden_size", 0) or 0
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if heads else 0)

    if getattr(config, "kv_lora_rank", None) is not None:
        # Multi-head latent attention (DeepSeek V3, Kimi K2/K3). Transformers'
        # implementations cache the up-projected keys and values rather than the
        # compressed latent, so the per-head shapes are the wide ones.
        key = (getattr(config, "qk_nope_head_dim", 0) or 0) + \
              (getattr(config, "qk_rope_head_dim", 0) or 0)
        value = getattr(config, "v_head_dim", 0) or head_dim
        per_token = heads * ((key or head_dim) + value)
    else:
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        per_token = 2 * kv_heads * head_dim

    return layers * per_token * tokens * _dtype_bytes(config)


def activation_bytes(config, tokens):
    """Rough size of the tensors that are live inside one decoder layer."""
    config = _text_config(config)
    hidden = getattr(config, "hidden_size", 0) or 0
    intermediate = max(getattr(config, "intermediate_size", 0) or 0,
                       getattr(config, "moe_intermediate_size", 0) or 0,
                       hidden)
    # A handful of tensors of the widest shape in the layer are alive at once,
    # plus whatever cuBLAS/attention kernels allocate as workspace.
    return 8 * tokens * (hidden + intermediate) * _dtype_bytes(config)


def _is_compressed(quantization_config):
    return (quantization_config.get("quant_method") == "compressed-tensors"
            and quantization_config.get("quantization_status") == "compressed")


def _compressed_quantization_config(config):
    """The quantization config, but only for a checkpoint stored compressed."""
    quantization_config = _quantization_config(config)
    return quantization_config if _is_compressed(quantization_config) else {}


def _compression_formats(config):
    quantization_config = _compressed_quantization_config(config)
    default = quantization_config.get("format")
    formats = set()
    for group in (quantization_config.get("config_groups") or {}).values():
        formats.add(group.get("format") or default)
    if not formats and default:
        formats.add(default)
    return {value for value in formats if value}


def weights_stay_packed(config):
    """Whether `keep_weights_packed` can work on this checkpoint.

    Only the `*-pack-quantized` formats stash their weights in a separate
    `weight_packed` tensor, which is what makes unpacking on demand possible.
    `float-quantized` (fp8) and friends keep a real `weight` of a dtype no matmul
    accepts, so compressed-tensors has to decompress those eagerly and the
    weights grow whatever we do. Knowing which case we are in *before* loading is
    what keeps the memory budget honest.
    """
    formats = _compression_formats(config)
    if not formats:
        return False
    # Anything unrecognised is assumed to grow: over-reserving VRAM costs
    # throughput, under-reserving it costs the run.
    return all(value.endswith("pack-quantized") for value in formats)


def matches(pattern, name):
    """compressed-tensors target/ignore matching, for the cases we can decide.

    Mirrors `compressed_tensors.utils.match.match_name`: a `re:` entry is a
    regex anchored at the start of the module name (`re.match`, not
    `re.search`), anything else is an exact module name. A bare class name such
    as "Linear" cannot be resolved without building the model and is reported as
    "no match" by design.
    """
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return pattern == name


def module_paths(tensor_names):
    """Every module that owns a tensor, plus all of its ancestors."""
    paths = set()
    for name in tensor_names:
        parts = name.split(".")[:-1]
        for end in range(1, len(parts) + 1):
            paths.add(".".join(parts[:end]))
    return paths


def quantized_modules(tensor_names):
    """The modules a compressed-tensors checkpoint actually stores compressed.

    A compressed module is the one that owns a `weight_packed` (the packed
    formats) or, failing that, a `weight_scale` (float/int-quantized).
    """
    packed = set()
    scaled = set()
    for name in tensor_names:
        prefix, _, leaf = name.rpartition(".")
        if leaf == "weight_packed":
            packed.add(prefix)
        elif leaf == "weight_scale":
            scaled.add(prefix)
    return packed or scaled


def select_modules(patterns, paths, ignore=()):
    """The modules a compressed-tensors `targets` list selects, out of `paths`."""
    return {path for path in paths
            if any(matches(pattern, path) for pattern in patterns)
            and not any(matches(entry, path) for entry in ignore)}


def leaf_pattern(modules):
    """A `re:` target that selects exactly the last name component of `modules`.

    Anchored on the dot in front of the leaf: without it a leaf like `down_proj`
    also selects `routed_expert_down_proj`, which is a different module that may
    well be in a different quantization group.
    """
    leaves = sorted({name.rsplit(".", 1)[-1] for name in modules})
    return r"re:.*\.(?:" + "|".join(re.escape(leaf) for leaf in leaves) + ")$"


def narrow_targets(quantization, quantized, paths):
    """Restrict over-broad `targets` regexes to the modules that are really quantized.

    llmcompressor can emit a group whose `targets` is a bare regex such as
    `re:.*block_sparse_moe.*`, losing the `Linear` restriction that was in the
    recipe. compressed-tensors then tries to quantize container modules,
    activation functions and norms as well, and the checkpoint cannot be loaded
    at all: `initialize_module_for_quantization` raises "Quantization of module
    type ... is not supported". Anchoring the regex to the leaf names that are
    actually packed fixes it.

    Modifies `quantization["config_groups"]` in place and returns
    (new_targets, over_matched), or None when there is nothing to do or the
    narrowing cannot be shown to be equivalent.
    """
    groups = quantization.get("config_groups") or {}
    if len(groups) != 1:
        # With several groups there is no way to tell from the tensor names
        # alone which group a module belongs to, so a narrowed regex could
        # silently move modules between groups.
        return None
    group = next(iter(groups.values()))
    targets = list(group.get("targets") or [])
    if not any(target.startswith("re:") for target in targets):
        return None  # class-name targets such as ["Linear"] are already precise

    ignore = quantization.get("ignore") or []

    def selection(patterns):
        return select_modules(patterns, paths, ignore)

    over_matched = selection(targets) - quantized
    if not over_matched:
        return None

    suffix = leaf_pattern(quantized).removeprefix("re:.*")  # keeps the "\." anchor
    narrowed = [target + suffix if target.startswith("re:") else target
                for target in targets]

    if selection(narrowed) != quantized:
        print("  WARNING: `targets` selects modules that are not quantized "
              f"(e.g. {sorted(over_matched)[0]}) and it could not be narrowed "
              "automatically. The checkpoint may fail to load; fix `targets` "
              "by hand.")
        return None

    group["targets"] = narrowed
    return narrowed, over_matched


def ignore_dense_targets(quantization, tensor_names):
    """Stop `targets` from selecting modules the exporter left dense.

    A `targets` of ["Linear"] selects *every* Linear in the model, including the
    ones that were never quantized. compressed-tensors then expects a
    `weight_packed` that is not in the checkpoint, and the failure is silent:
    transformers reports the real `weight` as UNEXPECTED, reports the packed pair
    as MISSING, and initialises them randomly. The module ends up full of noise
    and the model still loads.

    Published Kimi K3 does exactly this to `routed_expert_up_proj`,
    `routed_expert_down_proj` and the three attention-residual projections --
    372 modules, one of which is the residual writer this repo ablates.

    Modifies `quantization["ignore"]` in place. Returns (pattern, modules) or
    None when there is nothing to do or it cannot be expressed safely.
    """
    packed = quantized_modules(tensor_names)
    if not packed:
        return None
    dense = {name.rsplit(".", 1)[0] for name in tensor_names
             if name.rsplit(".", 1)[-1] == "weight"} - packed
    ignore = list(quantization.get("ignore") or [])
    outstanding = {module for module in dense
                   if not any(matches(pattern, module) for pattern in ignore)}
    if not outstanding:
        return None

    pattern = leaf_pattern(outstanding)
    if select_modules([pattern], packed):
        # A dense module shares its last name component with a packed one, so
        # ignoring the first by name would un-quantize the second.
        print("  WARNING: `targets` selects modules that the checkpoint stores "
              f"dense (e.g. {sorted(outstanding)[0]}), and they cannot be "
              "excluded by name without also excluding quantized ones. Their "
              "weights will be replaced by noise; fix `ignore` by hand.")
        return None

    quantization["ignore"] = ignore + [pattern]
    return pattern, outstanding


def checkpoint_tensor_names(model_id):
    """Every tensor name in the checkpoint, from the safetensors headers only.

    No weight data is read, so this stays instant on a multi-terabyte
    checkpoint. Returns an empty set when the names cannot be obtained without
    downloading the weights.
    """
    directory = Path(model_id)
    if directory.is_dir():
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            with open(index) as handle:
                return set(json.load(handle).get("weight_map") or {})
        from safetensors import safe_open

        names = set()
        for shard in sorted(directory.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as handle:
                names.update(handle.keys())
        return names

    # A repo on the hub: the shard index is a few hundred kilobytes and lists
    # every tensor. A single-shard repo has no index and would need the shard
    # header itself, which is not worth a download here.
    try:
        from transformers.utils import cached_file

        resolved = cached_file(model_id, "model.safetensors.index.json",
                               _raise_exceptions_for_missing_entries=False,
                               _raise_exceptions_for_connection_errors=False)
        if resolved:
            with open(resolved) as handle:
                return set(json.load(handle).get("weight_map") or {})
    except Exception:
        pass
    return set()


_LAYER_INDEX = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")


def checkpoint_tensor_sizes(model_id):
    """{tensor name: bytes}, read from the safetensors headers and nothing else.

    Empty when the checkpoint is not a local directory: the shard index lists the
    names but not the shapes, and the headers are not worth a download.
    """
    directory = Path(model_id)
    if not directory.is_dir():
        return {}

    from safetensors import safe_open

    widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "F32": 4, "I32": 4,
              "U32": 4, "F64": 8, "I64": 8, "U64": 8}
    sizes = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                view = handle.get_slice(name)
                count = 1
                for dimension in view.get_shape():
                    count *= dimension
                sizes[name] = count * widths.get(view.get_dtype(), 2)
    return sizes


def checkpoint_footprint(sizes, max_layers=None):
    """(bytes that end up resident, bytes of the largest indivisible block).

    Every architecture worth running this on puts its decoder layer in
    `_no_split_modules`, so accelerate places a layer whole -- and if the layer
    lands on the CPU, copies it to its execution device whole, for the duration
    of its own forward. That copy is the largest transient the run has, and
    nothing leaves room for it unless the budget does.
    """
    resident = 0
    per_layer = {}
    for name, size in sizes.items():
        match = _LAYER_INDEX.search(name)
        if match is None:
            resident += size
            continue
        layer = int(match.group(1))
        if max_layers is None or layer < max_layers:
            resident += size
            per_layer[layer] = per_layer.get(layer, 0) + size
    return resident, max(per_layer.values(), default=0)


def repair_quantization_targets(model_id, config):
    """Fix a quantization config whose `targets` regex selects too much.

    Such a checkpoint fails to load before anything else can happen, so the
    repair has to run before `from_pretrained`. The quantization config is
    mutated in place; pass the same `config` object to `from_pretrained` for the
    repair to take effect. Returns True when something was changed.
    """
    owner, quantization = _find_quantization_config(config)
    if not _is_compressed(quantization):
        return False

    tensor_names = checkpoint_tensor_names(model_id)
    quantized = quantized_modules(tensor_names)
    if not quantized:
        return False

    changed = []
    # First, because narrowing the targets checks its work against `ignore`.
    dense_fix = ignore_dense_targets(quantization, tensor_names)
    if dense_fix is not None:
        pattern, modules = dense_fix
        changed.append(
            f"Excluded {len(modules)} module(s) from quantization with "
            f"{pattern!r}: `targets` selected them but the checkpoint stores "
            f"them dense (e.g. {sorted(modules)[0]}), so compressed-tensors "
            "would have replaced their weights with noise")

    fixed = narrow_targets(quantization, quantized, module_paths(tensor_names))
    if fixed is None and not changed:
        return False

    # `quantization` is the live dict when the config holds one, and a detached
    # `to_dict()` copy when it holds a QuantizationConfigMixin. Writing it back
    # covers both; transformers accepts either form.
    setattr(owner, "quantization_config", quantization)

    if fixed is not None:
        narrowed, over_matched = fixed
        changed.append(
            f"Narrowed the quantization `targets` to {narrowed}: as written "
            f"they also selected {len(over_matched)} module(s) that are not "
            f"quantized (e.g. {sorted(over_matched)[0]}), which "
            "compressed-tensors refuses to load")
    for message in changed:
        print(message)
    return True


def attention_workspace_bytes(config, batch, sequence):
    """Peak VRAM of one eager attention score matrix.

    sdpa and flash-attention never materialise the [batch, heads, q, k] matrix.
    An eager implementation does, and a `trust_remote_code` model that ships only
    an eager path leaves no choice -- Kimi K3 falls back to eager whenever
    flash-attn 2 is not importable, and refuses `attn_implementation="sdpa"`
    outright. Being quadratic in the sequence length, it is then the largest
    single allocation of a long prefill: 36GiB for Kimi K3 at 8k tokens, against
    the 5GiB the rest of the activations need.

    Three copies are live at once, because `softmax` upcasts to float32 and the
    result is cast straight back: the scores, the fp32 tensor, and the cast.
    """
    config = _text_config(config)
    heads = getattr(config, "num_attention_heads", 0) or 0
    if not heads or not sequence:
        return 0
    return (3 * _dtype_bytes(config) + 4) * batch * heads * sequence * sequence


def decompression_workspace_bytes(config):
    """Peak VRAM taken by turning one packed weight matrix into a dense one.

    compressed-tensors unpacks 4-bit weights through float32 intermediates
    (`kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)` allocates three tensors
    the size of the dense matrix in fp32), so the transient cost is several times
    the dense matrix itself. Every GPU that holds quantized weights pays it:
    once per forward pass under `keep_weights_packed`, or once at load time when
    compressed-tensors decompresses the model instead. This is the allocation
    that fails first when accelerate has budgeted the cards for weights only.

    Zero for a checkpoint that is not stored compressed.
    """
    if not _compressed_quantization_config(config):
        return 0

    text_config = _text_config(config)
    hidden = getattr(text_config, "hidden_size", 0) or 0
    widest = max(getattr(text_config, "intermediate_size", 0) or 0,
                 getattr(text_config, "moe_intermediate_size", 0) or 0,
                 hidden)
    # One uint8 index tensor, up to three fp32 temporaries and the dense result,
    # per element of the widest matrix in the model.
    return 16 * hidden * widest


def decompressed_weight_growth(config):
    """How much a compressed-tensors checkpoint grows on the first forward.

    compressed-tensors keeps `weight_packed` around only until the first forward
    pass, which decompresses every quantized module in place. 4 bits per weight
    become a full bfloat16, i.e. four times the memory. Returns 1.0 for
    checkpoints that are not affected.
    """
    quantization_config = _compressed_quantization_config(config)
    if not quantization_config:
        return 1.0

    bits = [group.get("weights", {}).get("num_bits")
            for group in (quantization_config.get("config_groups") or {}).values()]
    bits = [value for value in bits if value]
    if not bits:
        return 1.0
    return max(1.0, _dtype_bytes(config) / (min(bits) / 8))


def _available_cpu_memory():
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0


def build_max_memory(config, tokens, headroom=None, headroom_fraction=0.1,
                     cpu_limit=None, weight_growth=1.0, sequence=None,
                     weights=0, onload=0):
    """Per-device weight budget that leaves room for the forward pass.

    accelerate's default budget is the free memory of each GPU, all of it, and
    it only counts weights. Whatever the forward pass needs on top of them --
    the KV cache, the activations, the buffers a quantized weight is unpacked
    through, the copy of an offloaded block that is being onloaded, the
    allocator's fragmentation -- has to come out of a reserve that nobody sets
    up. So the model loads, and then the first forward pass dies. Hence this.

    Spending VRAM on the reserve pushes more weights to CPU RAM, which costs
    throughput but is the only way the run finishes at all.
    """
    kv_cache = kv_cache_bytes(config, tokens)
    activations = activation_bytes(config, tokens)
    workspace = decompression_workspace_bytes(config)
    # Only the sequence length says how big the attention matrix gets; the token
    # count alone cannot tell one long prompt from many short ones.
    attention = attention_workspace_bytes(
        config, max(1, round(tokens / sequence)), sequence) if sequence else 0

    devices = list(range(torch.cuda.device_count()))
    shards = max(len(devices), 1)

    # A GPU runs one layer at a time, so it needs a full layer's activations and
    # a full weight's unpacking workspace no matter how few layers it holds;
    # only the KV cache scales with its share of them.
    reserve = activations + workspace + attention + kv_cache // shards
    # Device 0 is the execution device for every CPU-offloaded layer, so it also
    # holds the KV cache of all of them.
    reserve_0 = activations + workspace + attention + kv_cache

    def budgets(extra):
        out = {}
        for device in devices:
            free, total = torch.cuda.mem_get_info(device)
            if headroom is not None:
                device_headroom = parse_memory(headroom)
            else:
                # The fraction of the card is the backstop for everything the
                # terms above do not model: cuBLAS workspaces, NCCL buffers,
                # allocator fragmentation.
                device_headroom = max(int(headroom_fraction * total),
                                      (reserve_0 if device == 0 else reserve) + extra)
            out[device] = max(0, int((free - device_headroom) / weight_growth))
        return out

    max_memory = budgets(0)
    # Reserving room to onload an offloaded block costs a block per GPU, so it is
    # only worth doing when something is actually going to be offloaded. Which
    # GPU needs it is not knowable here -- accelerate picks the first device in
    # the map that holds weights, not necessarily device 0 -- so every GPU gets
    # it.
    offloading = weights and sum(max_memory.values()) < weights
    if offloading and onload:
        max_memory = budgets(onload)

    if cpu_limit is not None:
        max_memory["cpu"] = parse_memory(cpu_limit)
    else:
        # Offloaded weights are held in ordinary (non-pinned) RAM, and the
        # process needs some of it for itself: the shards are read through it,
        # and every onloaded block is copied through it as well.
        available = _available_cpu_memory()
        max_memory["cpu"] = max(0, int(available * 0.9 / weight_growth))

    print("Memory budget: " + ", ".join(
        f"{device}: {format_memory(size)}" for device, size in max_memory.items()))
    if headroom is None and devices:
        print(f"  reserved on GPU 0: at least {format_memory(reserve_0)}, "
              f"on the others at least {format_memory(reserve)} "
              f"({format_memory(kv_cache)} of KV cache for {tokens} tokens, "
              f"{format_memory(activations)} of activations, "
              f"{format_memory(workspace)} to unpack a weight through, "
              f"{format_memory(attention)} for an eager attention matrix), "
              f"{headroom_fraction:.0%} of the card where that is more")
    if weights:
        print(f"  weights to place: {format_memory(weights)}, "
              + (f"more than the {format_memory(sum(v for k, v in max_memory.items() if k != 'cpu'))} "
                 f"of VRAM budget, so {format_memory(onload)} per GPU is also "
                 "reserved to onload an offloaded block"
                 if offloading and onload else "which the GPUs have room for"))
    if weight_growth != 1.0:
        print(f"  budgets divided by {weight_growth:g}: the weights grow by that "
              "much when compressed-tensors decompresses them on the first "
              "forward pass")
    return max_memory


def keep_weights_packed(model):
    """Unpack compressed-tensors weights per forward instead of all at once.

    compressed-tensors registers a forward pre-hook that decompresses the whole
    model in place the first time it is used, because Transformers has no
    kernels for packed weights. For a 4-bit checkpoint that multiplies the
    resident weights by four, and it happens after accelerate has already filled
    the GPUs with the packed ones -- so the run dies with a decompression buffer
    as the straw that broke the camel's back.

    Dropping that hook and materialising `module.weight` on access instead keeps
    the resident weights at their checkpoint size, at the cost of unpacking a
    matrix every time it is used. On a model that is offloaded anyway, that cost
    is small next to the transfers. It also keeps CPU offloading working at all:
    accelerate looks the offloaded parameters up by name, and the decompressed
    model no longer has a `weight_packed` to look up.

    Returns the number of modules that were switched over.
    """
    hook = getattr(model, "ct_decompress_hook", None)
    if hook is None:
        return 0

    try:
        from compressed_tensors.compressors.base import (BaseCompressor,
                                                         infer_module_format)
        from compressed_tensors.config import CompressionFormat
        from compressed_tensors.quantization import QuantizationScheme
        from compressed_tensors.utils import get_direct_state_dict
    except ImportError as error:
        # The hook is there, so this is a compressed-tensors checkpoint, but the
        # internals it is unpacked with have moved. Leave the decompression to
        # compressed-tensors rather than guessing.
        print(f"Cannot keep the weights packed ({error}); compressed-tensors "
              "will decompress the model on the first forward pass")
        return 0

    modules = []
    for module in model.modules():
        scheme = getattr(module, "quantization_scheme", None)
        if not isinstance(scheme, QuantizationScheme):
            continue
        weight = module._parameters.get("weight")
        if weight is not None:
            if weight.dtype.is_floating_point and weight.element_size() >= 2:
                continue  # already dense, nothing to unpack
            # A format that keeps a `weight` of its own (float-quantized fp8,
            # for instance) cannot be fed to a plain matmul, so the eager
            # decompression is the only thing that makes this model run.
            print(f"Cannot keep {type(module).__name__} weights packed "
                  f"({weight.dtype}); leaving the decompression to "
                  "compressed-tensors")
            return 0
        modules.append((module, scheme))

    if not modules:
        return 0

    hook.remove()
    delattr(model, "ct_decompress_hook")

    lazy_classes = {}
    for module, scheme in modules:
        cls = type(module)
        # `decompress_module` resolves the format the same way; do it once here
        # so that the subclass below never has to (and never sees its own name
        # in `infer_module_format`, which only knows the original classes).
        scheme.format = CompressionFormat(
            scheme.format or infer_module_format(cls, scheme))
        module._compressor = BaseCompressor.get_value_from_registry(
            scheme.format.value)

        lazy_cls = lazy_classes.get(cls)
        if lazy_cls is None:
            def __getattr__(self, name, _cls=cls):
                # nn.Module.__getattr__ only runs for names that are not
                # registered parameters or buffers, which is exactly the state
                # `weight` is in while the module stays compressed.
                if name == "weight":
                    return self._compressor.decompress(
                        get_direct_state_dict(self),
                        self.quantization_scheme)["weight"]
                return _cls.__getattr__(self, name)

            lazy_cls = type(cls.__name__ + "Packed", (cls,),
                            {"__getattr__": __getattr__})
            lazy_classes[cls] = lazy_cls
        module.__class__ = lazy_cls

    return len(modules)


def pin_tiny_offloaded_modules(model, limit="1MiB"):
    """Keep small offloaded modules resident on their execution device.

    accelerate onloads an offloaded module's weights in the module's own forward
    pre-hook and drops them again afterwards. A module that is never called does
    not get that treatment, so a parent that reads `child.weight` directly --
    Kimi K3's `_apply_attn_res` does, and so do fused gates in other
    architectures -- reads a meta tensor and the forward pass dies.

    Such directly-read weights are always small (a norm, a scoring vector), so
    the fix is to pay the VRAM for the small offloaded modules and leave the
    matrices alone. Returns (module count, bytes made resident).
    """
    from accelerate.hooks import AlignDevicesHook
    from accelerate.utils import set_module_tensor_to_device
    from accelerate.utils.modeling import named_module_tensors

    limit = parse_memory(limit)
    pinned = 0
    pinned_bytes = 0
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if not isinstance(hook, AlignDevicesHook) or not hook.offload:
            continue
        if hook.execution_device is None or hook.weights_map is None:
            continue

        tensors = list(named_module_tensors(
            module, include_buffers=hook.offload_buffers,
            recurse=hook.place_submodules, remove_non_persistent=True))
        size = sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)
        if size > limit or size == 0:
            continue

        for name, _ in tensors:
            set_module_tensor_to_device(module, name, hook.execution_device,
                                       value=hook.weights_map[name])
        # Keep the hook itself: it is what moves this module's inputs to the
        # right device. It just has nothing left to onload or to evict.
        hook.offload = False
        pinned += 1
        pinned_bytes += size

    return pinned, pinned_bytes


def get_input_device(model):
    """Where the input ids have to be put.

    Not `model.device` (misleading on a sharded model) and not
    `model.get_input_embeddings().weight.device` either: that is the meta device
    once the embeddings are offloaded, and moving a tensor there fails with
    "Cannot copy out of meta tensor". accelerate stores the device the module
    actually runs on in its hook.
    """
    def as_device(value):
        return torch.device(f"cuda:{value}" if isinstance(value, int) else value)

    embeddings = model.get_input_embeddings()

    hook = getattr(embeddings, "_hf_hook", None)
    device = getattr(hook, "execution_device", None)
    device = as_device(device) if device is not None else embeddings.weight.device
    if device.type != "meta":
        return device

    for candidate in (getattr(model, "hf_device_map", None) or {}).values():
        if candidate not in ("cpu", "disk", "meta"):
            return as_device(candidate)

    return torch.device("cpu")


_cache_clears_suppressed = False


def suppress_per_tensor_cache_clears():
    """Stop accelerate from emptying the CUDA cache once per tensor.

    `set_module_tensor_to_device` ends with `torch.cuda.empty_cache()` -- its
    `clear_cache` argument defaults to True -- and accelerate calls it for every
    tensor it places. Twice over:

    * `dispatch_model` places every tensor in the model while it attaches the
      device hooks, so a load pays it once per tensor in the checkpoint;
    * `AlignDevicesHook.pre_forward` places every tensor of an offloaded block,
      so *every forward pass* pays it again for everything that did not fit in
      VRAM.

    `empty_cache` walks the allocator's block list and hands the blocks back to
    the driver, so it costs more the fuller the GPU is: measured 0.011ms with
    0.5GiB resident and 0.943ms with 12GiB, and it is 76% of `dispatch_model` on
    a model with 50k modules. Kimi K3 has half a million tensors, and five
    offloaded layers are another 27,000 of these per forward pass.

    Suppressing it is safe, and faster for exactly the reason it exists: the
    caching allocator reuses the blocks that the onload/offload cycle frees
    rather than returning them to the driver and asking for them back a
    millisecond later, and PyTorch still empties the cache itself before it
    raises an out-of-memory error. Applied once per process, and left in place
    because the forward passes need it as much as the load does.
    """
    global _cache_clears_suppressed
    if _cache_clears_suppressed:
        return False
    try:
        from accelerate import hooks as accelerate_hooks
        from accelerate.utils import modeling as accelerate_modeling
    except ImportError:
        return False

    patched = False
    for module in (accelerate_modeling, accelerate_hooks):
        if hasattr(module, "clear_device_cache"):
            module.clear_device_cache = lambda *args, **kwargs: None
            patched = True
    _cache_clears_suppressed = patched
    return patched


def load_model(model_id, dtype="auto", device_map="auto", max_memory=None,
               offload_dir=None, quantization_config=None, workload_tokens=1024,
               gpu_headroom=None, gpu_headroom_fraction=0.05, cpu_limit=None,
               keep_packed=True, pin_limit="1MiB", fix_targets=True,
               max_layers=None, workload_sequence=None, trust_remote_code=True):
    """Load `model_id` spread over the GPUs, RAM and `offload_dir`.

    `workload_tokens` is the batch size times the sequence length of the biggest
    forward pass that will be run, and it sizes the VRAM that is kept free.
    `workload_sequence` is that forward pass's sequence length on its own, which
    is what an eager attention matrix is quadratic in; without it that term is
    left out of the reserve.

    `max_layers` builds only that many decoder layers. Worth setting whenever the
    activation being read comes from the middle of the model, since the layers
    after it are never executed.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    dropped = truncate_layers(config, max_layers) if max_layers else 0
    if dropped:
        print(f"Building only the first {max_layers} of {max_layers + dropped} "
              f"decoder layers; the other {dropped} are never executed")

    # An over-broad `targets` regex stops the checkpoint from loading at all, so
    # this has to happen before `from_pretrained` and the repaired config has to
    # be handed to it explicitly.
    repaired = fix_targets and quantization_config is None and \
        repair_quantization_targets(model_id, config)

    # The budget has to assume the weights grow unless they provably do not: a
    # budget computed for packed weights on a checkpoint that gets decompressed
    # anyway is exactly the OOM this function exists to prevent, and by the time
    # `keep_weights_packed` reports back the weights are already loaded.
    stays_packed = keep_packed and weights_stay_packed(config)
    weight_growth = 1.0 if stays_packed else decompressed_weight_growth(config)

    # Only pass the optional arguments that are actually set. In particular
    # `quantization_config=None` is not the same as leaving it out: configs that
    # expose a `quantization_config` attribute (Kimi K3 lifts the one from
    # `text_config` up to the top level) have it overwritten with None, which
    # silently strips the checkpoint's own quantization on transformers 5.x and
    # raises `'NoneType' object has no attribute 'to_dict'` on 4.x.
    load_kwargs = {}
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    if offload_dir is not None:
        load_kwargs["offload_folder"] = offload_dir
    if repaired or dropped:
        # Only when the config was actually changed: passing `config` makes
        # transformers skip its own config load, and there is no reason to take
        # that path when the checkpoint's own config is fine.
        load_kwargs["config"] = config

    sharded = isinstance(device_map, str) and device_map in (
        "auto", "balanced", "balanced_low_0", "sequential")
    resident = 0
    if max_memory is None and sharded and torch.cuda.device_count() > 0:
        # Headers only, so this is instant even on a 1.5TB checkpoint. It says
        # how much has to be placed and how big the largest indivisible block is,
        # which is what decides whether an onload buffer has to be reserved.
        resident, onload = checkpoint_footprint(
            checkpoint_tensor_sizes(model_id), max_layers)
        max_memory = build_max_memory(config, workload_tokens,
                                      headroom=gpu_headroom,
                                      headroom_fraction=gpu_headroom_fraction,
                                      cpu_limit=cpu_limit,
                                      weight_growth=weight_growth,
                                      sequence=workload_sequence,
                                      weights=resident, onload=onload)
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    shards = len(list(Path(model_id).glob("*.safetensors"))) if Path(model_id).is_dir() else 0
    print(f"[pid {os.getpid()}] `kill -USR1 {os.getpid()}` prints the stack if "
          "this looks stuck")
    print("Building the model and loading the weights"
          + (f": {format_memory(resident)} across {shards} shard(s)" if resident and shards
             else "")
          + ". transformers reports its own progress below; the quiet stretch "
          "after the load report is accelerate placing the modules.")

    if suppress_per_tensor_cache_clears():
        print("  (accelerate's per-tensor torch.cuda.empty_cache() is suppressed; "
              "it costs more than everything else it does)")

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=trust_remote_code,
                                                 dtype=dtype,
                                                 device_map=device_map,
                                                 **load_kwargs)
    load_seconds = time.perf_counter() - started
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # once, instead of half a million times

    print(f"Loaded in {format_duration(load_seconds)}"
          + (f", {format_memory(resident / load_seconds)}/s" if resident else ""))

    placement_map = getattr(model, "hf_device_map", None)
    if placement_map:
        placement = {}
        for device in placement_map.values():
            placement[str(device)] = placement.get(str(device), 0) + 1
        print("Weights placed on: " + ", ".join(
            f"{device} ({count} modules)"
            for device, count in sorted(placement.items())))
        offloaded = [name for name, device in placement_map.items()
                     if str(device) in ("cpu", "disk")]
        if offloaded:
            print(f"  {len(offloaded)} module(s) are not on a GPU and will be "
                  "streamed in for every forward pass")

    if keep_packed:
        decompresses_itself = getattr(model, "ct_decompress_hook", None) is not None
        started = time.perf_counter()
        packed = keep_weights_packed(model)
        if packed:
            print(f"Keeping {packed} quantized modules packed "
                  f"({format_duration(time.perf_counter() - started)}); their "
                  "weights are unpacked per forward pass")
        elif stays_packed and decompresses_itself:
            # The budget was computed on the assumption that the weights stay at
            # their checkpoint size, and they will not. Say so now rather than
            # letting the first forward pass fail with an allocator error.
            growth = decompressed_weight_growth(config)
            if growth != 1.0:
                print(f"WARNING: the weights could not be kept packed after all "
                      f"and will grow by {growth:g}x on the first forward pass, "
                      f"which the memory budget did not allow for. Re-run with "
                      f"keep_packed=False to budget for the decompressed size.")

    if pin_limit is not None:
        started = time.perf_counter()
        pinned, pinned_bytes = pin_tiny_offloaded_modules(model, pin_limit)
        if pinned:
            print(f"Pinned {pinned} small offloaded modules "
                  f"({format_memory(pinned_bytes)}) to their execution device "
                  f"in {format_duration(time.perf_counter() - started)}")
    print("Model ready.")

    return model
