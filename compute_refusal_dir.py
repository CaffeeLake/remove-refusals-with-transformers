import random
import time

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from tqdm import tqdm

from model_loading import (format_duration, get_decoder_layers,
                           get_input_device, layer_count, load_model)

torch.inference_mode()

MODEL_ID = "../Kimi-K3"
# MODEL_ID = "tiiuae/Falcon3-1B-Instruct"
# MODEL_ID = "Qwen/Qwen3-1.7B"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
# MODEL_ID = "google/gemma-1.1-2b-it"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# 4-bit quantization packs weights into an opaque uint8 blob. Architectures that
# read `.weight` directly (e.g. Kimi K3's attention-residual projections) break
# under it, and small models do not need it anyway. Leave it off for a
# checkpoint that is already quantized.
LOAD_IN_4BIT = False

# "auto" follows the checkpoint. Forcing float16 on a bfloat16 model throws away
# most of its exponent range.
DTYPE = "auto"

# "auto" spreads the layers over every visible GPU, then over CPU RAM, then over
# OFFLOAD_DIR. Use "cuda:0" to pin everything to a single GPU instead.
DEVICE_MAP = "auto"
# Per-device caps for the *weights*, e.g. {0: "44GiB", 1: "44GiB",
# "cpu": "100GiB"}. `None` derives them from the free VRAM minus a reserve for
# the forward pass; see GPU_HEADROOM below. Setting this bypasses that reserve.
MAX_MEMORY = None
# Needed once the weights no longer fit in RAM + VRAM.
OFFLOAD_DIR = None

# VRAM to keep free per GPU for everything that is not weights: activations, the
# KV cache, the buffers a quantized weight is unpacked through, allocator
# fragmentation. accelerate budgets for weights only, so without a reserve the
# model loads and then the first forward pass OOMs with a few MiB free.
# `None` computes it from the model's own shapes and the batch below, with
# GPU_HEADROOM_FRACTION of the card as a floor -- a backstop for what the three
# terms do not model (cuBLAS workspaces, NCCL buffers, fragmentation), not the
# estimate itself, which is why 5% is enough. Raise it if a forward pass still
# runs out; the cost is more weights offloaded to CPU RAM, i.e. a slower run
# rather than a failed one.
GPU_HEADROOM = None
GPU_HEADROOM_FRACTION = 0.05
# Cap on offloaded weights in CPU RAM. `None` uses 90% of what is free.
CPU_LIMIT = None

# Keep compressed-tensors weights in their packed 4-bit form and unpack each one
# per forward pass. Otherwise compressed-tensors decompresses the whole model in
# place on the first forward, quadrupling the resident weights *after* they have
# been placed. Turning this off makes the budget above allow for that growth,
# which offloads roughly four times as much to CPU RAM.
KEEP_PACKED = True

# How deep into the model the refusal direction is read from. 0.6 is where the
# residual stream has usually settled into features but has not yet started
# turning back into logits.
LAYER_FRACTION = 0.6

# Build only the layers up to the one the activation is read from. Everything
# after it is never executed -- the capture hook aborts the forward pass -- so
# those layers are pure overhead: on Kimi K3 the decoder layers are 99.9% of the
# parameters, and dropping the last 40% of them takes about 40% off the
# checkpoint, VRAM and load time alike. The captured activation is unchanged; a
# layer cannot influence its own input.
TRUNCATE_LAYERS = True

# settings:
# How many prompts of each class the direction is averaged over. Enough of them
# that the direction is determined by refusal rather than by which prompts were
# drawn: measured on Qwen3, ablating along a 32-prompt direction leaves 26-36% of
# the refusal signal in held-out activations, against 22-24% for 256 and a floor
# of 22-23% (the class difference is not perfectly one-dimensional, so a single
# direction can never remove all of it). 32 also has a long tail -- one draw in
# twenty leaves nearly half of it. The cost is linear in forward passes and the
# capture hook aborts each one at LAYER_FRACTION, so this is cheap.
instructions = 512
# Dimensions excluded from the difference before it is normalised into the
# direction, chosen by activation magnitude. Some models develop a handful of
# dimensions three orders of magnitude larger than the rest from some layer on
# (Qwen3-14B: 85 to 11,776 at layer 20). A large dimension has a large variance,
# so it contributes a large *difference* between the two class means -- 94% of
# the whole difference vector on Qwen3-14B -- while its t is only 2.2, i.e. it is
# sampling noise. The direction then points at that dimension instead of at
# refusal, and agreement between two halves of the prompts falls from 0.94 to
# 0.19. Zeroing the two largest fixes it (back to 0.89) and is measurably free on
# models that do not have the problem: 0.938 -> 0.937 on Qwen3-14B's own layer
# 19, 0.886 -> 0.889 on Qwen3-8B, 0.850 -> 0.850 on Qwen3.5-9B. Raising it past 2
# changes nothing either (flat from 2 to 128); 0 turns it off.
DROP_LARGEST_DIMENSIONS = 0

# Share of the prompts kept out of the direction and used only to check it. The
# saved direction is still built from all of them; this only buys the honest
# number, and it costs nothing because the activations are collected either way.
# Set it to 0 to skip the check.
HELDOUT_FRACTION = 0.1
# Prompts per forward pass. On a model too big to fit in VRAM the weights have
# to be streamed from RAM/disk for every forward, so a larger batch directly
# divides the wall clock time. Lower it if activations blow up the VRAM.
batch_size = 64
pos = -1

print("Instruction count: " + str(instructions))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)


# `pos = -1` only points at the last real token if the padding is on the left.
tokenizer.padding_side = "left"


def tokenize(batch):
    return tokenizer.apply_chat_template(
        conversation=[[{"role": "user", "content": insn}] for insn in batch],
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt")


def batches(instructions):
    return [instructions[start:start + batch_size]
            for start in range(0, len(instructions), batch_size)]


# Tokenize before the model is loaded: the size of the biggest forward pass is
# what the VRAM reserve has to cover, and measuring it beats guessing it.
harmful_batches = [tokenize(batch) for batch in batches(harmful_instructions)]
harmless_batches = [tokenize(batch) for batch in batches(harmless_instructions)]
shapes = [tuple(inputs["input_ids"].shape)
          for inputs in harmful_batches + harmless_batches]
workload_tokens = max(rows * columns for rows, columns in shapes)
# The sequence length on its own, because an eager attention matrix is quadratic
# in it and the token count cannot tell one long prompt from many short ones.
workload_sequence = max(columns for _, columns in shapes)
print(f"Largest forward pass: {workload_tokens} tokens "
      f"({max(rows for rows, _ in shapes)} prompts of up to "
      f"{workload_sequence} tokens)")

# Only pass the optional arguments that are actually set. In particular
# `quantization_config=None` is not the same as leaving it out: configs that
# expose a `quantization_config` attribute (Kimi K3 lifts the one from
# `text_config` up to the top level) have it overwritten with None, which
# silently strips the checkpoint's own quantization on transformers 5.x and
# raises `'NoneType' object has no attribute 'to_dict'` on 4.x.
load_kwargs = {}
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if LOAD_IN_4BIT:
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError(
            "the checkpoint is already quantized; LOAD_IN_4BIT would stack "
            "bitsandbytes on top of it. Set LOAD_IN_4BIT = False.")
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# The layer index has to come from the config, not from the loaded model: with
# TRUNCATE_LAYERS the model no longer has the layers this fraction refers to.
total_layers = layer_count(config)
if not total_layers:
    raise RuntimeError(f"cannot tell how many decoder layers {MODEL_ID} has")
layer_idx = int(total_layers * LAYER_FRACTION)
print(f"Layer index: {layer_idx} of {total_layers}")
if TRUNCATE_LAYERS:
    load_kwargs["max_layers"] = layer_idx + 1

model = load_model(MODEL_ID,
                   dtype=DTYPE,
                   device_map=DEVICE_MAP,
                   max_memory=MAX_MEMORY,
                   offload_dir=OFFLOAD_DIR,
                   workload_tokens=workload_tokens,
                   workload_sequence=workload_sequence,
                   gpu_headroom=GPU_HEADROOM,
                   gpu_headroom_fraction=GPU_HEADROOM_FRACTION,
                   cpu_limit=CPU_LIMIT,
                   keep_packed=KEEP_PACKED,
                   **load_kwargs)

# With a sharded model `model.device` can be misleading, and the embedding
# weight's device is the meta device once the embeddings are offloaded. The
# inputs have to land on the device the embeddings actually execute on;
# accelerate moves them along from there.
input_device = get_input_device(model)

layers = get_decoder_layers(model)
if layer_idx >= len(layers):
    raise RuntimeError(
        f"the model built {len(layers)} decoder layers but the activation is "
        f"read from layer {layer_idx}; the config's layer count "
        f"({total_layers}) does not describe this architecture")

# On a model whose weights have to be unpacked (or streamed) for every forward
# pass, a batch takes minutes, so the bar is the only thing that says whether the
# run is progressing or wedged. It counts prompts, and tqdm turns that into a
# rate and an ETA.
bar = tqdm(total=instructions * 2, unit="prompt", smoothing=0.1,
           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                      "{rate_fmt}{postfix}]")

# `output_hidden_states=True` is unreliable across architectures (models still
# using the deprecated `check_model_inputs` decorator silently return None), so
# grab the activation straight off the layer with a forward pre-hook. The input
# of layer N is exactly `hidden_states[N]` in the usual HF convention.
captured = []


class StopForward(Exception):
    """Aborts the forward pass once the activation we want has been captured."""


def capture_hook(module, args, kwargs):
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None:
        hidden_states = args[0]
    captured.append(hidden_states.detach()[:, pos, :].to("cpu", torch.float32))
    # Everything past this layer is dead weight for us. On a model that streams
    # its weights from disk, skipping it saves a large part of the runtime.
    raise StopForward


handle = layers[layer_idx].register_forward_pre_hook(capture_hook, with_kwargs=True)


def collect_hidden(label, tokenized_batches):
    hidden = []
    for index, inputs in enumerate(tokenized_batches, 1):
        bar.set_description(f"{label} {index}/{len(tokenized_batches)}")
        started = time.perf_counter()
        captured.clear()
        try:
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(input_device),
                      attention_mask=inputs["attention_mask"].to(input_device),
                      use_cache=False)
        except StopForward:
            pass
        hidden.append(captured[0])
        bar.set_postfix_str(f"{format_duration(time.perf_counter() - started)}/batch")
        bar.update(n=inputs["input_ids"].shape[0])
    return torch.cat(hidden)


started = time.perf_counter()
harmful_hidden = collect_hidden("harmful", harmful_batches)
harmless_hidden = collect_hidden("harmless", harmless_batches)

handle.remove()
bar.close()
print(f"Collected the activations in {format_duration(time.perf_counter() - started)}")

print(f"Collected {tuple(harmful_hidden.shape)} harmful / "
      f"{tuple(harmless_hidden.shape)} harmless activations")

def largest_dimensions(harmful_rows, harmless_rows, count):
    """The `count` dimensions carrying the largest activations, or None."""
    if count <= 0:
        return None
    magnitude = torch.cat([harmful_rows, harmless_rows]).abs().mean(dim=0)
    return magnitude.argsort(descending=True)[:count]


def direction_between(harmful_rows, harmless_rows, drop=None):
    difference = harmful_rows.mean(dim=0) - harmless_rows.mean(dim=0)
    if drop is not None:
        difference = difference.clone()
        difference[drop] = 0
    return difference / difference.norm()


def separation_along(direction, harmful_rows, harmless_rows):
    """How far apart the two clusters are along `direction`.

    Two numbers, because the first one alone is misleading. The raw gap is in
    the units of the residual stream, which differ by orders of magnitude
    between models -- 56 on Qwen3-8B against 4 on a model where the direction is
    pure noise -- so it can only be compared with itself. The effect size
    divides by the spread of the two clusters and is therefore dimensionless and
    comparable: measured at 60% depth it is 3.6 on Qwen3-1.7B, 4.7 on Qwen3-4B,
    5.6 on Qwen3-8B, and -0.03 on a model with no refusal behaviour to find.
    """
    a = harmful_rows @ direction
    b = harmless_rows @ direction
    spread = torch.sqrt((a.var() + b.var()) / 2).clamp_min(1e-9)
    return ((a - b.mean()).mean().item(),
            ((a.mean() - b.mean()) / spread).item())


# The direction that gets saved is built from every prompt, because more of them
# is measurably better. But a direction is guaranteed to separate the prompts it
# was built from, so that number says nothing: on a model with no refusal
# behaviour it still reads 4.08, while the same direction scores -0.43 on prompts
# it has not seen. Holding a slice back costs nothing -- the activations are
# already collected -- and is the only thing here that can tell a real direction
# from noise.
drop = largest_dimensions(harmful_hidden, harmless_hidden, DROP_LARGEST_DIMENSIONS)
if drop is not None:
    # Report what was excluded and why it mattered. A dimension that holds most
    # of the difference while its t is barely above noise is the failure this
    # option exists for; one that holds almost none of it was excluded for
    # nothing, which is the normal case and costs nothing either.
    raw = harmful_hidden.mean(dim=0) - harmless_hidden.mean(dim=0)
    count = len(harmful_hidden)
    error = torch.sqrt(harmful_hidden.var(dim=0) / count
                       + harmless_hidden.var(dim=0) / count).clamp_min(1e-9)
    magnitude = torch.cat([harmful_hidden, harmless_hidden]).abs().mean(dim=0)
    print(f"Dropped {len(drop)} dimension(s) with the largest activations from "
          "the difference:")
    swamped = False
    for dimension in drop.tolist():
        share = (raw[dimension] ** 2 / raw.pow(2).sum()).item()
        t = (raw[dimension] / error[dimension]).item()
        print(f"  dim {dimension}: activation {magnitude[dimension]:.1f}, "
              f"{share:.1%} of the difference, t = {t:+.1f}")
        swamped |= share > 0.25 and abs(t) < 4
    if swamped:
        print("  One of them held most of the difference while being barely "
              "distinguishable from noise: without this option the direction "
              "would have been that dimension rather than the refusal.")

held_out = int(len(harmful_hidden) * HELDOUT_FRACTION)
if held_out >= 2:
    fit = slice(0, len(harmful_hidden) - held_out)
    check = slice(len(harmful_hidden) - held_out, None)
    probe = direction_between(harmful_hidden[fit], harmless_hidden[fit], drop)
    raw_fit, size_fit = separation_along(probe, harmful_hidden[fit], harmless_hidden[fit])
    raw_out, size_out = separation_along(probe, harmful_hidden[check], harmless_hidden[check])
    print(f"Separation, from {len(harmful_hidden) - held_out} prompts per class:")
    print(f"  on the prompts it was built from : {raw_fit:9.4f}  "
          f"(effect size {size_fit:6.2f})")
    print(f"  on {held_out} held-out prompts per class  : {raw_out:9.4f}  "
          f"(effect size {size_out:6.2f})")
    if size_out < 0.5:
        print("  WARNING: the direction does not generalise. It is noise, and "
              "ablating it will do nothing. Either this model does not refuse, "
              "or its weights did not load correctly.")
    elif size_out < 2.0:
        print("  The direction generalises weakly; more instructions would help.")

refusal_dir = direction_between(harmful_hidden, harmless_hidden, drop)
separation, effect_size = separation_along(refusal_dir, harmful_hidden, harmless_hidden)
print(f"Refusal direction: {tuple(refusal_dir.shape)}, "
      f"harmful/harmless separation {separation:.4f} (effect size {effect_size:.2f}, "
      "in-sample)")

out_path = MODEL_ID.replace("/", "_") + "_refusal_dir.pt"
torch.save(refusal_dir, out_path)
print(f"Saved {out_path}")
