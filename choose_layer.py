"""Pick LAYER_FRACTION by measuring it, instead of inheriting a constant.

compute_refusal_dir.py reads the residual stream at a fixed fraction of the way
through the model. 0.6 is what the upstream repo hardcodes; the technique it
implements picks the layer by searching. This is that search.

Every layer is evaluated from a single pass over the prompts -- one forward
captures all of them at once -- so the sweep costs about what one run of
compute_refusal_dir.py costs, not one run per layer.

Layers are ranked by the held-out effect size, because that is the one that was
checked against generation. Ablating with the direction from each layer in turn
and counting how often the model still refuses gives, on Qwen3-8B (11/16 refusals
before) and Qwen3-14B (10/16):

    depth   0.25  0.35  0.45  0.55  0.60  0.70  0.80
    8B      11/16 10/16  3/16  4/16  1/16  0/16  2/16
    14B     11/16 10/16  0/16  4/16  8/16  0/16  0/16
    effect   ~2.1  ~2.4  ~3.8  ~4.6  ~5.2  ~6.3  ~6.2

The effect size peaks at 0.70 on both, which is where the refusals go. The two
shallowest layers remove nothing at all, and their effect size says so: nothing
below about 4 works.

The other two numbers are reported but are not what the ranking uses. How much of
the difference the direction accounts for at *every* layer looked like the right
criterion -- the ablation is applied to the whole model -- but it does not
separate a layer that works from one that does nothing (19.3% at Qwen3-14B's
layer 10, which removes no refusals, against 21.1% at layer 18, which removes all
of them), and averaging over every layer of a deep model favours shallow
directions: on 93-layer Kimi K3 it ranked layer 23 first, where the effect size is
4.16 against 8.79 at layer 55.

The sweep also reports the largest activation and how concentrated the direction
is, because those are how the method fails. Past a certain layer some models
develop activations three orders of magnitude larger than the rest -- Qwen3-14B
goes from 84 to 12,096 at layer 21 -- and the difference of means becomes 97% one
dimension of noise. The refusal signal is still there underneath, but plain
difference-of-means cannot reach it, and the usable range ends at that layer.
"""

import random
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from model_loading import (format_duration, get_decoder_layers, get_input_device,
                           layer_count, load_model)

MODEL_ID = "../Kimi-K3"

DTYPE = "auto"
DEVICE_MAP = "auto"
MAX_MEMORY = None
OFFLOAD_DIR = None
GPU_HEADROOM = None
GPU_HEADROOM_FRACTION = 0.05
CPU_LIMIT = None
KEEP_PACKED = True

# Deepest fraction worth considering. Layers past it are neither built nor swept,
# which is what keeps this affordable on a model that does not fit in VRAM: the
# last tenth has never won on any model measured, and building it would cost
# another 140GiB on Kimi K3.
SWEEP_UNTIL = 0.9

# Fewer prompts than compute_refusal_dir.py needs. This is choosing between
# layers, not producing the direction, and the ranking settles long before the
# direction itself does.
instructions = 128
batch_size = 64

# A larger held-out share than compute_refusal_dir.py uses, for the same reason:
# every number here is an out-of-sample number, so it is worth spending prompts
# on.
HELDOUT_FRACTION = 0.375

# Must match compute_refusal_dir.py, or this picks a layer for a direction that
# will not be the one built there. See the comment on it in that file.
DROP_LARGEST_DIMENSIONS = 0

# An effect size below this is noise. Measured against generation: every layer
# scoring under about 4 removed no refusals and every layer over it removed most
# or all of them, so this floor is deliberately below that: it is a sanity check
# for "no direction here at all" (0.3 on a model that does not refuse), not the
# selection criterion, which is the ranking by effect size.
USABLE_EFFECT_SIZE = 2.0

print(f"Instruction count: {instructions}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"

with open("harmful.txt") as handle:
    harmful = handle.readlines()
with open("harmless.txt") as handle:
    harmless = handle.readlines()
harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)


def tokenize(batch):
    return tokenizer.apply_chat_template(
        conversation=[[{"role": "user", "content": insn}] for insn in batch],
        add_generation_prompt=True, tokenize=True, padding=True,
        return_dict=True, return_tensors="pt")


def batches(prompts):
    return [tokenize(prompts[start:start + batch_size])
            for start in range(0, len(prompts), batch_size)]


tokenized = {"harmful": batches(harmful_instructions),
             "harmless": batches(harmless_instructions)}
shapes = [tuple(inputs["input_ids"].shape)
          for group in tokenized.values() for inputs in group]
workload_tokens = max(rows * columns for rows, columns in shapes)
workload_sequence = max(columns for _, columns in shapes)
print(f"Largest forward pass: {workload_tokens} tokens "
      f"({max(rows for rows, _ in shapes)} prompts of up to "
      f"{workload_sequence} tokens)")

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
total_layers = layer_count(config)
if not total_layers:
    raise RuntimeError(f"cannot tell how many decoder layers {MODEL_ID} has")
deepest = min(total_layers - 1, int(total_layers * SWEEP_UNTIL))
print(f"Sweeping layers 1..{deepest} of {total_layers}")

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
                   max_layers=deepest + 1)

input_device = get_input_device(model)
layers = get_decoder_layers(model)
if deepest >= len(layers):
    raise RuntimeError(
        f"the model built {len(layers)} decoder layers but the sweep goes to "
        f"{deepest}; the config's layer count ({total_layers}) does not "
        "describe this architecture")

captured = {}


class StopForward(Exception):
    """Nothing past the deepest swept layer is read, so nothing past it runs."""


def capture_hook(index):
    def inner(module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            hidden = args[0]
        captured.setdefault(index, []).append(
            hidden.detach()[:, -1, :].to("cpu", torch.float32))
        if index == deepest:
            raise StopForward
    return inner


handles = [layer.register_forward_pre_hook(capture_hook(index), with_kwargs=True)
           for index, layer in enumerate(layers[:deepest + 1])]


def collect(label, group):
    rows = {}
    for number, inputs in enumerate(group, 1):
        print(f"  {label} batch {number}/{len(group)}", flush=True)
        captured.clear()
        started = time.perf_counter()
        try:
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(input_device),
                      attention_mask=inputs["attention_mask"].to(input_device),
                      use_cache=False)
        except StopForward:
            pass
        for index, chunk in captured.items():
            rows.setdefault(index, []).append(chunk[0])
        print(f"    {format_duration(time.perf_counter() - started)}", flush=True)
    return {index: torch.cat(chunks) for index, chunks in rows.items()}


started = time.perf_counter()
states = {label: collect(label, group) for label, group in tokenized.items()}
for handle in handles:
    handle.remove()
print(f"Collected the activations in {format_duration(time.perf_counter() - started)}")

held_out = max(2, int(instructions * HELDOUT_FRACTION))
fit = slice(0, instructions - held_out)
check = slice(instructions - held_out, None)
swept = sorted(index for index in states["harmful"] if index >= 1)


def direction_at(index, part):
    difference = (states["harmful"][index][part].mean(0)
                  - states["harmless"][index][part].mean(0))
    if DROP_LARGEST_DIMENSIONS > 0:
        magnitude = torch.cat([states["harmful"][index],
                               states["harmless"][index]]).abs().mean(0)
        difference = difference.clone()
        difference[magnitude.argsort(descending=True)[:DROP_LARGEST_DIMENSIONS]] = 0
    return difference / difference.norm()


def held_out_difference(index):
    """The difference the direction is scored against, excluded dimensions and all."""
    difference = (states["harmful"][index][check].mean(0)
                  - states["harmless"][index][check].mean(0))
    if DROP_LARGEST_DIMENSIONS > 0:
        magnitude = torch.cat([states["harmful"][index],
                               states["harmless"][index]]).abs().mean(0)
        difference = difference.clone()
        difference[magnitude.argsort(descending=True)[:DROP_LARGEST_DIMENSIONS]] = 0
    return difference


def effect_size(direction, index):
    a = states["harmful"][index][check] @ direction
    b = states["harmless"][index][check] @ direction
    spread = torch.sqrt((a.var() + b.var()) / 2).clamp_min(1e-9)
    return ((a.mean() - b.mean()) / spread).item()


print(f"\n{instructions - held_out} prompts per class build each direction, "
      f"{held_out} check it"
      + (f", with the {DROP_LARGEST_DIMENSIONS} largest dimensions excluded\n"
         if DROP_LARGEST_DIMENSIONS else "\n"))
print(f"{'layer':>5} {'fraction':>9} {'effect size':>12} {'explains here':>14} "
      f"{'explains everywhere':>20} {'max |h|':>10} {'top dim':>8}")

rows = []
for index in swept:
    probe = direction_at(index, fit)
    here = held_out_difference(index)
    explains_here = (torch.dot(here, probe).abs() / here.norm().clamp_min(1e-9)).item()
    scores = []
    for other in swept:
        elsewhere = held_out_difference(other)
        if elsewhere.norm() > 1e-6:
            scores.append((torch.dot(elsewhere, probe).abs()
                           / elsewhere.norm()).item())
    everywhere = sum(scores) / len(scores) if scores else 0.0
    peak = max(states[label][index].abs().max().item() for label in states)
    concentration = (probe.abs().max() / probe.norm()).item()
    size = effect_size(probe, index)
    rows.append((index, size, explains_here, everywhere, peak, concentration))
    print(f"{index:>5} {index / total_layers:>9.2f} {size:>12.2f} "
          f"{explains_here:>13.1%} {everywhere:>19.1%} {peak:>10.1f} "
          f"{concentration:>7.0%}")

usable = [row for row in rows if row[1] >= USABLE_EFFECT_SIZE]
if not usable:
    print(f"\nNo layer reaches an effect size of {USABLE_EFFECT_SIZE}. Either this "
          "model does not refuse, or its weights did not load correctly; check "
          "the load report for MISSING keys before reading anything into the "
          "numbers above.")
    raise SystemExit(1)

# The effect size rises to a noisy plateau rather than a peak, so taking its
# argmax lands wherever the noise is highest -- usually the deepest layer swept,
# and the deepest layers are where generation starts to degrade (Qwen3-8B at 0.80
# left refusals back at 2/16 and broke one answer in three, against 0/16 and none
# at 0.70). The shallowest layer that keeps most of the separation is the stable
# choice.
ceiling = max(row[1] for row in usable)
best = min((row for row in usable if row[1] >= 0.9 * ceiling),
           key=lambda row: row[0])
print(f"\nBest layer: {best[0]} of {total_layers} -> "
      f"LAYER_FRACTION = {best[0] / total_layers:.2f}")
print(f"  effect size {best[1]:.2f}, explains {best[2]:.1%} of its own layer's "
      f"difference and {best[3]:.1%} across all of them")
shallow = [row for row in usable if row[0] < total_layers * 0.4]
if best in shallow:
    print("  WARNING: that is unusually shallow. On every model checked, a "
          "direction from the first 40% of the layers removed no refusals at "
          "all, whatever it scored. Treat this as a sign that something is off "
          "rather than as a recommendation.")

deepest_usable = max(row[0] for row in usable)
if deepest_usable < swept[-1]:
    print(f"  usable range ends at layer {deepest_usable} "
          f"(fraction {deepest_usable / total_layers:.2f}); past it the effect "
          f"size falls below {USABLE_EFFECT_SIZE}")
concentrated = [row for row in rows if row[5] > 0.5]
if concentrated:
    print(f"  {len(concentrated)} layer(s) from {concentrated[0][0]} on have a "
          f"direction that is over half one dimension, next to activations up to "
          f"{max(row[4] for row in concentrated):.0f}. That is a massive-activation "
          "model: the difference of means there is noise in one huge dimension, "
          "and those layers cannot be used however good they look in-sample.")
