"""Pick LAYER_FRACTION by measuring it, instead of inheriting a constant.

compute_refusal_dir.py reads the residual stream at a fixed fraction of the way
through the model. 0.6 is what the upstream repo hardcodes; the technique it
implements picks the layer by searching. This is that search.

Every layer is evaluated from a single pass over the prompts -- one forward
captures all of them at once -- so the sweep costs about what one run of
compute_refusal_dir.py costs, not one run per layer.

Candidate layers are found by the held-out effect size, because that is the one
that was checked against generation. Ablating with the direction from each layer
in turn and counting how often the model still refuses gives, on Qwen3-8B (11/16
refusals before) and Qwen3-14B (10/16):

    depth   0.25  0.35  0.45  0.55  0.60  0.70  0.80
    8B      11/16 10/16  3/16  4/16  1/16  0/16  2/16
    14B     11/16 10/16  0/16  4/16  8/16  0/16  0/16
    effect   ~2.1  ~2.4  ~3.8  ~4.6  ~5.2  ~6.3  ~6.2

The effect size peaks at 0.70 on both, which is where the refusals go. The two
shallowest layers remove nothing at all, and their effect size says so: nothing
below about 4 works.

What it cannot do is pick between the layers that clear that bar, because it does
not rise to a peak but to a plateau, and the top of a plateau is wherever the
noise landed -- on Hy-Memory-30B-A3B layers 25 to 43 are all within 10% of each
other and the maximum moved from 36 to 42 between two draws of the prompts.
Choosing inside the plateau is what "explains everywhere" is for: how much of the
difference the direction accounts for at *every* layer, rather than only at its
own, which is what the direction is actually asked to do, since the ablation is
applied to the whole model. It is a sharper number than it looks -- on the two
runs above it chose 0.69 and 0.73 where the shallow edge of the plateau gave 0.52
and 0.71 -- and 0.70 is where the table above puts the optimum.

It is confined to the plateau on purpose, and `choose` says why: on its own it
barely separates a layer that works from one that does nothing, and its argmax on
93-layer Kimi K3 is layer 23, where the effect size is 4.16 against 8.79 at 55.

The sweep also reports the largest activation and how concentrated the direction
is, because those are how the method fails. Past a certain layer some models
develop activations three orders of magnitude larger than the rest -- Qwen3-14B
goes from 84 to 12,096 at layer 21 -- and the difference of means becomes 97% one
dimension of noise. The refusal signal is still there underneath, but plain
difference-of-means cannot reach it, and the usable range ends at that layer.

DROP_LARGEST_DIMENSIONS is the answer to that, and it is swept here too: it takes
a list, and every value in it is measured from the same captured activations,
because how many dimensions to exclude changes only the arithmetic that turns
those activations into a direction. Reloading the model to try another value buys
nothing. The value is reported with the layer, since the two interact -- excluding
the largest dimensions is what makes the deeper layers of a massive-activation
model usable at all -- and it is judged on split-half agreement rather than on
effect size, for the reason given on `agreement`.
"""

import random
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from model_loading import (format_duration, get_decoder_layers, get_input_device,
                           layer_count, load_model)

MODEL_ID = "../Hy-Memory-30B-A3B"

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
#
# It is also what keeps this model loadable at all on a unified-memory machine.
# At 0.9 the 44 layers built are 51.3GiB against a 52.3GiB budget; one layer more
# is 52.5GiB and does not fit, and 1.0 (all 48, 56.0GiB) is what the kernel used
# to answer by killing the process 96% of the way through the load. Sweeping
# deeper than the memory allows needs OFFLOAD_DIR, which streams the offloaded
# layers from disk on every one of the forward passes below.
SWEEP_UNTIL = 0.9

# The same count compute_refusal_dir.py uses, because at 128 the ranking was not
# settled: two unseeded runs on Hy-Memory-30B-A3B, same model and same code,
# disagreed by nine layers (0.52 against 0.71), and the per-layer effect sizes
# differed by a factor of two in both directions -- half as large as the other
# run below layer 10, twice as large above layer 30. The forward passes are
# aborted at the deepest swept layer and cost about three seconds per batch, so
# the extra prompts are the cheapest part of this script.
instructions = 1024
batch_size = 64

# Unseeded, the output of this script is a different constant every time it runs,
# and the constant gets hardcoded into compute_refusal_dir.py: it has to be
# reproducible to be worth writing down. The draw is from 1040 harmful and 63622
# harmless prompts, so the seed matters rather more than the usual token effort
# at determinism.
random.seed(1)

# A larger held-out share than compute_refusal_dir.py uses, for the same reason:
# every number here is an out-of-sample number, so it is worth spending prompts
# on.
HELDOUT_FRACTION = 0.375

# The same knob as compute_refusal_dir.py's, for the same reason (the comment on
# it in that file explains what it is for). A single value has to match the one
# set there, or this picks a layer for a direction that will not be the one built
# there.
#
# A list sweeps several values instead, and that costs nothing: the count only
# enters the arithmetic that turns the captured activations into a direction, so
# every value is measured from the same forward passes. Reloading the model to
# try another one is pure waste. The sweep then reports which value to set in
# compute_refusal_dir.py alongside which layer.
DROP_LARGEST_DIMENSIONS = [0, 1, 2, 3]

# An effect size below this is noise. Measured against generation: every layer
# scoring under about 4 removed no refusals and every layer over it removed most
# or all of them, so this floor is deliberately below that: it is a sanity check
# for "no direction here at all" (0.3 on a model that does not refuse), not the
# selection criterion.
USABLE_EFFECT_SIZE = 2.0

# How close to the best effect size a layer has to be to count as part of the
# plateau that the choice is then made inside. The effect size rises to a noisy
# plateau rather than a peak, so the top of it is wherever the noise landed: on
# Hy-Memory-30B-A3B the plateau runs from layer 25 to layer 43 and its maximum
# moved from 36 to 42 between two draws of the prompts. 10% is wide enough to
# hold the whole plateau and narrow enough to exclude the layers below it.
PLATEAU_FRACTION = 0.9

# Below this, the direction is being decided by which prompts were drawn rather
# than by refusal. 0.89 is what a healthy layer measures and 0.19 is what a
# massive-activation layer measures before the largest dimensions are excluded,
# so anything under about 0.8 means this many excluded dimensions is not enough.
GOOD_AGREEMENT = 0.8

drop_counts = sorted({max(0, int(count)) for count in (
    DROP_LARGEST_DIMENSIONS
    if isinstance(DROP_LARGEST_DIMENSIONS, (list, tuple, set, range))
    else (DROP_LARGEST_DIMENSIONS,))})

print(f"Instruction count: {instructions}")
print("Excluded dimensions to try: "
      + ", ".join(str(count) for count in drop_counts))

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


_magnitude_order = {}


def magnitude_order(index):
    """Layer `index`'s dimensions by activation magnitude, largest first.

    Cached because it does not depend on the number of dimensions dropped, only
    on how many of this list are taken, so the whole sweep over `drop_counts`
    shares one sort per layer.
    """
    if index not in _magnitude_order:
        magnitude = torch.cat([states["harmful"][index],
                               states["harmless"][index]]).abs().mean(0)
        _magnitude_order[index] = magnitude.argsort(descending=True)
    return _magnitude_order[index]


def difference_at(index, part, drop):
    """Difference of the class means over `part`, with `drop` dimensions zeroed."""
    difference = (states["harmful"][index][part].mean(0)
                  - states["harmless"][index][part].mean(0))
    if drop > 0:
        difference = difference.clone()
        difference[magnitude_order(index)[:drop]] = 0
    return difference


def direction_at(index, part, drop):
    difference = difference_at(index, part, drop)
    return difference / difference.norm().clamp_min(1e-9)


def effect_size(direction, index):
    a = states["harmful"][index][check] @ direction
    b = states["harmless"][index][check] @ direction
    spread = torch.sqrt((a.var() + b.var()) / 2).clamp_min(1e-9)
    return ((a.mean() - b.mean()) / spread).item()


def agreement(index, drop):
    """Cosine between the directions built from two disjoint halves of the fit prompts.

    This is the number that decides how many dimensions to drop, and the effect
    size is not: a direction that points at one huge noisy dimension still
    separates the prompts it was built from, so it can score well while being
    determined by the draw rather than by refusal. Agreement sees that directly
    -- on Qwen3-14B's layer 20 it reads 0.19 with nothing excluded and 0.89 with
    the largest two excluded, where the effect size moves far less.
    """
    half = (instructions - held_out) // 2
    first = direction_at(index, slice(0, half), drop)
    second = direction_at(index, slice(half, instructions - held_out), drop)
    return torch.dot(first, second).item()


def sweep(drop):
    """One row per layer, for directions with `drop` dimensions excluded."""
    # The held-out difference of every layer, because the "explains everywhere"
    # column needs all of them for each layer it scores.
    differences = {index: difference_at(index, check, drop) for index in swept}
    rows = []
    for index in swept:
        probe = direction_at(index, fit, drop)
        here = differences[index]
        explains_here = (torch.dot(here, probe).abs()
                         / here.norm().clamp_min(1e-9)).item()
        scores = [(torch.dot(elsewhere, probe).abs() / elsewhere.norm()).item()
                  for elsewhere in differences.values()
                  if elsewhere.norm() > 1e-6]
        everywhere = sum(scores) / len(scores) if scores else 0.0
        peak = max(states[label][index].abs().max().item() for label in states)
        concentration = (probe.abs().max() / probe.norm()).item()
        rows.append((index, effect_size(probe, index), agreement(index, drop),
                     explains_here, everywhere, peak, concentration))
    return rows


LAYER, EFFECT, AGREE, HERE, EVERYWHERE, PEAK, TOPDIM = range(7)


def choose(rows):
    """The layer to use out of `rows`, or None if none of them is usable.

    Two criteria, because neither works alone.

    The effect size decides which layers are candidates. It is the number that
    was checked against generation, and it is the only one that tells a layer
    with a direction from a layer with none. But it rises to a noisy plateau
    rather than a peak, so its argmax is wherever the noise landed -- usually one
    of the deepest layers swept, which is where generation starts to degrade
    (Qwen3-8B at 0.80 left refusals back at 2/16 and broke one answer in three,
    against 0/16 and none at 0.70).

    Inside the plateau the choice is by how much of the difference the direction
    accounts for at *every* layer, which is the criterion that matches what the
    direction is used for: the ablation is applied to the whole model, not to the
    layer the direction was read from. Taking the shallow edge of the plateau
    instead -- which is what this did before -- is what made it unstable, because
    the plateau's shallow edge is set by where the noise put the maximum: on
    Hy-Memory-30B-A3B two draws of the prompts chose 0.52 and 0.71 that way,
    against 0.69 and 0.73 by this criterion, and 0.70 is where the generation
    check in the module docstring puts the optimum on both Qwen3 models.

    What this deliberately does not do is rank *all* the layers by it. It barely
    separates a layer that works from one that does nothing (19.3% at Qwen3-14B's
    layer 10, which removes no refusals, against 21.1% at layer 18, which removes
    all of them), and on 93-layer Kimi K3 its argmax is layer 23, where the
    effect size is 4.16 against 8.79 at layer 55. Confined to the plateau, that
    failure cannot happen: those layers are not candidates.
    """
    usable = [row for row in rows if row[EFFECT] >= USABLE_EFFECT_SIZE]
    if not usable:
        return None
    ceiling = max(row[EFFECT] for row in usable)
    plateau = [row for row in usable if row[EFFECT] >= PLATEAU_FRACTION * ceiling]
    # Shallowest wins a tie, which is the old rule applied where it is harmless.
    return max(plateau, key=lambda row: (row[EVERYWHERE], -row[LAYER]))


print(f"\n{instructions - held_out} prompts per class build each direction, "
      f"{held_out} check it, and the agreement column splits the first "
      f"{instructions - held_out} in half")

results = {}
for drop in drop_counts:
    print("\n--- " + ("nothing excluded (the difference of means as-is)"
                       if drop == 0 else
                       f"{drop} largest dimension(s) excluded") + " ---")
    print(f"{'layer':>5} {'fraction':>9} {'effect size':>12} {'agreement':>10} "
          f"{'explains here':>14} {'explains everywhere':>20} {'max |h|':>10} "
          f"{'top dim':>8}")
    rows = sweep(drop)
    for row in rows:
        print(f"{row[LAYER]:>5} {row[LAYER] / total_layers:>9.2f} "
              f"{row[EFFECT]:>12.2f} {row[AGREE]:>10.2f} {row[HERE]:>13.1%} "
              f"{row[EVERYWHERE]:>19.1%} {row[PEAK]:>10.1f} {row[TOPDIM]:>7.0%}")
    best = choose(rows)
    results[drop] = (rows, best)
    if best is None:
        print(f"  no layer reaches an effect size of {USABLE_EFFECT_SIZE} with "
              f"{drop} dimension(s) excluded")
        continue
    print(f"  best layer {best[LAYER]} (fraction "
          f"{best[LAYER] / total_layers:.2f}), effect size {best[EFFECT]:.2f}, "
          f"agreement {best[AGREE]:.2f}")

if all(best is None for _, best in results.values()):
    print(f"\nNo layer reaches an effect size of {USABLE_EFFECT_SIZE} at any of "
          "the excluded-dimension counts tried. Either this model does not "
          "refuse, or its weights did not load correctly; check the load report "
          "for MISSING keys before reading anything into the numbers above.")
    raise SystemExit(1)

if len(drop_counts) > 1:
    print(f"\n{'dropped':>7} {'layer':>5} {'fraction':>9} {'effect size':>12} "
          f"{'agreement':>10} {'top dim':>8}")
    for drop, (_, best) in results.items():
        if best is None:
            print(f"{drop:>7} {'--':>5} {'':>9} {'unusable':>12}")
            continue
        print(f"{drop:>7} {best[LAYER]:>5} {best[LAYER] / total_layers:>9.2f} "
              f"{best[EFFECT]:>12.2f} {best[AGREE]:>10.2f} {best[TOPDIM]:>7.0%}")

# The fewest count that gets within 2% of the best effect size wins, rather than
# the argmax. Each count is scored at its own best layer, and those layers are
# not the same layer, so the argmax is partly measuring which count's plateau the
# noise happened to raise: on Hy-Memory-30B-A3B the four counts score 7.09, 7.35,
# 7.35 and 7.46 at their own best layers and 7.37, 7.35, 7.35 and 7.34 at one
# fixed layer. The spread between counts is 0.03 where it can be read honestly,
# and 0.37 where it cannot.
candidates = [(drop, best) for drop, (_, best) in results.items()
              if best is not None]
reachable = max(best[EFFECT] for _, best in candidates)
drop, best = min((pair for pair in candidates
                  if pair[1][EFFECT] >= 0.98 * reachable),
                 key=lambda pair: pair[0])
rows = results[drop][0]

print(f"\nBest layer: {best[LAYER]} of {total_layers} -> "
      f"LAYER_FRACTION = {best[LAYER] / total_layers:.2f}, "
      f"DROP_LARGEST_DIMENSIONS = {drop}")
print(f"  effect size {best[EFFECT]:.2f}, agreement {best[AGREE]:.2f}, explains "
      f"{best[HERE]:.1%} of its own layer's difference and {best[EVERYWHERE]:.1%} "
      "across all of them")

# The comparison that is not confounded by which layer each count won at. One
# layer, every count: whatever spread is left is what the exclusion is actually
# worth, and it is the number to compare against the difference between two runs
# before reading anything into it.
if len(drop_counts) > 1:
    at_chosen = []
    for count, (other_rows, _) in results.items():
        row = next((row for row in other_rows if row[LAYER] == best[LAYER]), None)
        if row is not None:
            at_chosen.append(f"{count}: {row[EFFECT]:.2f}")
    if at_chosen:
        print(f"  at layer {best[LAYER]} itself, excluding "
              + ", ".join(at_chosen)
              + " -- a count that looks better in the table above but not here "
                "won by moving the layer, not by improving the direction")

# A tie means this model does not have the problem the exclusion exists for, and
# saying so is worth a line: it is the difference between "0 is right here" and
# "0 and 2 are indistinguishable, and 2 is insurance that costs nothing".
tied = [count for count, pair in results.items()
        if count != drop and pair[1] is not None
        and pair[1][EFFECT] >= 0.98 * reachable]
if tied:
    print("  excluding "
          + ", ".join(str(count) for count in tied)
          + " instead is within 2% of that. The exclusion is insurance against "
            "one dimension dominating the direction, and the `top dim` column "
            "above says that is not happening at this layer, so the fewest is "
            "reported; raising it is equally defensible and costs nothing "
            "measurable here")

if best[AGREE] < GOOD_AGREEMENT:
    better = [(count, pair[1]) for count, pair in results.items()
              if pair[1] is not None and pair[1][AGREE] >= GOOD_AGREEMENT]
    print(f"  WARNING: agreement of {best[AGREE]:.2f} means two halves of the "
          "prompts largely disagree about where this direction points, so it is "
          "being set by the draw as much as by refusal"
          + (f". Excluding {better[0][0]} dimension(s) instead reaches "
             f"{better[0][1][AGREE]:.2f}; prefer that even at a lower effect size"
             if better else
             ", and no excluded-dimension count tried fixes it. Add larger "
             "values to DROP_LARGEST_DIMENSIONS."))

if best[LAYER] < total_layers * 0.4:
    print("  WARNING: that is unusually shallow. On every model checked, a "
          "direction from the first 40% of the layers removed no refusals at "
          "all, whatever it scored. Treat this as a sign that something is off "
          "rather than as a recommendation.")

usable = [row for row in rows if row[EFFECT] >= USABLE_EFFECT_SIZE]
deepest_usable = max(row[LAYER] for row in usable)
if deepest_usable < swept[-1]:
    print(f"  usable range ends at layer {deepest_usable} "
          f"(fraction {deepest_usable / total_layers:.2f}); past it the effect "
          f"size falls below {USABLE_EFFECT_SIZE}")
concentrated = [row for row in rows if row[TOPDIM] > 0.5]
if concentrated:
    print(f"  {len(concentrated)} layer(s) from {concentrated[0][LAYER]} on have "
          "a direction that is over half one dimension, next to activations up "
          f"to {max(row[PEAK] for row in concentrated):.0f}. That is a "
          "massive-activation model: the difference of means there is noise in "
          "one huge dimension, and those layers cannot be used however good they "
          "look in-sample.")
