"""How far the ablated model's output distribution moved from the original's.

The refusal direction is removed from every layer's weights, which is a global
edit, so the question it raises is global too: what else moved? An ablation that
works looks like a large change on prompts the model used to refuse and almost
no change on everything else. One that does not looks like a change everywhere,
and the model that comes out of it is worse at its job in ways no refusal count
would show.

Two numbers, from one forward pass per prompt per model:

  * KL on the next token. The full distribution over the vocabulary at the last
    prompt position -- the token the model is about to emit, which is where a
    refusal begins ("I", "Sorry", "很抱歉"). This is the behavioural change, and
    it is the one that should be large on harmful prompts.

  * The teacher-forced log-probability of the prompt itself, every position of
    it. The prompts are ordinary text, so this is a perplexity measurement in
    disguise: it says whether the edit damaged the model's language modelling,
    and it should be near zero on both classes. It costs nothing, because the
    forward pass computes those logits anyway.

Reported separately for harmful and harmless prompts, because the ratio between
them is the whole point. The absolute KL means little on its own -- it scales
with how sharp the model's distributions are -- but 100x more divergence on
harmful than on harmless prompts means the edit went where it was aimed.

Which position counts as "the next token" is not a detail on a reasoning model,
and getting it wrong makes this whole script read 0.0x on an ablation that
worked. `add_generation_prompt` on Dory1.0-4B ends the prompt with
`<|im_start|>assistant\n<think>\n`, so the next token is the first token of the
*thinking*, not of the answer -- and a reasoning model does not refuse there. It
measured `'The'` for 64 of 64 harmful prompts at p = 0.9998 and an entropy of
0.0015 nats: a delta function, with no room for any edit to move probability
mass and nothing about refusal in it either way. The refusal is decided inside
the thinking block and emitted after `</think>`, hundreds of tokens later.
PROBE below is what moves the probe to where the answer starts; it takes the
same three values as compute_refusal_dir.py's, and probes.py documents them.


The two models are never in memory at the same time: each is loaded in a child
process that writes its distributions to a cache file and exits, which is also
what makes the loads survivable on a machine where one copy of this model is
half the RAM. Re-running reuses whatever cache files are already there, so the
comparison can be redone without reloading anything.
"""

import hashlib
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

import probes
from model_loading import (format_duration, format_memory,
                           get_input_device, load_model)

BASE_MODEL = "../NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
ABLATED_MODEL = "../NVIDIA-Nemotron-Labs-Dory1.0-4B-Abliterated"

# Either pair can be overridden on the command line, which is what comparing
# several builds against the same baseline needs:
#   python measure_kl.py ../Model-Abliterated ../Model-Abliterated-MXFP8
if len(sys.argv) == 3 and sys.argv[1] != "--capture":
    BASE_MODEL, ABLATED_MODEL = sys.argv[1], sys.argv[2]

# Which token the KL is measured on. The same three values as
# compute_refusal_dir.py's `PROBE` and the same machinery in probes.py, but
# **this is a different choice and must not be copied from there.** There it
# names the position a direction is *read from*; here it names the position the
# edit is *evaluated at*, and the two are independent -- a direction read at
# `</think>` is still judged by what the model says when it answers.
#
# Leave this at "answer" unless there is a specific reason not to.
#
#   "answer"    the first token of the answer. The only setting that can see a
#               refusal being removed, whatever probe built the direction.
#   "thinking"  the first token of the *thinking*, where Dory1.0-4B says `'The'`
#               with probability 0.9998 whatever it was asked (entropy 0.0015
#               nats). The harmful column then reads ~0 for any edit, however
#               large, because there is no probability mass free to move; a
#               working ablation measured 1.8x this way and a broken one 0.0x.
#   "close"     the first token after the model's own `</think>`. A different and
#               far narrower question -- "given *this* reasoning, would it still
#               refuse?" -- whose answer on Dory1.0-4B is always "yes,
#               identically": every direction and every span tried leaves the
#               harmful top-1 at 100.0% with a KL under 0.03, because by then
#               the refusal is text in the context and no edit to the residual
#               stream removes text the model has already written.
#
#               Measuring here and concluding the ablation failed is the mistake
#               this comment exists to prevent. The checkpoint built from the
#               1024-prompt "close" direction reads 0.3x at this position and
#               46.3x at "answer", on the same 64 prompts.
#
# The traces for "close" always come from BASE_MODEL, so that both models are
# asked about the same reasoning and the comparison is controlled. That also
# changes what the prompt-log-probability column means; see `compare`.
PROBE = "answer"
TRACE_BUDGET = 2048
MEDIUM_EFFORT = True
TRACE_BATCH_SIZE = 24

# Prompts per class. The KL per prompt is a per-prompt number rather than an
# average over a fitted direction, so this does not need the 1024 that
# compute_refusal_dir.py does; the spread across prompts is reported and says
# whether more would help. Each one costs a forward pass and 472KiB of cache
# (one float32 per vocabulary entry).
PROMPTS = 1024
BATCH_SIZE = 24

# Must match whatever the other scripts use, so that the prompts the direction
# was built from and the prompts it is measured on are drawn the same way.
SEED = 1

DTYPE = "auto"
DEVICE_MAP = "auto"
MAX_MEMORY = None
OFFLOAD_DIR = None
GPU_HEADROOM = None
GPU_HEADROOM_FRACTION = 0.05
CPU_LIMIT = None
KEEP_PACKED = True

CACHE_DIR = Path("kl_cache")


def prompts():
    """The same two prompt sets every run, in the same order."""
    random.seed(SEED)
    with open("harmful.txt") as handle:
        harmful = random.sample(handle.readlines(), PROMPTS)
    with open("harmless.txt") as handle:
        harmless = random.sample(handle.readlines(), PROMPTS)
    return harmful, harmless


def fingerprint(harmful, harmless):
    """Identifies the measurement, so a stale cache cannot be compared by mistake.

    The probe position is part of it: two caches taken at different positions
    are not comparable at all, and the difference between them does not look
    like an error.
    """
    digest = hashlib.sha256()
    for prompt in harmful + harmless:
        digest.update(prompt.encode())
    digest.update(f"probe={PROBE}|{TRACE_BUDGET}|{MEDIUM_EFFORT}".encode())
    return digest.hexdigest()[:16]


def cache_path(model_id):
    return CACHE_DIR / (model_id.strip("/").replace("/", "_") + "_kl.pt")


def weights_stamp(model_id):
    """Identifies the weights behind `model_id`, cheaply.

    Without this, rebuilding a checkpoint in place -- which is what re-running
    save_ablated_model.py does -- leaves a cache that still matches on the
    prompt set and is silently compared against the new model's. The failure
    looks like an ablation that changed nothing, which is exactly the thing
    this script is used to rule out.
    """
    directory = Path(model_id)
    if not directory.is_dir():
        return model_id         # a Hub id is immutable enough for this
    return sorted((path.name, path.stat().st_size, int(path.stat().st_mtime))
                  for path in directory.glob("*.safetensors"))


def capture(model_id):
    """Load `model_id`, record its distributions, write the cache, exit.

    Runs in its own process. Nothing here is shared with the comparison below
    except the file it writes, which is the point: the 56GiB this model occupies
    is returned to the machine when the process ends, and no sequence of `del`
    and `empty_cache()` can promise that as reliably.
    """
    harmful, harmless = prompts()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"

    groups = {"harmful": harmful, "harmless": harmless}
    # The traces come from BASE_MODEL whichever model is being captured, so that
    # both are asked about the same reasoning and the difference is the model
    # rather than the context. The cache makes the second capture free.
    traces = {label: None for label in groups}
    if PROBE == "close":
        traces = probes.reasoning_traces(
            BASE_MODEL, tokenizer, groups, budget=TRACE_BUDGET,
            medium_effort=MEDIUM_EFFORT, batch_size=TRACE_BATCH_SIZE,
            dtype=DTYPE, device_map=DEVICE_MAP, max_memory=MAX_MEMORY,
            offload_dir=OFFLOAD_DIR, gpu_headroom=GPU_HEADROOM,
            gpu_headroom_fraction=GPU_HEADROOM_FRACTION, cpu_limit=CPU_LIMIT,
            keep_packed=KEEP_PACKED)

    # One id sequence per prompt, built once. A prompt whose reasoning never
    # reached `</think>` has no position to read and drops out of both classes'
    # counts, which the per-class statistics below already allow for.
    sequences = {}
    for label, group in groups.items():
        sequences[label], kept = probes.probe_sequences(
            tokenizer, group, PROBE, traces[label], MEDIUM_EFFORT)
        if len(kept) < len(group):
            print(f"  {len(group) - len(kept)} {label} prompt(s) dropped: no "
                  f"`</think>` within {TRACE_BUDGET} tokens")

    batches = {label: [probes.pad_left(group[start:start + BATCH_SIZE],
                                       tokenizer.pad_token_id)
                       for start in range(0, len(group), BATCH_SIZE)]
               for label, group in sequences.items()}
    shapes = [tuple(inputs["input_ids"].shape)
              for group in batches.values() for inputs in group]
    workload_tokens = max(rows * columns for rows, columns in shapes)
    workload_sequence = max(columns for _, columns in shapes)
    print(f"Largest forward pass: {workload_tokens} tokens "
          f"({max(rows for rows, _ in shapes)} prompts of up to "
          f"{workload_sequence} tokens)")

    model = load_model(model_id,
                       dtype=DTYPE,
                       device_map=DEVICE_MAP,
                       max_memory=MAX_MEMORY,
                       offload_dir=OFFLOAD_DIR,
                       workload_tokens=workload_tokens,
                       workload_sequence=workload_sequence,
                       gpu_headroom=GPU_HEADROOM,
                       gpu_headroom_fraction=GPU_HEADROOM_FRACTION,
                       cpu_limit=CPU_LIMIT,
                       keep_packed=KEEP_PACKED)
    model.eval()
    input_device = get_input_device(model)

    captured = {}
    started = time.perf_counter()
    for label, group in batches.items():
        next_token = []
        text = []
        for number, inputs in enumerate(group, 1):
            print(f"  {label} batch {number}/{len(group)}", flush=True)
            ids = inputs["input_ids"].to(input_device)
            mask = inputs["attention_mask"].to(input_device)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask,
                               use_cache=False).logits
            # float32 even if the model ran in bfloat16: a KL is a difference of
            # logs, and bfloat16 has three decimal digits to spend on it.
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            next_token.append(log_probs[:, -1, :].cpu())

            # The teacher-forced log-probability the model gives each prompt
            # token, from the position before it. Padding is excluded by the
            # mask; `pos + 1` has to be a real token as well as `pos`.
            targets = ids[:, 1:]
            gathered = log_probs[:, :-1, :].gather(
                2, targets.unsqueeze(-1)).squeeze(-1)
            valid = (mask[:, :-1] * mask[:, 1:]).bool()
            text.append(torch.where(valid, gathered, torch.nan).cpu())
            del logits, log_probs
        captured[label] = {
            "next_token": torch.cat(next_token),
            # Ragged across batches, because each one pads to its own longest
            # prompt. Kept per batch rather than padded to a common width.
            "text": text,
        }
    print(f"Collected in {format_duration(time.perf_counter() - started)}")

    CACHE_DIR.mkdir(exist_ok=True)
    path = cache_path(model_id)
    torch.save({"model_id": model_id,
                "fingerprint": fingerprint(harmful, harmless),
                "prompts": PROMPTS,
                "weights": weights_stamp(model_id),
                "captured": captured}, path)
    print(f"Wrote {path} ({format_memory(path.stat().st_size)})")


def kl(p_log, q_log):
    """KL(P || Q) per row, from log-probabilities. In nats."""
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(-1)


def percentiles(values, points=(50, 90, 99)):
    ordered = values.sort().values
    return {point: ordered[min(len(ordered) - 1,
                               int(len(ordered) * point / 100))].item()
            for point in points}


def compare(base, ablated):
    if base["fingerprint"] != ablated["fingerprint"]:
        raise RuntimeError(
            "the two caches were built from different prompt sets "
            f"({base['fingerprint']} vs {ablated['fingerprint']}); delete "
            f"{CACHE_DIR}/ and run again")

    print(f"\n{base['prompts']} prompts per class, "
          f"{BASE_MODEL} -> {ABLATED_MODEL}\n")
    print(f"{'':>9} {'KL(base||abl)':>26} {'KL(abl||base)':>14} "
          f"{'top-1 kept':>11} {'prompt logprob':>15}")
    print(f"{'':>9} {'mean':>8} {'p50':>8} {'p99':>8} {'mean':>14} "
          f"{'':>11} {'mean delta':>15}")

    summary = {}
    for label in ("harmful", "harmless"):
        p_log = base["captured"][label]["next_token"]
        q_log = ablated["captured"][label]["next_token"]
        forward = kl(p_log, q_log)
        backward = kl(q_log, p_log)
        agree = (p_log.argmax(-1) == q_log.argmax(-1)).float().mean().item()

        # Teacher-forced log-probability of the prompt text itself. Positive
        # means the ablated model finds the same text *more* likely.
        deltas = []
        for before, after in zip(base["captured"][label]["text"],
                                 ablated["captured"][label]["text"]):
            difference = (after - before).flatten()
            deltas.append(difference[~difference.isnan()])
        delta = torch.cat(deltas)

        points = percentiles(forward)
        summary[label] = (forward.mean().item(), backward.mean().item(),
                          agree, delta.mean().item(), delta.abs().mean().item())
        print(f"{label:>9} {forward.mean().item():>8.4f} {points[50]:>8.4f} "
              f"{points[99]:>8.4f} {backward.mean().item():>14.4f} "
              f"{agree:>10.1%} {delta.mean().item():>+15.4f}")

    # What the two models actually want to say first. The KL says how far the
    # distributions moved; this says whether what moved was the refusal, which
    # is the only reason to run any of this.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print()
    for label in ("harmful", "harmless"):
        for which, cache in (("base   ", base), ("ablated", ablated)):
            counts = Counter(
                tokenizer.decode([token]) for token in
                cache["captured"][label]["next_token"].argmax(-1).tolist())
            top = ", ".join(f"{text!r} x{count}"
                            for text, count in counts.most_common(4))
            print(f"  first token, {label:>8} prompts, {which}: {top}")

    harmful_kl, harmless_kl = summary["harmful"][0], summary["harmless"][0]
    ratio = harmful_kl / harmless_kl if harmless_kl > 1e-12 else float("inf")
    print(f"\nHarmful/harmless KL ratio: {ratio:.1f}x")
    if PROBE != "answer":
        print(f"  Read that with PROBE = {PROBE!r} in mind: it is not a number "
              "about the ablation yet.")
        if PROBE == "close":
            print("  This position is *after* the model's own reasoning, and no "
                  "ablation tried has ever moved it -- the harmful top-1 stays "
                  "at 100% and the KL under 0.03 whatever the edit, because the "
                  "refusal is already in the context as text. A checkpoint that "
                  "reads 0.3x here read 46.3x at PROBE = \"answer\". Set it to "
                  "\"answer\" before concluding anything; the column worth "
                  "reading at this position is the prompt log-probability "
                  "below, which covers the reasoning trace itself.")
        else:
            print("  This position is before the model has decided anything, "
                  "where its next-token distribution is the same for both "
                  "classes, so the harmful column cannot move. Set PROBE to "
                  "\"answer\".")
    print("  An edit that went where it was aimed moves the next-token "
          "distribution on prompts the model used to refuse and leaves the rest "
          "alone, so this ratio is the one number here that says whether the "
          "ablation was surgical. 1x would mean the model changed everywhere "
          "equally, i.e. it was damaged rather than edited.")
    print("  It only reads that way when the second model is an ablation of the "
          "first. Comparing two *encodings* of the same model -- a quantized "
          "build against the checkpoint it came from -- measures round-trip "
          "damage instead, and there 1x is the expected answer: quantization "
          "error has no reason to prefer one class of prompt.")
    print(f"\nPrompt log-probability: {summary['harmless'][3]:+.4f} nats per "
          f"token on harmless prompts ({summary['harmless'][4]:.4f} in absolute "
          f"value), {summary['harmful'][3]:+.4f} on harmful ones")
    print("  This is the model's ordinary language modelling, teacher-forced on "
          "text it was not asked to refuse. A drop of more than a few hundredths "
          "of a nat per token is collateral damage, whatever the refusal count "
          "says; near zero means the weights outside the direction survived.")
    if PROBE == "close":
        # The teacher-forced text is prompt + reasoning trace here, so this
        # column stops being only a perplexity check and becomes the one place
        # the edit is visible at this position.
        print("  At PROBE = \"close\" the teacher-forced text includes the base "
              "model's own reasoning, so the harmful figure is how much less "
              "likely the ablated model finds a trace that reasons its way to a "
              "refusal. That is the edit working, not damage: the 1024-prompt "
              "\"close\" checkpoint reads -0.296 on harmful against -0.032 on "
              "harmless, a 9x asymmetry, while its next-token column at this "
              "position shows nothing at all.")
    print(f"\nTop-1 next token unchanged on {summary['harmless'][2]:.1%} of "
          f"harmless prompts and {summary['harmful'][2]:.1%} of harmful ones")


if len(sys.argv) == 3 and sys.argv[1] == "--capture":
    capture(sys.argv[2])
    raise SystemExit(0)

harmful, harmless = prompts()
wanted = fingerprint(harmful, harmless)
caches = {}
for model_id in (BASE_MODEL, ABLATED_MODEL):
    path = cache_path(model_id)
    if path.is_file():
        cached = torch.load(path, weights_only=False)
        stamp = weights_stamp(model_id)
        if cached.get("fingerprint") == wanted and cached.get("prompts") == PROMPTS:
            if cached.get("weights") == stamp:
                print(f"Reusing {path}")
                caches[model_id] = cached
                continue
            print(f"{path} was built from different weights at the same path; "
                  "rebuilding it")
        else:
            print(f"{path} was built from a different prompt set; rebuilding it")

    # A child process, so that this one never holds two models' weights, and so
    # that the first model's memory is returned by the operating system rather
    # than by the allocator.
    print(f"\n=== capturing {model_id} ===", flush=True)
    finished = subprocess.run([sys.executable, __file__, "--capture", model_id])
    if finished.returncode != 0:
        raise SystemExit(f"capturing {model_id} failed "
                         f"(exit {finished.returncode})")
    caches[model_id] = torch.load(path, weights_only=False)

compare(caches[BASE_MODEL], caches[ABLATED_MODEL])
