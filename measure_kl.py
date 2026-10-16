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
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_loading import format_duration, format_memory, get_input_device, load_model

BASE_MODEL = "../Hy-Memory-30B-A3B"
ABLATED_MODEL = "../Hy-Memory-30B-A3B-Abliterated"

# Prompts per class. The KL per prompt is a per-prompt number rather than an
# average over a fitted direction, so this does not need the 1024 that
# compute_refusal_dir.py does; the spread across prompts is reported and says
# whether more would help. Each one costs a forward pass and 472KiB of cache
# (one float32 per vocabulary entry).
PROMPTS = 128
BATCH_SIZE = 8

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
    """Identifies the prompt set, so a stale cache cannot be compared by mistake."""
    digest = hashlib.sha256()
    for prompt in harmful + harmless:
        digest.update(prompt.encode())
    return digest.hexdigest()[:16]


def cache_path(model_id):
    return CACHE_DIR / (model_id.strip("/").replace("/", "_") + "_kl.pt")


def tokenize(tokenizer, batch):
    return tokenizer.apply_chat_template(
        conversation=[[{"role": "user", "content": insn}] for insn in batch],
        add_generation_prompt=True, tokenize=True, padding=True,
        return_dict=True, return_tensors="pt")


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
    batches = {label: [tokenize(tokenizer, group[start:start + BATCH_SIZE])
                       for start in range(0, len(group), BATCH_SIZE)]
               for label, group in groups.items()}
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

    harmful_kl, harmless_kl = summary["harmful"][0], summary["harmless"][0]
    ratio = harmful_kl / harmless_kl if harmless_kl > 1e-12 else float("inf")
    print(f"\nHarmful/harmless KL ratio: {ratio:.1f}x")
    print("  An edit that went where it was aimed moves the next-token "
          "distribution on prompts the model used to refuse and leaves the rest "
          "alone, so this ratio is the one number here that says whether the "
          "ablation was surgical. 1x would mean the model changed everywhere "
          "equally, i.e. it was damaged rather than edited.")
    print(f"\nPrompt log-probability: {summary['harmless'][3]:+.4f} nats per "
          f"token on harmless prompts ({summary['harmless'][4]:.4f} in absolute "
          "value)")
    print("  This is the model's ordinary language modelling, teacher-forced on "
          "text it was not asked to refuse. A drop of more than a few hundredths "
          "of a nat per token is collateral damage, whatever the refusal count "
          "says; near zero means the weights outside the direction survived.")
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
        if cached.get("fingerprint") == wanted and cached.get("prompts") == PROMPTS:
            print(f"Reusing {path}")
            caches[model_id] = cached
            continue
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
