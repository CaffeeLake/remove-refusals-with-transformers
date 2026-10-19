"""Which token of the sequence the refusal direction is read from.

LAYER_FRACTION picks the depth to read the residual stream at. This picks the
position along the sequence, which on an ordinary instruct model is not a choice
worth making -- `add_generation_prompt=True` puts the last prompt token
immediately before the answer, which is where a refusal begins -- and on a
reasoning model is three different choices with very different results.

  "thinking"  The last prompt token with the thinking block opened, i.e.
              `<|im_start|>assistant\\n<think>\\n`. This is what a script written
              for a non-reasoning model measures by default, and it is the
              traditional method: on a model with no thinking mode it is the
              only position there is, and `"answer"` collapses onto it.

              On Dory1.0-4B it is the position *before* any reasoning, and the
              model has decided nothing there: its next-token distribution is
              `'The'` at p = 0.9998 for 64 of 64 harmful and 62 of 64 harmless
              prompts, an entropy of 0.0015 nats. A difference-of-means
              direction read there does separate the classes, because the topic
              is in the residual stream, but it is a topic direction. Ablating
              with it removes 99.3% of itself from all 73 residual writers, cuts
              the harmful/harmless gap along it from +0.4992 to +0.0111, and
              changes no behaviour at all: 6 of 6 spot-checked harmful prompts
              still refused, in the base model's own words.

  "answer"    `enable_thinking=False`, so the prompt ends `<think></think>` and
              the next token is the first token of the answer. The refusal is in
              the residual stream there because it is about to be emitted:
              the base model opens with `'I'` or `'私は'` on harmful prompts and
              a content word on harmless ones.

  "close"     The `</think>` token at the end of the model's *own* reasoning
              trace. This is the position the model refuses from when it is
              served normally, and it is the only probe here that needs
              generation: each prompt's trace has to be produced before the
              activation can be read. `reasoning_traces` does that, with a
              resumable cache, because it is by far the most expensive step in
              this repo.

Measured on Dory1.0-4B at LAYER_FRACTION = 0.76, ablating all 73 residual
writers with each probe's direction and reading the first answer token on 48
held-out prompts per class:

    probe      KL harmful  KL harmless   ratio  harmful top-1 kept  harmless
    thinking        0.106        0.058    1.8x               95.3%     89.1%
    answer          2.663        0.163   16.4x                8.3%     89.6%
    close           3.032        0.124   24.4x                2.1%     95.8%

`"close"` is the best of the three on every column -- it moves the harmful
distribution more *and* leaves the harmless one alone more -- and behaviourally
it is the only one that stops the model refusing the hardest of the six
spot-checked prompts. The directions are related but distinct:
cosine(thinking, answer) = 0.41 and cosine(answer, close) = 0.68.

Two negative results, recorded so they are not rediscovered:

* **Removing more than one of them is worse.** Ablating the span of two
  directions was tried twice -- {thinking, answer} and {answer, close} -- and
  both times the harmful KL was unchanged while the harmless KL rose (0.170 ->
  0.196 and 0.124 -> 0.430) and harmless top-1 agreement fell (to 87.5% and to
  75.0%). Each extra direction costs one of `hidden_size` dimensions in all 73
  matrices whether or not it buys anything.

* **No weight ablation removes the refusal once the model has reasoned.**
  Teacher-forced on the base model's own traces and measured at `</think>`, all
  three directions and both spans leave the harmful first-answer-token top-1
  **100%** unchanged (KL 0.002 to 0.022). By then the refusal is in the context
  as text -- the trace says "I should refuse" in so many words -- and no edit to
  the residual stream can remove text the model has already written. The
  thinking-mode refusal that survives a successful ablation is this, and it is a
  different mechanism from the one the technique addresses.
"""

import hashlib
import time
from pathlib import Path

import torch

__all__ = [
    "PROBES",
    "template_flags",
    "encode",
    "prompt_tokens",
    "pad_left",
    "close_token",
    "reasoning_traces",
    "probe_sequences",
]

PROBES = ("thinking", "answer", "close")

# Whether the chat template opens a thinking block for each probe. "close"
# needs it open, since the trace is what it reads the end of. A template that
# does not know `enable_thinking` ignores it, which is what makes "thinking"
# and "answer" the same position on a model with no thinking mode.
_THINKING = {"thinking": True, "answer": False, "close": True}

CACHE_DIR = Path("trace_cache")


def template_flags(probe, medium_effort=False):
    """The chat-template arguments a probe needs."""
    if probe not in PROBES:
        raise ValueError(f"unknown probe {probe!r}; use one of "
                         + ", ".join(PROBES))
    # `medium_effort` is Dory's "{reasoning effort: efficient}" tag. It only
    # matters for "close", where it decides how long the traces are, and it is
    # ignored by templates that do not define it.
    return {"enable_thinking": _THINKING[probe], "medium_effort": medium_effort}


def encode(tokenizer, prompts, probe, medium_effort=False):
    """A padded batch for a probe that needs no generation."""
    return tokenizer.apply_chat_template(
        conversation=[[{"role": "user", "content": prompt}] for prompt in prompts],
        add_generation_prompt=True, tokenize=True, padding=True,
        return_dict=True, return_tensors="pt",
        **template_flags(probe, medium_effort))


def prompt_tokens(tokenizer, prompt, probe, medium_effort=False):
    """One conversation as a plain list of token ids.

    `return_dict=True` on purpose: without it this version of
    `apply_chat_template` hands back a `tokenizers.Encoding` for a single
    conversation, which has a length and so passes silently into `torch.tensor`
    before failing there.
    """
    ids = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        **template_flags(probe, medium_effort))["input_ids"]
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def pad_left(sequences, pad_token_id):
    """Left-pad id sequences into a batch, as `{input_ids, attention_mask}`.

    Left, because everything in this repo reads the activation at the last
    position and a right-padded batch's last position is padding.
    """
    width = max(len(ids) for ids in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, ids in enumerate(sequences):
        input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        mask[row, width - len(ids):] = 1
    return {"input_ids": input_ids, "attention_mask": mask}


def close_token(tokenizer):
    """The id of `</think>`, which has to be a single token to be found."""
    ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"`</think>` tokenizes to {len(ids)} tokens ({ids}) on this "
            "tokenizer, so the end of a reasoning trace cannot be located by "
            "token id. The \"close\" probe needs a model whose thinking block "
            "is delimited by a single special token.")
    return ids[0]


def _fingerprint(prompts, budget, medium_effort):
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode())
    digest.update(f"{budget}|{medium_effort}".encode())
    return digest.hexdigest()[:16]


def reasoning_traces(model_id, tokenizer, groups, budget=1024,
                     medium_effort=True, batch_size=64, cache_dir=None,
                     load=None, **load_kwargs):
    """The model's own reasoning trace per prompt, to and including `</think>`.

    `groups` is `{label: [prompt, ...]}` and the result has the same shape, with
    `None` for a prompt whose trace did not reach `</think>` inside `budget`
    tokens. Those prompts are dropped by `probe_sequences`; the caller is told
    how many.

    Cached on disk and **resumable at batch granularity**, which is not
    defensive programming but a requirement: this is the one step in the repo
    measured in tens of minutes rather than minutes, and on the machine it was
    written on the process was repeatedly killed part-way through with no
    traceback and no cgroup OOM event. Re-running picks up where it stopped.

    `medium_effort` defaults to True because it roughly halves the wall clock
    (2m14s against 5m23s per batch of 48 on Dory1.0-4B) by asking for shorter
    reasoning. It does not change the direction much: the harmless prompts that
    reach `</think>` inside a budget are the ones that reason briefly either
    way -- 61 of 240 against 184 of 240 harmful -- but length-matching the two
    classes before taking the difference of means moves the direction by
    cosine 0.9991 and its effect size by 0.01, so the selection is not what the
    direction encodes.
    """
    directory = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    directory.mkdir(exist_ok=True)
    name = Path(str(model_id).rstrip("/")).name or "model"
    every = [prompt for prompts in groups.values() for prompt in prompts]
    path = directory / (f"{name}_{_fingerprint(every, budget, medium_effort)}"
                        "_traces.pt")

    state = (torch.load(path, weights_only=False) if path.is_file()
             else {"traces": {label: [None] * len(prompts)
                              for label, prompts in groups.items()},
                   "done": []})
    traces, done = state["traces"], {tuple(entry) for entry in state["done"]}
    outstanding = [(label, start) for label, prompts in groups.items()
                   for start in range(0, len(prompts), batch_size)
                   if (label, start) not in done]

    if not outstanding:
        print(f"Reusing {path}")
    else:
        from transformers import StoppingCriteria, StoppingCriteriaList

        closing = close_token(tokenizer)

        class Closed(StoppingCriteria):
            """Mark each row done once it has emitted `</think>`.

            Per row, because the API wants a BoolTensor of shape (batch,) and a
            scalar silently means "never finish early" -- which is most of the
            cost of this function.
            """

            def __init__(self, start):
                self.start = start

            def __call__(self, input_ids, scores, **kwargs):
                return (input_ids[:, self.start:] == closing).any(dim=1)

        if load is None:
            from model_loading import load_model as load
        print(f"Generating reasoning traces for the \"close\" probe: "
              f"{len(outstanding)} batch(es) of up to {batch_size}, "
              f"{len(done)} already cached. This is the slow step; it is "
              f"written to {path} after every batch and re-running resumes.")
        longest = max(len(prompt_tokens(tokenizer, prompt, "close", medium_effort))
                      for prompt in every)
        model = load(model_id,
                     workload_tokens=batch_size * (longest + budget),
                     workload_sequence=longest + budget, **load_kwargs)
        model.eval()
        started = time.perf_counter()
        for label, start in outstanding:
            prompts = groups[label][start:start + batch_size]
            inputs = encode(tokenizer, prompts, "close", medium_effort)
            width = inputs["input_ids"].shape[1]
            device = next(model.parameters()).device
            try:
                from model_loading import get_input_device

                device = get_input_device(model)
            except Exception:
                pass
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=budget, do_sample=False,
                    stopping_criteria=StoppingCriteriaList([Closed(width)]))
            for row in range(len(prompts)):
                tail = generated[row][width:].tolist()
                traces[label][start + row] = (tail[:tail.index(closing) + 1]
                                              if closing in tail else None)
            done.add((label, start))
            torch.save({"traces": traces, "done": sorted(done)}, path)
            reached = sum(trace is not None for trace in traces[label])
            print(f"  {label} {start + len(prompts)}/{len(groups[label])}, "
                  f"{reached} reached </think>, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for label, group in traces.items():
        lengths = sorted(len(trace) for trace in group if trace is not None)
        if not lengths:
            raise RuntimeError(
                f"not one {label} prompt's reasoning reached `</think>` within "
                f"{budget} tokens. Raise the budget, or set medium_effort=True "
                "to ask for shorter reasoning.")
        print(f"  {label}: {len(lengths)}/{len(group)} traces reached "
              f"</think>, median {lengths[len(lengths) // 2]} tokens, "
              f"max {lengths[-1]}")
    return traces


def probe_sequences(tokenizer, prompts, probe, traces=None,
                    medium_effort=False):
    """(id sequences to run, which prompts they came from).

    The second element only matters for `"close"`, where a prompt whose
    reasoning never closed has no position to read and is dropped.
    """
    if probe != "close":
        return ([prompt_tokens(tokenizer, prompt, probe, medium_effort)
                 for prompt in prompts], list(range(len(prompts))))
    if traces is None:
        raise ValueError('the "close" probe needs `traces`; call '
                         "`reasoning_traces` first")
    kept = [index for index, trace in enumerate(traces) if trace]
    return ([prompt_tokens(tokenizer, prompts[index], probe, medium_effort)
             + list(traces[index]) for index in kept], kept)
