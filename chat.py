"""Chat with a model directory that already has the ablation baked into it.

inference.py removes the refusal direction at runtime with forward hooks, which
means it needs the original model *and* the `refusal_dir.pt` that goes with it,
and it pays for the hooks on every layer of every token. save_ablated_model.py
instead rewrites the weights, and what it writes is an ordinary model: no hooks,
no direction file, nothing this repo has to be present for.

This script is that ordinary load, kept here only because a checkpoint too big
for VRAM still needs the memory budget from model_loading.py to be loadable at
all. For a 4-bit Kimi K3 that is the difference between running and an OOM after
the weights are already placed. Anything that fits in VRAM can be run with a
plain `AutoModelForCausalLM.from_pretrained` instead, and any serving runtime
(vLLM, SGLang, llama.cpp after conversion) will be far faster than this on a
large model, because compressed-tensors has to unpack every 4-bit weight it
touches on every forward pass.
"""

import sys

import torch
from transformers import AutoTokenizer, TextStreamer

from model_loading import get_input_device, load_model

MODEL_ID = "../Kimi-K3-Abliterated"

# "auto" follows the checkpoint. Forcing float16 on a bfloat16 model throws away
# most of its exponent range.
DTYPE = "auto"

# "auto" spreads the layers over every visible GPU, then over CPU RAM, then over
# OFFLOAD_DIR. Use "cuda:0" to pin everything to a single GPU instead.
DEVICE_MAP = "auto"
# Per-device caps for the *weights*. `None` derives them from the free VRAM minus
# a reserve for generation; see GPU_HEADROOM below.
MAX_MEMORY = None
# Needed once the weights no longer fit in RAM + VRAM.
OFFLOAD_DIR = None

# Longest conversation that will be generated, prompt plus completion. The KV
# cache and, on a model without flash-attention, the quadratic attention matrix
# are both sized from this, so a value that is too small brings the OOM back
# late in a long answer and one that is too large offloads weights for nothing.
MAX_NEW_TOKENS = 1337
WORKLOAD_TOKENS = 4096 + MAX_NEW_TOKENS

# VRAM to keep free per GPU for everything that is not weights. accelerate
# budgets for weights only, so without a reserve the model loads and then
# generation OOMs with a few MiB free. `None` computes it from the model's own
# shapes and WORKLOAD_TOKENS, with GPU_HEADROOM_FRACTION of the card as a floor.
GPU_HEADROOM = None
GPU_HEADROOM_FRACTION = 0.05
# Cap on offloaded weights in CPU RAM. `None` uses 90% of what is free.
CPU_LIMIT = None

# Keep compressed-tensors weights packed and unpack each one per forward pass,
# instead of letting compressed-tensors decompress the whole model in place and
# quadruple the resident weights. Mandatory on anything that does not have room
# for the dense version: Kimi K3 is 1.4TiB packed and 5.1TiB decompressed.
KEEP_PACKED = True

# Sampling. `None` for do_sample leaves the checkpoint's generation_config alone.
DO_SAMPLE = None
TEMPERATURE = None
TOP_P = None

model = load_model(MODEL_ID,
                   dtype=DTYPE,
                   device_map=DEVICE_MAP,
                   max_memory=MAX_MEMORY,
                   offload_dir=OFFLOAD_DIR,
                   workload_tokens=WORKLOAD_TOKENS,
                   workload_sequence=WORKLOAD_TOKENS,
                   gpu_headroom=GPU_HEADROOM,
                   gpu_headroom_fraction=GPU_HEADROOM_FRACTION,
                   cpu_limit=CPU_LIMIT,
                   keep_packed=KEEP_PACKED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Not `model.device`, which is misleading on a sharded model, and not the
# embedding weight's device either, which is the meta device once the embeddings
# are offloaded. accelerate moves the inputs on from there by itself.
input_device = get_input_device(model)

generate_kwargs = {}
if DO_SAMPLE is not None:
    generate_kwargs["do_sample"] = DO_SAMPLE
if TEMPERATURE is not None:
    generate_kwargs["temperature"] = TEMPERATURE
if TOP_P is not None:
    generate_kwargs["top_p"] = TOP_P

# `skip_special_tokens` so that what is streamed is what ends up in the
# conversation history below; without it the two diverge on any model whose chat
# template emits channel markers.
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
conversation = []

print(f"Chat with {MODEL_ID}. Ctrl-D to quit, /reset to forget the conversation.")
while True:
    try:
        prompt = input("\n> ")
    except EOFError:
        print()
        break
    if not prompt.strip():
        continue
    if prompt.strip() == "/reset":
        conversation.clear()
        print("(conversation cleared)")
        continue

    conversation.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(conversation=conversation,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_dict=True,
                                           return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length + MAX_NEW_TOKENS > WORKLOAD_TOKENS:
        print(f"(warning: {prompt_length} prompt + {MAX_NEW_TOKENS} new tokens is "
              f"more than the {WORKLOAD_TOKENS} the VRAM reserve was sized for)",
              file=sys.stderr)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs["input_ids"].to(input_device),
            attention_mask=inputs["attention_mask"].to(input_device),
            streamer=streamer,
            # some configs ship with `use_cache: false`, which makes generation
            # quadratic and blows up VRAM on long completions
            use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
            **generate_kwargs)

    reply = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": reply})
