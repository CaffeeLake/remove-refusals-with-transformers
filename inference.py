import einops
import jaxtyping
import torch
from transformers import (AutoConfig, AutoTokenizer, TextStreamer,
                          BitsAndBytesConfig)

from model_loading import get_decoder_layers, get_input_device, load_model

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
# Per-device caps for the *weights*, e.g. {0: "44GiB", "cpu": "100GiB"}. `None`
# derives them from the free VRAM minus a reserve for generation; see
# GPU_HEADROOM below. Setting this bypasses that reserve.
MAX_MEMORY = None
OFFLOAD_DIR = None

# Longest conversation that will be generated, prompt plus completion. This is
# what the KV cache is sized from, so a value that is too small brings the OOM
# back late in a long answer.
MAX_NEW_TOKENS = 1337
WORKLOAD_TOKENS = 4096 + MAX_NEW_TOKENS
# One conversation at a time, so the token count above is also the sequence
# length. That matters because an eager attention matrix is quadratic in it:
# Kimi K3 needs 36GiB for one at 8k tokens, against 5GiB for everything else.

# VRAM to keep free per GPU for everything that is not weights: the KV cache,
# activations, the buffers a quantized weight is unpacked through, allocator
# fragmentation. accelerate budgets for weights only, so without a reserve the
# model loads and then generation OOMs with a few MiB free. `None` computes it
# from the model's own shapes and WORKLOAD_TOKENS, with
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
# per forward pass, instead of letting compressed-tensors decompress the whole
# model in place on the first forward and quadruple the resident weights.
KEEP_PACKED = True

# Only pass the optional arguments that are actually set. In particular
# `quantization_config=None` is not the same as leaving it out: configs that
# expose a `quantization_config` attribute (Kimi K3 lifts the one from
# `text_config` up to the top level) have it overwritten with None, which
# silently strips the checkpoint's own quantization on transformers 5.x and
# raises `'NoneType' object has no attribute 'to_dict'` on 4.x.
load_kwargs = {}
if LOAD_IN_4BIT:
    if getattr(AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True),
               "quantization_config", None) is not None:
        raise ValueError(
            "the checkpoint is already quantized; LOAD_IN_4BIT would stack "
            "bitsandbytes on top of it. Set LOAD_IN_4BIT = False.")
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

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
                   keep_packed=KEEP_PACKED,
                   **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# With a sharded model `model.device` can be misleading, and the embedding
# weight's device is the meta device once the embeddings are offloaded. The
# inputs have to land on the device the embeddings actually execute on;
# accelerate moves them along from there.
input_device = get_input_device(model)

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt").view(-1)


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    direction = direction.to(device=activation.device, dtype=activation.dtype)
    proj = einops.einsum(activation, direction.view(-1, 1),
                         '... d_act, d_act single -> ... single') * direction
    return activation - proj


layers = get_decoder_layers(model)


# Ablate the refusal direction from the residual stream on the way into every
# decoder layer. A forward pre-hook is used rather than splicing dummy layers
# into the ModuleList: layer signatures and return protocols differ wildly
# between architectures (tensor vs. tuple vs. extra residual streams), and
# inserting modules also breaks per-layer cache indexing.
def ablation_pre_hook(module, args, kwargs):
    if kwargs.get("hidden_states") is not None:
        kwargs["hidden_states"] = direction_ablation_hook(
            kwargs["hidden_states"], refusal_dir)
    elif args:
        args = (direction_ablation_hook(args[0], refusal_dir),) + args[1:]
    return args, kwargs


for layer in layers:
    layer.register_forward_pre_hook(ablation_pre_hook, with_kwargs=True)

conversation = []

streamer = TextStreamer(tokenizer)

print(f"Chat with {MODEL_ID}: (Ctrl-D to quit)")
while True:
    try:
        prompt = input()
    except EOFError:
        print()
        break
    conversation.append({"role": "user", "content": prompt})
    toks = tokenizer.apply_chat_template(conversation=conversation,
                                         add_generation_prompt=True,
                                         tokenize=True,
                                         return_dict=False,
                                         return_tensors="pt")

    # some configs ship with `use_cache: false`, which makes generation
    # quadratic and blows up VRAM on long completions
    gen = model.generate(toks.to(input_device), streamer=streamer,
                         use_cache=True, max_new_tokens=MAX_NEW_TOKENS)

    decoded = tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": "".join(decoded)})
