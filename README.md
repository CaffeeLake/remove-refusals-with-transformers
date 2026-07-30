# Removing refusals with HF Transformers

This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens. This means, that this supports every model that HF Transformers supports*.

The code was tested on a RTX 2060 6GB, thus mostly <3B models have been tested, but the code has been tested to work with bigger models as well.

*While most models are compatible, some models are not. Mainly because of custom model implementations. Some Qwen implementations for example don't work. Because `model.model.layers` can't be used for getting layers. They call the variables so that, `model.transformer.h` must be used, if I'm not mistaken.

## Usage
1. Set model and quantization in compute_refusal_dir.py and inference.py (Quantization can apparently be mixed)
2. Run compute_refusal_dir.py (Some settings in that file may be changed depending on your use-case)
3. Run inference.py and ask the model how to build an army of rabbits, that will overthrow your local government one day, by stealing all the carrots.
4. Optionally run choose_layer.py first to pick `LAYER_FRACTION` for your model, rather than inheriting 0.6.
5. Optionally run save_ablated_model.py to bake the ablation into the weights and write a standalone model directory, which no longer needs the runtime hooks from inference.py.
6. Run chat.py against that directory to talk to the baked model.

inference.py and chat.py do the same thing from opposite ends. inference.py needs the original model *and* its `refusal_dir.pt`, and removes the direction with a forward hook on every layer of every token. chat.py loads a directory that save_ablated_model.py has already rewritten, so there is no hook, no direction file and nothing in this repo that has to be importable — it exists only because a checkpoint too large for VRAM still needs the memory budget below to load at all. Anything that fits in VRAM can be run with a plain `AutoModelForCausalLM.from_pretrained`, and any serving runtime will be much faster than either script on a large quantized model, since compressed-tensors unpacks every 4-bit weight it touches on every forward pass.

### Large models
compute_refusal_dir.py, inference.py and chat.py load through model_loading.py and default to `DEVICE_MAP = "auto"`, which spreads the layers over every visible GPU, then over CPU RAM, then over `OFFLOAD_DIR`.

accelerate's own budget is every free byte of every GPU, and it counts weights only. A checkpoint that does not fit in VRAM therefore loads successfully and then dies on the first forward pass, with a handful of MiB free on a 141GB card, because nothing was left for the activations, the KV cache or the buffers a quantized weight is unpacked through. `build_max_memory` reserves that space up front:

- per GPU: one layer's activations, one weight's unpacking workspace and one eager attention matrix (a GPU only runs one layer at a time, so none of them shrink as GPUs are added), plus its share of the KV cache,
- on GPU 0: the whole KV cache, since it is the execution device for every CPU-offloaded layer,
- at least `GPU_HEADROOM_FRACTION` of each card (5% by default) regardless, as a backstop for cuBLAS workspaces, NCCL buffers and allocator fragmentation. The floor can be this low only because the three terms above are modelled properly, the attention one in particular; it is a backstop, not the estimate.

The attention term is the one that bites on long prompts. sdpa and flash-attention never build the `[batch, heads, q, k]` score matrix, but a `trust_remote_code` model that ships only an eager path does — Kimi K3 falls back to eager whenever flash-attn 2 is not importable, and rejects `attn_implementation="sdpa"` outright. Being quadratic in the sequence length it dominates everything else: for Kimi K3 one conversation of 5433 tokens needs 26GiB for that matrix alone against 3GiB for the rest of the activations, and 16k tokens would need 240GiB, more than one H200 holds. `WORKLOAD_TOKENS` in inference.py is what sizes it, so it has to be set to the longest conversation that will actually be generated.

### Which layer to read from
`LAYER_FRACTION` is 0.6 because that is what the upstream repo hardcodes. The technique it implements picks the layer by searching, and choose_layer.py is that search: it captures every layer in one pass over the prompts, builds a direction from each, and scores them on prompts none of them saw. Measured optima:

| model | layers | best layer | `LAYER_FRACTION` |
|---|---|---|---|
| Qwen3-1.7B | 28 | 19 | 0.68 |
| Qwen3-4B | 36 | 26 | 0.72 |
| Qwen3-8B | 36 | 25 | 0.69 |
| Qwen3.5-9B | 32 | 21 | 0.66 |
| Qwen3-14B | 40 | 24 | 0.60 |

0.6 is a little shallow on most of them and costs 3-13%, which is not worth much on its own. It is worth running once per model anyway, because the failure it catches is not a few percent — see below — and it is not free on a checkpoint that does not fit in VRAM, since it has to build layers up to `SWEEP_UNTIL` rather than up to `LAYER_FRACTION`.

### Massive activations
Some models grow a handful of hidden dimensions three orders of magnitude larger than the rest, from one layer onwards. Qwen3-14B goes from 85 to 11,776 at layer 20. A large dimension has a large variance, so the difference between the two class means is large there too — 94% of the whole difference vector — while its t is 2.2, which is to say it is sampling noise. The direction then points at that dimension instead of at refusal, and agreement between two halves of the prompts collapses from 0.94 to 0.19. Nothing fails: the separation over the prompts the direction was built from reads 150, which looks excellent.

`DROP_LARGEST_DIMENSIONS` (0 by default) zeroes that many dimensions, chosen by activation magnitude, before the difference is normalised into the direction. Measured effect on the train/test agreement of the direction:

| | Qwen3-14B L24 | Qwen3-14B L19 | Qwen3-8B L25 | Qwen3.5-9B L21 |
|---|---|---|---|---|
| largest activation | 10432 | 79 | 46 | 17 |
| difference of means | 0.187 | 0.938 | 0.886 | 0.850 |
| two largest zeroed | **0.894** | 0.937 | 0.889 | 0.850 |

So it rescues the broken case and is free on the others; raising it past 2 changes nothing (flat from 2 to 128), and 0 turns it off. With it on, Qwen3-14B's usable range extends to the whole model and its best layer moves from 19 to 24 — the same 0.60 as everything else. compute_refusal_dir.py reports which dimensions it dropped, what share of the difference they held and how significant they were, so it is visible whether the option did anything:

```
Dropped 2 dimension(s) with the largest activations from the difference:
  dim 731: activation 10460.6, 94.2% of the difference, t = -2.4
  dim 2994: activation 1606.0, 2.2% of the difference, t = -2.2
  One of them held most of the difference while being barely distinguishable from
  noise: without this option the direction would have been that dimension rather
  than the refusal.
```

Zeroing coordinates by significance instead (`|t| < 2`) does not work: the offending dimension passes at 2.2. Dividing each coordinate by its standard error does work, but the result is no longer the displacement between the two clusters, and the displacement is what the ablation has to remove.

### Did the direction come out?
A direction built from a set of prompts is guaranteed to separate *those* prompts, so the separation over them says nothing. `HELDOUT_FRACTION` keeps a slice out of the fit and reports the separation over it as well, which costs nothing because the activations are collected either way, and the saved direction is still built from all of them.

Both are reported twice: the raw gap, which is in the units of the residual stream and therefore only comparable with itself, and the effect size, which divides by the spread of the two clusters and is comparable across models. At 60% depth the effect size is 6.2 on Qwen3-8B, 4.7 on Qwen3-4B and 3.6 on Qwen3-1.7B. Below 0.5 the direction is noise and ablating it does nothing — which is also how a model whose weights did not load correctly shows up, so it is worth reading before spending an hour on the conversion.

The inherited `Kimi-K3-0.40B-MXFP4` is such a case: raw separation 3.0 in-sample and 5.5 held out, both of which look plausible, against an effect size of 0.18 and 0.33. It has no refusal behaviour to find and is only useful for exercising the code paths.

### Only load the layers that run
compute_refusal_dir.py reads the residual stream at 60% depth and aborts the forward pass there, so the last 40% of the layers are never executed. `TRUNCATE_LAYERS` (on by default) stops them from being built at all. On Kimi K3 the decoder layers are 99.9% of the parameters, so this takes roughly 40% off the checkpoint, the VRAM and the load time together, and the captured activation is bit-for-bit identical — a layer cannot influence its own input. inference.py cannot use it, since generation needs the whole model.

The reserve is sized from the actual workload: compute_refusal_dir.py tokenizes its batches *before* loading the model and passes the real token count and sequence length, inference.py and chat.py use `WORKLOAD_TOKENS`. Raise `GPU_HEADROOM_FRACTION`, or pin an absolute number with `GPU_HEADROOM = "30GiB"`, if a forward pass still runs out. The cost is only that more weights end up in CPU RAM, so the run gets slower rather than failing. `MAX_MEMORY` still overrides the whole calculation, and `CPU_LIMIT` caps the RAM used for offloaded weights.

For reference, the published `moonshotai/Kimi-K3` is 1.420TiB across 96 shards, of which the decoder layers are 99.6%, and only the routed experts are packed (its `ignore` list covers `self_attn`, `shared_experts`, `mlp.*_proj`, `lm_head` and the vision tower, so every residual writer is already dense bf16 and nothing has to leave the quantized format). With `TRUNCATE_LAYERS` that is 878GiB for compute_refusal_dir.py, which fits on 8xB200 or 8xB300 outright and on 8xH200 at the default 5% headroom. Generation needs all 93 layers, i.e. 1454GiB, so chat.py fits on 8xB300 only; `KEEP_PACKED` is not optional either way, since the decompressed model is 5178GiB.

save_ablated_model.py never instantiates the model: it streams the safetensors shards one at a time, so its peak memory is the size of a single shard rather than the size of the checkpoint. Set `DRY_RUN = True` to print the list of tensors it would rewrite; that only reads the shard headers and is instant even on a multi-terabyte checkpoint.

### Quantized checkpoints
compressed-tensors checkpoints (`nvfp4-pack-quantized`, `mxfp4-pack-quantized`, `mxfp8-quantized`, `float-quantized`) are supported, including a `quantization_config` nested inside `text_config`.

A quantized module never holds a usable `<module>.weight`. The packed formats put the values in `weight_packed`; the fp8 ones do keep a `weight`, but it holds the values *divided by* `weight_scale`, in a dtype no matmul accepts. save_ablated_model.py therefore decodes every quantized residual writer before projecting, and writes a fresh `weight_scale` only if the result stays a scaled tensor. Projecting the stored tensor directly is not the same operation — the per-group scale varies along the output axis as well — and on a Kimi K3 MXFP8 build it removes 91% of the refusal component instead of 99.9%, while reporting a plausible-looking leakage because it measures in the scaled domain too.

Such a checkpoint is not *run* compressed: compressed-tensors installs a forward pre-hook that decompresses the entire model in place the first time it is used, turning 4-bit weights into bfloat16 and quadrupling the resident weights — after accelerate has already filled the GPUs with the packed ones. `KEEP_PACKED = True` (the default) drops that hook and unpacks each weight on demand instead, which keeps the resident size equal to the checkpoint size and also keeps CPU offloading working at all, since accelerate looks offloaded parameters up by name and a decompressed model no longer has a `weight_packed`. With `KEEP_PACKED = False` the memory budget is divided by the growth factor instead, which offloads roughly four times as much to CPU RAM. Formats that keep a real `weight` of an unusable dtype (`float-quantized` fp8) cannot be kept packed; that is detected from the config before loading, so the budget allows for their growth either way.

### Output precision of the ablated writers
`WRITER_PRECISION` in save_ablated_model.py decides how the ablated residual writers are encoded again. The ablation itself is always computed on the decoded weight in float32; this only affects storage.

| value | what it writes |
|---|---|
| `None` (default) | dense, in the checkpoint's own dtype |
| `"bf16"` / `"fp16"` / `"fp32"` | dense, in that dtype |
| `"keep"` | back into the checkpoint's own quantized format |
| `"mxfp4"` / `"nvfp4"` / `"mxfp8"` | back into that format; if it differs from the checkpoint's, the writers get a second config group and the top-level format becomes `mixed-precision`, which transformers cannot load with CPU offloading |

Keeping a writer quantized costs most of the ablation, and the reason is a size comparison rather than anything subtle. The edit `W <- (I - r rᵀ)W` has relative size `alignment / sqrt(hidden_size)`, where the alignment is how much more of the refusal direction the matrix carries than an arbitrary one. **That alignment measures ~1.0 on every model tried** — Qwen3 0.6B/1.7B/4B/8B all give a median of 1.01-1.07 despite harmful/harmless separations of 12-85 — so the edit is essentially `1/sqrt(hidden_size)` and *shrinks* as models get wider:

| hidden_size | 1024 | 2048 | 2560 | 4096 | 7168 (Kimi K3) |
|---|---|---|---|---|---|
| edit size | 3.26% | 2.28% | 2.03% | 1.58% | ~1.2% |

A format whose round-trip error exceeds the edit rounds it straight back. Measured on real weights, the errors do not depend on the model: MXFP4 11.3%, NVFP4 9.5%, MXFP8 2.7%. The refusal component the format puts back, normalised so 1.0 is a matrix that was never ablated:

| writers written as | round-trip error | component put back (h=1024 → 4096) | removal at h=4096, end to end |
|---|---|---|---|
| dense bf16 / fp32 | — | ~0.3% | 86.7% |
| MXFP8 | 2.7% | 16% → 34% | 76.5% |
| NVFP4 | 9.5% | 57% → 85% | 33.8% |
| MXFP4 | 11.3% | 73% → 92% | 25.5% |

NVFP4's finer scale (fp8 per 16 values plus a per-tensor global scale, against MXFP4's power-of-two per 32) buys only 16% less round-trip error, because the error is dominated by the 4-bit E2M1 element grid rather than by the scale. Both 4-bit formats leave the residual stream *more* aligned with the refusal direction than a random direction would be (160-460% of isotropic), i.e. the ablation is effectively undone. MXFP8 tracked dense exactly up to hidden 2560 and then broke down at 4096, precisely where the edit fell below its 2.7% error — so at Kimi K3's 7168 there is no quantized format left that works.

Bigger models make this worse, not better. The runtime dtype puts a floor under everything anyway — a bf16 model rounds even fp32 writers back to the same leak — so `None` is the right default and `"fp32"` only helps a runtime that computes in fp32.

Every run prints the residual table for the writers it actually wrote, decoded back out of whatever it encoded them into, so the cost of the setting is never a guess. A residual writer stored in 4 bits cannot stay in 4 bits, though: MXFP4 has a round-trip error near 11%, which dwarfs the refusal component being removed, so re-packing the ablated matrix restores about 70% of it. Those matrices are decompressed to dense instead and added to the quantization config's `ignore` list. In the usual layout only the routed expert output projection is affected, since attention output projections, shared experts, dense MLPs and the embeddings are already on the `ignore` list.

Some exports lose the `Linear` restriction on a quantization group and leave a bare regex such as `re:.*block_sparse_moe.*` in `targets`. That also selects containers, activations and norms, and the checkpoint then fails to load with *"Quantization of module type ... is not supported"* — before any ablation, and for any runtime, not just this repo. `narrow_targets` in model_loading.py detects this from the tensor names alone and anchors the regex to the leaf names that are really packed, but only when the narrowed regex provably selects the same module set; otherwise it warns and leaves `targets` alone.

Both ends use it. compute_refusal_dir.py and inference.py repair the config in memory before `from_pretrained` (the repair has to happen first, since the failure is during module initialisation), and save_ablated_model.py writes the repaired `targets` into the output `config.json` under `FIX_QUANTIZATION_TARGETS`, so the ablated model loads anywhere without it. Only the safetensors headers are read either way, so the check costs nothing.

### trust_remote_code models and the transformers version
Models that ship their own modeling code are written against one transformers release and usually only assert a lower bound. Kimi K3, for example, imports `OutputRecorder` from `transformers.utils.generic`, which moved to `transformers.utils.output_capturing` in transformers 5.x, so it needs `transformers==4.56.*`. save_ablated_model.py is unaffected either way: it only reads the config and the safetensors shards, never the modeling code.

## Credits
- [Harmful instructions](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
- [Harmless instructions](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Technique](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
