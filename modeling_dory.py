"""Dory (NVIDIA-Nemotron-Labs-Dory1.0) for transformers.

Dory is not in transformers. It is in vLLM (PR vllm-project/vllm#60149, which is
what the model card tells you to install) and in llama.cpp, and the checkpoint
ships no `modeling_*.py`, so `trust_remote_code=True` has nothing to trust:
`AutoConfig.from_pretrained` falls back to the base `PreTrainedConfig`, whose
`validate_layer_type` rejects the checkpoint's `"mlp"` entries, and the load dies
before the tokenizer is even built. That is the whole of the error this file
exists to fix; the architecture itself is ordinary enough to write out.

What is unusual about it, and what the rest of this repo has to be told about:

* **One sublayer per layer.** A Dory "layer" is an attention block *or* an MLP
  block, not both, which is what `layer_types` is saying when it alternates
  `sliding_attention, mlp, sliding_attention, mlp, ...`. 72 layers are therefore
  36 attention blocks and 36 MLPs, i.e. the depth of an 18-layer llama.

* **The layers are looped.** The stack is split into an input block
  (`n_input_layers`), a shared recurrent block (`n_recurrent_layers`) and an
  output block (`n_output_layers`). The recurrent block runs
  `n_recurrent_loops` times, with the input block's output re-injected at the
  start of every loop. So the 72 physical layers execute 112 times at the
  default 2 loops, and a physical layer in the recurrent block is a *different
  point in the computation* on each of its passes. `execution_schedule` in
  model_loading.py is what turns one into the other, and it is why
  choose_layer.py sweeps 112 positions rather than 72.

* **nGPT.** Everything is normalized: no LayerNorms at all, the residual update
  is a learned spherical interpolation rather than an addition, and the learned
  scales are stored divided by `initializer_range` and multiplied back up here.
  None of that is optional, so the arithmetic below follows vLLM's
  `model_executor/models/dory.py` term for term.

What that was checked against, because an architecture written out by hand is
worth nothing until it is: the 4B checkpoint's prompt logprobs from a vLLM
server (`--hf-overrides '{"n_recurrent_loops": 2}'`) against this file, which
agree to 0.012 nats on average and 0.09 at worst -- the size of the difference
between this file's own float32 and bfloat16 answers. All 113 residual states of
one forward pass match a naive float32 transcription of vLLM's module to 3e-6,
and prefilling through the KV cache and then stepping one token matches a single
forward over the whole sequence to 1.4e-05 in float32.

Importing this module registers the architecture with the `Auto*` classes, so
`AutoConfig`/`AutoModelForCausalLM`/`AutoTokenizer` work on a Dory checkpoint
afterwards. model_loading.py imports it, which means every script in this repo
gets it.

Only `recurrent_kv_cache_mode="per_loop"` is implemented. The `"last_loop"`
mode has each loop *overwrite* the previous loop's KV, and transformers' `Cache`
only ever appends, so there is nothing here it could be built out of; it is
rejected at load time rather than silently approximated.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import (create_causal_mask,
                                        create_sliding_window_causal_mask)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

__all__ = [
    "DoryConfig",
    "DoryPreTrainedModel",
    "DoryModel",
    "DoryForCausalLM",
    "DoryCache",
    "register_dory",
    "attention_layer_types",
    "physical_layer_schedule",
]

# vLLM's `dory.py` uses 1e-6 inside every normalization. It is not a
# configurable epsilon in the checkpoint, so it is not one here either.
EPS = 1e-6

ATTENTION_TYPES = ("sliding_attention", "full_attention")


@strict
class DoryConfig(PreTrainedConfig):
    """Dory's config, as vLLM defines it.

    Kept deliberately close to `vllm/transformers_utils/configs/dory.py`: the
    checkpoint on the Hub was written against that class, and a field this one
    spells differently is a field that silently takes its default instead of the
    checkpoint's value.

    `validate_layer_type` has to be overridden because the base
    `PreTrainedConfig` only allows attention-shaped layer types, and half of
    Dory's layers are plain MLPs.
    """

    model_type = "dory"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 10_000.0

    vocab_size: int = 131072
    hidden_size: int = 2560
    num_hidden_layers: int = 72
    intermediate_size: int = 7168
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    # Divides every learned nGPT scale at inference time, not just at init; see
    # the factors in `DoryAttention` and `DoryForCausalLM.forward`.
    initializer_range: float = 0.02
    sliding_window: int | None = 1024
    rope_parameters: dict | None = None
    layer_types: list[str] | None = None
    ngpt_alpha_init: float = 0.05
    attention_output_gate_logit_scale_init: float | None = None
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    n_input_layers: int = 0
    n_recurrent_layers: int | None = None
    n_output_layers: int = 0
    n_recurrent_loops: int = 1
    # "last_loop" reuses one cache, overwritten by each recurrent loop. vLLM
    # implements it; see the module docstring for why this file does not.
    recurrent_kv_cache_mode: str = "per_loop"

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "mlp" if i % 2
                else "full_attention" if i % 8 == 6
                else "sliding_attention"
                for i in range(self.num_hidden_layers)]
        if self.rope_parameters is None:
            self.rope_parameters = {
                kind: {"rope_type": "default", "partial_rotary_factor": 1.0}
                for kind in ATTENTION_TYPES if kind in self.layer_types}
        if self.n_recurrent_layers is None:
            self.n_recurrent_layers = (self.num_hidden_layers
                                       - self.n_input_layers
                                       - self.n_output_layers)
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        valid = {"sliding_attention", "full_attention", "mlp"}
        if self.layer_types is None or any(kind not in valid
                                           for kind in self.layer_types):
            raise ValueError("layer_types must be a list of sliding_attention, "
                             "full_attention, or mlp")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")

    def validate_architecture(self):
        if self.layer_types is None or self.n_recurrent_layers is None:
            raise ValueError("Dory layer layout must be initialized")
        partitions = (self.n_input_layers, self.n_recurrent_layers,
                      self.n_output_layers)
        if any(n < 0 for n in partitions) or sum(partitions) != self.num_hidden_layers:
            raise ValueError("Dory input, recurrent, and output layers must sum "
                             "to num_hidden_layers")
        if self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        if any(n <= 0 for n in (self.head_dim, self.num_attention_heads,
                                self.num_key_value_heads)):
            raise ValueError("Attention dimensions must be positive integers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by "
                             "num_key_value_heads")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if "sliding_attention" in self.layer_types and (
                self.sliding_window is None or self.sliding_window <= 0):
            raise ValueError("sliding_window must be a positive integer")
        if self.attention_bias or self.mlp_bias:
            raise ValueError("Dory requires bias-free projections")
        if self.hidden_act != "silu" or self.tie_word_embeddings:
            raise ValueError("Dory requires SiLU and untied embeddings")

    def validate_rope(self):
        if self.layer_types is None:
            raise ValueError("Dory layer layout must be initialized")
        attention_types = set(self.layer_types) - {"mlp"}
        if self.rope_parameters is None or (
                set(self.rope_parameters) - set(ATTENTION_TYPES)
                or attention_types - self.rope_parameters.keys()):
            raise ValueError("rope_parameters must define each attention block type")
        for rope in self.rope_parameters.values():
            if (not isinstance(rope, dict) or rope.get("rope_type") != "default"
                    or rope.get("rope_theta", 0) <= 0):
                raise ValueError("Dory requires default RoPE with a positive rope_theta")
            fraction = rope.get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * fraction)
            if not 0 < fraction <= 1 or rotary_dim <= 0 or rotary_dim % 2:
                raise ValueError("Rotary dimensions must be positive, even, and "
                                 "<= head_dim")


# --- the looped layout ------------------------------------------------------
# Two pure functions, because three different things need to agree about it: the
# model (which layer runs when), the KV cache (one entry per *execution*, not
# per layer) and choose_layer.py (which point in the computation an activation
# came from).


def physical_layer_schedule(config):
    """Physical layer index for every sublayer the forward pass executes.

    `[0, 1, ..., 15, 16, ..., 55, 16, ..., 55, 56, ..., 71]` for the 4B
    checkpoint at two loops: 112 executions of 72 modules. The identity for a
    config with one loop.
    """
    total = config.num_hidden_layers
    first = config.n_input_layers
    last = first + (config.n_recurrent_layers or 0)
    return (list(range(first))
            + list(range(first, last)) * max(1, config.n_recurrent_loops)
            + list(range(last, total)))


def attention_layer_types(config):
    """`layer_types` of the unrolled model, i.e. one entry per execution.

    This is the KV cache's layout: a recurrent attention block holds a separate
    cache per loop (`recurrent_kv_cache_mode="per_loop"`), because each loop
    sees a different residual stream and therefore writes different keys.
    """
    return [config.layer_types[index] for index in physical_layer_schedule(config)]


class DoryCache(DynamicCache):
    """A `DynamicCache` with one layer per execution rather than per module.

    `DynamicCache(config=...)` would build 72 entries from `config.layer_types`
    and index them by physical layer, which is wrong for a looped model: the two
    passes through a recurrent attention block would append into the same entry
    and each would then attend to the other's keys.
    """

    def __init__(self, config, offloading=False, offload_only_non_sliding=False):
        if config.recurrent_kv_cache_mode != "per_loop":
            raise NotImplementedError(
                "recurrent_kv_cache_mode="
                f"{config.recurrent_kv_cache_mode!r} is not supported here: it "
                "has each loop overwrite the previous loop's KV, and a "
                "transformers Cache only appends. Load the checkpoint with "
                "per_loop instead (it is the default) or use vLLM, which "
                "implements both.")
        layers = [
            DynamicSlidingWindowLayer(sliding_window=config.sliding_window)
            if kind == "sliding_attention" else DynamicLayer()
            for kind in attention_layer_types(config)]
        # Skips `DynamicCache.__init__`, which only knows how to build its
        # layers from a config, and hands the list straight to `Cache`.
        Cache.__init__(self, layers=layers, offloading=offloading,
                       offload_only_non_sliding=offload_only_non_sliding)


# --- nGPT arithmetic --------------------------------------------------------


def justnorm(x, dim=-1):
    """nGPT's normalization: project onto the unit sphere, accumulating in fp32."""
    wide = x.float()
    normalized = wide * torch.rsqrt(wide.pow(2).sum(dim=dim, keepdim=True) + EPS)
    return normalized.to(x.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    """Dory's residual update: a learned step from `a` towards `b` on the sphere.

    Not `a + b`. Both ends are normalized first, the step length is a learned
    per-channel `alpha` (stored divided by `initializer_range`), and the result
    is normalized again, so the residual stream stays on the unit sphere all the
    way through the model.
    """
    rate = (alpha.float() * alpha_scale).abs()
    start, end = justnorm(a).float(), justnorm(b).float()
    return justnorm(start + rate * (end - start)).to(a.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states, n_rep):
    batch, kv_heads, length, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, n_rep, length, head_dim)
    return hidden_states.reshape(batch, kv_heads * n_rep, length, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask,
                            dropout=0.0, scaling=None, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    keys = repeat_kv(key, module.num_key_value_groups)
    values = repeat_kv(value, module.num_key_value_groups)

    weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : keys.shape[-2]]
    weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, values).transpose(1, 2).contiguous()
    return output, weights


def _default_rope_parameters(config, device=None, layer_type=None, **kwargs):
    """Unscaled RoPE inverse frequencies for one attention block type."""
    parameters = config.rope_parameters[layer_type]
    base = parameters["rope_theta"]
    dim = int(config.head_dim * parameters.get("partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq.to(device), 1.0


class DoryRotaryEmbedding(nn.Module):
    """One set of inverse frequencies per attention block type.

    Dory's sliding blocks use theta 10,000 and its full blocks 10,000,000, which
    is how a 1024-token window and a 256k context live in the same model.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.layer_types = sorted(set(config.layer_types) - {"mlp"})
        for layer_type in self.layer_types:
            inv_freq = self.inverse_frequencies(layer_type, device)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", 1.0)

    def inverse_frequencies(self, layer_type, device=None):
        """Also called from `_init_weights`; see the comment there."""
        parameters = self.config.rope_parameters[layer_type]
        # `ROPE_INIT_FUNCTIONS` holds the *scaled* variants only; plain RoPE is
        # not in it, which is why every model with per-layer rope carries its
        # own copy of that arithmetic.
        initialize = (_default_rope_parameters
                      if parameters["rope_type"] == "default"
                      else ROPE_INIT_FUNCTIONS[parameters["rope_type"]])
        inv_freq, scaling = initialize(self.config, device, layer_type=layer_type)
        setattr(self, f"{layer_type}_attention_scaling", scaling)
        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        scaling = getattr(self, f"{layer_type}_attention_scaling")
        frequencies = position_ids[..., None].float() * inv_freq.to(
            device=x.device, dtype=torch.float)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return ((embedding.cos() * scaling).to(dtype=x.dtype),
                (embedding.sin() * scaling).to(dtype=x.dtype))


class DoryAttention(nn.Module):
    """GQA with nGPT qk-normalization, a sigmoid output gate, and no LayerNorm.

    Three things make it not a llama attention:

    1. The queries and keys are normalized *after* RoPE, not before, and the
       queries are then scaled by `sqk**2` rather than by a constant. There is
       no `q_norm`/`k_norm` weight: `sqk` is the whole of it.
    2. `g_proj` produces a per-(head, channel) sigmoid gate on the attention
       output, applied before `o_proj`.
    3. The gated output is projected onto the unit sphere over *all* heads at
       once, which is what makes the residual update below a well-defined
       interpolation.
    """

    def __init__(self, config, layer_idx, cache_indices):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # One cache entry per pass through this block; see `attention_layer_types`.
        self.cache_indices = cache_indices
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.query_size = self.num_heads * self.head_dim
        self.key_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = (config.sliding_window
                               if self.layer_type == "sliding_attention" else None)

        self.q_proj = nn.Linear(config.hidden_size, self.query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.key_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.key_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, self.query_size, bias=False)
        self.o_proj = nn.Linear(self.query_size, config.hidden_size, bias=False)

        self.sqk = nn.Parameter(torch.empty(self.query_size))
        self.s_gate = nn.Parameter(torch.empty(self.query_size))
        # The learned scales are stored divided by `initializer_range`, so every
        # read of them multiplies it back. `attention_output_gate_logit_scale_init`
        # defaults to sqrt(hidden_size) when the checkpoint leaves it null.
        deviation = config.initializer_range
        self._sqk_factor = self.head_dim ** 0.5 / deviation
        gate_scale = config.attention_output_gate_logit_scale_init
        if gate_scale is None:
            gate_scale = config.hidden_size ** 0.5
        self._s_gate_factor = gate_scale ** 0.5 / deviation

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, loop_idx=0, **kwargs):
        shape = hidden_states.shape[:-1]
        heads = (*shape, -1, self.head_dim)

        query = self.q_proj(hidden_states).view(heads).transpose(1, 2)
        key = self.k_proj(hidden_states).view(heads).transpose(1, 2)
        value = self.v_proj(hidden_states).view(heads).transpose(1, 2)
        gate = self.g_proj(hidden_states)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # nGPT qk-norm, after RoPE. `sqk` is squared because it scales a vector
        # that has just been made unit-length, i.e. it is the only thing left
        # setting the logit magnitude.
        sqk = (self.sqk.float() * self._sqk_factor).view(1, self.num_heads, 1,
                                                         self.head_dim)
        query = (sqk.square() * justnorm(query).float()).to(query.dtype)
        key = justnorm(key)

        if past_key_values is not None:
            index = self.cache_indices[loop_idx if len(self.cache_indices) > 1 else 0]
            key, value = past_key_values.update(key, value, index)

        attention: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        output, _ = attention(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)

        output = output.reshape(*shape, -1)
        scale = (self.s_gate.float() * self._s_gate_factor).square()
        output = (output.float() * torch.sigmoid(gate.float() * scale)).to(
            hidden_states.dtype)
        return self.o_proj(justnorm(output))


class DoryMLP(nn.Module):
    """SwiGLU with nGPT's `suv` scaling and a normalized activation."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,
                                   bias=False)
        self.suv_gate = nn.Parameter(torch.empty(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.empty(config.intermediate_size))
        self._suv_factor = config.hidden_size ** 0.5

    def forward(self, hidden_states):
        gate = (self.gate_proj(hidden_states).float()
                * (self.suv_gate.float() * self._suv_factor))
        up = (self.up_proj(hidden_states).float()
              * (self.suv_up.float() * self._suv_factor))
        activated = (F.silu(gate) * up).to(hidden_states.dtype)
        return self.down_proj(justnorm(activated))


class DoryDecoderLayer(GradientCheckpointingLayer):
    """One physical layer: a single sublayer (attention *or* MLP), then a slerp.

    `mixer` rather than `self_attn`/`mlp` because the checkpoint calls it that,
    and because which of the two it is depends on `layer_types[layer_idx]`.
    """

    def __init__(self, config, layer_idx, cache_indices=()):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_attention = self.layer_type != "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(config, layer_idx, cache_indices)
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.mixer = DoryMLP(config)
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None, loop_idx=0, **kwargs):
        if not self.is_attention:
            return slerp_alpha(hidden_states, self.mixer(hidden_states),
                               self.mlp_alpha, self._alpha_scale)
        mixed = self.mixer(hidden_states,
                           position_embeddings=position_embeddings,
                           attention_mask=attention_mask,
                           past_key_values=past_key_values,
                           loop_idx=loop_idx, **kwargs)
        return slerp_alpha(hidden_states, mixed, self.attn_alpha,
                           self._alpha_scale)


class DoryPreTrainedModel(PreTrainedModel):
    config: DoryConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DoryDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # The nGPT scales are multiplicative and stored divided by
        # `initializer_range`, so "1" is what an untrained one has to be, not 0.
        deviation = self.config.initializer_range
        if isinstance(module, DoryAttention):
            init.constant_(module.sqk, deviation / module.head_dim ** 0.5)
            init.constant_(module.s_gate, deviation / module._s_gate_factor)
        elif isinstance(module, DoryMLP):
            init.constant_(module.suv_gate, 1.0 / module._suv_factor)
            init.constant_(module.suv_up, 1.0 / module._suv_factor)
        elif isinstance(module, DoryDecoderLayer):
            alpha = getattr(module, "attn_alpha", None)
            init.constant_(module.mlp_alpha if alpha is None else alpha, deviation)
        elif isinstance(module, DoryRotaryEmbedding):
            # Not an optional nicety. The inverse frequencies are
            # non-persistent buffers, so they are not in the checkpoint and
            # `from_pretrained` does not fill them: the module is built on the
            # meta device and only the tensors the checkpoint names are
            # materialized. Without this they are whatever the allocator had,
            # which does not raise -- it quietly rotates every query and key by
            # a garbage angle. Positions 0-10 still look sane, because position
            # 0 is unrotated whatever the frequencies are.
            for layer_type in module.layer_types:
                init.copy_(getattr(module, f"{layer_type}_inv_freq"),
                           module.inverse_frequencies(layer_type))


class DoryModel(DoryPreTrainedModel):
    """The looped stack. `embeddings` and `layers`, as the checkpoint names them."""

    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Which cache entries each physical layer owns. A recurrent attention
        # block owns one per loop; everything else owns one.
        schedule = physical_layer_schedule(config)
        cache_indices = {}
        for step, layer_idx in enumerate(schedule):
            cache_indices.setdefault(layer_idx, []).append(step)
        self.layers = nn.ModuleList([
            DoryDecoderLayer(config, index, cache_indices.get(index, [index]))
            for index in range(config.num_hidden_layers)])

        self.rotary_emb = DoryRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_hidden_states=None, **kwargs) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        config = self.config
        if use_cache and past_key_values is None:
            past_key_values = DoryCache(config)

        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1],
                                         device=inputs_embeds.device) + seen).unsqueeze(0)

        if not isinstance(mask_mapping := attention_mask, dict):
            unrolled = attention_layer_types(config)
            mask_kwargs = {"config": config, "inputs_embeds": inputs_embeds,
                           "attention_mask": attention_mask,
                           "past_key_values": past_key_values,
                           "position_ids": position_ids}
            # The cache entry to size each mask against has to be named: left
            # to itself, `create_causal_mask` takes the first non-sliding entry,
            # which in a Dory cache is an MLP placeholder holding no tokens.
            mask_mapping = {
                kind: builder(**mask_kwargs,
                              layer_idx=unrolled.index(kind) if kind in unrolled else 0)
                for kind, builder in (("full_attention", create_causal_mask),
                                      ("sliding_attention",
                                       create_sliding_window_causal_mask))
                if kind in unrolled}

        position_embeddings = {
            kind: self.rotary_emb(inputs_embeds, position_ids, kind)
            for kind in set(config.layer_types) - {"mlp"}}

        def run(layer, hidden_states, loop_idx=0):
            kind = layer.layer_type
            return layer(hidden_states=hidden_states,
                         position_embeddings=position_embeddings.get(kind),
                         attention_mask=mask_mapping.get(kind),
                         past_key_values=past_key_values,
                         position_ids=position_ids,
                         loop_idx=loop_idx, **kwargs)

        collected = [] if output_hidden_states else None
        first = config.n_input_layers
        last = first + config.n_recurrent_layers

        hidden_states = inputs_embeds
        for layer in self.layers[:first]:
            if collected is not None:
                collected.append(hidden_states)
            hidden_states = run(layer, hidden_states)

        # The input block's output is re-injected at the start of every loop --
        # that is what makes this a latent-recurrent model rather than a stack
        # with shared weights.
        injected = hidden_states
        state = injected
        for loop in range(config.n_recurrent_loops):
            hidden_states = justnorm(state + injected)
            for layer in self.layers[first:last]:
                if collected is not None:
                    collected.append(hidden_states)
                hidden_states = run(layer, hidden_states, loop_idx=loop)
            state = justnorm(hidden_states)

        hidden_states = state
        for layer in self.layers[last:]:
            if collected is not None:
                collected.append(hidden_states)
            hidden_states = run(layer, hidden_states)

        if collected is not None:
            collected.append(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(collected) if collected is not None else None)


class DoryForCausalLM(DoryPreTrainedModel, GenerationMixin):
    config: DoryConfig
    _tp_plan = {"lm_head": "colwise_gather_output"}

    def __init__(self, config: DoryConfig):
        super().__init__(config)
        if config.recurrent_kv_cache_mode != "per_loop":
            raise NotImplementedError(
                f"recurrent_kv_cache_mode={config.recurrent_kv_cache_mode!r} is "
                "not implemented here; see modeling_dory.py's docstring. Set it "
                "to per_loop, or serve the model with vLLM.")
        self.backbone = DoryModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Per-token output temperature, and the only nGPT scale outside a layer.
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def set_input_embeddings(self, value):
        self.backbone.embeddings = value

    def get_decoder(self):
        return self.backbone

    def _prepare_cache_for_generation(self, generation_config, model_kwargs,
                                      *args, **kwargs):
        # `generate` would otherwise build a `DynamicCache` from the config,
        # which has one entry per physical layer instead of one per execution.
        if (model_kwargs.get("past_key_values") is None
                and generation_config.use_cache is not False
                and generation_config.cache_implementation is None):
            model_kwargs["past_key_values"] = DoryCache(self.config)
        return super()._prepare_cache_for_generation(
            generation_config, model_kwargs, *args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, **kwargs) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

        hidden_states = outputs.last_hidden_state
        indices = (slice(-logits_to_keep, None) if isinstance(logits_to_keep, int)
                   else logits_to_keep)
        logits = self.lm_head(hidden_states[:, indices, :]).float()
        # There is no final norm: the residual stream is already on the unit
        # sphere, and `logit_scale` is what turns a cosine into a logit.
        logits = logits * (self.lm_head.logit_scale.float()
                           / self.config.initializer_range)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states)


def register_dory():
    """Make the `Auto*` classes recognise `model_type: dory`. Idempotent."""
    try:
        AutoConfig.register("dory", DoryConfig)
    except ValueError:
        return False            # already registered, by us or by transformers
    AutoModel.register(DoryConfig, DoryModel)
    AutoModelForCausalLM.register(DoryConfig, DoryForCausalLM)
    return True


register_dory()
