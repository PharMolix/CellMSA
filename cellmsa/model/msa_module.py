# type: ignore
from functools import partial, wraps

import torch
from torch import nn
from torch import Tensor, is_tensor
import torch.nn.functional as F

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from beartype.typing import (
    Any,
    Literal,
    Tuple,
)

from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    typecheck,
    checkpoint,
)

from .attention import (
    Attention,
    pad_to_multiple,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed
)

from alphafold3_pytorch.utils.model_utils import pack_one
from alphafold3_pytorch.utils.utils import get_gpu_type, not_exists

# personal libraries

from hyper_connections.hyper_connections_with_multi_input_streams import HyperConnections

# other external libs

# einstein notation related

import einx
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# constants

MAX_CONCURRENT_TENSOR_ELEMENTS = int(2e9) if "ROCm" in get_gpu_type() else float("inf")

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def max_neg_value(t: Tensor):
    return -torch.finfo(t.dtype).max

def exclusive_cumsum(t, dim = -1):
    return t.cumsum(dim = dim) - t


@typecheck
def should_checkpoint(
    self: Module,
    inputs: Tensor | Tuple[Tensor, ...],
    check_instance_variable: str | None = 'checkpoint'
) -> bool:
    if is_tensor(inputs):
        inputs = (inputs,)

    return (
        self.training and
        any([i.requires_grad for i in inputs]) and
        (not exists(check_instance_variable) or getattr(self, check_instance_variable, False))
    )

# decorators

def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)
    return inner

@typecheck
def pad_and_window(
    t: Float['b n ...'] | Int['b n ...'],
    window_size: int
):
    t = pad_to_multiple(t, window_size, dim = 1)
    t = rearrange(t, 'b (n w) ... -> b n w ...', w = window_size)
    return t

# packed atom representation functions

@typecheck
def lens_to_mask(
    lens: Int['b ...'],
    max_len: int | None = None
) -> Bool['... m']:

    device = lens.device
    if not exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device = device)
    return einx.less('m, ... -> ... m', arange, lens)

@typecheck
def to_pairwise_mask(
    mask_i: Bool['... n'],
    mask_j: Bool['... n'] | None = None
) -> Bool['... n n']:

    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)

@typecheck
def sum_pool_with_lens(
    feats: Float['b m d'],
    lens: Int['b n']
) -> tuple[Float['b n d'], Bool['b n']]:

    seq_len = feats.shape[1]

    mask = lens > 0
    assert (lens.sum(dim = -1) <= seq_len).all(), 'one of the lengths given exceeds the total sequence length of the features passed in'

    cumsum_feats = feats.cumsum(dim = 1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value = 0.)

    cumsum_indices = lens.cumsum(dim = 1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value = 0)


    cumsum_indices = repeat(cumsum_indices, 'b n -> b n d', d = cumsum_feats.shape[-1])
    sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    return summed, mask

@typecheck
def batch_repeat_interleave(
    feats: Float['b n ...'] | Bool['b n ...'] | Bool['b n'] | Int['b n'],
    lens: Int['b n'],
    output_padding_value: float | int | bool | None = None, # this value determines what the output padding value will be
) -> Float['b m ...'] | Bool['b m ...'] | Bool['b m'] | Int['b m']:

    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device = device)

    offsets = exclusive_cumsum(lens)
    indices = einx.add('w, b n -> b n w', arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.clamp(min = 0).sum(dim = -1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device = device, dtype = torch.long)

    indices = indices.masked_fill(~mask, max_len) # scatter to sink position for padding
    indices = rearrange(indices, 'b n w -> b (n w)')

    # scatter

    seq_arange = torch.arange(seq, device = device)
    seq_arange = repeat(seq_arange, 'n -> b (n w)', b = batch, w = window_size)


    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather


    feats, unpack_one = pack_one(feats, 'b n *')
    output_indices = repeat(output_indices, 'b m -> b m d', d = feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value

    output_padding_value = default(output_padding_value, False if dtype == torch.bool else 0)

    output = einx.where(
        'b n, b n ..., -> b n ...',
        output_mask, output, output_padding_value
    )

    return output

# classic feedforward, SwiGLU variant
# they name this 'transition' in their paper
# Algorithm 11

class SwiGLU(Module):
    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float[' ... (d//2)']:

        x, gates = x.chunk(2, dim = -1)
        return F.silu(gates) * x

class Transition(Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor = 2
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float['... d']:

        return self.ff(x)

# dropout
# they seem to be using structured dropout - row / col wise in triangle modules

class Dropout(Module):
    @typecheck
    def __init__(
        self,
        prob: float,
        *,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    @typecheck
    def forward(
        self,
        t: Tensor
    ) -> Tensor:

        if self.dropout_type in {'row', 'col'}:
            assert t.ndim == 4, 'tensor must be 4 dimensions for row / col structured dropout'

        if not exists(self.dropout_type):
            return self.dropout(t)

        if self.dropout_type == 'row':
            batch, _, col, dim = t.shape
            ones_shape = (batch, 1, col, dim)

        elif self.dropout_type == 'col':
            batch, row, _, dim = t.shape
            ones_shape = (batch, row, 1, dim)

        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped

# normalization
# both pre layernorm as well as adaptive layernorm wrappers

class PreLayerNorm(Module):
    @typecheck
    def __init__(
        self,
        fn,
        *,
        dim,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    @typecheck
    def forward(
        self,
        x: Float['... n d'],
        **kwargs
    ) -> (
        Float['... n d'] |
        tuple[Float['... n d'] | Any]
    ):

        x = self.norm(x)
        return self.fn(x, **kwargs)

class AdaptiveLayerNorm(Module):
    """ Algorithm 26 """

    def __init__(
        self,
        *,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)
        self.norm_cond = nn.LayerNorm(dim_cond, bias = False)

        self.to_gamma = nn.Sequential(
            Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    @typecheck
    def forward(
        self,
        x: Float['b n d'],
        cond: Float['b n dc']
    ) -> Float['b n d']:

        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta


# there are two types of attention in this paper, triangle and attention-pair-bias
# they differ by how the attention bias is computed
# triangle is axial attention w/ itself projected for bias

class AttentionPairBias(Module):
    """An Attention module with pair bias computation."""

    def __init__(self, *, heads, dim_pairwise, window_size=None, num_memory_kv=0, **attn_kwargs):
        super().__init__()
        assert dim_pairwise == heads, f"Expected dim_pairwise ({dim_pairwise}) to equal heads ({heads}) when using pairwise_repr directly as attention bias."

        self.window_size = window_size

        self.attn = Attention(
            heads = heads,
            window_size = window_size,
            num_memory_kv = num_memory_kv,
            **attn_kwargs
        )

        # line 8 of Algorithm 24

        self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
        self.to_attn_bias = nn.Sequential(
            Rearrange("b ... h -> b h ...")
        )

    @typecheck
    def forward(
        self,
        single_repr: Float["b n ds"],  # type: ignore
        *,
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        attn_bias: Float["b n n"] | Float["b nw w (w*2)"] | None = None,  # type: ignore
        return_values: bool = False,
        return_attn: bool = False,
        value_residual: Float['b _ _ _'] | None = None,
        **kwargs,
    ) -> (
        Float['b n ds'] |
        tuple[Float['b n ds'], Float['b _ _ _']]
    ):  # type: ignore

        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param attn_bias: The attention bias tensor.
        :return: The output tensor.
        """
        b, dp = pairwise_repr.shape[0], pairwise_repr.shape[-1]
        dtype, device = pairwise_repr.dtype, pairwise_repr.device
        w, has_window_size = self.window_size, exists(self.window_size)


        windowed_pairwise = pairwise_repr.ndim == 5

        windowed_attn_bias = None

        if exists(attn_bias):
            windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        if has_window_size:
            if not windowed_pairwise:
                pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size=w)
            if exists(attn_bias):
                attn_bias = full_attn_bias_to_windowed(attn_bias, window_size=w)
        else:
            assert (
                not windowed_pairwise
            ), "Cannot pass in windowed pairwise representation if no `window_size` given to `AttentionPairBias`."
            assert (
                not_exists(windowed_attn_bias) or not windowed_attn_bias
            ), "Cannot pass in windowed attention bias if no `window_size` is set for `AttentionPairBias`."

        # attention bias preparation with further addition from pairwise repr

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "b ... -> b 1 ...")
        else:
            attn_bias = 0.0

        if pairwise_repr.numel() > MAX_CONCURRENT_TENSOR_ELEMENTS:
            # create a stub tensor and normalize it to maintain gradients to `to_attn_bias_norm`
            stub_pairwise_repr = torch.zeros((b, dp), dtype=dtype, device=device)
            stub_attn_bias_norm = self.to_attn_bias_norm(stub_pairwise_repr) * 0.0

            # adjust `attn_bias_norm` dimensions to match `pairwise_repr`
            attn_bias_norm = pairwise_repr + (
                stub_attn_bias_norm[:, None, None, None, :]
                if windowed_pairwise
                else stub_attn_bias_norm[:, None, None, :]
            )

            # apply bias transformation
            attn_bias = self.to_attn_bias(attn_bias_norm) + attn_bias
        else:
            attn_bias = self.to_attn_bias(pairwise_repr) + attn_bias

        out, values, pair_rep = self.attn(
            single_repr,
            attn_bias = attn_bias,
            value_residual = value_residual,
            return_values = True,
            **kwargs
        )

        # whether to return values for value residual learning

        if not return_values:
            return out, pair_rep

        return out, values, pair_rep

# PairwiseBlock
# used in both MSAModule and Pairformer
# consists of all the "Triangle" modules + Transition

class PairwiseBlock(Module):
    def __init__(
        self,
        *,
        dim_pairwise = 8,
    ):
        super().__init__()

        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)
        self.pairwise_transition = pre_ln(nn.Linear(dim_pairwise, dim_pairwise))

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        *,
        mask: Bool['b n'] | None = None,
        return_values = False,
    ):
        # Keep only the pairwise transition MLP.
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr

        if not return_values:
            return pairwise_repr

        # Preserve the original interface when callers request residual values.
        return pairwise_repr, (None, None)

# msa module

class OuterProductMean(Module):
    """ Algorithm 9 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 8,
        dim_hidden = 32,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim_msa)
        self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_pairwise_repr = nn.Linear(dim_hidden ** 2, dim_pairwise)

    @typecheck
    def forward(
        self,
        msa: Float['b s n d'],
        *,
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None
    ) -> Float['b n n dp']:

        dtype = msa.dtype

        msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim = -1)

        # maybe masked mean for outer product

        if exists(msa_mask):
            a = einx.multiply('b s i d, b s -> b s i d', a, msa_mask.type(dtype))
            b = einx.multiply('b s j e, b s -> b s j e', b, msa_mask.type(dtype))

            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')

            num_msa = reduce(msa_mask.type(dtype), '... s -> ...', 'sum')

            outer_product_mean = einx.divide('b i j d e, b', outer_product, num_msa.clamp(min = self.eps))
        else:
            num_msa = msa.shape[1]
            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')
            outer_product_mean = outer_product / num_msa

        # flatten

        outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')

        # masking for pairwise repr

        if exists(mask):
            mask = to_pairwise_mask(mask)
            outer_product_mean = einx.multiply(
                'b i j d, b i j', outer_product_mean, mask.type(dtype)
            )

        pairwise_repr = self.to_pairwise_repr(outer_product_mean)
        return pairwise_repr


class MSAPairWeightedAveraging(Module):
    """ Algorithm 10 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_head = 32,
        heads = 8,
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()
        assert dim_pairwise == heads, f"Expected dim_pairwise ({dim_pairwise}) to equal heads ({heads}) when using pairwise_repr directly as attention bias."
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = nn.Sequential(
            nn.LayerNorm(dim_msa),
            LinearNoBias(dim_msa, dim_inner * 2),
            Rearrange('b s n (gv h d) -> gv b h s n d', gv = 2, h = heads)
        )

        self.pairwise_repr_to_attn = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            Rearrange('b h s n d -> b s n (h d)'),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        *,
        msa: Float['b s n d'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b s n d']:

        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        # line 3

        b = self.pairwise_repr_to_attn(pairwise_repr)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            b = b.masked_fill(~mask, max_neg_value(b))

        # line 5

        weights = b.softmax(dim = -1)

        # line 6

        out = einsum(weights, values, 'b h i j, b h s j d -> b h s i d')

        out = out * gates

        # combine heads

        return self.to_out(out)

class MSAModule(Module):
    """ Algorithm 8 """

    def __init__(
        self,
        *,
        dim_single = 512,
        dim_pairwise = 8,
        depth = 4,
        dim_msa = 64,
        dim_msa_input=512,
        dim_additional_msa_feats=2,
        outer_product_mean_dim_hidden = 32,
        msa_pwa_dropout_row_prob = 0.15,
        msa_pwa_heads = 8,
        msa_pwa_dim_head = 32,
        checkpoint = False,
        pairwise_block_kwargs: dict = dict(),
        max_num_msa: int | None = None,
        layerscale_output: bool = True
    ):
        super().__init__()

        self.max_num_msa = default(
            max_num_msa, float('inf')
        )  # cap the number of MSAs, will do sample without replacement if exceeds

        self.msa_init_proj = LinearNoBias(dim_msa_input + dim_additional_msa_feats, dim_msa)

        self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)

        layers = ModuleList([])

        for _ in range(depth):

            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)

            outer_product_mean = OuterProductMean(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                dim_hidden = outer_product_mean_dim_hidden
            )

            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                heads = msa_pwa_heads,
                dim_head = msa_pwa_dim_head,
                dropout = msa_pwa_dropout_row_prob,
                dropout_type = 'row'
            )

            msa_transition = Transition(dim = dim_msa)

            layers.append(ModuleList([
                outer_product_mean,
                msa_pair_weighted_avg,
                msa_pre_ln(msa_transition),
            ]))

        self.checkpoint = checkpoint

        self.layers = layers

        # Do not start from exact zero, otherwise the MSA branch is fully gated off
        # and its internal parameters receive no gradient at initialization.
        self.layerscale_output = nn.Parameter(torch.full((dim_pairwise,), 1.)) if layerscale_output else 1.

        self.pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

        self.dmi = dim_additional_msa_feats

    @typecheck
    def to_layers(
        self,
        *,
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
        ) in self.layers:

            # communication between msa and pairwise rep

            pairwise_repr = outer_product_mean(msa, mask=mask, msa_mask=msa_mask) + pairwise_repr

            msa = msa_pair_weighted_avg(msa=msa, pairwise_repr=pairwise_repr, mask=mask) + msa
            msa = msa_transition(msa) + msa

        return pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        inputs = (pairwise_repr, mask, msa, msa_mask)

        wrapped_layers = []

        def outer_product_mean_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                pairwise_repr = fn(msa = msa, mask = mask, msa_mask = msa_mask) + pairwise_repr
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def msa_pair_weighted_avg_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                msa = fn(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def msa_transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                msa = fn(msa) + msa
                return pairwise_repr, mask, msa, msa_mask
            return inner

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
        ) in self.layers:

            wrapped_layers.append(outer_product_mean_wrapper(outer_product_mean))
            wrapped_layers.append(msa_pair_weighted_avg_wrapper(msa_pair_weighted_avg))
            wrapped_layers.append(msa_transition_wrapper(msa_transition))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        pairwise_repr, *_ = inputs
        return pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        additional_msa_feats: Float['b s n {self.dmi}'] | None = None,
    ) -> Float['b n n dp']:

        batch, num_msa, device = *msa.shape[:2], msa.device

        # sample without replacement

        if num_msa > self.max_num_msa:
            rand = torch.randn((batch, num_msa), device = device)

            if exists(msa_mask):
                rand.masked_fill_(~msa_mask, max_neg_value(msa))

            indices = rand.topk(self.max_num_msa, dim = -1).indices

            msa, unpack_one = pack_one(msa, 'b s *')
            msa_indices = repeat(indices, 'b sampled -> b sampled d', d = msa.shape[-1])
            msa = msa.gather(1, msa_indices)
            msa = unpack_one(msa)

            if exists(msa_mask):
                msa_mask = msa_mask.gather(1, indices)

            if exists(additional_msa_feats):
                additional_msa_feats, unpack_one = pack_one(additional_msa_feats, 'b s *')
                additional_msa_indices = repeat(
                    indices, 'b sampled -> b sampled d', d=additional_msa_feats.shape[-1]
                )
                additional_msa_feats = additional_msa_feats.gather(1, additional_msa_indices)
                additional_msa_feats = unpack_one(additional_msa_feats)

        # account for no msa

        if exists(msa_mask):
            has_msa = reduce(msa_mask, 'b s -> b', 'any')

        # account for additional msa features

        if exists(additional_msa_feats):
            msa = torch.cat((msa, additional_msa_feats), dim=-1)

        # process msa

        msa = self.msa_init_proj(msa)

        single_msa_feats = self.single_to_msa_feats(single_repr)

        msa = rearrange(single_msa_feats, 'b n d -> b 1 n d') + msa

        # going through the layers

        if should_checkpoint(self, (pairwise_repr, msa)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        pairwise_repr = to_layers_fn(
            msa = msa,
            mask = mask,
            pairwise_repr = pairwise_repr,
            msa_mask = msa_mask
        )
        pairwise_repr = self.pairwise_block(pairwise_repr=pairwise_repr, mask=mask)

        # final masking and then layer scale

        if exists(msa_mask):
            pairwise_repr = einx.where(
                'b, b ..., -> b ...',
                has_msa, pairwise_repr, 0.
            )

        return pairwise_repr * self.layerscale_output

class PairformerStack(Module):
    """ Algorithm 17 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 48,
        recurrent_depth = 1, # effective depth will be depth * recurrent_depth
        pair_bias_attn_dim_head = 64,
        pair_bias_attn_heads = 16,
        dropout_row_prob = 0.25,
        num_register_tokens = 0,
        checkpoint = False,
        add_value_residual = False,
        num_residual_streams = 1,
        pairwise_block_kwargs: dict = dict(),
        pair_bias_attn_kwargs: dict = dict()
    ):
        super().__init__()

        # residual / hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # layers

        layers = ModuleList([])

        pair_bias_attn_kwargs = dict(
            dim = dim_single,
            dim_pairwise = dim_pairwise,
            heads = pair_bias_attn_heads,
            dim_head = pair_bias_attn_dim_head,
            dropout = dropout_row_prob,
            **pair_bias_attn_kwargs
        )

        for i in range(depth):

            is_first = i == 0
            accept_value_residual = add_value_residual and not is_first

            single_pre_ln = partial(PreLayerNorm, dim = dim_single)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            pair_bias_attn = AttentionPairBias(accept_value_residual = accept_value_residual, **pair_bias_attn_kwargs)
            single_transition = Transition(dim = dim_single)

            layers.append(ModuleList([
                init_hyper_conn(dim = dim_pairwise, branch = pairwise_block),
                init_hyper_conn(dim = dim_single, additional_input_paths = [('pairwise_repr', dim_pairwise)], branch = single_pre_ln(pair_bias_attn)),
                init_hyper_conn(dim = dim_single, branch = single_pre_ln(single_transition)),
            ]))
        self.layers = layers

        self.add_value_residual = add_value_residual

        # checkpointing

        self.checkpoint = checkpoint

        # https://arxiv.org/abs/2405.16039 and https://arxiv.org/abs/2405.15071
        # although possibly recycling already takes care of this

        assert recurrent_depth > 0
        self.recurrent_depth = recurrent_depth

        self.num_registers = num_register_tokens
        self.has_registers = num_register_tokens > 0

        if self.has_registers:
            self.single_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_single))
            self.pairwise_row_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))
            self.pairwise_col_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))

    def _update_pairwise_repr_with_pair_rep(
        self,
        pairwise_repr: Tensor,
        pair_rep: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        pair_rep = rearrange(pair_rep, 'b h i j -> b i j h')
        assert pairwise_repr.shape == pair_rep.shape, f"Expected pair bias attention output shape {pairwise_repr.shape} to match pair_rep shape {pair_rep.shape}"

        # Keep the Uni-Mol style qk update, but never write mask sentinels
        # (e.g. -finfo.max) back into the persistent pairwise state.
        pair_rep = torch.where(torch.isfinite(pair_rep), pair_rep, 0.)

        if exists(mask):
            pair_mask = rearrange(mask, 'b i -> b i 1 1') & rearrange(mask, 'b j -> b 1 j 1')
            pair_rep = pair_rep.masked_fill(~pair_mask, 0.)

        return pair_rep

    @typecheck
    def to_layers(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
        use_embedding: bool = False
    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        single_repr = self.expand_streams(single_repr)
        pairwise_repr = self.expand_streams(pairwise_repr)

        for _ in range(self.recurrent_depth):

            value_residual = None
            pairwise_value_residuals = None

            used_layers = self.layers if not use_embedding else self.layers[:-1] # if using embedding, skip the last layer
            for (
                pairwise_block,
                pair_bias_attn,
                single_transition
            ) in used_layers:

                single_repr, attn_values, pair_rep = pair_bias_attn(single_repr, pairwise_repr = pairwise_repr, mask = mask, return_values = True, value_residual = value_residual)

                pairwise_repr = self._update_pairwise_repr_with_pair_rep(pairwise_repr, pair_rep, mask)

                if self.add_value_residual:
                    value_residual = default(value_residual, attn_values)

                single_repr = single_transition(single_repr)

        single_repr = self.reduce_streams(single_repr)
        pairwise_repr = self.reduce_streams(pairwise_repr)

        return single_repr, pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
        use_embedding: bool = False
    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        def pairwise_block_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals = inputs
                pairwise_repr, pairwise_attn_values = layer(
                    pairwise_repr,
                    mask = mask,
                    return_values = True
                )

                if self.add_value_residual:
                    maybe_pairwise_value_residuals = default(maybe_pairwise_value_residuals, pairwise_attn_values)

                return single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals
            return inner

        def pair_bias_attn_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals  = inputs
                single_repr, attn_values, pair_rep = layer(
                    single_repr,
                    pairwise_repr = pairwise_repr,
                    mask = mask,
                    return_values = True,
                    value_residual = maybe_value_residual
                )

                pairwise_repr = self._update_pairwise_repr_with_pair_rep(pairwise_repr, pair_rep, mask)

                if self.add_value_residual:
                    maybe_value_residual = default(maybe_value_residual, attn_values)

                return single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals
            return inner

        def single_transition_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals = inputs
                single_repr = layer(single_repr)
                return single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals
            return inner

        wrapped_layers = []

        used_layers = self.layers if not use_embedding else self.layers[:-1] # if using embedding, skip the last layer
        for (
            pairwise_block,
            pair_bias_attn,
            single_transition
        ) in used_layers:

            wrapped_layers.append(pair_bias_attn_wrapper(pair_bias_attn))
            wrapped_layers.append(single_transition_wrapper(single_transition))

        single_repr = self.expand_streams(single_repr)
        pairwise_repr = self.expand_streams(pairwise_repr)

        for _ in range(self.recurrent_depth):
            inputs = (single_repr, pairwise_repr, mask, None, None)

            for layer in wrapped_layers:
                inputs = checkpoint(layer, inputs)

            single_repr, pairwise_repr, *_ = inputs

        single_repr = self.reduce_streams(single_repr)
        pairwise_repr = self.reduce_streams(pairwise_repr)

        return single_repr, pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
        use_embedding: bool = False
    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        # prepend register tokens

        if self.has_registers:
            batch_size, num_registers = single_repr.shape[0], self.num_registers
            single_registers = repeat(self.single_registers, 'r d -> b r d', b = batch_size)
            single_repr = torch.cat((single_registers, single_repr), dim = 1)

            row_registers = repeat(self.pairwise_row_registers, 'r d -> b r n d', b = batch_size, n = pairwise_repr.shape[-2])
            pairwise_repr = torch.cat((row_registers, pairwise_repr), dim = 1)
            col_registers = repeat(self.pairwise_col_registers, 'r d -> b n r d', b = batch_size, n = pairwise_repr.shape[1])
            pairwise_repr = torch.cat((col_registers, pairwise_repr), dim = 2)

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value = True)

        to_layers_fn = self.to_layers
        # main transformer block layers

        single_repr, pairwise_repr = to_layers_fn(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask,
            use_embedding = use_embedding
        )

        # splice out registers

        if self.has_registers:
            single_repr = single_repr[:, num_registers:]
            pairwise_repr = pairwise_repr[:, num_registers:, num_registers:]

        return single_repr, pairwise_repr
