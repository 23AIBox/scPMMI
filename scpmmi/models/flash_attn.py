

from typing import Optional, Callable
from einops import rearrange

import torch
import torch.nn as nn
from torch import Tensor
from flash_attn import flash_attn_qkvpacked_func

from ..activations import ACT2FN


def _get_clones(factory_fn: Callable[[], nn.Module], N: int) -> nn.ModuleList:
    return nn.ModuleList([factory_fn() for _ in range(N)])


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of the FlashTransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = FlashTransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        encoder_layer: "FlashTransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        mask_check: bool = True
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.mask_check = mask_check


    def forward(self, src: Tensor) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """

        output = src

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        num_heads,
        softmax_scale=None,
        causal=True,
        intermediate_size=2048,
        dropout=0.1,
        activation="silu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        bias: bool = True,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.norm1 = RMSNorm(d_model, layer_norm_eps, **factory_kwargs)
        self.self_attn = FlashMHA(
            d_model, num_heads, 
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
            causal=causal,
            batch_first=batch_first,
            out_proj_bias=bias,
            qkv_proj_bias=bias,
            **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(d_model, layer_norm_eps, **factory_kwargs)
        
        self.FFN = FeedforwardNetwork(d_model, intermediate_size, bias, activation, **factory_kwargs)

        self.dropout2 = nn.Dropout(dropout)
        
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")
    
    def forward(
        self,
        src: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).

        Shape:
            see the docs in Transformer class.
        """
        
        if self.norm_scheme == "pre":
            # first layer
            src = self.norm1(src)
            src2 = self.self_attn(src)
            src = src + self.dropout1(src2)
            
            # second layer
            src = self.norm2(src)
            src = self.FFN(src)
            src = src + self.dropout2(src2)
        else:
            # first layer
            src2 = self.self_attn(src)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # second layer
            src = self.FFN(src)
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class FeedforwardNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
        activation="silu",
        device = None,
        dtype = None
    ):
        r"""
        Initialize the FeedForward module.

        Args:
            hidden_size (int): Input dimension.
            intermediate_size (int): Hidden dimension of the feedforward layer.
            bias (bool): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.act_fn = ACT2FN[activation]

    def forward(self, x:torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(device=device, dtype=dtype)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class FlashMHA(nn.Module):
    r"""Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        d_model (int): the number of expected features in the input (required).
        num_heads (int): the number of heads in multi-heads attention models (required).
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(
        self,
        d_model,
        num_heads,
        attention_dropout=0.0,
        causal=False,
        layer_idx=None,
        softmax_scale=None,
        batch_first=True,
        out_proj_bias=True,
        qkv_proj_bias=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.causal = causal
        self.layer_idx=layer_idx
        self.batch_first = batch_first

        assert self.d_model % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        assert self.head_dim %8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, **factory_kwargs)

    def forward(self, x, inference_params=None, return_attn_probs=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        attn_out = flash_attn_qkvpacked_func(qkv,  
                                            self.attention_dropout if self.training else 0.0, 
                                            softmax_scale=self.softmax_scale, 
                                            causal=self.causal,
                                            return_attn_probs=return_attn_probs) # type: ignore
        return self.out_proj(rearrange(attn_out, 'b s h d -> b s (h d)'))
