import torch
import torch.nn as nn
from torch import Tensor
from functools import partial

from .flash_attn import FeedforwardNetwork, RMSNorm
from .encoder import _init_weights
from .decoder import MixerModel
from .config import DecoderConfig
from ..activations import ACT2FN


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        hidden_dim: int = 512,
        activation: str = "silu"
    ):
        super().__init__()
        self.act_fn = ACT2FN[activation] if isinstance(activation, str) else activation
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            if i == 0:
                self._decoder.append(nn.Linear(d_model, hidden_dim))
            else:
                self._decoder.append(nn.Linear(hidden_dim, hidden_dim))
            self._decoder.append(self.act_fn)
            self._decoder.append(nn.LayerNorm(hidden_dim))
        self.out_layer = nn.Linear(hidden_dim, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
        activation="silu",
        device=None,
        dtype=None
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, device=device, dtype=dtype) # type: ignore
        self.ffn = FeedforwardNetwork(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
            activation=activation,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ffn(x)
        x = self.norm(x)
        return residual + x


# class ModalityDecoder(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         vocab_size,
#         d_intermediate,
#         n_layers,
#         bias=True,
#         activation='silu',
#         device=None,
#         dtype=None
#     ):
#         super().__init__()
#         layers = []
#         for _ in range(n_layers):
#             layers.append(
#                 DecoderLayer(
#                     hidden_size=d_model,
#                     intermediate_size=d_intermediate,
#                     bias=bias,
#                     activation=activation,
#                     device=device,
#                     dtype=dtype,
#                 )
#             )
#         self.layers = nn.ModuleList(layers)
#         self.Wo = nn.Linear(d_model, vocab_size, 
#                         bias=bias, device=device, dtype=dtype)

#     def forward(self, x: torch.Tensor):
#         for layer in self.layers:
#             x = layer(x)
#         return self.Wo(x)
    

class ModalityDecoder(nn.Module):
    def __init__(
        self,
        config: DecoderConfig,
        activation="silu",
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.pool = config.pool

        # Backbone
        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            n_features=config.n_features,
            patch_size=config.patch_size,
            vocab_size=config.vocab_size,
            dropout_prob=config.dropout_prob,
            padding_idx=config.padding_idx,
            append_cls=config.append_cls,
            d_intermediate=config.d_intermediate,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            norm_epsilon=config.norm_epsilon,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            **factory_kwargs
        )

        self.act_fn = ACT2FN[activation] if isinstance(activation, str) else activation
        self.out_head = nn.Linear(config.d_model, config.n_features, bias=True, **factory_kwargs)

        # Weight init
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs) -> Tensor:
        """Forward pass for MambaLMHeadModel."""
        hidden_states = self.backbone(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = hidden_states[:, 0].reshape(hidden_states.shape[0], -1)
        hidden_states = self.act_fn(hidden_states)
        y = self.out_head(hidden_states)
        return y
