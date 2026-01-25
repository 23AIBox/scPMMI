import math
from functools import partial
import json
import os
import copy
from typing import Optional, Tuple

import scanpy as sc
from tqdm import tqdm
from anndata import AnnData, concat
from collections import namedtuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Subset

from mamba_ssm.utils.generation import GenerationMixin
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .encoder import create_block, _init_weights
from .config import DecoderConfig
from ..activations import ACT2FN
from ..utils.hf import load_config_hf, load_state_dict_hf
from .. import logger



class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        n_features: int,
        vocab_size: int,
        patch_size: int,
        d_intermediate: int,
        dropout_prob = 0.1,
        padding_idx: Optional[int] = None, 
        append_cls: bool = True,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs # type: ignore
        ) # type: ignore

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm) # type: ignore
            ) # type: ignore
        return hidden_states
    

class MixerDecoder(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: DecoderConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.pool = config.pool
        activation = config.activation

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
        self.pred_proj = nn.Sequential(
            # nn.Linear(config.d_model, config.d_model * 2, bias=False, **factory_kwargs),
            ACT2FN[activation] if isinstance(activation, str) else activation,
            nn.Linear(config.d_model, config.patch_size, bias=False, **factory_kwargs),
        )
        # self.act_fn = ACT2FN[activation] if isinstance(activation, str) else activation
        # self.out_head = nn.Linear(config.d_model, config.patch_size, bias=False, **factory_kwargs)

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
        if self.config.append_cls:
            hidden_states = hidden_states[:, 1:]
        return self.pred_proj(hidden_states)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = DecoderConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        save_directory = os.path.join(save_directory, self.config.mod)
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=4)