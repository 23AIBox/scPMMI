from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EncoderConfig:
    mod: str
    d_model: int
    d_intermediate: int
    n_layer: int
    d_embed: int
    patch_size: int
    n_features: int
    vocab_size: int
    pool: str = 'cls'
    activation: str = 'silu'
    dropout_prob: float = 0.1
    padding_idx: int = 0
    append_cls: bool = True
    norm_epsilon: float = 0.00001
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

@dataclass
class DecoderConfig:
    mod: str
    d_model: int
    d_intermediate: int
    n_layer: int
    d_embed: int
    patch_size: int
    n_features: int
    vocab_size: int
    pool: str = 'cls'
    activation: str = 'silu'
    dropout_prob: float = 0.1
    padding_idx: int = 0
    append_cls: bool = True
    norm_epsilon: float = 0.00001
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True