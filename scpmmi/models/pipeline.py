import math
from functools import partial
import json
import os
import copy
from typing import Optional, Tuple, Dict, Union

from tqdm import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
from torch import Tensor
from mamba_ssm.utils.generation import GenerationMixin

from .config import EncoderConfig, DecoderConfig
from .encoder import MixerEncoder
from .decoder import MixerDecoder
from .header import ClsDecoder, ModalityDecoder
from ..utils.hf import load_config_hf, load_state_dict_hf
from .scPMMI import BatchLabelEncoder, MVCDecoder



class ProteinPrediction(nn.Module, GenerationMixin):
    def __init__(self, 
        config, 
        nlayers_cls: int = 3,
        n_cls: int = 1,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        device=None, 
        dtype=None
    ) -> None:
        super().__init__()
        self.config = config
        self.do_dab = do_dab
        self.use_batch_labels = use_batch_labels
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config_mod1 = EncoderConfig(**config["encoder_mod1"])
        self.config_mod2 = EncoderConfig(**config["encoder_mod2"])
        
        self.encoder_mod1 = MixerEncoder(config=EncoderConfig(**config["encoder_mod1"]),  **factory_kwargs)
        self.encoder_mod2 = MixerEncoder(config=EncoderConfig(**config["encoder_mod2"]),  **factory_kwargs)

        self.decoder_mod1 = MixerDecoder(config=DecoderConfig(**config["decoder_mod1"]), **factory_kwargs)
        self.decoder_mod2 = MixerDecoder(config=DecoderConfig(**config["decoder_mod2"]), **factory_kwargs)

        self.decoder_mod1t2 = ModalityDecoder(config=DecoderConfig(**config["decoder_mod2"]), activation='silu', **factory_kwargs)

        # Batch Encoder
        if use_batch_labels and num_batch_labels is not None:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, self.config_mod1.d_embed)


        if do_mvc:
            self.mvc_decoder_mod1t2 = MVCDecoder(
                self.config_mod2.d_embed,
                self.config_mod2.n_features,
                nlayers=2,
                use_batch_labels=use_batch_labels
            )
            self.mvc_decoder_mod2t2 = MVCDecoder(
                self.config_mod2.d_embed,
                self.config_mod2.n_features,
                nlayers=2,
                use_batch_labels=use_batch_labels
            )

        self.cls_decoder_mod1 = ClsDecoder(self.config_mod1.d_embed, n_cls, nlayers=nlayers_cls)
        self.cls_decoder_mod2 = ClsDecoder(self.config_mod1.d_embed, n_cls, nlayers=nlayers_cls)


    def forward(
        self,
        input_ids_mod1, input_values_mod1,
        input_ids_mod2, input_values_mod2, 
        batch_labels: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        include_hidden_states=True, inference_params=None, **mixer_kwargs
    ) -> Dict[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output

        Returns:
            dict of output Tensors.
        """
        output = {}
        emb_mod1, hidden_states_mod1 = self.encoder_mod1(input_ids_mod1, input_values_mod1, include_hidden_states=include_hidden_states, inference_params=inference_params, **mixer_kwargs)
        emb_mod2, hidden_states_mod2 = self.encoder_mod2(input_ids_mod2, input_values_mod2,  include_hidden_states=include_hidden_states,inference_params=inference_params, **mixer_kwargs)
        output['cell_emb_mod1'] = emb_mod1
        output['cell_emb_mod2'] = emb_mod2
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        pred_mod1 = self.decoder_mod1(hidden_states_mod1, inference_params=None, **mixer_kwargs)
        pred_mod2 = self.decoder_mod2(hidden_states_mod2, inference_params=None, **mixer_kwargs)
        pred_mod1 = pred_mod1.reshape(pred_mod1.size(0), -1)[:, :self.config['decoder_mod1']['n_features']]
        pred_mod2 = pred_mod2.reshape(pred_mod2.size(0), -1)[:, :self.config['decoder_mod2']['n_features']]
        output['pred_mod1'] = pred_mod1
        output['pred_mod2'] = pred_mod2
        output['pred_mod1t2'] = self.decoder_mod1t2(hidden_states_mod1)

        if CLS:
            output["cls_output_mod1"] = self.cls_decoder_mod1(emb_mod1)  # (batch, n_cls)
            output["cls_output_mod2"] = self.cls_decoder_mod2(emb_mod2)  # (batch, n_cls)
        if MVC:
            output["mvc_output_mod1t2"] = self.mvc_decoder_mod1t2(
                emb_mod1
                if not self.use_batch_labels
                else torch.cat([emb_mod2, batch_emb], dim=1), # type: ignore
            )
            output["mvc_output_mod2t2"] = self.mvc_decoder_mod2t2(
                emb_mod2
                if not self.use_batch_labels
                else torch.cat([emb_mod1, batch_emb], dim=1), # type: ignore
            )

        return output
        

    def generate_protein(
        self,
        input_ids_mod1, input_values_mod1,
        batch_labels: Optional[Tensor] = None,
        MVC: bool = False,
        include_hidden_states=True, inference_params=None, **mixer_kwargs
    ) -> Dict[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        output = {}
        emb_mod1, hidden_states_mod1 = self.encoder_mod1(input_ids_mod1, input_values_mod1, include_hidden_states=include_hidden_states, inference_params=inference_params, **mixer_kwargs)
        output['cell_emb_mod1'] = emb_mod1
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        pred_mod1 = self.decoder_mod1(hidden_states_mod1, inference_params=None, **mixer_kwargs)
        pred_mod1 = pred_mod1.reshape(pred_mod1.size(0), -1)[:, :self.config['decoder_mod1']['n_features']]
        output['pred_mod1'] = pred_mod1
        output['pred_mod1t2'] = self.decoder_mod1t2(hidden_states_mod1)

        if MVC:
            output["mvc_output_mod2"] = self.mvc_decoder_mod1t2(
                emb_mod1
                if not self.use_batch_labels
                else torch.cat([emb_mod1, batch_emb], dim=1), # type: ignore
            )

        return output


    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        model = cls(config_data, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))
        
        config = {
            "encoder_mod1": self.encoder_mod1.config.__dict__,
            "encoder_mod2": self.encoder_mod2.config.__dict__,
            "decoder_mod1": self.decoder_mod1.config.__dict__,
            "decoder_mod2": self.decoder_mod2.config.__dict__,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=4)


