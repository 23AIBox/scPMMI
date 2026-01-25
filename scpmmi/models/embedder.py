import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class scEmbeddings(nn.Module):
    def __init__(
            self, 
            d_model,
            n_embs,
            n_features, 
            patch_size,
            dropout_prob=0.1,
            padding_idx: int = 0,
            append_cls: bool = True,
            device=None,
            dtype=None
        ) -> None:
        super().__init__()
        num_patches = math.ceil(
            n_features / patch_size
        )
        
        self.pad_size = num_patches * patch_size - n_features
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.append_cls = append_cls
        self.padding_idx = padding_idx

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # self.T_token = nn.Parameter(torch.randn(1, 1, d_model))
        # nn.init.trunc_normal_(self.T_token, std=0.02)
        # self.S_token = nn.Parameter(torch.randn(1, 1, d_model))
        # nn.init.trunc_normal_(self.S_token, std=0.02)

        self.embedding = nn.Embedding(
            n_embs, d_model, padding_idx=padding_idx, device=device, dtype=dtype
        )

        self.patch_position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, d_model)
        )
        nn.init.trunc_normal_(self.patch_position_embeddings, std=0.02)

        
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, token_ids:Tensor, values:Tensor) -> Tensor:
        dtype = self.embedding.weight.dtype
        values = values.to(dtype=dtype)
        # total_counts = values[:, 0]
        # # scaled_counts = values[:, 1]
        # values = values[:, 1:]
        batch_size, _ = token_ids.shape[0], token_ids.shape[1]

        token_ids = F.pad(token_ids, (self.padding_idx, self.pad_size)).reshape(
            token_ids.shape[0], self.num_patches, self.patch_size
        )
        values = F.pad(values, (0, self.pad_size)).reshape(
            values.shape[0], self.num_patches, self.patch_size
        )
        values = self.dropout(values)

        token_embeddings = self.embedding(token_ids)
        values = values.unsqueeze(-1)
        weighted_embeddings = token_embeddings * values
        patch_embs = weighted_embeddings.sum(dim=2)
        # patch_embs = patch_embs + self.patch_position_embeddings.to(values)
        
        # T_tokens = self.T_token.expand(batch_size, -1, -1).to(values) * total_counts[:, None, None]
        # S_tokens = self.S_token.expand(batch_size, -1, -1).to(values) * scaled_counts[:, None, None]
        # patch_embs = torch.cat([T_tokens, patch_embs], dim=1)

        if self.append_cls: 
        # add the [CLS] token to the embedded patch tokens
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(values)
            patch_embs = torch.cat([cls_tokens, patch_embs], dim=1)

        return patch_embs