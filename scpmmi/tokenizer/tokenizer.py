import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pickle
import json
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any
from typing_extensions import Self
from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab


def tokenize(
        data: np.ndarray,
        token_ids: np.ndarray,
        cls_id: int,
        return_pt: bool = True,
        append_cls: bool = False,
        include_zero_gene: bool = True,
    ) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        token_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of token_ids and counts,
            default to True.
        append_cls (bool): Whether to append the <cls> token.
        include_zero_gene (bool): Whether to contain zero gene expressions.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data.shape[1] != len(token_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of token_ids ({len(token_ids)})."
        )
    
    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        if include_zero_gene:
            values = row
            genes = token_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = token_ids[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
        tokenized_data.append((genes, values))
    return tokenized_data


def pad(
    batch: List[Tuple],
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    max_len: int = 0,
    cls_appended: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (token_ids, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    token_ids_list = []
    values_list = []

    for i in range(len(batch)):
        token_ids, values = batch[i]

        # if the length of batch is greater than the max length,
        # randomly remove some samples
        # if len(token_ids) > max_len:
        #     # sample max_len genes
        #     if not cls_appended:
        #         idx = np.random.choice(len(token_ids), max_len, replace=False)
        #     else:
        #         idx = np.random.choice(len(token_ids) - 1, max_len - 1, replace=False)
        #         idx = idx + 1
        #         idx = np.insert(idx, 0, 0)
        #     token_ids = token_ids[idx]
        #     values = values[idx]

        # elif len(token_ids) < max_len:
        #     token_ids = torch.cat(
        #         [
        #             token_ids,
        #             torch.full(
        #                 (max_len - len(token_ids),), pad_id, dtype=token_ids.dtype
        #             ),
        #         ]
        #     )
        #     values = torch.cat(
        #         [
        #             values,
        #             torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
        #         ]
        #     )
        token_ids_list.append(token_ids)
        values_list.append(values)
        

    batch_padded = {
        "token_ids": torch.stack(token_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }

    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    token_ids: np.ndarray,
    vocab: Vocab,
    max_len: int = 0,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    append_cls: bool = False,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    
    tokenized_data = tokenize(
        data,
        token_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
    )

    batch_padded = pad(
        tokenized_data,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        max_len=max_len
    )
    return batch_padded


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: float = -1.0,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly mask elements in a batch of data while keeping padding unchanged.

    Args:
        values (Union[torch.Tensor, np.ndarray]):
            Input batch of data with shape (batch_size, n_features).
        mask_ratio (float): Fraction of non-padding elements to mask. Default: 0.15.
        mask_value (float): Value to assign to masked elements. Default: -1.0.
        pad_value (float): Value representing padding, which will remain unchanged. Default: 0.0.

    Returns:
        masked_values (torch.Tensor): Tensor with randomly masked elements.
        pad_mask (torch.Tensor): Boolean tensor indicating padding positions.
    """
    # Convert to NumPy array for indexing, clone if tensor
    if isinstance(values, torch.Tensor):
        values_np = values.clone().detach().cpu().numpy()
    else:
        values_np = values.copy()

    batch_size = values_np.shape[0]
    masked_values = values_np.copy()

    for i in range(batch_size):
        row = masked_values[i]
        # Indices of non-padding elements
        non_pad_idx = np.nonzero(row - pad_value)[0]
        # non_pad_idx = np.flatnonzero(row != pad_value)
        n_mask = max(1, int(len(non_pad_idx) * mask_ratio))  # at least 1 element
        if n_mask > 0:
            mask_idx = np.random.choice(non_pad_idx, n_mask, replace=False)
            row[mask_idx] = mask_value

    mask = (masked_values == mask_value)

    # # Convert back to tensor
    # masked_values = torch.tensor(masked_values, dtype=torch.float32)
    # pad_mask = torch.tensor(mask, dtype=torch.bool)

    return masked_values, mask


def random_mask_atac_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: float = -1.0,
    pad_value: float = 0.0,
    zero_mask_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly mask elements in a batch of data while keeping most padding unchanged,
    but allowing a small fraction of zero (padding) elements to also be masked.

    Args:
        values (Union[torch.Tensor, np.ndarray]):
            Input batch of data with shape (batch_size, n_features).
        mask_ratio (float): Fraction of non-padding (non-zero) elements to mask. Default: 0.15.
        mask_value (float): Value to assign to masked elements. Default: -1.0.
        pad_value (float): Value representing padding. Default: 0.0.
        zero_mask_ratio (float): Fraction of zero (padding) elements to also mask,
            relative to number of non-zero masked elements. Default: 0.1.

    Returns:
        masked_values (np.ndarray): Array with randomly masked elements.
        mask (np.ndarray): Boolean mask indicating which elements are masked.
    """
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    else:
        values_np = values.copy()

    batch_size, _ = values_np.shape
    masked_values = values_np.copy()

    for i in range(batch_size):
        row = masked_values[i]

        non_zero_idx = np.flatnonzero(row != pad_value)
        zero_idx = np.flatnonzero(row == pad_value)

        n_mask_nonzero = max(1, int(len(non_zero_idx) * mask_ratio))
        n_mask_zero = int(n_mask_nonzero * zero_mask_ratio)

        mask_idx_nonzero = (
            np.random.choice(non_zero_idx, n_mask_nonzero, replace=False)
            if len(non_zero_idx) > 0
            else np.array([], dtype=int)
        )
        mask_idx_zero = (
            np.random.choice(zero_idx, n_mask_zero, replace=False)
            if len(zero_idx) > 0
            else np.array([], dtype=int)
        )

        mask_idx = np.concatenate([mask_idx_nonzero, mask_idx_zero])
        if len(mask_idx) > 0:
            row[mask_idx] = mask_value

    mask = (masked_values == mask_value)
    return masked_values, mask