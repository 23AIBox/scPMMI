import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from typing_extensions import Self


class Vocab:
    def __init__(
        self,
        token_list_or_dict: Union[List[str], Dict[str, int]],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        pad_token: str = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.

        Args:
            token_list_or_dict (List[str] or dict): List of tokens or
                dict object such as {'chromatin_names': chromatin_idx}.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            pad_token (str): Padding token. Will be placed at index 0.
        """
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._specials = list(dict.fromkeys([pad_token] + (specials or [])))

        if isinstance(token_list_or_dict, dict):
            self.token2idx = dict(token_list_or_dict)
            self.idx2token = {i: t for t, i in self.token2idx.items()}
            # the idx of pad token must be 0
            if pad_token not in self.token2idx:
                if 0 in self.idx2token:
                    token_zero = self.idx2token[0]
                    self.token2idx[pad_token] = 0
                    self.idx2token[0] = pad_token
                    self.append_token(token_zero)
                else:
                    self.idx2token[0] = pad_token
                    self.token2idx[pad_token] = 0
            else:
                if self.token2idx[pad_token] != 0:
                    token_zero = self.idx2token[0]
                    pad_token_id = self.token2idx[pad_token]
                    self.token2idx[token_zero] = pad_token_id
                    self.idx2token[pad_token_id] = token_zero
                    self.token2idx[pad_token] = 0
                    self.idx2token[0] = pad_token

            if specials is not None:
                raise ValueError("`specials` must be None when initializing from a dict.")

        elif isinstance(token_list_or_dict, list):
            if specials is None:
                specials = []

            if pad_token not in specials:
                specials = [pad_token] + specials

            if special_first:
                all_tokens = specials + token_list_or_dict
            else:
                all_tokens = token_list_or_dict + specials

            seen = set()
            unique_tokens = []
            for tok in all_tokens:
                if tok not in seen:
                    seen.add(tok)
                    unique_tokens.append(tok)

            for idx, token in enumerate(unique_tokens):
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        else:
            raise ValueError("token_list_or_dict must be a list of tokens or a dict object.")


        if pad_token not in self.token2idx:
            raise ValueError(f"Pad token `{pad_token}` not found in vocabulary.")
        if self.token2idx[pad_token] != 0:
            raise ValueError(f"Pad token `{pad_token}` must have index 0.")

        self._pad_token = pad_token
        self._default_token = pad_token

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                token2idx = pickle.load(f)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
        else:
            raise ValueError("Only .pkl and .json are supported.")
        return cls(token2idx)

    def append_token(self, token: str) -> None:
        if token not in self.token2idx:
            idx = max(self.token2idx.values(), default=-1) + 1
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def pad_index(self) -> int:
        return self.token2idx[self._pad_token]

    def get_token_list(self) -> List[str]:
        """
        Return only non-special tokens.
        """
        return [tok for tok in self.token2idx.keys() if tok not in self._specials]

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.token2idx, f, indent=2)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx

    def __getitem__(self, token: str) -> int:
        if token in self.token2idx:
            return self.token2idx[token]
        return self.token2idx[self._default_token]

    def __len__(self) -> int:
        return len(self.token2idx)

    def idx_to_token(self, idx: int) -> str:
        return self.idx2token[idx]

    def __call__(self, tokens: Union[List[str], np.ndarray]) -> List[int]:
        """
        Convert a list of tokens to their corresponding indices.
        Unknown tokens will return the default_token index (if set).
        
        Args:
            tokens (List[str] or np.ndarray): List of token names.
            
        Returns:
            List[int]: List of indices.
        """
        return [self[token] for token in tokens]
