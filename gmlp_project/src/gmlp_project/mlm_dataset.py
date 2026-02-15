#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for masked language modeling (MLM) datasets.
- Tokenize raw text and chunk into fixed-length sequences.
- Provide an MLM collator that applies 15% masking with the Hugging Face recipe
  (80% [MASK], 10% random token, 10% original).
"""

from typing import Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(name: str, *, local_files_only: bool = False) -> PreTrainedTokenizerBase:
    """Load a tokenizer with masking tokens."""
    tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)
    if tok.mask_token is None:
        raise ValueError(f"Tokenizer {name} has no [MASK] token configured.")
    if tok.pad_token is None:
        # Most decoder-only tokenizers need an explicit pad token for batching.
        tok.pad_token = tok.eos_token
    return tok


def tokenize_and_chunk(
    text_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> Dataset:
    """
    Tokenize text and pack into fixed-length sequences.

    Args:
        text_dataset: Hugging Face Dataset with a "text" column.
        tokenizer: tokenizer that provides mask/pad tokens.
        seq_len: sequence length after chunking.

    Returns:
        Dataset with columns:
            - input_ids: int64 tensor (seq_len,)
            - attention_mask: int64 tensor (seq_len,)
    """

    def _tokenize(batch):
        return tokenizer(batch["text"])

    tokenized = text_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=text_dataset.column_names,
    )

    def _group(batch):
        # Flatten then split into equal-length chunks
        concatenated = sum(batch["input_ids"], [])
        total_length = (len(concatenated) // seq_len) * seq_len
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}
        ids = [concatenated[i : i + seq_len] for i in range(0, total_length, seq_len)]
        masks = [[1] * seq_len for _ in range(len(ids))]
        return {"input_ids": ids, "attention_mask": masks}

    chunked = tokenized.map(_group, batched=True)
    chunked.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return chunked


def build_wikitext_dataset(
    tokenizer_name: str = "bert-base-uncased",
    seq_len: int = 128,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    cache_dir: Optional[str] = None,
    *,
    dataset_dir: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    offline: bool = False,
) -> Tuple[Dataset, Dataset, PreTrainedTokenizerBase]:
    """
    Convenience loader for WikiText.

    Args:
        dataset_dir: ローカルに展開済みの HF データセットディレクトリを指定する場合に使う。
        tokenizer_path: ローカルに展開済みのトークナイザーを指定する場合に使う。
        offline: True の場合は local_files_only でロードしネットアクセスしない。

    Returns:
        train_ds, val_ds, tokenizer
    """
    tok_source = tokenizer_path or tokenizer_name
    tokenizer = load_tokenizer(tok_source, local_files_only=offline)

    ds_source = dataset_dir or dataset_name
    raw = load_dataset(
        ds_source,
        dataset_config,
        cache_dir=cache_dir,
        local_files_only=offline,
    )
    if "validation" in raw:
        val_split = "validation"
    elif "test" in raw:
        val_split = "test"
    else:
        raise ValueError(f"Dataset {dataset_name}/{dataset_config} has no validation/test split.")
    train_ds = tokenize_and_chunk(raw["train"], tokenizer, seq_len)
    val_ds = tokenize_and_chunk(raw[val_split], tokenizer, seq_len)
    return train_ds, val_ds, tokenizer


def mask_tokens(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    mlm_probability: float = 0.15,
    generator: Optional[torch.Generator] = None,
):
    """
    Apply BERT-style MLM masking.
    Returns masked_input_ids, labels, mask_bool.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id.")

    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=input_ids.device)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
        for val in labels
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=input_ids.device)
    probability_matrix.masked_fill_(special_tokens_mask, 0.0)

    if tokenizer.pad_token_id is not None:
        probability_matrix.masked_fill_(labels == tokenizer.pad_token_id, 0.0)

    mask = torch.bernoulli(probability_matrix, generator=generator).to(dtype=torch.bool)
    labels[~mask] = -100

    # 80% replace with [MASK]
    replace_prob = torch.full(labels.shape, 0.8, device=input_ids.device)
    indices_replaced = torch.bernoulli(replace_prob, generator=generator).to(dtype=torch.bool) & mask
    input_ids = input_ids.clone()
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% replace with random token
    random_prob = torch.full(labels.shape, 0.5, device=input_ids.device)
    indices_random = (
        torch.bernoulli(random_prob, generator=generator).to(dtype=torch.bool)
        & mask
        & ~indices_replaced
    )
    random_tokens = torch.randint(
        low=0,
        high=len(tokenizer),
        size=labels.shape,
        device=input_ids.device,
        generator=generator,
    )
    input_ids[indices_random] = random_tokens[indices_random]

    return input_ids, labels, mask


class MLMDataCollator:
    """Minimal collator for MLM training/eval."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm_probability: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch]).long()
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).long()
        masked, labels, _ = mask_tokens(
            input_ids, self.tokenizer, self.mlm_probability, generator=self.generator
        )
        return {
            "input_ids": masked,
            "attention_mask": attention_mask,
            "labels": labels,
        }
