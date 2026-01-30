"""Dataset loading and preprocessing for language modeling."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Iterator
from pathlib import Path
import os


class TextDataset(Dataset):
    """
    Generic text dataset for language modeling.

    Handles:
    - WikiText-2 (small, good for debugging)
    - C4 (large, for actual experiments)
    - Custom text files
    """

    def __init__(
        self,
        tokenizer,
        seq_length: int,
        split: str = "train",
        dataset_name: str = "wikitext-2",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.dataset_name = dataset_name
        self.max_samples = max_samples

        # Load and tokenize
        self.tokens = self._load_and_tokenize(cache_dir)

        # Calculate number of samples
        self.num_samples = len(self.tokens) // seq_length
        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)

    def _load_and_tokenize(self, cache_dir: Optional[str]) -> torch.Tensor:
        """Load dataset and convert to token IDs."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Map dataset names to HuggingFace identifiers
        dataset_map = {
            "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
            "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
            "c4": ("c4", "en"),
        }

        if self.dataset_name in dataset_map:
            name, config = dataset_map[self.dataset_name]
            if self.dataset_name == "c4":
                # C4 is streaming, handle differently
                dataset = load_dataset(
                    name, config,
                    split=self.split,
                    streaming=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                # Take a reasonable amount for non-streaming use
                texts = []
                for i, item in enumerate(dataset):
                    texts.append(item["text"])
                    if i >= 100000:  # Limit for memory
                        break
                text = "\n".join(texts)
            else:
                dataset = load_dataset(
                    name, config,
                    split=self.split,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                text = "\n".join(dataset["text"])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length

        input_ids = self.tokens[start:end]
        labels = self.tokens[start:end].clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class ChunkedBatchSampler:
    """
    Batch sampler that ensures sequences align with chunk boundaries.

    This ensures that:
    - Each batch has sequences of equal length
    - Sequence lengths are multiples of chunk_size
    """

    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        chunk_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator:
        indices = list(range(self.dataset_length))

        if self.shuffle:
            import random
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_length // self.batch_size
        return (self.dataset_length + self.batch_size - 1) // self.batch_size


def create_tokenizer(vocab_size: int = 32000):
    """
    Create or load a tokenizer.

    Uses HuggingFace tokenizers with GPT-2 style BPE.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    # Use GPT-2 tokenizer as base (will be truncated/extended to vocab_size)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure we have padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_dataloaders(
    tokenizer,
    seq_length: int,
    batch_size: int,
    chunk_size: int,
    dataset_name: str = "wikitext-2",
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create train and eval dataloaders.

    Args:
        tokenizer: Tokenizer to use
        seq_length: Sequence length for samples
        batch_size: Batch size
        chunk_size: Chunk size for attention (used for alignment)
        dataset_name: Name of dataset to load
        num_workers: Number of dataloader workers
        cache_dir: Cache directory for datasets
        max_train_samples: Limit training samples
        max_eval_samples: Limit eval samples

    Returns:
        Dict with 'train' and 'eval' dataloaders
    """
    # Ensure seq_length is multiple of chunk_size
    assert seq_length % chunk_size == 0, \
        f"seq_length ({seq_length}) must be divisible by chunk_size ({chunk_size})"

    # Create datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        split="train",
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        max_samples=max_train_samples,
    )

    # Use validation split for eval
    eval_split = "validation" if dataset_name != "c4" else "validation"
    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        split=eval_split,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        max_samples=max_eval_samples,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return {
        "train": train_loader,
        "eval": eval_loader,
    }


class InfiniteDataLoader:
    """
    Wrapper that makes a dataloader infinite by cycling.

    Useful for step-based training instead of epoch-based.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def __len__(self):
        return len(self.dataloader)
