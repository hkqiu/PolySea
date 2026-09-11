"""PolyTAO (T5) dataset — prompt/target CSV formatting."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PolyTAODataset(Dataset):
    """prompt (condition) -> target (polymer SMILES), T5 seq2seq."""

    def __init__(
        self,
        prompt_list: list[str],
        answer_list: list[str],
        tokenizer,
        max_length_prompt: int = 1024,
        max_length_answer: int = 256,
    ):
        self.tokenizer = tokenizer
        tokenizer.padding_side = "right"

        encodings_list1 = [
            tokenizer("<s>" + str(txt), truncation=True, max_length=max_length_prompt, padding="max_length")
            for txt in prompt_list
        ]
        self.input_ids = [torch.tensor(e["input_ids"]) for e in encodings_list1]
        self.attn_masks = [torch.tensor(e["attention_mask"]) for e in encodings_list1]

        encodings_list2 = [
            tokenizer("<s>" + str(ans), truncation=True, max_length=max_length_answer, padding="max_length")
            for ans in answer_list
        ]
        self.answer_ids = [torch.tensor(e["input_ids"]) for e in encodings_list2]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.attn_masks[idx], self.answer_ids[idx]
