"""Fine-tune PolyTAO (hkqiu/PolymerGenerationPretrainedModel) — T5 conditional generation."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer, get_cosine_schedule_with_warmup

from ...training_status import TrainingCancelled, polytao_epoch
from ... import training_status as ts_mod
from .dataset import PolyTAODataset

CUSTOM_TOKENS = [
    "Cl",
    "Br",
    "Se",
    "se",
    "Na",
    "Ge",
    "Si",
    "Te",
    "Fe",
    "Pb",
    "Sn",
    "Ca",
    "Co",
    "Ni",
    "Zn",
    "As",
    "Cd",
]


def _train_loop(
    model,
    tokenizer,
    train_dataloader,
    validation_dataloader,
    epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    epsilon: float,
    device: torch.device,
    on_epoch_end: Callable[[int, int, float, float], None] | None = None,
) -> list[dict[str, Any]]:
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    training_stats: list[dict[str, Any]] = []
    sample_every = max(50, len(train_dataloader) // 5)

    for epoch_i in range(epochs):
        if ts_mod.is_cancel_requested():
            ts_mod.finish_cancelled("PolyTAO 微调已由用户中止")
            raise TrainingCancelled()
        t0 = time.time()
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(
                b_input_ids,
                labels=b_labels,
                attention_mask=b_masks,
                output_attentions=False,
            )
            loss = outputs.loss
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_dataloader), 1)

        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_masks = batch[1].to(device)
                b_labels = batch[2].to(device)
                outputs = model(
                    b_input_ids,
                    attention_mask=b_masks,
                    labels=b_labels,
                    output_attentions=False,
                )
                loss = outputs.loss
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss = loss.mean()
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / max(len(validation_dataloader), 1)
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "train_loss": float(avg_train_loss),
                "valid_loss": float(avg_val_loss),
                "time_s": time.time() - t0,
            }
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch_i + 1, epochs, float(avg_train_loss), float(avg_val_loss))

    return training_stats


def finetune_polytao(
    csv_path: str | Path,
    output_dir: str | Path,
    base_model_id: str = "hkqiu/PolymerGenerationPretrainedModel",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    epsilon: float = 1e-8,
    max_length_prompt: int = 128,
    max_length_answer: int = 256,
    val_fraction: float = 0.1,
) -> dict[str, Any]:
    """
    CSV must contain columns `prompt` and `target` (polymer SMILES), per user spec.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "prompt" not in df.columns or "target" not in df.columns:
        raise ValueError("微调数据需包含列: prompt, target")
    df = df.dropna(subset=["prompt", "target"])

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        pad_token="[PAD]",
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    tokenizer.add_tokens(CUSTOM_TOKENS)

    configuration = T5Config.from_pretrained(base_model_id, output_hidden_states=False)
    model = T5ForConditionalGeneration.from_pretrained(base_model_id, config=configuration)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    n = len(df)
    idx = np.random.RandomState(42).permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = set(idx[:n_val])
    train_idx = [i for i in range(n) if i not in val_idx]

    prompt_train = [str(x) for x in df.iloc[train_idx]["prompt"].tolist()]
    target_train = df.iloc[train_idx]["target"].tolist()
    prompt_val = [str(x) for x in df.iloc[list(val_idx)]["prompt"].tolist()]
    target_val = df.iloc[list(val_idx)]["target"].tolist()

    train_ds = PolyTAODataset(
        prompt_train,
        target_train,
        tokenizer,
        max_length_prompt=max_length_prompt,
        max_length_answer=max_length_answer,
    )
    val_ds = PolyTAODataset(
        prompt_val,
        target_val,
        tokenizer,
        max_length_prompt=max_length_prompt,
        max_length_answer=max_length_answer,
    )

    train_loader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        sampler=SequentialSampler(val_ds),
        batch_size=batch_size,
    )

    stats = _train_loop(
        model,
        tokenizer,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        epsilon=epsilon,
        device=device,
        on_epoch_end=polytao_epoch,
    )

    to_save = model.module if hasattr(model, "module") else model
    to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "output_dir": str(output_dir.resolve()),
        "base_model": base_model_id,
        "epochs": epochs,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "training_stats": stats,
        "device": str(device),
    }
