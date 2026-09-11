"""Generate polymer SMILES from a property prompt (template-free inverse design)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration


def generate_smiles(
    model_path: str | Path,
    prompt: str,
    num_return_sequences: int = 5,
    max_length: int = 300,
    top_k: int = 100,
    top_p: float = 0.999,
    device: str | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)
    tok = AutoTokenizer.from_pretrained(str(model_path), pad_token="[PAD]", padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    configuration = T5Config.from_pretrained(str(model_path), output_hidden_states=False)
    model = T5ForConditionalGeneration.from_pretrained(str(model_path), config=configuration)
    model.eval()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    enc = tok.encode(str(prompt))
    generated = torch.tensor(enc).unsqueeze(0).to(dev)

    with torch.no_grad():
        sample_outputs = model.generate(
            generated,
            do_sample=True,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

    texts = []
    for out in sample_outputs:
        texts.append(tok.decode(out, skip_special_tokens=True))

    return {"prompt": prompt, "sequences": texts, "model_path": str(model_path.resolve())}
