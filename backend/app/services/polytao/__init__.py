def generate_smiles(*args, **kwargs):
    from .generate import generate_smiles as implementation
    return implementation(*args, **kwargs)


def finetune_polytao(*args, **kwargs):
    from .train import finetune_polytao as implementation
    return implementation(*args, **kwargs)

__all__ = ["finetune_polytao", "generate_smiles"]
