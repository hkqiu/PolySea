"""CSV inspection: detect numeric features vs SMILES columns for property modeling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SMILES_LIKE = re.compile(r"[\[\]\*]|C[lr]|Br?|[Nn]=?|=[Nn]|%?\d+")


def _try_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except ImportError:
        return None


def _fraction_valid_smiles(series: pd.Series, sample: int = 40) -> float:
    Chem = _try_rdkit()
    if Chem is None:
        return 0.0
    vals = series.dropna().astype(str).head(sample)
    if len(vals) == 0:
        return 0.0
    ok = 0
    for v in vals:
        m = Chem.MolFromSmiles(v.strip())
        if m is not None:
            ok += 1
    return ok / len(vals)


def _is_likely_smiles_column(name: str, series: pd.Series) -> bool:
    name_l = name.lower()
    if "smiles" in name_l or "selfies" in name_l or "bigsmiles" in name_l:
        return _fraction_valid_smiles(series) >= 0.5
    if _fraction_valid_smiles(series) >= 0.85:
        return True
    # Heuristic: many strings with chemistry-like tokens
    vals = series.dropna().astype(str).head(30)
    if len(vals) == 0:
        return False
    score = sum(1 for v in vals if _SMILES_LIKE.search(v) and 5 <= len(v) <= 500)
    return score / len(vals) >= 0.6 and _fraction_valid_smiles(series) >= 0.4


def _numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, int(0.5 * len(df))):
            # At least half non-null after coercion or 3+ valid
            ratio = s.notna().sum() / max(len(df), 1)
            if ratio >= 0.4:
                cols.append(c)
    return cols


def analyze_csv(
    path: str | Path,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Return structured hints for the LLM and training pipeline."""
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    smiles_cols = [c for c in df.columns if _is_likely_smiles_column(c, df[c])]
    exclude_for_numeric = set(smiles_cols)

    numeric_cols = _numeric_columns(df, exclude_for_numeric)

    # Target guess
    if target_column and target_column not in df.columns:
        raise ValueError(f"目标列不存在: {target_column}")
    if target_column is None:
        # Prefer last numeric column not in smiles
        candidates = [c for c in df.columns if c in numeric_cols or pd.api.types.is_numeric_dtype(df[c])]
        if not candidates:
            raise ValueError("无法推断目标列，请显式指定 target_column。")
        target_column = candidates[-1]

    y = df[target_column]
    y_num = pd.to_numeric(y, errors="coerce")
    is_regression = y_num.notna().sum() >= max(5, int(0.7 * len(df)))
    if is_regression:
        task_type = "regression"
        n_classes = None
    else:
        task_type = "classification"
        le_vals = y.dropna().astype(str).unique()
        n_classes = len(le_vals)

    feature_numeric = [c for c in numeric_cols if c != target_column]

    has_clear_tabular_features = len(feature_numeric) >= 1

    only_smiles_as_input = (
        len(smiles_cols) >= 1
        and not has_clear_tabular_features
        and target_column not in smiles_cols
    )

    recommended_mode = "tabular"
    if only_smiles_as_input:
        recommended_mode = "rdkit_descriptors"
    elif has_clear_tabular_features and len(smiles_cols) >= 1:
        recommended_mode = "tabular_plus_smiles_fp"
    elif has_clear_tabular_features:
        recommended_mode = "tabular"

    return {
        "path": str(path.resolve()),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "detected_smiles_columns": smiles_cols,
        "numeric_feature_columns": feature_numeric,
        "target_column": target_column,
        "task_type": task_type,
        "n_classes": n_classes,
        "has_clear_tabular_features": has_clear_tabular_features,
        "only_smiles_and_target": only_smiles_as_input,
        "recommended_mode": recommended_mode,
        "summary": (
            f"数据 {len(df)} 行；目标 `{target_column}` 为{task_type}。"
            f" 数值特征列: {feature_numeric or '无'}；"
            f" SMILES 列: {smiles_cols or '未识别'}。"
            f" 建议建模模式: {recommended_mode}。"
        ),
    }


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df
