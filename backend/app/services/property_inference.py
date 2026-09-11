"""Load saved property models and run single-sample inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .automl import META_FILENAME, PIPELINE_FILENAME, PROPERTY_BUNDLE_VERSION, _rdkit_morgan

LATEST_REL = Path("property_models") / "LATEST"


def write_latest_property_artifact(outputs_root: Path, artifact_dir: Path) -> None:
    root = outputs_root.resolve()
    latest = root / LATEST_REL
    latest.parent.mkdir(parents=True, exist_ok=True)
    ad = artifact_dir.resolve()
    if not ad.is_relative_to(root):
        raise ValueError("artifact_dir must be under outputs_root")
    latest.write_text(str(ad), encoding="utf-8")


def read_latest_property_artifact(outputs_root: Path) -> Path | None:
    latest = outputs_root.resolve() / LATEST_REL
    if not latest.is_file():
        return None
    raw = latest.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_dir():
        return None
    return p


def _build_X_prediction(
    meta: dict[str, Any],
    *,
    smiles: str | None,
    tabular_features: dict[str, float] | None,
) -> np.ndarray:
    mode = meta["mode"]
    tcols: list[str] = list(meta.get("tabular_feature_columns") or [])
    sm_col: str | None = meta.get("smiles_column")
    tf = dict(tabular_features or {})

    if mode == "tabular":
        missing = [c for c in tcols if c not in tf]
        if missing:
            raise ValueError(f"tabular 模式缺少数值特征: {missing}")
        row = {c: [float(tf[c])] for c in tcols}
        df = pd.DataFrame(row)
        X = df.values.astype(np.float64)
        return np.nan_to_num(X, nan=0.0)

    if mode == "rdkit_descriptors":
        if not smiles or not str(smiles).strip():
            raise ValueError("rdkit_descriptors 模式需要 polymer SMILES（字符串）")
        if not sm_col:
            raise ValueError("模型元数据缺少 smiles_column")
        return _rdkit_morgan([str(smiles).strip()])

    if mode == "tabular_plus_smiles_fp":
        if not smiles or not str(smiles).strip():
            raise ValueError("需要 polymer SMILES")
        if not sm_col:
            raise ValueError("模型元数据缺少 smiles_column")
        missing = [c for c in tcols if c not in tf]
        if missing:
            raise ValueError(f"缺少数值特征: {missing}")
        Xt = np.array([[float(tf[c]) for c in tcols]], dtype=np.float64)
        Xt = np.nan_to_num(Xt, nan=0.0)
        Xm = _rdkit_morgan([str(smiles).strip()])
        return np.hstack([Xt, Xm]).astype(np.float32)

    raise ValueError(f"当前 bundle 不支持推理模式: {mode}")


def predict_property(
    artifact_dir: Path,
    *,
    smiles: str | None = None,
    tabular_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    ad = artifact_dir.resolve()
    pipe_path = ad / PIPELINE_FILENAME
    meta_path = ad / META_FILENAME
    if not pipe_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"无效的模型目录（缺少 {PIPELINE_FILENAME} 或 {META_FILENAME}）")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("bundle_version", 0)) != PROPERTY_BUNDLE_VERSION:
        raise ValueError("模型 bundle 版本不匹配，请重新训练")

    pipeline = joblib.load(pipe_path)
    X = _build_X_prediction(meta, smiles=smiles, tabular_features=tabular_features)

    preds = pipeline.predict(X)
    task = meta["task"]

    if task == "regression":
        value = float(preds.flat[0])
        return {
            "task": task,
            "prediction": value,
            "target_column": meta["target_column"],
            "mode": meta["mode"],
        }

    if task == "classification":
        classes = meta.get("label_classes") or []
        idx = int(preds.flat[0])
        label = classes[idx] if 0 <= idx < len(classes) else str(idx)
        out: dict[str, Any] = {
            "task": task,
            "prediction_class_index": idx,
            "prediction_label": label,
            "target_column": meta["target_column"],
            "mode": meta["mode"],
        }
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X)[0]
                out["class_probabilities"] = {
                    str(classes[i]): float(proba[i]) for i in range(min(len(classes), len(proba)))
                }
            except Exception:
                pass
        return out

    raise ValueError(f"未知 task: {task}")
