"""Property prediction: Optuna-tuned base models + stacking ensemble."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import optuna
import optuna.trial
import pandas as pd

from .. import training_status as ts
from ..training_status import TrainingCancelled
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
# sklearn 1.6+ 与 joblib 并行时的提示，与建模正确性无关，避免刷屏
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

Mode = Literal["tabular", "rdkit_descriptors", "tabular_plus_smiles_fp", "deepchem_gnn"]

PROPERTY_BUNDLE_VERSION = 1
PIPELINE_FILENAME = "pipeline.joblib"
META_FILENAME = "meta.json"


def _tabular_columns_for_meta(mode: Mode, feat_names: list[str]) -> list[str]:
    if mode == "tabular":
        return list(feat_names)
    if mode == "rdkit_descriptors":
        return []
    if mode == "tabular_plus_smiles_fp":
        return [f for f in feat_names if not str(f).startswith("morgan_")]
    return []


def save_property_bundle(
    artifact_dir: Path,
    pipeline: Any,
    *,
    task: str,
    mode: Mode,
    target_column: str,
    smiles_column: str | None,
    feat_names: list[str],
    label_classes: list[str] | None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / PIPELINE_FILENAME)
    meta = {
        "bundle_version": PROPERTY_BUNDLE_VERSION,
        "task": task,
        "mode": mode,
        "target_column": target_column,
        "smiles_column": smiles_column,
        "tabular_feature_columns": _tabular_columns_for_meta(mode, feat_names),
        "label_classes": label_classes,
    }
    (artifact_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _rdkit_morgan(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
        X[i] = np.array(fp, dtype=np.float32)
    return X


def _build_feature_matrix(
    df: pd.DataFrame,
    target: str,
    mode: Mode,
    smiles_column: str | None,
) -> tuple[np.ndarray, list[str]]:
    y_raw = df[target]
    feature_names: list[str] = []

    numeric_cols = [
        c
        for c in df.columns
        if c != target
        and c != smiles_column
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]
    # refine: drop columns that are all NaN after coercion
    usable_num = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, int(0.3 * len(df))):
            usable_num.append(c)

    if mode == "tabular":
        X = df[usable_num].apply(pd.to_numeric, errors="coerce").values
        feature_names = usable_num
        return np.nan_to_num(X, nan=np.nanmedian(X, axis=0) if X.size else 0), feature_names

    if mode == "rdkit_descriptors":
        if not smiles_column or smiles_column not in df.columns:
            raise ValueError("rdkit_descriptors 模式需要有效的 smiles_column")
        smi = df[smiles_column].astype(str).tolist()
        X = _rdkit_morgan(smi)
        feature_names = [f"morgan_{i}" for i in range(X.shape[1])]
        return X, feature_names

    if mode == "tabular_plus_smiles_fp":
        if not smiles_column or smiles_column not in df.columns:
            raise ValueError("需要 smiles_column")
        smi = df[smiles_column].astype(str).tolist()
        Xm = _rdkit_morgan(smi)
        Xt = df[usable_num].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
        Xt = np.nan_to_num(Xt, nan=np.nanmedian(Xt, axis=0) if Xt.size else 0)
        X = np.hstack([Xt, Xm])
        feature_names = usable_num + [f"morgan_{i}" for i in range(Xm.shape[1])]
        return X.astype(np.float32), feature_names

    if mode == "deepchem_gnn":
        raise NotImplementedError("请在 train_with_deepchem_gnn 中调用")

    raise ValueError(f"未知模式: {mode}")


def train_with_deepchem_gnn(
    df: pd.DataFrame,
    target: str,
    smiles_column: str,
    task_type: str,
    max_epochs: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    """预留：DeepChem 图网络路径依赖版本与 CUDA 环境差异较大，后续版本将固化为可复现实验。"""
    raise RuntimeError(
        "DeepChem GNN 路径在当前开源骨架中尚未启用；请使用 mode=rdkit_descriptors 或 "
        "tabular_plus_smiles_fp。安装 deepchem 后可在 automl 中扩展 train_with_deepchem_gnn。"
    )


def run_automl(
    csv_path: str | Path,
    target_column: str,
    mode: Mode,
    smiles_column: str | None = None,
    n_optuna_trials: int = 12,
    random_state: int = 42,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Train stacked ensemble with internal Optuna tuning."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if target_column not in df.columns:
        raise ValueError(f"缺少目标列 {target_column}")

    analysis_mode = mode
    if analysis_mode != "deepchem_gnn":
        ts.start_property(n_optuna_trials)

    if mode == "deepchem_gnn":
        if not smiles_column:
            raise ValueError("deepchem_gnn 需要 smiles_column")
        task = (
            "regression"
            if pd.to_numeric(df[target_column], errors="coerce").notna().mean() > 0.7
            else "classification"
        )
        return train_with_deepchem_gnn(df, target_column, smiles_column, task)

    if mode == "tabular_plus_smiles_fp" and not smiles_column:
        for c in df.columns:
            if c != target_column and str(c).lower() == "smiles":
                smiles_column = c
                break

    X, feat_names = _build_feature_matrix(df, target_column, analysis_mode, smiles_column)
    y_raw = df[target_column]

    y_num = pd.to_numeric(y_raw, errors="coerce")
    is_regression = y_num.notna().sum() >= max(5, int(0.7 * len(df)))

    if is_regression:
        y = y_num.values
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if len(y) < 10:
            raise ValueError("回归任务有效样本过少")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        def make_stack_rf_hgb_xgb(trial: optuna.Trial) -> StackingRegressor:
            rf_n = trial.suggest_int("rf_n_estimators", 80, 400)
            rf_d = trial.suggest_int("rf_max_depth", 4, 24)
            hgb_lr = trial.suggest_float("hgb_lr", 0.02, 0.2, log=True)
            hgb_depth = trial.suggest_int("hgb_max_depth", 3, 12)
            try:
                import xgboost as xgb  # noqa: PLC0415

                xgb_est = xgb.XGBRegressor(
                    n_estimators=trial.suggest_int("xgb_n", 80, 400),
                    max_depth=trial.suggest_int("xgb_depth", 3, 10),
                    learning_rate=trial.suggest_float("xgb_eta", 0.02, 0.2, log=True),
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
            except Exception:
                xgb_est = HistGradientBoostingRegressor(random_state=random_state)

            estimators = [
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=rf_n,
                        max_depth=rf_d,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        learning_rate=hgb_lr,
                        max_depth=hgb_depth,
                        random_state=random_state,
                    ),
                ),
                ("xgb", xgb_est),
            ]
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=trial.suggest_float("ridge_alpha", 0.1, 10.0, log=True)),
                cv=5,
                n_jobs=-1,
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("stack", stack)])
            return pipe

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))

        def _reg_obj(trial: optuna.Trial) -> float:
            return cross_val_score(
                make_stack_rf_hgb_xgb(trial),
                X_train,
                y_train,
                cv=5,
                scoring="r2",
                n_jobs=1,
            ).mean()

        def _reg_optuna_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            val = trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None
            ts.optuna_trial(trial.number + 1, n_optuna_trials, val)

        def _reg_cancel_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if ts.is_cancel_requested():
                study.stop()

        study.optimize(
            _reg_obj,
            n_trials=n_optuna_trials,
            callbacks=[_reg_optuna_cb, _reg_cancel_cb],
        )

        if ts.is_cancel_requested():
            ts.finish_cancelled("性质预测训练已由用户中止")
            raise TrainingCancelled()

        ts.phase_fit_best()
        if ts.is_cancel_requested():
            ts.finish_cancelled("性质预测训练已由用户中止")
            raise TrainingCancelled()
        best = make_stack_rf_hgb_xgb(study.best_trial)
        best.fit(X_train, y_train)
        pred = best.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

        artifact_saved: str | None = None
        if artifact_dir is not None:
            best_full = make_stack_rf_hgb_xgb(study.best_trial)
            best_full.fit(X, y)
            save_property_bundle(
                artifact_dir,
                best_full,
                task="regression",
                mode=mode,
                target_column=target_column,
                smiles_column=smiles_column,
                feat_names=feat_names,
                label_classes=None,
            )
            artifact_saved = str(artifact_dir.resolve())

        out: dict[str, Any] = {
            "task": "regression",
            "mode": mode,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "feature_names_sample": feat_names[:20],
            "optuna_best_r2_cv": float(study.best_value),
            "test_r2": float(r2),
            "test_rmse": rmse,
            "metric_primary": "r2",
            "best_params": study.best_params,
            "ensemble": "StackingRegressor(RF + HGB + XGB) + Ridge meta",
        }
        if artifact_saved:
            out["artifact_dir"] = artifact_saved
        return out

    # classification
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    u, cnt = np.unique(y, return_counts=True)
    strat = y if (len(u) > 1 and cnt.min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=strat
    )

    def make_stack_clf(trial: optuna.Trial) -> StackingClassifier:
        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 80, 400),
            max_depth=trial.suggest_int("rf_max_depth", 4, 24),
            random_state=random_state,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("hgb_lr", 0.02, 0.2, log=True),
            max_depth=trial.suggest_int("hgb_max_depth", 3, 12),
            random_state=random_state,
        )
        estimators = [("rf", rf), ("hgb", hgb)]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
            n_jobs=-1,
        )
        return Pipeline([("scaler", StandardScaler()), ("stack", stack)])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))

    def objective(trial: optuna.Trial) -> float:
        clf = make_stack_clf(trial)
        return cross_val_score(
            clf,
            X_train,
            y_train,
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
        ).mean()

    def _clf_optuna_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        val = trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None
        ts.optuna_trial(trial.number + 1, n_optuna_trials, val)

    def _clf_cancel_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if ts.is_cancel_requested():
            study.stop()

    study.optimize(
        objective,
        n_trials=n_optuna_trials,
        callbacks=[_clf_optuna_cb, _clf_cancel_cb],
    )

    if ts.is_cancel_requested():
        ts.finish_cancelled("性质预测训练已由用户中止")
        raise TrainingCancelled()

    ts.phase_fit_best()
    if ts.is_cancel_requested():
        ts.finish_cancelled("性质预测训练已由用户中止")
        raise TrainingCancelled()
    best = make_stack_clf(study.best_trial)
    best.fit(X_train, y_train)
    pred = best.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    artifact_saved: str | None = None
    if artifact_dir is not None:
        best_full = make_stack_clf(study.best_trial)
        best_full.fit(X, y)
        save_property_bundle(
            artifact_dir,
            best_full,
            task="classification",
            mode=mode,
            target_column=target_column,
            smiles_column=smiles_column,
            feat_names=feat_names,
            label_classes=le.classes_.tolist(),
        )
        artifact_saved = str(artifact_dir.resolve())

    out = {
        "task": "classification",
        "mode": mode,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "classes": le.classes_.tolist(),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "metric_primary": "f1_macro",
        "optuna_best_f1_cv": float(study.best_value),
        "best_params": study.best_params,
        "ensemble": "StackingClassifier(RF + HGB) + LogisticRegression meta",
    }
    if artifact_saved:
        out["artifact_dir"] = artifact_saved
    return out
