"""长任务训练进度（单用户本地 MVP，供前端轮询）。"""

from __future__ import annotations

import time
from threading import RLock
from typing import Any

_lock = RLock()
_state: dict[str, Any] = {
    "cancel_requested": False,
    "agent": {
        "phase": "idle",
        "message": "",
        "detail": "",
    },
    "active": False,
    "kind": "",
    "phase": "",
    "optuna_current": 0,
    "optuna_total": 0,
    "optuna_best_value": None,
    "epoch_current": 0,
    "epoch_total": 0,
    "train_loss": None,
    "valid_loss": None,
    "message": "",
    "logs": [],
    "started_at": None,
}


class TrainingCancelled(Exception):
    """用户中止性质预测或 PolyTAO 训练。"""


def snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def request_cancel() -> None:
    """前端「中止任务」：训练在下一 trial/epoch 边界停止；对话在下一轮 Agent 循环前结束。"""
    with _lock:
        _state["cancel_requested"] = True


def is_cancel_requested() -> bool:
    with _lock:
        return bool(_state.get("cancel_requested"))


def clear_cancel_request() -> None:
    with _lock:
        _state["cancel_requested"] = False


def set_agent(phase: str, message: str = "", detail: str = "") -> None:
    """Agent 阶段：idle | thinking | tool | training | optimizing | done"""
    with _lock:
        _state["agent"] = {
            "phase": phase,
            "message": message,
            "detail": detail,
        }


def clear_agent() -> None:
    set_agent("idle", "", "")


def _log(line: str) -> None:
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {line}"
    with _lock:
        logs = list(_state.get("logs") or [])
        logs.append(entry)
        _state["logs"] = logs[-100:]


def reset() -> None:
    with _lock:
        _state.update(
            active=True,
            kind="",
            phase="",
            optuna_current=0,
            optuna_total=0,
            optuna_best_value=None,
            epoch_current=0,
            epoch_total=0,
            train_loss=None,
            valid_loss=None,
            message="",
            logs=[],
            started_at=time.time(),
        )


def start_property(total_trials: int) -> None:
    reset()
    set_agent("training", "正在训练性质预测集成模型…", "Optuna + Stacking")
    with _lock:
        _state["kind"] = "property"
        _state["phase"] = "optuna"
        _state["optuna_total"] = int(total_trials)
        _state["message"] = f"性质预测：Optuna 搜索（最多 {total_trials} 次试验）"
    _log("开始性质预测自动建模")


def optuna_trial(current: int, total: int, value: float | None) -> None:
    detail = f"第 {current}/{total} 次试验"
    if value is not None:
        detail += f"，CV 得分≈{value:.4f}"
    set_agent("optimizing", "正在优化性质预测集成模型超参数…", detail)
    with _lock:
        _state["optuna_current"] = int(current)
        _state["optuna_total"] = int(total)
        _state["optuna_best_value"] = value
        _state["message"] = f"Optuna 试验 {current}/{total}" + (
            f"，当前 CV 目标≈{value:.4f}" if value is not None else ""
        )
    _log(f"Optuna trial {current}/{total} 完成" + (f"，score={value:.5f}" if value is not None else ""))


def phase_fit_best() -> None:
    set_agent("optimizing", "正在拟合最优集成模型并评估测试集…", "Stacking + 元学习器")
    with _lock:
        _state["phase"] = "fit_best"
        _state["message"] = "正在用最优超参在训练集上拟合集成模型并评估测试集…"
    _log("开始拟合最优 Stacking 模型")


def start_polytao(total_epochs: int) -> None:
    reset()
    set_agent("training", "正在训练 PolyTAO 生成模型…", f"共 {total_epochs} 个 epoch")
    with _lock:
        _state["kind"] = "polytao"
        _state["phase"] = "epoch"
        _state["epoch_total"] = int(total_epochs)
        _state["message"] = f"PolyTAO 微调：共 {total_epochs} 个 epoch"
    _log("开始 PolyTAO 微调")


def polytao_epoch(epoch: int, total: int, train_loss: float, valid_loss: float) -> None:
    set_agent(
        "training",
        f"正在训练 PolyTAO 模型… Epoch {epoch}/{total}",
        f"train_loss={train_loss:.4f}  valid_loss={valid_loss:.4f}",
    )
    with _lock:
        _state["epoch_current"] = int(epoch)
        _state["epoch_total"] = int(total)
        _state["train_loss"] = float(train_loss)
        _state["valid_loss"] = float(valid_loss)
        _state["message"] = f"Epoch {epoch}/{total}  train_loss={train_loss:.4f}  valid_loss={valid_loss:.4f}"
    _log(f"Epoch {epoch}/{total} train={train_loss:.4f} val={valid_loss:.4f}")


def finish_cancelled(msg: str = "已由用户中止") -> None:
    set_agent("idle", msg, "")
    with _lock:
        _state["active"] = False
        _state["phase"] = "cancelled"
        _state["message"] = msg
        _state["cancel_requested"] = False
    _log(msg)


def finish_ok(summary: str = "完成") -> None:
    with _lock:
        _state["cancel_requested"] = False
        kind = _state.get("kind", "")
    if kind == "property":
        set_agent("done", "性质预测模型训练结束", summary)
    elif kind == "polytao":
        set_agent("done", "PolyTAO 模型训练结束", summary)
    else:
        set_agent("done", summary, "")
    with _lock:
        _state["active"] = False
        _state["phase"] = "done"
        _state["message"] = summary
    _log(summary)


def finish_error(msg: str) -> None:
    set_agent("done", "训练失败", msg[:200])
    with _lock:
        _state["cancel_requested"] = False
        _state["active"] = False
        _state["phase"] = "error"
        _state["message"] = msg
    _log(f"错误: {msg}")
