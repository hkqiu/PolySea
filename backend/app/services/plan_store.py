"""单机单用户结构化任务计划：data/active_plan.json（与对话轮次无关，服务端单例）。"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
PLAN_PATH = _DATA_ROOT / "active_plan.json"

_LOCK = threading.Lock()

STEP_STATUSES = frozenset({"pending", "in_progress", "done", "skipped"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


@dataclass
class PlanStep:
    id: str
    text: str
    status: str = "pending"

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "status": self.status}


@dataclass
class ActivePlan:
    plan_id: str
    title: str
    created: datetime
    updated: datetime
    steps: list[PlanStep] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "created": _fmt_iso(self.created),
            "updated": _fmt_iso(self.updated),
            "steps": [s.to_json() for s in self.steps],
        }


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _plan_to_doc(p: ActivePlan) -> dict[str, Any]:
    return {
        "plan_id": p.plan_id,
        "title": p.title,
        "created": _fmt_iso(p.created),
        "updated": _fmt_iso(p.updated),
        "steps": [asdict(s) for s in p.steps],
    }


def _doc_to_plan(d: dict[str, Any]) -> ActivePlan | None:
    try:
        pid = str(d.get("plan_id") or "")
        if not pid:
            return None
        title = str(d.get("title") or "")
        created = _parse_iso(str(d["created"]))
        updated = _parse_iso(str(d["updated"]))
        raw_steps = d.get("steps") or []
        steps: list[PlanStep] = []
        if not isinstance(raw_steps, list):
            return None
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "")
            text = str(item.get("text") or "").strip()
            st = str(item.get("status") or "pending").lower()
            if st not in STEP_STATUSES:
                st = "pending"
            if sid and text:
                steps.append(PlanStep(id=sid, text=text, status=st))
        return ActivePlan(
            plan_id=pid,
            title=title,
            created=created,
            updated=updated,
            steps=steps,
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_plan() -> ActivePlan | None:
    with _LOCK:
        if not PLAN_PATH.is_file():
            return None
        try:
            d = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(d, dict):
            return None
        return _doc_to_plan(d)


def save_plan(p: ActivePlan) -> None:
    with _LOCK:
        _write_atomic(PLAN_PATH, _plan_to_doc(p))


def delete_plan_file() -> None:
    with _LOCK:
        try:
            PLAN_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def plan_summary_text(p: ActivePlan) -> str:
    lines = [
        f"计划 ID: {p.plan_id}",
        f"标题: {p.title or '（无）'}",
        f"步骤数: {len(p.steps)}",
    ]
    for i, s in enumerate(p.steps):
        lines.append(f"  [{i}] id={s.id} status={s.status} | {s.text[:200]}")
    return "\n".join(lines)


def plan_create(*, title: str | None, steps: list[str]) -> dict[str, Any]:
    texts = [str(t).strip() for t in steps if str(t).strip()]
    if not texts:
        return {"error": "steps 至少包含一条非空描述", "summary": "创建计划失败：无有效步骤"}
    now = _utc_now()
    pid = f"p{_utc_now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2)}"
    plan_steps: list[PlanStep] = []
    for i, tx in enumerate(texts):
        plan_steps.append(PlanStep(id=str(i + 1), text=tx, status="pending"))
    p = ActivePlan(
        plan_id=pid,
        title=(title or "").strip(),
        created=now,
        updated=now,
        steps=plan_steps,
    )
    save_plan(p)
    return {
        "success": True,
        "summary": f"已创建计划 {pid}，共 {len(plan_steps)} 步。",
        "plan": p.to_json(),
        "plan_text": plan_summary_text(p),
    }


def plan_read() -> dict[str, Any]:
    p = load_plan()
    if not p:
        return {
            "success": True,
            "summary": "当前没有活动计划。",
            "plan": None,
            "plan_text": "",
        }
    return {
        "success": True,
        "summary": plan_summary_text(p),
        "plan": p.to_json(),
        "plan_text": plan_summary_text(p),
    }


def _find_step_index(p: ActivePlan, *, step_index: int | None, step_id: str | None) -> int | None:
    if step_id is not None and str(step_id).strip():
        sid = str(step_id).strip()
        for i, s in enumerate(p.steps):
            if s.id == sid:
                return i
    if step_index is not None:
        idx = int(step_index)
        if 0 <= idx < len(p.steps):
            return idx
    return None


def plan_set_step_status(
    *,
    status: str,
    step_index: int | None = None,
    step_id: str | None = None,
) -> dict[str, Any]:
    st = str(status or "").strip().lower()
    if st not in STEP_STATUSES:
        return {
            "error": f"status 须为 {', '.join(sorted(STEP_STATUSES))}",
            "summary": "无效的状态值",
        }
    p = load_plan()
    if not p:
        return {"error": "当前没有活动计划", "summary": "请先 plan_create"}
    idx = _find_step_index(p, step_index=step_index, step_id=step_id)
    if idx is None:
        return {
            "error": "未找到对应步骤：请提供有效 step_index（从 0 起）或 step_id（与 plan 中 id 一致）",
            "summary": "步骤定位失败",
        }
    now = _utc_now()
    new_steps: list[PlanStep] = []
    for i, s in enumerate(p.steps):
        if i == idx:
            new_steps.append(PlanStep(id=s.id, text=s.text, status=st))
        else:
            ns = s.status
            if st == "in_progress" and s.status == "in_progress":
                ns = "pending"
            new_steps.append(PlanStep(id=s.id, text=s.text, status=ns))
    p2 = ActivePlan(
        plan_id=p.plan_id,
        title=p.title,
        created=p.created,
        updated=now,
        steps=new_steps,
    )
    save_plan(p2)
    return {
        "success": True,
        "summary": f"步骤 [{idx}] id={p2.steps[idx].id} 已设为 {st}。",
        "plan": p2.to_json(),
        "plan_text": plan_summary_text(p2),
    }


def plan_clear() -> dict[str, Any]:
    p = load_plan()
    delete_plan_file()
    if p:
        return {
            "success": True,
            "summary": f"已清除计划 {p.plan_id}。",
            "plan": None,
        }
    return {"success": True, "summary": "当前本就没有活动计划。", "plan": None}
