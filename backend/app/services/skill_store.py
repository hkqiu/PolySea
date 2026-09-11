"""PolySea 领域技能：data/skills/<id>/SKILL.md（YAML frontmatter + Markdown，可与 Cursor SKILL 风格对齐）。"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
SKILLS_ROOT = _DATA_ROOT / "skills"
SKILL_FILENAME = "SKILL.md"

# 可调
DEFAULT_SKILL_INJECT_MAX_CHARS = 1500
MAX_SKILL_BODY_CHARS = 48000
MAX_SKILLS_INJECTED = 2

_SKILL_ID_SAFE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9]{16,}"),
    re.compile(r"(?i)api[_\s-]*key\s*[:=]\s*\S+"),
    re.compile(r"(?i)password\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9._-]{20,}"),
    re.compile(r"(?i)openai[_\s-]*api[_\s-]*key\s*[:=]\s*\S+"),
    re.compile(r"\b\d{17}[\dXx]\b"),
]


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def skill_inject_enabled() -> bool:
    return _env_bool("POLYSEA_SKILL_INJECT_ENABLED", True)


def skill_inject_max_chars() -> int:
    raw = (os.environ.get("POLYSEA_SKILL_INJECT_MAX_CHARS") or "").strip()
    if raw.isdigit():
        return max(200, min(8000, int(raw)))
    return DEFAULT_SKILL_INJECT_MAX_CHARS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _skill_body_check(text: str) -> str | None:
    if not text or not str(text).strip():
        return "内容为空"
    s = str(text)
    if len(s) > MAX_SKILL_BODY_CHARS:
        return f"技能正文过长（>{MAX_SKILL_BODY_CHARS} 字符）"
    for pat in _SENSITIVE_PATTERNS:
        if pat.search(s):
            return "内容疑似包含密钥或敏感标识，禁止写入技能"
    return None


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    idx = text.find("\n", 3)
    if idx == -1:
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_raw = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in fm_raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k:
            meta[k] = v
    return meta, body


def _serialize_skill_md(
    *,
    skill_id: str,
    title: str,
    version: int,
    status: str,
    triggers: str,
    tags: str,
    created: str,
    updated: str,
    body: str,
) -> str:
    return (
        "---\n"
        f"id: {skill_id}\n"
        f"title: {title}\n"
        f"version: {version}\n"
        f"status: {status}\n"
        f"triggers: {triggers}\n"
        f"tags: {tags}\n"
        f"created: {created}\n"
        f"updated: {updated}\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


@dataclass
class SkillMeta:
    skill_id: str
    title: str
    version: int
    status: str
    triggers: str
    tags: str
    path: Path

    def trigger_list(self) -> list[str]:
        return [t.strip() for t in self.triggers.split(",") if t.strip()]

    def tag_list(self) -> list[str]:
        return [t.strip() for t in self.tags.split(",") if t.strip()]


def _parse_meta_file(path: Path) -> SkillMeta | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, _ = _split_frontmatter(raw)
    sid = (meta.get("id") or path.parent.name).strip()
    if not sid or not _SKILL_ID_SAFE.match(sid):
        return None
    try:
        ver = int(meta.get("version", "1"))
    except ValueError:
        ver = 1
    return SkillMeta(
        skill_id=sid,
        title=meta.get("title", sid),
        version=max(1, ver),
        status=(meta.get("status") or "active").lower(),
        triggers=meta.get("triggers", ""),
        tags=meta.get("tags", ""),
        path=path,
    )


def ensure_skills_root() -> None:
    SKILLS_ROOT.mkdir(parents=True, exist_ok=True)


def list_skill_metas() -> list[SkillMeta]:
    ensure_skills_root()
    out: list[SkillMeta] = []
    with _LOCK:
        if not SKILLS_ROOT.is_dir():
            return out
        for d in sorted(SKILLS_ROOT.iterdir()):
            if not d.is_dir():
                continue
            p = d / SKILL_FILENAME
            if not p.is_file():
                continue
            m = _parse_meta_file(p)
            if m:
                out.append(m)
    return out


def read_skill_file(skill_id: str) -> tuple[dict[str, str], str] | None:
    if not _SKILL_ID_SAFE.match(skill_id):
        return None
    p = SKILLS_ROOT / skill_id / SKILL_FILENAME
    with _LOCK:
        if not p.is_file():
            return None
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            return None
    meta, body = _split_frontmatter(raw)
    return meta, body


def skill_create(
    *,
    title: str,
    body: str,
    triggers: str = "",
    tags: list[str] | None = None,
    skill_id: str | None = None,
    status: str = "active",
) -> dict[str, Any]:
    err = _skill_body_check(body)
    if err:
        return {"error": err, "summary": err}
    t = (title or "").strip()
    if not t:
        return {"error": "需要非空 title", "summary": "缺少标题"}
    st = (status or "active").strip().lower()
    if st not in ("active", "draft"):
        st = "active"
    trig = (triggers or "").strip().replace("\n", " ")
    tag_s = ", ".join(str(x).strip() for x in (tags or []) if str(x).strip())
    now = _utc_now()
    created_s = _fmt_iso(now)
    if skill_id and str(skill_id).strip():
        sid = str(skill_id).strip()
    else:
        sid = f"polysea-s-{now.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2)}"
    if not _SKILL_ID_SAFE.match(sid):
        return {"error": "skill_id 仅允许字母数字、下划线、连字符", "summary": "id 非法"}

    ensure_skills_root()
    dir_path = SKILLS_ROOT / sid
    file_path = dir_path / SKILL_FILENAME
    with _LOCK:
        if file_path.is_file():
            return {"error": f"技能 id 已存在: {sid}", "summary": "请换 skill_id 或先 skill_delete"}
        dir_path.mkdir(parents=True, exist_ok=False)
        content = _serialize_skill_md(
            skill_id=sid,
            title=t.replace("\n", " "),
            version=1,
            status=st,
            triggers=trig,
            tags=tag_s,
            created=created_s,
            updated=created_s,
            body=body,
        )
        file_path.write_text(content, encoding="utf-8")

    return {
        "success": True,
        "summary": f"已创建技能 {sid}（version=1）。",
        "skill_id": sid,
        "title": t,
        "version": 1,
    }


def skill_update(
    skill_id: str,
    *,
    title: str | None = None,
    triggers: str | None = None,
    tags: list[str] | None = None,
    body: str | None = None,
    append_to_body: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    sid = (skill_id or "").strip()
    if not _SKILL_ID_SAFE.match(sid):
        return {"error": "非法 skill_id", "summary": "id 无效"}
    parsed = read_skill_file(sid)
    if not parsed:
        return {"error": f"未找到技能: {sid}", "summary": "请先 skill_list 查看 id"}
    meta, old_body = parsed
    try:
        ver = int(meta.get("version", "1"))
    except ValueError:
        ver = 1
    new_body = old_body
    if body is not None and str(body).strip():
        err = _skill_body_check(str(body))
        if err:
            return {"error": err, "summary": err}
        new_body = str(body).strip()
    if append_to_body is not None and str(append_to_body).strip():
        chunk = str(append_to_body).strip()
        err = _skill_body_check(new_body + "\n\n" + chunk)
        if err:
            return {"error": err, "summary": err}
        new_body = (new_body.rstrip() + "\n\n## 迭代补充\n\n" + chunk).strip()

    nt = meta.get("title", sid)
    if title is not None and str(title).strip():
        nt = str(title).strip().replace("\n", " ")
    ntrig = meta.get("triggers", "")
    if triggers is not None:
        ntrig = str(triggers).strip().replace("\n", " ")
    ntags = meta.get("tags", "")
    if tags is not None:
        ntags = ", ".join(str(x).strip() for x in tags if str(x).strip())
    nst = (meta.get("status") or "active").lower()
    if status is not None and str(status).strip():
        s = str(status).strip().lower()
        if s in ("active", "draft"):
            nst = s

    bump = 0
    if body is not None or append_to_body is not None:
        bump = 1
    if title is not None or triggers is not None or tags is not None or status is not None:
        bump = max(bump, 1)
    new_ver = ver + bump if bump else ver

    now_s = _fmt_iso(_utc_now())
    created_s = meta.get("created") or now_s
    content = _serialize_skill_md(
        skill_id=sid,
        title=nt,
        version=new_ver,
        status=nst,
        triggers=ntrig,
        tags=ntags,
        created=created_s,
        updated=now_s,
        body=new_body,
    )
    with _LOCK:
        (SKILLS_ROOT / sid / SKILL_FILENAME).write_text(content, encoding="utf-8")

    return {
        "success": True,
        "summary": f"已更新技能 {sid}，version={new_ver}。",
        "skill_id": sid,
        "version": new_ver,
    }


def skill_delete(skill_id: str) -> dict[str, Any]:
    sid = (skill_id or "").strip()
    if not _SKILL_ID_SAFE.match(sid):
        return {"error": "非法 skill_id", "summary": "id 无效"}
    dir_path = SKILLS_ROOT / sid
    with _LOCK:
        if not dir_path.is_dir():
            return {"error": f"未找到技能目录: {sid}", "summary": "不存在"}
        shutil.rmtree(dir_path, ignore_errors=True)
    return {"success": True, "summary": f"已删除技能 {sid}。", "skill_id": sid}


def skill_list_json() -> dict[str, Any]:
    metas = list_skill_metas()
    rows = []
    for m in metas:
        rows.append(
            {
                "skill_id": m.skill_id,
                "title": m.title,
                "version": m.version,
                "status": m.status,
                "triggers": m.triggers,
                "tags": m.tags,
            }
        )
    return {
        "success": True,
        "summary": f"共 {len(rows)} 条技能。",
        "skills": rows,
    }


def skill_read_full(skill_id: str) -> dict[str, Any]:
    sid = (skill_id or "").strip()
    if not _SKILL_ID_SAFE.match(sid):
        return {"error": "非法 skill_id", "summary": "id 无效"}
    parsed = read_skill_file(sid)
    if not parsed:
        return {"error": f"未找到: {sid}", "summary": "无此技能"}
    meta, body = parsed
    return {
        "success": True,
        "summary": f"技能 {sid} v{meta.get('version', '?')}",
        "meta": meta,
        "body": body,
        "full_markdown": _serialize_skill_md(
            skill_id=sid,
            title=meta.get("title", sid),
            version=int(meta.get("version", "1") or 1),
            status=meta.get("status", "active"),
            triggers=meta.get("triggers", ""),
            tags=meta.get("tags", ""),
            created=meta.get("created", ""),
            updated=meta.get("updated", ""),
            body=body,
        ),
    }


def _user_blob_from_messages(user_messages: list[dict[str, Any]], n: int = 4) -> str:
    parts: list[str] = []
    for m in user_messages:
        if m.get("role") == "user":
            parts.append(str(m.get("content") or ""))
    return "\n".join(parts[-n:] if n > 0 else parts)


def _score_skill(meta: SkillMeta, blob_lower: str) -> float:
    s = 0.0
    for t in meta.trigger_list():
        if t.lower() in blob_lower:
            s += 2.0
    for t in meta.tag_list():
        if t.lower() in blob_lower:
            s += 0.5
    return s


def build_skill_inject(user_messages: list[dict[str, Any]]) -> str:
    """按用户近期话术与 triggers/tags 匹配，注入技能正文摘要。"""
    if not skill_inject_enabled():
        return ""
    blob = _user_blob_from_messages(user_messages, 4).strip()
    if not blob:
        return ""
    blob_lower = blob.lower()
    metas = [m for m in list_skill_metas() if m.status == "active"]
    if not metas:
        return ""
    scored = [( _score_skill(m, blob_lower), m) for m in metas]
    scored.sort(key=lambda x: -x[0])
    top = [(s, m) for s, m in scored if s >= 1.0][:MAX_SKILLS_INJECTED]
    if not top:
        return ""

    budget = skill_inject_max_chars()
    lines: list[str] = ["【匹配到的领域技能（请优先遵循，除非与用户当前指令冲突）】"]
    used = len(lines[0]) + 20
    for sc, m in top:
        parsed = read_skill_file(m.skill_id)
        if not parsed:
            continue
        _, body = parsed
        header = f"### {m.title} (id={m.skill_id}, v{m.version}, match≈{sc:.1f})"
        chunk = header + "\n" + body.strip()
        if used + len(chunk) > budget:
            remain = budget - used - len(header) - 30
            if remain < 80:
                break
            chunk = header + "\n" + body.strip()[:remain] + "…\n（正文已截断，完整请 skill_read）"
        lines.append(chunk)
        used += len(chunk) + 2

    return "\n\n".join(lines).strip()
