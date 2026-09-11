"""PolySea 单机单用户长期记忆：data/memory.md，艾宾浩斯启发式衰减与 reflection。"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 可调参数（直接改这里的常量即可；注入摘要长度也可用环境变量覆盖）
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_INJECT_MAX_CHARS = 500

# 距「上次强化」越久，强度乘性衰减越快（艾宾浩斯启发，单位：天）
TAU_DECAY_SINCE_TOUCH_DAYS = 8.0
# 距「创建」的慢衰减，防止单条记忆永久占坑（单位：天）
TAU_DECAY_SINCE_CREATED_DAYS = 45.0

# 强度阈值：低于则压缩层级；低于 FORGET 则删除
STRENGTH_COMPRESS_FULL = 0.42
STRENGTH_COMPRESS_AGAIN = 0.20
STRENGTH_FORGET = 0.05

# 正文长度上限（单条）
MAX_BODY_CHARS = 8000

# 备份文件保留个数
MEMORY_BACKUP_KEEP = 20

_MEMORY_FILE_LOCK = threading.Lock()

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
MEMORY_PATH = _DATA_ROOT / "memory.md"

_ENTRY_HEADER = "### ENTRY"

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9]{16,}"),
    re.compile(r"(?i)api[_\s-]*key\s*[:=]\s*\S+"),
    re.compile(r"(?i)password\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9._-]{20,}"),
    re.compile(r"(?i)openai[_\s-]*api[_\s-]*key\s*[:=]\s*\S+"),
    re.compile(r"\b\d{17}[\dXx]\b"),  # 中国 18 位身份证
]


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def memory_inject_enabled() -> bool:
    """默认开启：新会话首条用户消息时注入长期记忆摘要。"""
    return _env_bool("POLYSEA_MEMORY_INJECT_ENABLED", True)


def memory_inject_max_chars() -> int:
    raw = (os.environ.get("POLYSEA_MEMORY_INJECT_MAX_CHARS") or "").strip()
    if raw.isdigit():
        return max(100, min(8000, int(raw)))
    return DEFAULT_MEMORY_INJECT_MAX_CHARS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_dt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_dt(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


@dataclass
class MemoryEntry:
    id: str
    created: datetime
    last_touched: datetime
    strength: float
    layer: str  # full | compressed | minimal
    tags: list[str] = field(default_factory=list)
    body: str = ""
    # 上次执行 memory_reflect 衰减的时间；用于增量衰减，避免短时间多次 reflect 重复套用「记忆年龄」
    last_decay_applied: datetime | None = None

    def display_body(self) -> str:
        return self.body.strip()


def _default_header() -> str:
    return (
        "# PolySea 长期记忆\n\n"
        "<!-- 固定格式：每条以单独一行「### ENTRY」开头；"
        "下一行起为「键: 值」元数据，一个空行后为正文，直至下一条 ENTRY 或文件结束。"
        "勿写入 API Key、密码、完整隐私数据或整表 CSV。 -->\n\n"
    )


def _parse_memory_file(text: str) -> tuple[str, list[MemoryEntry]]:
    """返回 (头部 markdown（含首个 ENTRY 之前）, 条目列表)。"""
    if _ENTRY_HEADER not in text:
        return text.rstrip() + "\n\n" if text.strip() else _default_header(), []

    parts = text.split(_ENTRY_HEADER)
    header = parts[0]
    entries: list[MemoryEntry] = []
    for raw in parts[1:]:
        block = raw.strip()
        if not block:
            continue
        body_lines = block.split("\n")
        meta: dict[str, str] = {}
        body_start = 0
        for i, line in enumerate(body_lines):
            if line.strip() == "":
                body_start = i + 1
                break
            m = re.match(r"^([a-zA-Z_]+)\s*:\s*(.*)$", line.strip())
            if m:
                meta[m.group(1).lower()] = m.group(2).strip()
            else:
                body_start = i
                break
        body = "\n".join(body_lines[body_start:]).strip()
        try:
            eid = meta.get("id") or ""
            if not eid:
                continue
            created = _parse_dt(meta.get("created", _fmt_dt(_utc_now())))
            last_touched = _parse_dt(meta.get("last_touched", meta.get("created", _fmt_dt(created))))
            strength = float(meta.get("strength", "1"))
            layer = (meta.get("layer") or "full").lower()
            if layer not in ("full", "compressed", "minimal"):
                layer = "full"
            tags_raw = meta.get("tags", "")
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            lda_raw = meta.get("last_decay_applied")
            lda = _parse_dt(lda_raw) if lda_raw else None
            entries.append(
                MemoryEntry(
                    id=eid,
                    created=created,
                    last_touched=last_touched,
                    strength=max(0.0, min(1.0, strength)),
                    layer=layer,
                    tags=tags,
                    body=body,
                    last_decay_applied=lda,
                )
            )
        except (ValueError, KeyError, TypeError):
            continue

    if not header.strip():
        header = _default_header()
    return header, entries


def _serialize_entry(e: MemoryEntry) -> str:
    tags_s = ", ".join(e.tags)
    lines = [
        _ENTRY_HEADER,
        f"id: {e.id}",
        f"created: {_fmt_dt(e.created)}",
        f"last_touched: {_fmt_dt(e.last_touched)}",
    ]
    if e.last_decay_applied:
        lines.append(f"last_decay_applied: {_fmt_dt(e.last_decay_applied)}")
    lines += [
        f"strength: {e.strength:.4f}",
        f"layer: {e.layer}",
        f"tags: {tags_s}",
        "",
        e.body.strip(),
        "",
    ]
    return "\n".join(lines)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def load_memory_document() -> tuple[str, list[MemoryEntry]]:
    with _MEMORY_FILE_LOCK:
        if not MEMORY_PATH.is_file():
            return _default_header(), []
        text = MEMORY_PATH.read_text(encoding="utf-8")
    return _parse_memory_file(text)


def save_memory_document(header: str, entries: list[MemoryEntry]) -> None:
    parts = [header.rstrip() + "\n\n"]
    for e in entries:
        parts.append(_serialize_entry(e))
    content = "".join(parts)
    with _MEMORY_FILE_LOCK:
        _write_atomic(MEMORY_PATH, content)


def ensure_memory_file() -> None:
    with _MEMORY_FILE_LOCK:
        if not MEMORY_PATH.is_file():
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(MEMORY_PATH, _default_header())


def _backup_before_write() -> str | None:
    """在改写 memory.md 前复制一份。返回备份路径字符串，若无可备份则 None。"""
    if not MEMORY_PATH.is_file():
        return None
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    bak = MEMORY_PATH.parent / f"memory.backup.{stamp}.md"
    shutil.copy2(MEMORY_PATH, bak)
    _prune_backups()
    return str(bak)


def _prune_backups() -> None:
    pattern = "memory.backup.*.md"
    root = MEMORY_PATH.parent
    backups = sorted(root.glob("memory.backup.*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in backups[MEMORY_BACKUP_KEEP:]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def sensitive_check(text: str) -> str | None:
    """若命中敏感模式，返回拒绝原因；否则 None。"""
    if not text or not str(text).strip():
        return "内容为空"
    s = str(text)
    if len(s) > MAX_BODY_CHARS:
        return f"单条记忆过长（>{MAX_BODY_CHARS} 字符），请只保存摘要"
    for pat in _SENSITIVE_PATTERNS:
        if pat.search(s):
            return "内容疑似包含密钥、口令或敏感标识，禁止写入长期记忆"
    return None


def _compress_body(body: str, *, target_layer: str) -> str:
    b = body.strip()
    if target_layer == "compressed":
        if len(b) <= 160:
            return b
        return b[:120].rstrip() + "… [已压缩]"
    if target_layer == "minimal":
        one = b.split("\n", 1)[0].strip()
        if len(one) > 64:
            one = one[:61] + "…"
        return one + " [极简]"
    return b


def _decay_factor_for_interval(dt_days: float) -> float:
    """艾宾浩斯启发：间隔越久遗忘越多；用两段指数模拟「近因 + 长期痕迹消退」。"""
    dt_days = max(0.0, dt_days)
    f1 = math.exp(-dt_days / TAU_DECAY_SINCE_TOUCH_DAYS)
    f2 = math.exp(-dt_days / TAU_DECAY_SINCE_CREATED_DAYS)
    return f1 * f2


def apply_time_decay(entries: list[MemoryEntry], now: datetime | None = None) -> list[MemoryEntry]:
    """按距上次 reflect 的间隔做增量衰减，并调整压缩层级。"""
    now = now or _utc_now()
    out: list[MemoryEntry] = []
    for e in entries:
        lda = e.last_decay_applied or e.created
        dt_step = (now - lda).total_seconds() / 86400.0
        factor = _decay_factor_for_interval(dt_step)
        new_s = max(0.0, min(1.0, e.strength * factor))
        layer = e.layer
        body = e.body
        if new_s < STRENGTH_FORGET:
            continue
        if new_s < STRENGTH_COMPRESS_AGAIN and layer == "compressed":
            layer = "minimal"
            body = _compress_body(body, target_layer="minimal")
        elif new_s < STRENGTH_COMPRESS_FULL and layer == "full":
            layer = "compressed"
            body = _compress_body(body, target_layer="compressed")
        ne = MemoryEntry(
            id=e.id,
            created=e.created,
            last_touched=e.last_touched,
            strength=new_s,
            layer=layer,
            tags=e.tags,
            body=body,
            last_decay_applied=now,
        )
        out.append(ne)
    return out


def run_memory_reflect() -> dict[str, Any]:
    """备份 → 时间衰减 → 压缩/删除 → 写回。"""
    ensure_memory_file()
    header, entries = load_memory_document()
    bak = _backup_before_write()
    now = _utc_now()
    decayed = apply_time_decay(entries, now=now)
    save_memory_document(header, decayed)
    return {
        "success": True,
        "summary": (
            f"记忆整理完成：原有 {len(entries)} 条，整理后 {len(decayed)} 条。"
            + (f" 已备份至 {bak}。" if bak else "")
        ),
        "before": len(entries),
        "after": len(decayed),
        "backup_path": bak,
    }


def effective_strength_now(e: MemoryEntry, now: datetime | None = None) -> float:
    """当前时刻有效强度（用于排序与摘要，不写回文件）。"""
    now = now or _utc_now()
    lda = e.last_decay_applied or e.created
    dt_step = (now - lda).total_seconds() / 86400.0
    return max(0.0, min(1.0, e.strength * _decay_factor_for_interval(dt_step)))


def build_inject_summary(max_chars: int | None = None) -> str:
    """按有效强度排序，生成注入 system 的短摘要（纯文本）。"""
    max_chars = max_chars if max_chars is not None else memory_inject_max_chars()
    ensure_memory_file()
    _, entries = load_memory_document()
    if not entries:
        return ""
    now = _utc_now()
    scored = []
    for e in entries:
        eff = effective_strength_now(e, now)
        scored.append((eff, e))
    scored.sort(key=lambda x: -x[0])
    lines: list[str] = ["【长期记忆摘要】"]
    used = len(lines[0]) + 1
    for eff, e in scored:
        line = f"- [{e.layer}, s≈{eff:.2f}] {e.id}: {e.display_body()[:200]}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _new_id() -> str:
    return f"m{_utc_now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"


def memory_append(
    body: str,
    *,
    tags: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    err = sensitive_check(body)
    if err:
        return {"error": err, "summary": err}
    if title:
        err2 = sensitive_check(title)
        if err2:
            return {"error": err2, "summary": err2}
    ensure_memory_file()
    header, entries = load_memory_document()
    now = _utc_now()
    text = body.strip()
    if title and str(title).strip():
        text = f"{str(title).strip()}\n{text}"
    eid = _new_id()
    entry = MemoryEntry(
        id=eid,
        created=now,
        last_touched=now,
        strength=1.0,
        layer="full",
        tags=list(tags or []),
        body=text,
        last_decay_applied=now,
    )
    entries.append(entry)
    save_memory_document(header, entries)
    return {
        "success": True,
        "summary": f"已写入长期记忆 id={eid}。",
        "id": eid,
    }


def memory_update(entry_id: str, new_body: str) -> dict[str, Any]:
    err = sensitive_check(new_body)
    if err:
        return {"error": err, "summary": err}
    ensure_memory_file()
    header, entries = load_memory_document()
    now = _utc_now()
    found = False
    new_list: list[MemoryEntry] = []
    for e in entries:
        if e.id != entry_id:
            new_list.append(e)
            continue
        found = True
        new_list.append(
            MemoryEntry(
                id=e.id,
                created=e.created,
                last_touched=now,
                strength=min(1.0, e.strength + 0.08),
                layer="full",
                tags=e.tags,
                body=new_body.strip(),
                last_decay_applied=e.last_decay_applied,
            )
        )
    if not found:
        msg = f"未找到 id={entry_id}"
        return {"error": msg, "summary": msg}
    save_memory_document(header, new_list)
    return {"success": True, "summary": f"已更新记忆 {entry_id}（强度略上调并视为强化）。"}


def memory_delete(entry_id: str) -> dict[str, Any]:
    ensure_memory_file()
    header, entries = load_memory_document()
    new_list = [e for e in entries if e.id != entry_id]
    if len(new_list) == len(entries):
        msg = f"未找到 id={entry_id}"
        return {"error": msg, "summary": msg}
    save_memory_document(header, new_list)
    return {"success": True, "summary": f"已删除记忆 {entry_id}。"}


def memory_read(*, mode: str = "summary", query: str | None = None) -> dict[str, Any]:
    """mode: summary | full。可选 query 子串过滤（不区分大小写）。"""
    ensure_memory_file()
    header, entries = load_memory_document()
    q = (query or "").strip().lower()
    if q:
        entries = [
            e
            for e in entries
            if q in e.id.lower()
            or q in e.body.lower()
            or any(q in t.lower() for t in e.tags)
        ]
    now = _utc_now()
    entries_sorted = sorted(entries, key=lambda e: -effective_strength_now(e, now))

    if mode == "full":
        blocks = []
        touched: list[MemoryEntry] = []
        for e in entries_sorted:
            blocks.append(
                json.dumps(
                    {
                        "id": e.id,
                        "created": _fmt_dt(e.created),
                        "last_touched": _fmt_dt(e.last_touched),
                        "strength": round(e.strength, 4),
                        "effective_strength": round(effective_strength_now(e, now), 4),
                        "layer": e.layer,
                        "tags": e.tags,
                        "body": e.display_body(),
                    },
                    ensure_ascii=False,
                )
            )
            touched.append(
                MemoryEntry(
                    id=e.id,
                    created=e.created,
                    last_touched=now,
                    strength=min(1.0, e.strength + 0.04),
                    layer=e.layer,
                    tags=e.tags,
                    body=e.body,
                    last_decay_applied=e.last_decay_applied,
                )
            )
        # 未出现在结果里的条目保持原样
        id_set = {e.id for e in entries_sorted}
        rest = [e for e in entries if e.id not in id_set]
        merged_map = {e.id: e for e in rest}
        for e in touched:
            merged_map[e.id] = e
        merged = list(merged_map.values())
        save_memory_document(header, merged)
        text = "\n".join(blocks) if blocks else "（无匹配记忆）"
        return {
            "success": True,
            "summary": f"已返回 {len(blocks)} 条完整记忆，并已轻微强化（last_touched/strength）。",
            "content": text,
        }

    # summary
    lines = [f"共 {len(entries_sorted)} 条（已按有效强度排序）。"]
    for e in entries_sorted[:24]:
        preview = e.display_body().replace("\n", " ")[:120]
        lines.append(
            f"- id={e.id} eff≈{effective_strength_now(e, now):.2f} layer={e.layer} | {preview}"
        )
    content = "\n".join(lines)
    return {
        "success": True,
        "summary": "长期记忆摘要（未写入强化；需要细节请用 mode=full 并可用 query 过滤）。",
        "content": content,
    }


def is_first_user_message_in_session(user_messages: list[dict[str, Any]]) -> bool:
    return sum(1 for m in user_messages if m.get("role") == "user") == 1
