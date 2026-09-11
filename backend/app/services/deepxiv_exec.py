"""DeepXiv (deepxiv-sdk) tool implementations for literature search and reading."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from deepxiv_sdk import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    Reader,
    ServerError,
)

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_MAX_RAW_MARKDOWN_CHARS = 100_000
_MAX_JSON_STR_CHARS = 200_000
_DIGEST_MAX_PAPERS = 12
_DEEP_DIVE_MAX_SECTIONS = 8
_DEFAULT_DEEPXIV_BASE = "https://data.rag.ac.cn"


def _load_deepxiv_dotenv() -> None:
    """与 deepxiv CLI 一致：从用户目录与当前工作目录加载 .env（不覆盖已存在的环境变量）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for env_path in (Path.home() / ".env", Path.cwd() / ".env"):
        try:
            if env_path.is_file():
                load_dotenv(env_path, override=False)
        except OSError as e:
            logger.debug("跳过加载 %s：%s", env_path, e)


def _effective_deepxiv_token(base_url: str) -> str | None:
    _load_deepxiv_dotenv()
    t = (os.environ.get("DEEPXIV_TOKEN") or "").strip() or None
    if t:
        return t
    # 仅对官方 API 端尝试与 CLI 相同的自动注册；自建网关需自行配置 TOKEN
    if base_url.rstrip("/") != _DEFAULT_DEEPXIV_BASE.rstrip("/"):
        logger.warning(
            "未检测到 DEEPXIV_TOKEN，且 DEEPXIV_BASE_URL 非官方地址，无法自动申请 token；"
            "请在环境变量或 ~/.env 中配置 DEEPXIV_TOKEN。"
        )
        return None
    try:
        from deepxiv_sdk.cli import ensure_token

        reg = ensure_token(None, auto_create=True)
        if reg and str(reg).strip():
            return str(reg).strip()
    except Exception:
        logger.exception("DeepXiv 自动申请 token（ensure_token）失败")
    return None


def _reader() -> Reader:
    base = (os.environ.get("DEEPXIV_BASE_URL") or _DEFAULT_DEEPXIV_BASE).strip().rstrip("/")
    token = _effective_deepxiv_token(base)
    return Reader(token=token, base_url=base)


def _normalize_arxiv_id(raw: str | None) -> str | dict[str, Any]:
    if not raw or not str(raw).strip():
        return {"error": "缺少 arxiv_id", "summary": "缺少 arxiv_id"}
    s = str(raw).strip()
    m = _ARXIV_ID_RE.search(s)
    if m:
        return m.group(1)
    # bare id like 2409.05591
    s2 = s.replace("arXiv:", "").replace("arxiv:", "").strip()
    if re.fullmatch(r"\d{4}\.\d{4,5}", s2):
        return s2
    return {"error": f"无法解析 arXiv ID：{raw!r}", "summary": "arxiv_id 格式无效"}


def _normalize_pmc_id(raw: str | None) -> str | dict[str, Any]:
    if not raw or not str(raw).strip():
        return {"error": "缺少 pmc_id", "summary": "缺少 pmc_id"}
    s = re.sub(r"\s+", "", str(raw).strip().upper())
    s = re.sub(r"^PMC", "", s)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return {"error": f"无法解析 PMC ID：{raw!r}", "summary": "pmc_id 格式无效"}
    return "PMC" + digits


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _dxv_err(e: Exception) -> dict[str, Any]:
    if isinstance(e, AuthenticationError):
        msg = (
            "DeepXiv 鉴权失败（401）。请检查："
            "1) 用户主目录下 ~/.env 或项目目录 .env 中是否已有 DEEPXIV_TOKEN（与 deepxiv CLI 共用）；"
            "2) 或在运行后端的终端设置环境变量 DEEPXIV_TOKEN；"
            "3) 若 token 过期，可执行：`deepxiv config --token <新 token>` 或任意 `deepxiv search \"test\"` 触发自动注册；"
            "4) 使用自建 DEEPXIV_BASE_URL 时，须使用与该网关匹配的 token。"
            f" 原始信息：{e!s}"
        )
    elif isinstance(e, RateLimitError):
        msg = "DeepXiv 已达速率或日限额，请稍后再试或申请提高限额。"
    elif isinstance(e, NotFoundError):
        msg = f"DeepXiv 未找到资源：{e!s}"
    elif isinstance(e, ServerError):
        msg = f"DeepXiv 服务端错误：{e!s}"
    elif isinstance(e, APIError):
        msg = f"DeepXiv API 错误：{e!s}"
    else:
        logger.exception("DeepXiv 调用异常")
        msg = f"DeepXiv 调用失败：{e!s}"
    return {"error": msg, "summary": msg}


def _truncate_str(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return (
        text[:max_chars] + "\n\n…[内容已截断；可改用 section、preview 或更小范围读取]…",
        True,
    )


def _maybe_truncate_value(val: Any, max_chars: int) -> tuple[Any, bool]:
    if isinstance(val, str):
        t, cut = _truncate_str(val, max_chars)
        return t, cut
    if isinstance(val, dict):
        dumped = json.dumps(val, ensure_ascii=False, default=str)
        if len(dumped) <= max_chars:
            return val, False
        t, cut = _truncate_str(dumped, max_chars)
        return {"_truncated_json_string": t}, cut
    dumped = json.dumps(val, ensure_ascii=False, default=str)
    t, cut = _truncate_str(dumped, max_chars)
    return {"_truncated": t}, cut


def tool_deepxiv_search(args: dict[str, Any]) -> dict[str, Any]:
    q = args.get("query")
    if not q or not str(q).strip():
        return {"error": "缺少 query", "summary": "缺少检索关键词或问题"}
    try:
        r = _reader()
        mc = args.get("min_citation")
        min_citation = None
        if mc is not None:
            try:
                min_citation = int(mc)
                if min_citation < 0:
                    min_citation = None
            except (TypeError, ValueError):
                min_citation = None
        out = r.search(
            str(q).strip(),
            size=_clamp_int(args.get("size"), 10, 1, 100),
            offset=max(0, _clamp_int(args.get("offset"), 0, 0, 10_000)),
            search_mode=str(args.get("search_mode") or "hybrid"),
            bm25_weight=float(args.get("bm25_weight", 0.5)),
            vector_weight=float(args.get("vector_weight", 0.5)),
            categories=args.get("categories") if isinstance(args.get("categories"), list) else None,
            authors=args.get("authors") if isinstance(args.get("authors"), list) else None,
            min_citation=min_citation,
            date_from=(str(args["date_from"]).strip() if args.get("date_from") else None),
            date_to=(str(args["date_to"]).strip() if args.get("date_to") else None),
        )
        return {"success": True, "summary": "检索完成，可将候选 arXiv ID 交给后续工具精读。", "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_trending(args: dict[str, Any]) -> dict[str, Any]:
    try:
        r = _reader()
        out = r.trending(
            days=_clamp_int(args.get("days"), 7, 1, 365),
            limit=_clamp_int(args.get("limit"), 30, 1, 100),
        )
        return {"success": True, "summary": "热点列表已获取。", "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_paper_brief(args: dict[str, Any]) -> dict[str, Any]:
    aid = _normalize_arxiv_id(args.get("arxiv_id"))
    if isinstance(aid, dict):
        return aid
    try:
        out = _reader().brief(aid)
        return {"success": True, "summary": f"已获取 {aid} 的 brief。", "arxiv_id": aid, "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_paper_head(args: dict[str, Any]) -> dict[str, Any]:
    aid = _normalize_arxiv_id(args.get("arxiv_id"))
    if isinstance(aid, dict):
        return aid
    try:
        out = _reader().head(aid)
        return {"success": True, "summary": f"已获取 {aid} 的结构与章节概览。", "arxiv_id": aid, "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_paper_section(args: dict[str, Any]) -> dict[str, Any]:
    aid = _normalize_arxiv_id(args.get("arxiv_id"))
    if isinstance(aid, dict):
        return aid
    sec = args.get("section_name")
    if not sec or not str(sec).strip():
        return {"error": "缺少 section_name", "summary": "缺少章节名称（如 Introduction、Method）"}
    try:
        text = _reader().section(aid, str(sec).strip())
        body, cut = _truncate_str(text, _MAX_RAW_MARKDOWN_CHARS)
        return {
            "success": True,
            "summary": f"已读取 {aid} 的章节 {sec!r}。",
            "arxiv_id": aid,
            "section_name": str(sec).strip(),
            "truncated": cut,
            "text": body,
        }
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_paper_fetch(args: dict[str, Any]) -> dict[str, Any]:
    aid = _normalize_arxiv_id(args.get("arxiv_id"))
    if isinstance(aid, dict):
        return aid
    mode = str(args.get("content_mode") or "preview").strip().lower()
    try:
        reader = _reader()
        if mode == "preview":
            out = reader.preview(aid)
            return {"success": True, "summary": f"已获取 {aid} 的 preview。", "arxiv_id": aid, "data": out}
        if mode == "raw":
            raw = reader.raw(aid)
            body, cut = _truncate_str(raw, _MAX_RAW_MARKDOWN_CHARS)
            return {
                "success": True,
                "summary": f"已获取 {aid} 的 raw 全文（可能截断）。",
                "arxiv_id": aid,
                "truncated": cut,
                "text": body,
            }
        if mode == "json":
            out = reader.json(aid)
            val, cut = _maybe_truncate_value(out, _MAX_JSON_STR_CHARS)
            return {
                "success": True,
                "summary": f"已获取 {aid} 的结构化 JSON。",
                "arxiv_id": aid,
                "truncated": cut,
                "data": val,
            }
        if mode == "markdown":
            md = reader.markdown(aid)
            body, cut = _truncate_str(md, _MAX_RAW_MARKDOWN_CHARS)
            return {
                "success": True,
                "summary": f"已获取 {aid} 的 Markdown。",
                "arxiv_id": aid,
                "truncated": cut,
                "text": body,
            }
        return {"error": f"未知 content_mode: {mode}", "summary": "content_mode 须为 preview/raw/json/markdown"}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_paper_social(args: dict[str, Any]) -> dict[str, Any]:
    aid = _normalize_arxiv_id(args.get("arxiv_id"))
    if isinstance(aid, dict):
        return aid
    try:
        out = _reader().social_impact(aid)
        if out is None:
            return {
                "success": True,
                "summary": f"{aid} 暂无可用传播/热度数据。",
                "arxiv_id": aid,
                "data": None,
            }
        return {"success": True, "summary": f"已获取 {aid} 的传播/热度信号。", "arxiv_id": aid, "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_websearch(args: dict[str, Any]) -> dict[str, Any]:
    q = args.get("query")
    if not q or not str(q).strip():
        return {"error": "缺少 query", "summary": "缺少 web 检索 query"}
    try:
        out = _reader().websearch(str(q).strip())
        return {
            "success": True,
            "summary": "Web 检索完成（注意：每次调用会消耗较高 API quota）。",
            "data": out,
        }
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_semantic_scholar(args: dict[str, Any]) -> dict[str, Any]:
    sid = args.get("semantic_scholar_id")
    if not sid or not str(sid).strip():
        return {"error": "缺少 semantic_scholar_id", "summary": "缺少 Semantic Scholar 论文 ID"}
    try:
        out = _reader().semantic_scholar(str(sid).strip())
        return {"success": True, "summary": "Semantic Scholar 元数据已获取。", "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_pmc_head(args: dict[str, Any]) -> dict[str, Any]:
    pid = _normalize_pmc_id(args.get("pmc_id"))
    if isinstance(pid, dict):
        return pid
    try:
        out = _reader().pmc_head(pid)
        return {"success": True, "summary": f"已获取 {pid} 的元数据。", "pmc_id": pid, "data": out}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_pmc_fetch(args: dict[str, Any]) -> dict[str, Any]:
    pid = _normalize_pmc_id(args.get("pmc_id"))
    if isinstance(pid, dict):
        return pid
    mode = str(args.get("content_mode") or "json").strip().lower()
    try:
        reader = _reader()
        if mode == "full":
            out = reader.pmc_full(pid)
            val, cut = _maybe_truncate_value(out, _MAX_JSON_STR_CHARS)
            return {
                "success": True,
                "summary": f"已获取 {pid} 的全文 JSON（可能截断）。",
                "pmc_id": pid,
                "truncated": cut,
                "data": val,
            }
        if mode == "json":
            out = reader.pmc_json(pid)
            val, cut = _maybe_truncate_value(out, _MAX_JSON_STR_CHARS)
            return {
                "success": True,
                "summary": f"已获取 {pid} 的 JSON。",
                "pmc_id": pid,
                "truncated": cut,
                "data": val,
            }
        return {"error": f"未知 content_mode: {mode}", "summary": "content_mode 须为 full 或 json"}
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_deep_dive(args: dict[str, Any]) -> dict[str, Any]:
    """单篇论文多粒度打包，便于多轮深挖。"""
    aid = _normalize_arxiv_id(args.get("arxiv_id"))
    if isinstance(aid, dict):
        return aid
    include_brief = bool(args.get("include_brief", True))
    include_head = bool(args.get("include_head", True))
    include_preview = bool(args.get("include_preview", False))
    include_raw = bool(args.get("include_raw", False))
    section_names = args.get("section_names")
    if section_names is not None and not isinstance(section_names, list):
        return {"error": "section_names 须为字符串数组", "summary": "参数 section_names 无效"}
    names = [str(x).strip() for x in (section_names or []) if str(x).strip()][: _DEEP_DIVE_MAX_SECTIONS]

    try:
        reader = _reader()
        bundle: dict[str, Any] = {"arxiv_id": aid}
        if include_brief:
            bundle["brief"] = reader.brief(aid)
        if include_head:
            bundle["head"] = reader.head(aid)
        if include_preview:
            bundle["preview"] = reader.preview(aid)
        if include_raw:
            raw = reader.raw(aid)
            body, cut = _truncate_str(raw, _MAX_RAW_MARKDOWN_CHARS)
            bundle["raw"] = body
            bundle["raw_truncated"] = cut
        if names:
            bundle["sections"] = {}
            for sn in names:
                try:
                    bundle["sections"][sn] = reader.section(aid, sn)
                except NotFoundError:
                    bundle["sections"][sn] = {"error": "章节未找到或不可用"}
                except APIError as e:
                    bundle["sections"][sn] = {"error": str(e)}
        return {
            "success": True,
            "summary": f"已对 {aid} 完成深度打包（brief/head/章节等按参数）。",
            "data": bundle,
        }
    except (AuthenticationError, RateLimitError, NotFoundError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


def tool_deepxiv_digest_pack(args: dict[str, Any]) -> dict[str, Any]:
    """多篇论文批量拉取，供模型写对比/综述（最终表述仍由你生成）。"""
    ids = args.get("arxiv_ids")
    if not isinstance(ids, list) or not ids:
        return {"error": "arxiv_ids 须为非空数组", "summary": "缺少 arxiv_ids 列表"}
    tier = str(args.get("tier") or "brief_only").strip().lower()
    if tier not in ("brief_only", "brief_and_head"):
        return {"error": "tier 须为 brief_only 或 brief_and_head", "summary": "tier 参数无效"}

    normalized: list[str] = []
    for raw in ids[:_DIGEST_MAX_PAPERS]:
        a = _normalize_arxiv_id(raw)
        if isinstance(a, dict):
            continue
        if a not in normalized:
            normalized.append(a)
    if not normalized:
        return {"error": "没有有效的 arxiv_id", "summary": "arxiv_ids 均无效"}

    reader = _reader()
    papers: list[dict[str, Any]] = []
    try:
        for a in normalized:
            entry: dict[str, Any] = {"arxiv_id": a}
            try:
                entry["brief"] = reader.brief(a)
                if tier == "brief_and_head":
                    entry["head"] = reader.head(a)
            except NotFoundError:
                entry["error"] = "未找到该论文"
            except APIError as e:
                entry["error"] = str(e)
            papers.append(entry)
        return {
            "success": True,
            "summary": f"已打包 {len(papers)} 篇文献素材（{tier}），请据此用中文组织对比或综述。",
            "tier": tier,
            "papers": papers,
            "hint": "你应基于下列结构化结果自行撰写最终回答；勿编造未出现在素材中的具体数值或结论。",
        }
    except (AuthenticationError, RateLimitError, ServerError, APIError) as e:
        return _dxv_err(e)
    except Exception as e:
        return _dxv_err(e)


DEEPXIV_TOOL_FUNCTIONS: dict[str, Any] = {
    "deepxiv_search": tool_deepxiv_search,
    "deepxiv_trending": tool_deepxiv_trending,
    "deepxiv_paper_brief": tool_deepxiv_paper_brief,
    "deepxiv_paper_head": tool_deepxiv_paper_head,
    "deepxiv_paper_section": tool_deepxiv_paper_section,
    "deepxiv_paper_fetch": tool_deepxiv_paper_fetch,
    "deepxiv_paper_social": tool_deepxiv_paper_social,
    "deepxiv_websearch": tool_deepxiv_websearch,
    "deepxiv_semantic_scholar": tool_deepxiv_semantic_scholar,
    "deepxiv_pmc_head": tool_deepxiv_pmc_head,
    "deepxiv_pmc_fetch": tool_deepxiv_pmc_fetch,
    "deepxiv_deep_dive": tool_deepxiv_deep_dive,
    "deepxiv_digest_pack": tool_deepxiv_digest_pack,
}
