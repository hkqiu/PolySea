"""PolySea FastAPI 入口：本地 Web 后端 + LLM Agent + 工具。"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services.agent import run_agent_turn
from .state import state

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="PolySea API", version="0.1.0")


@app.get("/")
def root():
    """避免浏览器直接访问 http://127.0.0.1:8000/ 时出现 404 误解。"""
    return {
        "service": "PolySea API",
        "health": "/api/health",
        "docs": "/docs",
        "note": "前端开发请用 Vite 的 5173 端口；本端口仅 API。",
    }

# 注意：allow_credentials=True 时不能使用 allow_origins="*"（浏览器会拒绝），故默认关闭 credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMConfigBody(BaseModel):
    base_url: str | None = Field(default=None, description="OpenAI 兼容 API 根路径，如 http://host/v1")
    model: str | None = None
    api_key: str | None = None
    timeout_s: float | None = None


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "polysea"}


@app.get("/api/training/status")
def training_status():
    """长任务训练进度（性质预测 / PolyTAO 微调），供前端轮询。"""
    from .training_status import snapshot

    return snapshot()


@app.post("/api/training/cancel")
def training_cancel():
    """请求中止：训练在下一 Optuna trial / PolyTAO epoch 边界停止；对话在下一轮 Agent 循环前结束。"""
    from .training_status import request_cancel

    request_cancel()
    return {"ok": True}


@app.get("/api/settings")
def get_settings():
    with state.lock:
        return {
            "base_url": state.llm.base_url,
            "model": state.llm.model,
            "api_key_set": bool(state.llm.api_key_effective()),
            "timeout_s": state.llm.timeout_s,
            "polyopus_configured": state.llm.polyopus_configured(),
            "polyopus_model": state.llm.polyopus_model if state.llm.polyopus_configured() else None,        }


@app.post("/api/settings")
def post_settings(body: LLMConfigBody):
    with state.lock:
        if body.base_url is not None:
            state.llm.base_url = body.base_url.strip()
        if body.model is not None:
            state.llm.model = body.model.strip()
        if body.api_key is not None:
            state.llm.api_key = body.api_key.strip()
        if body.timeout_s is not None:
            state.llm.timeout_s = float(body.timeout_s)
    return {"ok": True}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    locale: str = "zh"


@app.post("/api/chat")
def chat(req: ChatRequest):
    msgs = [m.model_dump() for m in req.messages]
    locale = req.locale if req.locale in ("zh", "en") else "zh"
    try:
        text, _debug = run_agent_turn(msgs, locale=locale)
    except Exception as e:
        if locale == "en":
            error_reply = (
                f"Backend processing error: {e!s}\n\n"
                "If your gateway does not support function calling / tools, use the latest backend "
                "(automatic fallback to plain chat is supported). Also verify the Base URL, Model, and API Key."
            )
        else:
            error_reply = (
                f"后端处理出错：{e!s}\n\n"
                "若网关不支持「函数调用 / tools」，请拉取最新后端（已支持自动降级为纯对话）。"
                "并请确认 Base URL、Model 与 API Key 正确。"
            )
        return JSONResponse(
            status_code=200,
            content={"reply": error_reply, "error": True},
        )
    return {"reply": text}


UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "data" / "uploads"


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        raise HTTPException(400, "无文件名")
    safe = Path(file.filename).name
    dest = UPLOAD_ROOT / safe
    content = await file.read()
    dest.write_bytes(content)
    return {
        "filename": safe,
        "path": str(dest.resolve()),
        "size_bytes": len(content),
    }


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
