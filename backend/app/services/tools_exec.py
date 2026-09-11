"""Executable tools for the LLM agent."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .. import training_status as ts
from ..state import state
from ..training_status import TrainingCancelled
from . import dataset as dataset_mod
from .automl import PIPELINE_FILENAME, run_automl
from .polytao import finetune_polytao, generate_smiles
from .property_inference import (
    predict_property,
    read_latest_property_artifact,
    write_latest_property_artifact,
)
from .deepxiv_exec import DEEPXIV_TOOL_FUNCTIONS
from .memory_store import (
    memory_append,
    memory_delete,
    memory_read,
    run_memory_reflect,
    memory_update,
)
from .plan_store import plan_clear, plan_create, plan_read, plan_set_step_status
from .skill_store import (
    skill_create,
    skill_delete,
    skill_list_json,
    skill_read_full,
    skill_update,
)

logger = logging.getLogger(__name__)

UPLOAD_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"

_MAX_TRAINING_STATS_IN_RESPONSE = 24


def _ensure_dirs() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _upload_root_resolved() -> Path:
    _ensure_dirs()
    return UPLOAD_ROOT.resolve()


def _output_root_resolved() -> Path:
    _ensure_dirs()
    return OUTPUT_ROOT.resolve()


def _err(message: str) -> dict[str, Any]:
    """User-facing error (no stack trace in payload). Log with warning for validation errors."""
    logger.warning("Tool rejected: %s", message)
    return {
        "error": message,
        "summary": message,
    }


def tool_polyopus_chat(args: dict[str, Any]) -> dict[str, Any]:
    """Call the optional OpenAI-compatible PolyOpus natural-language endpoint."""
    prompt = args.get("prompt") or args.get("query")
    if prompt is None or not str(prompt).strip():
        return _err("PolyOpus 调用缺少自然语言 prompt")
    language = str(args.get("language") or "zh").strip().lower()
    if language not in {"zh", "en"}:
        language = "zh"

    with state.lock:
        if not state.llm.polyopus_configured():
            return _err(
                "PolyOpus 尚未配置。请设置 POLYOPUS_BASE_URL（可选再设置 "
                "POLYOPUS_MODEL、POLYOPUS_API_KEY、POLYOPUS_TIMEOUT_S）；当前不会上传权重。"
            )
        base_url = state.llm.polyopus_base_url.rstrip("/")
        model = state.llm.polyopus_model
        api_key = state.llm.polyopus_api_key_effective()
        timeout_s = state.llm.polyopus_timeout_s

    language_rule = (
        "Answer in English. Translate explanations and knowledge-base content into English; keep SMILES, URLs, code, and proper nouns unchanged."
        if language == "en"
        else "请使用中文回答；解释和知识库内容使用中文，SMILES、URL、代码和专有名词可保持原样。"
    )
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 PolyOpus，一个面向聚合物科学的专用语言模型，接收自然语言任务。"
                        + language_rule
                    ),
                },
                {"role": "user", "content": str(prompt).strip()},
            ],
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return _err("PolyOpus 返回了空响应")
        return {
            "success": True,
            "summary": text,
            "result": {"text": text, "model": model},
        }
    except Exception as e:
        logger.exception("PolyOpus request failed")
        return _err(f"PolyOpus 调用失败：{e!s}")



def _require_csv_under_uploads(csv_path: str | None) -> Path | dict[str, Any]:
    if not csv_path or not str(csv_path).strip():
        return _err("缺少 csv_path")
    raw = Path(str(csv_path).strip()).expanduser()
    try:
        full = raw.resolve()
    except (OSError, RuntimeError) as e:
        logger.exception("CSV path resolve failed: %s", csv_path)
        return _err(f"路径无效: {csv_path}（{e!s}）")
    root = _upload_root_resolved()
    if not full.is_relative_to(root):
        return _err(
            f"拒绝访问：数据文件仅允许位于上传目录内（{root}），请勿使用路径穿越或访问其他目录。"
        )
    if not full.is_file():
        return _err(f"文件不存在或不是文件: {full.name}")
    return full


def _require_output_dir(path_str: str | None, *, default_name: str) -> Path | dict[str, Any]:
    root = _output_root_resolved()
    if not path_str or not str(path_str).strip():
        out = (root / default_name).resolve()
    else:
        raw = Path(str(path_str).strip()).expanduser()
        try:
            out = raw.resolve()
        except (OSError, RuntimeError) as e:
            logger.exception("Output path resolve failed: %s", path_str)
            return _err(f"输出路径无效: {path_str}（{e!s}）")
        if not out.is_relative_to(root):
            return _err(
                f"拒绝访问：输出目录必须位于 {root} 下（可省略 output_dir 使用默认子目录）。"
            )
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.exception("mkdir failed: %s", out)
        return _err(f"无法创建输出目录: {out}（{e!s}）")
    return out


def _resolve_property_artifact_dir(artifact_dir: str | None) -> Path | dict[str, Any]:
    """性质预测 bundle 目录：须位于 outputs；省略则使用最近一次 train 写入的 LATEST。"""
    root = _output_root_resolved()
    if artifact_dir and str(artifact_dir).strip():
        raw = Path(str(artifact_dir).strip()).expanduser()
        try:
            full = raw.resolve()
        except (OSError, RuntimeError) as e:
            logger.exception("artifact_dir resolve failed: %s", artifact_dir)
            return _err(f"artifact_dir 无效: {artifact_dir}（{e!s}）")
        if not full.is_relative_to(root):
            return _err(f"artifact_dir 必须位于服务端输出目录内：{root}")
        if not (full / PIPELINE_FILENAME).is_file():
            return _err("该目录下没有已保存的性质预测模型（缺少 pipeline.joblib）")
        return full
    latest = read_latest_property_artifact(root)
    if latest is None or not (latest / PIPELINE_FILENAME).is_file():
        return _err(
            "未指定 artifact_dir，且尚未通过 train_property_model 保存过模型。"
            "请先完成训练，或传入训练结果中的 artifact_dir。"
        )
    if not latest.resolve().is_relative_to(root):
        return _err("内部错误：LATEST 指针不在允许的输出目录内")
    return latest.resolve()


def _parse_tabular_features(raw: Any) -> dict[str, float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            return None
    return out


def _require_model_dir_under_outputs(model_path: str | None) -> Path | dict[str, Any]:
    if not model_path or not str(model_path).strip():
        return _err("缺少 model_path")
    raw = Path(str(model_path).strip()).expanduser()
    try:
        full = raw.resolve()
    except (OSError, RuntimeError) as e:
        logger.exception("Model path resolve failed: %s", model_path)
        return _err(f"模型路径无效: {model_path}（{e!s}）")
    root = _output_root_resolved()
    if not full.is_relative_to(root):
        return _err(
            f"拒绝访问：模型目录必须位于输出目录内（{root}），通常为 PolyTAO 微调保存路径。"
        )
    if not full.is_dir():
        return _err(f"模型路径不是有效目录: {full.name}")
    return full


def _compact_polytao_result(r: dict[str, Any]) -> dict[str, Any]:
    stats = r.get("training_stats")
    if not isinstance(stats, list) or len(stats) <= _MAX_TRAINING_STATS_IN_RESPONSE:
        return r
    omitted = len(stats) - _MAX_TRAINING_STATS_IN_RESPONSE
    tail = stats[-_MAX_TRAINING_STATS_IN_RESPONSE :]
    out = {**r, "training_stats": tail, "training_stats_omitted": omitted}
    return out


def _summary_train(result: dict[str, Any]) -> str:
    mode = result.get("mode", "")
    if result.get("task") == "regression":
        base = (
            f"性质预测（回归）完成，模式 {mode}："
            f"test R²={result.get('test_r2')!s}，RMSE={result.get('test_rmse')!s}，"
            f"Optuna CV R²≈{result.get('optuna_best_r2_cv')!s}。"
        )
    else:
        base = (
            f"性质预测（分类）完成，模式 {mode}："
            f"准确率={result.get('test_accuracy')!s}，macro-F1={result.get('test_f1_macro')!s}，"
            f"Optuna CV macro-F1≈{result.get('optuna_best_f1_cv')!s}。"
        )
    ad = result.get("artifact_dir")
    if ad:
        base += (
            f" 已保存可推理模型，目录 artifact_dir={ad}。"
            "对新聚合物结构请调用 predict_property（可省略 artifact_dir 使用本次模型），不要再次 train。"
        )
    return base


def _summary_polytao_finetune(r: dict[str, Any], *, truncated: bool) -> str:
    base = (
        f"PolyTAO 微调完成：输出 {r.get('output_dir')!s}，epochs={r.get('epochs')!s}，"
        f"样本数 train/val={r.get('n_train')!s}/{r.get('n_val')!s}。"
    )
    if truncated:
        base += f" training_stats 已截断，仅保留最近 {_MAX_TRAINING_STATS_IN_RESPONSE} 条。"
    return base


def _summary_inverse(r: dict[str, Any]) -> str:
    seqs = r.get("sequences") or []
    n = len(seqs) if isinstance(seqs, list) else 0
    return f"逆向设计完成：共 {n} 条候选 SMILES（详见 result.sequences）。"


def tool_analyze_dataset(args: dict[str, Any]) -> dict[str, Any]:
    raw_path = args.get("csv_path") or args.get("path")
    csv_path = str(raw_path).strip() if raw_path is not None else None
    checked = _require_csv_under_uploads(csv_path)
    if isinstance(checked, dict):
        return checked
    p = checked
    target = args.get("target_column")
    try:
        out = dataset_mod.analyze_csv(p, target_column=target)
        if isinstance(out, dict) and "summary" not in out:
            out = {
                **out,
                "summary": (
                    f"分析完成：{out.get('n_rows')} 行，目标列 `{out.get('target_column')}`，"
                    f"建议模式 {out.get('recommended_mode')}。"
                ),
            }
        return out
    except Exception as e:
        logger.exception("analyze_dataset failed")
        return _err(f"分析失败：{e!s}")


def tool_train_property_model(args: dict[str, Any]) -> dict[str, Any]:
    csv_path = args.get("csv_path")
    target = args.get("target_column")
    mode = args.get("mode", "auto")
    smiles_column = args.get("smiles_column")

    if not target:
        return _err("需要 target_column")

    csv_s = str(csv_path).strip() if csv_path is not None else None
    checked = _require_csv_under_uploads(csv_s)
    if isinstance(checked, dict):
        return checked
    p = checked

    if mode == "auto":
        a = dataset_mod.analyze_csv(p, target_column=target)
        mode = a["recommended_mode"]
        if mode == "tabular_plus_smiles_fp":
            if not smiles_column and a["detected_smiles_columns"]:
                smiles_column = a["detected_smiles_columns"][0]
        if mode == "rdkit_descriptors" and not smiles_column and a["detected_smiles_columns"]:
            smiles_column = a["detected_smiles_columns"][0]

    try:
        if mode == "deepchem_gnn" and not smiles_column:
            a = dataset_mod.analyze_csv(p, target_column=target)
            smiles_column = (a.get("detected_smiles_columns") or [None])[0]
        trials = int(args.get("n_optuna_trials", 12))
        _ensure_dirs()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact_dir = OUTPUT_ROOT / "property_models" / f"train_{stamp}"
        result = run_automl(
            p,
            target,
            mode=mode,  # type: ignore[arg-type]
            smiles_column=smiles_column,
            n_optuna_trials=trials,
            artifact_dir=artifact_dir,
        )
        if result.get("artifact_dir"):
            write_latest_property_artifact(OUTPUT_ROOT, Path(result["artifact_dir"]))
        ts.finish_ok("性质预测训练完成")
        return {
            "success": True,
            "summary": _summary_train(result),
            "result": result,
        }
    except TrainingCancelled:
        return {
            "error": "训练已由用户中止",
            "summary": "性质预测训练已中止，未完成保存。",
        }
    except Exception as e:
        ts.finish_error(str(e))
        logger.exception("train_property_model failed")
        return _err(f"训练失败：{e!s}")


def tool_polytao_finetune(args: dict[str, Any]) -> dict[str, Any]:
    raw_csv = args.get("csv_path")
    csv_path = str(raw_csv).strip() if raw_csv is not None else None
    checked_csv = _require_csv_under_uploads(csv_path)
    if isinstance(checked_csv, dict):
        return checked_csv
    p = checked_csv

    out_raw = args.get("output_dir")
    out_dir = _require_output_dir(
        str(out_raw).strip() if out_raw is not None else None,
        default_name="polytao_finetune",
    )
    if isinstance(out_dir, dict):
        return out_dir
    out = str(out_dir)

    try:
        epochs = int(args.get("epochs", 10))
        ts.start_polytao(epochs)
        r = finetune_polytao(
            p,
            out,
            base_model_id=args.get("base_model_id", "hkqiu/PolymerGenerationPretrainedModel"),
            epochs=epochs,
            batch_size=int(args.get("batch_size", 16)),
            learning_rate=float(args.get("learning_rate", 2e-5)),
        )
        ts.finish_ok("PolyTAO 微调完成")
        compact = _compact_polytao_result(r)
        truncated = compact is not r or compact.get("training_stats_omitted", 0) > 0
        return {
            "success": True,
            "summary": _summary_polytao_finetune(compact, truncated=truncated),
            "result": compact,
        }
    except TrainingCancelled:
        return {
            "error": "微调已由用户中止",
            "summary": "PolyTAO 微调已中止。",
        }
    except Exception as e:
        ts.finish_error(str(e))
        logger.exception("polytao_finetune failed")
        return _err(f"微调失败：{e!s}")


def tool_polytao_inverse_design(args: dict[str, Any]) -> dict[str, Any]:
    model_path = args.get("model_path")
    prompt = args.get("prompt")
    if prompt is None:
        return _err("需要 prompt（性质条件字符串或 PolyTAO 15 维逗号分隔）")
    mp_check = _require_model_dir_under_outputs(
        str(model_path).strip() if model_path is not None else None
    )
    if isinstance(mp_check, dict):
        return mp_check
    mp = mp_check
    try:
        r = generate_smiles(
            mp,
            str(prompt),
            num_return_sequences=int(args.get("num_sequences", 5)),
        )
        return {
            "success": True,
            "summary": _summary_inverse(r),
            "result": r,
        }
    except Exception as e:
        logger.exception("polytao_inverse_design failed")
        return _err(f"逆向设计失败：{e!s}")


def tool_memory_read(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode") or "summary").strip().lower()
    if mode not in ("summary", "full"):
        return _err("mode 须为 summary 或 full")
    query = args.get("query")
    q = str(query).strip() if query is not None else None
    return memory_read(mode=mode, query=q or None)


def tool_memory_append(args: dict[str, Any]) -> dict[str, Any]:
    body = args.get("body")
    if body is None or not str(body).strip():
        return _err("需要非空 body")
    tags = args.get("tags")
    tag_list: list[str] | None = None
    if isinstance(tags, list):
        tag_list = [str(t).strip() for t in tags if str(t).strip()]
    title = args.get("title")
    t = str(title).strip() if title is not None else None
    return memory_append(str(body), tags=tag_list, title=t or None)


def tool_memory_update(args: dict[str, Any]) -> dict[str, Any]:
    eid = args.get("entry_id") or args.get("id")
    body = args.get("body")
    if not eid or not str(eid).strip():
        return _err("需要 entry_id")
    if body is None or not str(body).strip():
        return _err("需要非空 body")
    return memory_update(str(eid).strip(), str(body))


def tool_memory_delete(args: dict[str, Any]) -> dict[str, Any]:
    eid = args.get("entry_id") or args.get("id")
    if not eid or not str(eid).strip():
        return _err("需要 entry_id")
    return memory_delete(str(eid).strip())


def tool_memory_reflect(args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    return run_memory_reflect()


def tool_plan_create(args: dict[str, Any]) -> dict[str, Any]:
    raw_steps = args.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return _err("steps 须为非空数组，每项为一步的简短描述")
    title = args.get("title")
    t = str(title).strip() if title is not None else None
    return plan_create(title=t or None, steps=[str(x) for x in raw_steps])


def tool_plan_read(args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    return plan_read()


def tool_plan_set_step_status(args: dict[str, Any]) -> dict[str, Any]:
    status = args.get("status")
    if not status:
        return _err("需要 status")
    si = args.get("step_index")
    sid = args.get("step_id")
    idx: int | None = None
    if si is not None and str(si).strip() != "":
        try:
            idx = int(si)
        except (TypeError, ValueError):
            return _err("step_index 须为整数")
    sid_s = str(sid).strip() if sid is not None and str(sid).strip() else None
    if idx is None and not sid_s:
        return _err("需要提供 step_index 或 step_id 之一")
    return plan_set_step_status(
        status=str(status),
        step_index=idx,
        step_id=sid_s,
    )


def tool_plan_clear(args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    return plan_clear()


def tool_skill_list(args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    return skill_list_json()


def tool_skill_read(args: dict[str, Any]) -> dict[str, Any]:
    sid = args.get("skill_id") or args.get("id")
    if not sid:
        return _err("需要 skill_id")
    return skill_read_full(str(sid).strip())


def tool_skill_create(args: dict[str, Any]) -> dict[str, Any]:
    title = args.get("title")
    body = args.get("body")
    if not title or not str(title).strip():
        return _err("需要 title")
    if body is None or not str(body).strip():
        return _err("需要 body")
    triggers = args.get("triggers")
    tr = str(triggers).strip() if triggers is not None else ""
    tags = args.get("tags")
    tag_list: list[str] | None = None
    if isinstance(tags, list):
        tag_list = [str(t).strip() for t in tags if str(t).strip()]
    sid = args.get("skill_id")
    sid_s = str(sid).strip() if sid is not None and str(sid).strip() else None
    st = args.get("status")
    status_s = str(st).strip() if st is not None else "active"
    return skill_create(
        title=str(title).strip(),
        body=str(body),
        triggers=tr,
        tags=tag_list,
        skill_id=sid_s,
        status=status_s,
    )


def tool_skill_update(args: dict[str, Any]) -> dict[str, Any]:
    sid = args.get("skill_id") or args.get("id")
    if not sid or not str(sid).strip():
        return _err("需要 skill_id")
    title = args.get("title")
    triggers = args.get("triggers")
    tags = args.get("tags")
    body = args.get("body")
    append = args.get("append_to_body")
    st = args.get("status")
    tag_list: list[str] | None = None
    if isinstance(tags, list):
        tag_list = [str(t).strip() for t in tags if str(t).strip()]
    return skill_update(
        str(sid).strip(),
        title=str(title).strip() if title is not None and str(title).strip() else None,
        triggers=str(triggers).strip() if triggers is not None and str(triggers).strip() else None,
        tags=tag_list,
        body=str(body) if body is not None and str(body).strip() else None,
        append_to_body=str(append) if append is not None and str(append).strip() else None,
        status=str(st).strip() if st is not None and str(st).strip() else None,
    )


def tool_skill_delete(args: dict[str, Any]) -> dict[str, Any]:
    sid = args.get("skill_id") or args.get("id")
    if not sid or not str(sid).strip():
        return _err("需要 skill_id")
    return skill_delete(str(sid).strip())


def tool_predict_property(args: dict[str, Any]) -> dict[str, Any]:
    raw_ad = args.get("artifact_dir")
    ad_check = _resolve_property_artifact_dir(
        str(raw_ad).strip() if raw_ad is not None else None
    )
    if isinstance(ad_check, dict):
        return ad_check
    ad = ad_check

    smiles = args.get("smiles")
    smiles_s = str(smiles).strip() if smiles is not None else None
    if smiles_s == "":
        smiles_s = None

    tabular = _parse_tabular_features(args.get("tabular_features"))

    try:
        r = predict_property(ad, smiles=smiles_s, tabular_features=tabular)
        task = r.get("task", "")
        if task == "regression":
            summ = (
                f"预测 {r.get('target_column')} ≈ {r.get('prediction')!s} "
                f"（模式 {r.get('mode')}）。"
            )
        else:
            summ = (
                f"预测类别：{r.get('prediction_label')!s} "
                f"（目标列 {r.get('target_column')}，模式 {r.get('mode')}）。"
            )
        return {"success": True, "summary": summ, "result": r}
    except Exception as e:
        logger.exception("predict_property failed")
        return _err(f"预测失败：{e!s}")


TOOL_FUNCTIONS = {
    "polyopus_chat": tool_polyopus_chat,
    "analyze_dataset": tool_analyze_dataset,
    "train_property_model": tool_train_property_model,
    "predict_property": tool_predict_property,
    "polytao_finetune": tool_polytao_finetune,
    "polytao_inverse_design": tool_polytao_inverse_design,
    "memory_read": tool_memory_read,
    "memory_append": tool_memory_append,
    "memory_update": tool_memory_update,
    "memory_delete": tool_memory_delete,
    "memory_reflect": tool_memory_reflect,
    "plan_create": tool_plan_create,
    "plan_read": tool_plan_read,
    "plan_set_step_status": tool_plan_set_step_status,
    "plan_clear": tool_plan_clear,
    "skill_list": tool_skill_list,
    "skill_read": tool_skill_read,
    "skill_create": tool_skill_create,
    "skill_update": tool_skill_update,
    "skill_delete": tool_skill_delete,
    **DEEPXIV_TOOL_FUNCTIONS,
}


def dispatch_tool(name: str, arguments_json: str) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"参数 JSON 无效: {e}"}, ensure_ascii=False)
    out = fn(args)
    return json.dumps(out, ensure_ascii=False, default=str)
