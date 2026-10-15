"""Capability-based uncertainty policy. Labels are categorical, never calibrated CIs."""
from __future__ import annotations

import re
from typing import Any

POLICY_VERSION = "capability-policy-1"
SUPPORTED_PROPERTIES = frozenset({"tg", "band_gap", "atomization_energy", "heat_resistance"})
ALIASES = {
    "tg": r"\bt[_ -]?g\b|glass[ -]transition|玻璃化(?:转变)?温度|玻璃化温度",
    "band_gap": r"band[ -]?gap|\beg\b|带隙|禁带宽度",
    "atomization_energy": r"atomization|atomisation|原子化能",
    "heat_resistance": r"heat[ -]resistance\s*(?:class|classification)|耐热等级|热稳定性(?:分类|等级)|thermal[ -]stability\s*(?:class|classification)",
    "thermal_conductivity": r"thermal[ -]conductivity|热导率|导热系数",
    "melting_temperature": r"melting\s*(?:point|temperature)|\bt[_ -]?m\b|熔点|熔融温度",
    "decomposition_temperature": r"decomposition\s*temperature|\bt[_ -]?d\b|热分解温度|分解温度",
    "dielectric_constant": r"dielectric\s*(?:constant|permittivity)|介电常数",
    "breakdown_strength": r"breakdown\s*(?:strength|field)|击穿(?:场强|强度)",
    "youngs_modulus": r"young.?s\s*modulus|elastic\s*modulus|杨氏模量|弹性模量",
    "tensile_strength": r"tensile\s*strength|拉伸强度|抗拉强度",
    "solubility": r"solubility|溶解度",
    "gas_permeability": r"gas\s*permeability|气体渗透|透气率",
    "thermal_stability_unspecified": r"thermal[ -]stability|热稳定性",
}
_PREDICT = re.compile(r"\b(?:predict|estimate|calculate|forecast)\b|预测|估算|估计|计算|给.*(?:数值|值)|(?:give|provide).*(?:value|number)", re.I)
_STRUCTURE = re.compile(r"smiles|\*|repeat\s*unit|重复单元|结构式", re.I)
_VALUE_QUESTION = re.compile(r"what\s+(?:is|are).*(?:\bof\b|\bfor\b)|how\s+(?:large|high|much)|是多少|有多大|数值", re.I)
_DEFINITION = re.compile(r"\bdefine\b|definition|meaning\s+of|定义|含义|是什么概念", re.I)


def property_ids(text: str) -> list[str]:
    text = text.replace("_", " ")
    found = [key for key, pattern in ALIASES.items() if re.search(pattern, text, re.I)]
    if "heat_resistance" in found and "thermal_stability_unspecified" in found:
        found.remove("thermal_stability_unspecified")
    return found


def classify_request(text: str) -> dict[str, Any]:
    ids = property_ids(text)
    quantitative = bool(_PREDICT.search(text) or (ids and (_STRUCTURE.search(text) or (_VALUE_QUESTION.search(text) and not _DEFINITION.search(text)))))
    if quantitative and not ids:
        ids = ["unrecognized_property"]
    unsupported = [x for x in ids if x not in SUPPORTED_PROPERTIES]
    return {
        "policy_version": POLICY_VERSION,
        "quantitative_request": quantitative,
        "requested_properties": ids,
        "polyopus_supported_properties": sorted(SUPPORTED_PROPERTIES),
        "unsupported_properties": unsupported,
        "specialist_scope": "out_of_scope" if quantitative and unsupported else ("in_scope" if quantitative else "not_assessed"),
        "calibrated": False,
        "limitations": ["Property coverage is not structural applicability.",
                        "Heat-resistance classes are Tg-derived polyimide categories.",
                        "No numerical confidence interval is supplied by this policy."],
    }


def request_text(messages: list[dict[str, Any]]) -> str:
    user = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    if not user:
        return ""
    latest = user[-1]
    # Short followups may refer to a previous prediction. A new explicit property wins.
    if re.fullmatch(r"\s*(?:continue|yes|go ahead|继续|好|可以|开始|按默认)[.!。！\s]*", latest, re.I) and len(user)>1:
        return user[-2] + "\n" + latest
    return latest


def _covered_properties(tool: str, payload: dict[str, Any]) -> set[str]:
    if not payload.get("success") or payload.get("error"):
        return set()
    result = payload.get("result") or {}
    if tool == "polyopus_chat":
        return set(result.get("property_ids") or []) & SUPPORTED_PROPERTIES
    if tool == "predict_property":
        return set(property_ids(str(result.get("target_column") or "")))
    return set()


def assess_answer(text: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    scope = classify_request(text)
    covered: set[str] = set()
    sources: list[str] = []
    failures: list[str] = []
    for obs in observations:
        name = str(obs.get("tool") or "")
        payload = obs.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        props = _covered_properties(name, payload)
        if props:
            covered.update(props); sources.append(name)
        if name in {"polyopus_chat", "predict_property"} and payload.get("error"):
            failures.append(name)
    missing = sorted(set(scope["requested_properties"]) - covered)
    if not scope["quantitative_request"]:
        level, basis, fallback = "not_assessed", "nonquantitative_request", False
    elif missing:
        level, basis, fallback = "high", "no_matching_successful_predictor", True
    else:
        level, basis, fallback = "unquantified", "matching_predictor_uncalibrated", False
    return {**scope, "level": level, "basis": basis, "llm_fallback": fallback,
            "covered_properties": sorted(covered), "uncovered_properties": missing,
            "tool_sources": sorted(set(sources)), "failed_prediction_tools": failures,
            "specialist_extrapolation_refused": bool(scope["quantitative_request"] and scope["unsupported_properties"])}


def system_policy(text: str) -> str:
    scope = classify_request(text)
    if not scope["quantitative_request"]:
        return ""
    return (
        "[Capability and uncertainty policy]\n"
        f"Requested properties: {', '.join(scope['requested_properties'])}. "
        f"PolyOpus does not cover: {', '.join(scope['unsupported_properties']) or 'none of the detected properties'}. "
        "Use a successful tool matching the requested target when available. "
        "Do not ask PolyOpus to extrapolate to unsupported targets or substitute Tg heat-resistance classes for decomposition stability. "
        "When no matching predictor succeeds, you may give general-LLM heuristic estimates only with high-uncertainty and unvalidated labels. "
        "State missing experimental conditions. Do not invent calibrated probabilities or confidence intervals. "
        "A tool's availability does not establish structural applicability."
    )


def format_answer(answer: str, assessment: dict[str, Any], locale: str = "zh") -> str:
    if not assessment["quantitative_request"]:
        return answer
    if assessment["level"] == "high":
        labels = {
            "tg": ("glass-transition temperature", "玻璃化转变温度"),
            "band_gap": ("band gap", "带隙"),
            "atomization_energy": ("atomization energy", "原子化能"),
            "heat_resistance": ("heat-resistance class", "耐热等级"),
            "thermal_conductivity": ("thermal conductivity", "热导率"),
            "melting_temperature": ("melting temperature", "熔融温度"),
            "decomposition_temperature": ("decomposition temperature", "热分解温度"),
            "dielectric_constant": ("dielectric constant", "介电常数"),
            "breakdown_strength": ("breakdown strength", "击穿强度"),
            "youngs_modulus": ("Young's modulus", "杨氏模量"),
            "tensile_strength": ("tensile strength", "拉伸强度"),
            "thermal_stability_unspecified": ("thermal stability", "热稳定性"),
            "unrecognized_property": ("the requested property", "所请求的性质"),
        }
        targets = ", ".join(labels.get(p, (p.replace('_',' '),p.replace('_',' ')))[0 if locale == 'en' else 1] for p in assessment["uncovered_properties"])
        notice = (
            f"**High uncertainty — unvalidated LLM fallback.** No successful predictor matching {targets} was used. "
            "Any numerical estimate below is heuristic; no calibrated confidence interval is available."
            if locale == "en" else
            f"**高不确定性：通用 LLM 启发式估计，未经验证。** 本次未取得覆盖 {targets} 的有效预测工具结果。"
            "下文如包含数值，仅为启发式估计；当前没有经过校准的置信区间。"
        )
    else:
        notice = (
            "**Tool-based prediction; uncertainty unquantified.** A matching predictor returned a result, "
            "but structural applicability and numerical confidence intervals have not been calibrated."
            if locale == "en" else
            "**工具预测，不确定性尚未定量校准。** 已取得对应性质的工具结果，但结构适用域与数值置信区间尚未校准。"
        )
    return notice + "\n\n" + answer
