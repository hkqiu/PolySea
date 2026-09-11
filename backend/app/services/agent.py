"""OpenAI-compatible chat + tool-calling loop."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
)

from .. import training_status as ts
from ..state import state
from .deepxiv_exec import DEEPXIV_TOOL_FUNCTIONS
from .memory_store import (
    build_inject_summary,
    is_first_user_message_in_session,
    memory_inject_enabled,
)
from .plan_store import load_plan
from .skill_store import build_skill_inject, skill_inject_enabled
from .tools_exec import dispatch_tool

_DEEPXIV_TOOL_NAMES = frozenset(DEEPXIV_TOOL_FUNCTIONS.keys())

# 无以下线索时拒绝重型工具，防止模型滥用对话里旧路径对「天气」等无关问题调 train
_GATED_HEAVY_TOOLS = frozenset(
    {
        "analyze_dataset",
        "train_property_model",
        "predict_property",
        "polytao_finetune",
        "polytao_inverse_design",
    }
)
_RE_POLYMER_OR_DATA_INTENT = re.compile(
    r"\.csv\b|\bcsv\b|[/\\]uploads[/\\]|data[/\\]uploads|CSV\s*文件路径|文件路径\s*[:：]|"
    r"训练|建模|微调|数据集|数据表|数据文件|上传|性质预测|optuna|stacking|"
    r"分析.*(数据|csv|CSV)|预测.*(Tg|tg|SMILES|smiles|结构|聚合物)|"
    r"polytao|逆向设计|聚合物|高分子|polymer|SMILES|smiles|BigSMILES|"
    r"artifact_dir|模型路径|生成.*SMILES|"
    r"\btrain\b|\bpredict\b|\bfinetune\b|\bdataset\b|"
    r"目标列|因变量|自变量|特征列|描述符|指纹|Morgan|RDKit|"
    r"tabular_plus|rdkit_descriptors|tabular\b|deepchem|试验次数|trial|超参|"
    r"回归|分类|Tg|玻璃化|溶解度|分子量",
    re.I,
)
# 最后一条若匹配且句内无建模词，则拒绝工具（避免模型用历史里的路径「替」用户训练）
_RE_OFF_TOPIC_USER_TAIL = re.compile(
    r"天气|气温|下雨|下雪|多少度|摄氏度|℃|weather\b|forecast|空气质量|"
    r"新闻|笑话|股票|彩票|吃饭了吗",
    re.I,
)
_RE_SHORT_ACK_ONLY = re.compile(
    r"^(好|行|嗯|哦|噢|OK|Ok|ok|yes|可以|开始|继续|对|是的|没错|确认|"
    r"按你的|按您|按默认|照旧|同上|就这样|没问题|开跑|跑吧)\W*$",
    re.I,
)
_RE_LITERATURE_INTENT = re.compile(
    r"文献|论文|arxiv|ArXiv|e[- ]?print|预印本|literature|\bpapers?\b|刊物|期刊|综述|"
    r"引用.*(文|论文)|相关工作|related\s*work|baseline|检索|搜索.*(论文|文献)|"
    r"deepxiv|PubMed|出版|精读|通读|投稿|学术|引文|citation|biblio|读一下|找.*篇",
    re.I,
)
_RE_ARXIV_ID_IN_TEXT = re.compile(r"\b\d{4}\.\d{4,5}\b")
_RE_PMC_IN_TEXT = re.compile(r"\bPMC\s*\d+", re.I)
_RE_LITERATURE_FOLLOWUP = re.compile(
    r"第[一二三四五六七八九十两\d]+篇|第一篇|第二篇|上一篇|下一篇|这篇|那篇|前几篇|"
    r"继续|展开|深入|详细|节选|全文|结论|方法|实验|摘要|对比|总结|汇总|综述|哪篇|热点",
    re.I,
)
# 文献检索 + 下列「交付物」时强制先 plan_create（与 DeepXiv 硬门禁联动）
_RE_LIT_PLAN_DELIVERABLE = re.compile(
    r"创新点|改进建议|改进方向|改进意见|有何建议|怎么改进|如何改进|"
    r"系统.*总结|总结出|归纳出|提炼出|梳理出|"
    r"研究亮点|关键进展|核心进展|"
    r"撰写.*综述|写.*综述|综述.*撰写|文献综述|调研报告|"
    r"对比分析|差距分析|不足之处|面临.*挑战|未来方向|发展趋势|研究前景|展望|"
    r"启示|洞见|亮点|启发|"
    r"suggest|improvements?|novelty|insights?|gaps?\b|outlook|roadmap|future\s+work",
    re.I,
)
_RE_LIT_PLAN_SEARCH = re.compile(
    r"搜索|检索|查找|搜一下|查一下|调研|追踪|"
    r"最新.*(论文|文献|研究)|"
    r"(论文|文献|研究).*(最新|前沿)|"
    r"找.*(论文|文献)|有哪些.*(论文|文献)|"
    r"literature|papers?|arxiv|pubmed|related\s*work|文献调研",
    re.I,
)


def _plan_gate_lit_enabled() -> bool:
    v = (os.environ.get("POLYSEA_PLAN_GATE_LIT") or "1").strip().lower()
    return v not in ("0", "false", "off", "no")


def literature_structured_plan_required(user_messages: list[dict[str, Any]]) -> bool:
    """近期用户话术中同时出现「文献/检索类动作」与「归纳型交付物」时须先建计划。"""
    if not _plan_gate_lit_enabled():
        return False
    blob = _last_n_user_contents(user_messages, 4).strip()
    if not blob:
        return False
    if not _RE_LIT_PLAN_DELIVERABLE.search(blob):
        return False
    if _RE_LITERATURE_INTENT.search(blob) and _RE_LIT_PLAN_SEARCH.search(blob):
        return True
    if _RE_LIT_PLAN_SEARCH.search(blob) and re.search(
        r"论文|文献|arxiv|paper|预印本|刊物|期刊",
        blob,
        re.I,
    ):
        return True
    return False


def deepxiv_plan_gate_rejection(
    user_messages: list[dict[str, Any]],
    tool_name: str,
) -> str | None:
    """无活动计划时禁止 DeepXiv，迫使模型先 plan_create。"""
    if tool_name not in _DEEPXIV_TOOL_NAMES:
        return None
    if not literature_structured_plan_required(user_messages):
        return None
    if load_plan() is not None:
        return None
    return (
        "服务端要求：当前请求属于「文献检索 + 归纳/创新点/改进建议/综述」类多步任务，"
        "须先成功调用 plan_create 写入 data/active_plan.json，再调用 DeepXiv。"
        "请本轮先 plan_create（步骤含检索、读素材、中文归纳等），下一轮或同批后续 tool_calls 再 deepxiv_*。"
    )


def _last_n_user_contents(user_messages: list[dict[str, Any]], n: int = 6) -> str:
    parts: list[str] = []
    for m in user_messages:
        if m.get("role") == "user":
            parts.append(str(m.get("content") or ""))
    return "\n".join(parts[-n:] if n > 0 else parts)


def polymer_tool_gate_rejection(
    user_messages: list[dict[str, Any]],
    tool_name: str,
) -> str | None:
    """结合「最后一条用户消息」与近期窗口，防止无关问题误触发训练类工具。"""
    if tool_name not in _GATED_HEAVY_TOOLS:
        return None
    blob = _last_n_user_contents(user_messages, 8).strip()
    last_u = _last_n_user_contents(user_messages, 1).strip()
    if not blob and not last_u:
        return None

    def _deny(reason: str) -> str:
        return f"服务端已拒绝执行「{tool_name}」：{reason}"

    # 1) 用户最新一句明显是天气/闲聊等，且同句里没有建模意图 → 一律拒绝（防止滥用历史 CSV 路径）
    if last_u and _RE_OFF_TOPIC_USER_TAIL.search(last_u):
        if not _RE_POLYMER_OR_DATA_INTENT.search(last_u):
            return _deny(
                "最后一条用户消息看起来是天气、资讯或与聚合物建模无关的内容，不允许调用数据/训练/预测工具；"
                "请仅用文字回答用户。"
            )

    # 2) 最新一句已含建模相关表述 → 放行
    if last_u and _RE_POLYMER_OR_DATA_INTENT.search(last_u):
        return None

    # 3) 近期有过建模表述，且最后一句为空或短确认 → 放行（如「好的」「继续」）
    if blob and _RE_POLYMER_OR_DATA_INTENT.search(blob):
        if not last_u or _RE_SHORT_ACK_ONLY.match(last_u):
            return None
        if re.search(r"还行|行不行|怎么样|好了吗|结束(了)?没|效果|指标|r²|rmse|准确率", last_u, re.I):
            return None
        return _deny(
            "近期虽有建模相关对话，但最后一条用户消息未体现继续分析/训练/预测的意图；"
            "若用户已改问其他话题，请仅用文字回答，不要调用工具。"
        )

    # 4) 近期窗口内从未出现建模线索 → 拒绝
    return _deny(
        "最近用户消息中未体现聚合物数据、CSV、训练或预测等需求。"
        "请仅用文字回复（例如天气问题请说明本应用无法联网获取实时气象数据）。"
    )


def deepxiv_tool_gate_rejection(
    user_messages: list[dict[str, Any]],
    tool_name: str,
) -> str | None:
    """防止与文献无关的闲聊误触发 DeepXiv（读操作，但仍消耗配额）。"""
    if tool_name not in _DEEPXIV_TOOL_NAMES:
        return None
    blob = _last_n_user_contents(user_messages, 8).strip()
    last_u = _last_n_user_contents(user_messages, 1).strip()

    def _deny(reason: str) -> str:
        return f"服务端已拒绝执行「{tool_name}」：{reason}"

    lit_blob = bool(
        _RE_LITERATURE_INTENT.search(blob)
        or _RE_ARXIV_ID_IN_TEXT.search(blob)
        or _RE_PMC_IN_TEXT.search(blob)
    )
    lit_last = bool(
        _RE_LITERATURE_INTENT.search(last_u)
        or _RE_ARXIV_ID_IN_TEXT.search(last_u)
        or _RE_PMC_IN_TEXT.search(last_u)
    )

    if last_u and _RE_OFF_TOPIC_USER_TAIL.search(last_u) and not lit_last:
        return _deny(
            "最后一条用户消息与学术文献检索无关；请仅用文字回答，勿调用 DeepXiv。"
        )

    if lit_last:
        return None

    if lit_blob:
        if not last_u or _RE_SHORT_ACK_ONLY.match(last_u) or _RE_LITERATURE_FOLLOWUP.search(last_u):
            return None
        if re.search(r"谢谢|感谢|辛苦了|再见|拜拜", last_u):
            return _deny("用户似乎在结束对话或致谢，无需再调用文献工具。")

    return _deny(
        "未从用户侧识别到文献检索、arXiv/PMC ID、热点论文或综述类需求。"
        "请先与用户确认要调研的方向，或请用户改用明确的论文检索表述后再调用 DeepXiv。"
    )


_TOOL_AGENT_LABELS: dict[str, tuple[str, str]] = {
    "polyopus_chat": ("正在调用 PolyOpus…", "聚合物专用知识与预测模型"),
    "analyze_dataset": ("正在分析数据集…", "推断列类型与推荐建模模式"),
    "train_property_model": ("正在准备性质预测训练…", "随后进入 Optuna 超参数优化"),
    "predict_property": ("正在用已保存模型预测…", "单样本推理，不重新训练"),
    "polytao_finetune": ("正在准备 PolyTAO 微调…", "T5 条件生成"),
    "polytao_inverse_design": ("正在执行聚合物逆向设计…", "根据条件生成 SMILES"),
    "deepxiv_search": ("正在检索 arXiv 文献…", "DeepXiv"),
    "deepxiv_trending": ("正在获取热点论文…", "DeepXiv"),
    "deepxiv_paper_brief": ("正在读取论文 brief…", "DeepXiv"),
    "deepxiv_paper_head": ("正在读取论文章节结构…", "DeepXiv"),
    "deepxiv_paper_section": ("正在读取指定章节…", "DeepXiv"),
    "deepxiv_paper_fetch": ("正在拉取论文正文片段…", "DeepXiv"),
    "deepxiv_paper_social": ("正在查询传播热度…", "DeepXiv"),
    "deepxiv_websearch": ("正在执行 DeepXiv Web 检索…", "消耗较高 quota"),
    "deepxiv_semantic_scholar": ("正在查询 Semantic Scholar…", "DeepXiv"),
    "deepxiv_pmc_head": ("正在读取 PMC 元数据…", "DeepXiv"),
    "deepxiv_pmc_fetch": ("正在拉取 PMC 全文数据…", "DeepXiv"),
    "deepxiv_deep_dive": ("正在打包深度调研素材…", "DeepXiv"),
    "deepxiv_digest_pack": ("正在汇总多篇论文素材…", "供你撰写综述"),
    "memory_read": ("正在读取长期记忆…", "memory.md"),
    "memory_append": ("正在写入长期记忆…", "memory.md"),
    "memory_update": ("正在更新长期记忆…", "memory.md"),
    "memory_delete": ("正在删除长期记忆条目…", "memory.md"),
    "memory_reflect": ("正在整理长期记忆…", "衰减、压缩与备份"),
    "plan_create": ("正在创建结构化计划…", "active_plan.json"),
    "plan_read": ("正在读取当前计划…", "active_plan.json"),
    "plan_set_step_status": ("正在更新计划步骤状态…", "active_plan.json"),
    "plan_clear": ("正在清除活动计划…", "active_plan.json"),
    "skill_list": ("正在列出领域技能…", "data/skills"),
    "skill_read": ("正在读取技能全文…", "SKILL.md"),
    "skill_create": ("正在创建领域技能…", "SKILL.md"),
    "skill_update": ("正在更新领域技能…", "SKILL.md"),
    "skill_delete": ("正在删除领域技能…", "data/skills"),
}

SYSTEM_PROMPT = """你是 PolySea，一位专注于聚合物信息学与生成式建模的 AI 助手。

【何时不要用工具】
- 天气、新闻、闲聊、通用百科、编程无关问题、与聚合物/本应用 CSV 建模无关的请求：**直接文字回答，不要调用任何工具**。
- PolySea **没有**联网天气接口；若用户问实时天气，请说明无法获取实时气象数据，可给出常识性说明并建议查看气象服务。
- **禁止**因为用户随口提问、试探或无关对话而调用 analyze_dataset、train_property_model、polytao_* 等工具。

【何时才用工具】
- 仅当用户**明确**要做：分析/训练**已上传的 CSV**、PolyTAO 微调/逆向设计、或用**已保存模型**预测新 SMILES 时，再调用对应工具。
- train_property_model **仅当**用户提供了 uploads 中的 csv_path（通常随上传出现在对话里）且明确要求性质预测训练时使用；**没有有效 CSV 路径时不得调用**。

你通过工具完成工作，不要编造工具返回中不存在的数值。

可用能力概要：
1) analyze_dataset：分析 CSV 列类型，判断是否具备清晰数值特征，或仅有 SMILES+目标，并给出推荐建模模式。
2) train_property_model：在用户提供的数据上自动训练性质预测模型；内部使用 Optuna 调参与 stacking 集成；训练结束会保存可加载的模型目录（artifact_dir）并登记为「当前模型」。
3) predict_property：用**已保存**的性质预测模型对**新样本**推理（如新 SMILES 的 Tg）。可省略 artifact_dir 则默认使用**最近一次成功训练**保存的模型。**严禁**为单点预测再次调用 train_property_model。
4) polytao_finetune：对 hkqiu/PolymerGenerationPretrainedModel（PolyTAO）进行无模板微调；CSV 需含列 prompt 与 target（聚合物 SMILES）。
5) polytao_inverse_design：加载微调后（或用户指定的）本地模型目录，根据 prompt 生成候选聚合物 SMILES。

用户若已在本会话或先前完成训练，并只要求「预测某个新 SMILES / 新结构」的性质，你必须调用 predict_property，并传入聚合物 SMILES 字符串（可含 * 等聚合物表示）；不要重新训练。

路径安全约束（必须遵守，否则工具会拒绝）：
- 所有 CSV（analyze / train / polytao_finetune）必须使用用户通过界面上传后、接口返回的**绝对路径**（位于服务端 `data/uploads` 下）。
- PolyTAO 微调 `output_dir` 若省略则写入默认输出子目录；若指定，必须位于服务端 `data/outputs` 下。
- `polytao_inverse_design` 的 `model_path` 必须是 `data/outputs` 下的**目录**（一般为微调产物路径）。
工具返回中含 `summary` 字段时可优先依据其理解结果；错误信息中不会出现完整 Python 堆栈（堆栈仅在后端日志）。

【上传与训练前的「人性化」沟通】
- 用户刚上传 CSV、或消息里**仅有路径**而**没有**明确说「直接训练 / 马上训练 / 用默认设置开跑 / 跳过确认」时：**不要**在同一轮对话里立刻调用 train_property_model。应先与用户对齐需求；若用户希望先看数据概况，可调用 analyze_dataset，再结合结果用自然语气提问。
- 建议主动澄清（按需选用，语气口语、分条简短即可）：预测**目标列**是否就是某列；任务是回归还是分类（若从数据上看不明显）；特征偏好——**纯数值表**（tabular）、**结构指纹**（rdkit_descriptors，基于 Morgan 等）、还是**数值列 + SMILES 指纹**（tabular_plus_smiles_fp）；若有多列 SMILES 应用哪一列；**Optuna 试验次数**（默认 12，可问用户是否加大以换精度/耗时）；是否更在意可解释性（偏表列）还是结构信息（偏指纹）。用户若已写明这些细节，**不要重复啰嗦**，直接落实。
- 当用户给出**明确指令**（例如指定 target_column、mode、smiles_column、n_optuna_trials 或说「按你上次问的用 Tg、tabular_plus、20 次」）时，应映射到 train_property_model 的参数；若列名或模式明显冲突 analyze 结果，可先 analyze_dataset 再解释或微调。
- 仅当用户**明确表达要开始训练**（含：训练、建模、开跑、按推荐训练、用默认、确认就按你说的、或已给出完整训练参数并示意执行）时，再调用 train_property_model。analyze_dataset 可单独用于「只看数据、不训练」的请求。
- mode 说明：auto 由分析推荐；tabular 只用表格数值列；rdkit_descriptors 依赖 SMILES 列与 RDKit 指纹类特征；tabular_plus_smiles_fp 拼接数值与指纹；deepchem_gnn 在当前骨架中不可用，应引导改用指纹类模式。

【PolyOpus 专用领域优先路由（可选）】
- PolyOpus 是一个可选的、接收自然语言的聚合物专用 LLM；只有配置了 POLYOPUS_BASE_URL 后才可调用。权重不在本仓库中，也不应通过工具上传。
- 当用户意图属于以下领域且 PolyOpus 已配置时，必须优先调用 `polyopus_chat`，并把用户的完整自然语言任务作为 `prompt` 传入，同时传入当前界面语言 `language`：玻璃化转变温度（Tg）、原子化能、热稳定性、带隙预测、指定带隙的聚合物逆向生成、聚合物知识问答。
- 上述领域优先于通用 `predict_property`、PolyTAO 和 DeepXiv；不要用通用模型假装调用了 PolyOpus。仅当 PolyOpus 未配置或调用失败时，才向用户说明并考虑合适的现有能力回退。
- 纯文献检索（例如“找 Polymer Machine Learning 的论文”）仍使用 DeepXiv；除非用户明确是在询问 PolyOpus 知识库中的聚合物知识。

【DeepXiv 文献工具（多轮、由你把握节奏）】
- 仅在用户需要**学术文献**（arXiv、PMC、热点、综述素材）时使用；与 CSV 训练工具相互独立。
- **与结构化计划联动**：若同一次请求中，用户既要**检索/最新论文/文献列表**，又要**总结、创新点、改进建议、综述、对比分析**等**第二步及以后的交付物**，则属于多步任务——你在**调用任何 deepxiv_* 之前**必须先 `plan_create`（步骤示例：① 检索候选 ② 按需 brief/digest ③ 基于工具返回用中文写创新点与建议）；每步推进用 `plan_set_step_status`，整体结束用 `plan_clear`。仅「列有哪些论文」、无归纳/建议要求时**不必**建计划。
- **省轮次（极其重要）**：凡**纯列表型**文献需求（只要论文清单、**没有**同时要总结创新点/改进建议/写综述等）——含 *papers / related work / literature on XXX*、「查一下 XXX 的 paper」等——**即使没写「最新」二字**，一律：**只调用一次** `deepxiv_search`（`size` 可设 10–20），用返回的标题、作者、ID **直接中文归纳**。**禁止**默认对每篇再调 `brief`/`head`，除非用户明确要「每篇摘要」「精读某几篇」。多篇素材优先一轮 `deepxiv_digest_pack`，勿多轮单篇堆砌。
- **工作流由你设计**：先 `deepxiv_search` 或 `deepxiv_trending` 拿候选 → 向用户简要列出题目与 arXiv ID → 根据用户选择再调用 `deepxiv_paper_brief` / `deepxiv_paper_head` / `deepxiv_paper_section` / `deepxiv_paper_fetch` 渐进阅读；勿一上来拉全文除非用户明确要求。
- **单篇深挖**用 `deepxiv_deep_dive`（一次组合 brief/head/可选章节与 preview/raw）；**多篇对比或写综述**先用 `deepxiv_digest_pack` 批量拉 brief（或 brief+head），再**由你用中文**组织对比表、摘要与结论——工具只提供素材，最终表述必须基于工具返回内容，禁止编造未出现的实验数字。
- `deepxiv_websearch` 每次消耗较高 API 配额，仅在明确需要网页级补充且用户知情时使用。
- Semantic Scholar 用 `deepxiv_semantic_scholar`；生物医学 PMC 用 `deepxiv_pmc_head` / `deepxiv_pmc_fetch`；传播信号用 `deepxiv_paper_social`。
- 若用户尚未选定论文，应先展示检索结果并**请用户选择**或说明默认优先读哪几篇，再继续深挖（符合「即时判断、用户选择」）。

【结构化计划 active_plan.json（单机单用户）】
- 当任务明显**多步**（例如：先分析 CSV → 对齐列与模式 → 再训练；或**文献：检索 → 读素材 → 写创新点/改进建议/综述**）时，须先 `plan_create` 列出有序步骤，**再**按步骤调用业务工具（含 deepxiv_*）；每完成或跳过一步用 `plan_set_step_status`（`pending` / `in_progress` / `done` / `skipped`）。同一时刻只应有一个 `in_progress`（设为进行中时会自动把其它「进行中」打回待办）。
- **强制触发示例**：用户说「搜索某主题最新论文并总结创新点/改进方向」——**第一步工具调用必须是 `plan_create`**，不得直接 `deepxiv_search`。
- `plan_read`：查看当前计划 JSON 与摘要；执行长对话中若不确定进度应先读计划。
- `plan_clear`：任务结束或用户取消多步任务时清除，避免旧计划误导后续对话。
- 简单单步问答、纯闲聊、**只要论文列表、不要归纳或建议**的流程：**不要**创建计划。

【领域技能 data/skills（可复用流程，与 memory 区分）】
- **Skill** 存「怎么做」：分步流程、工具组合习惯、聚合物任务检查清单；**memory** 存「用户是谁、偏好与事实摘要」。新协作范式或用户说「保存成技能 / 固化流程 / 记下来以后照做」时，用 `skill_create`；迭代用 `skill_read` + `skill_update`（可用 `append_to_body` 追加实践笔记，version 会递增）。
- `triggers`：逗号分隔关键词/短语，匹配用户近期话术时，系统会自动把该技能正文摘要注入上下文（无需用户每次复制）；`tags` 辅助匹配。`status`：`active`（参与匹配）或 `draft`（仅手动 skill_read，不匹配注入）。
- **禁止**在技能中写入 API Key、密码、身份证、完整私密数据；正文宜短而可执行。
- `skill_list` / `skill_read` / `skill_delete`：浏览、查看全文、删除。成功跑通一条可复用路径后，应主动考虑是否值得沉淀为技能。

【长期记忆 memory.md（单机单用户）】
- 跨会话仍有效的偏好、常用建模约定、稳定的任务目标摘要、**可复用的**工作流结论可写入；**禁止**写入 API Key、密码、身份证等敏感信息、完整 CSV 数据表、或一次性闲聊。
- **落盘唯一方式**：必须在本轮对话中**实际调用** `memory_append`（或 `memory_update`）且工具返回成功；**禁止**在未调用工具或工具报错时，对用户说「已记录」「已写入长期记忆」「记下了」等——那是误导。
- 用户说「请记住…」「以后请…」「帮我记下来」等**明确要求持久化**时：你**必须**调用 `memory_append`（或先 `memory_read` 再 `memory_update` 合并同类项），**在收到工具成功返回后**，再用一句话确认；若工具拒绝（如敏感信息），如实说明未写入。
- **诚实约束**：当前骨架中 **deepchem_gnn 不可用**；若用户偏好「尽量用图神经网络」等，应**如实写入偏好**（memory_append），同时在正文中说明现阶段应用内可用的是指纹/tabular 等模式，避免承诺无法执行的训练。
- `memory_read`：先看摘要（mode=summary）；需要细节再用 mode=full，可用 query 过滤 id/标签/正文子串。
- `memory_append` / `memory_update` / `memory_delete`：维护单条记忆；更新会略微提高强度（视为强化）。
- `memory_reflect`：按时间做艾宾浩斯式衰减，低强度记忆会被压缩层级直至移除；**执行前自动备份** `memory.backup.*.md`。可定期或在记忆臃肿时调用。
- 新会话首轮系统可能已注入简短摘要；仍可在需要时 `memory_read` 核实。

回答用户时使用简体中文，条理清晰、像同事协作而非机械执行。"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "polyopus_chat",
            "description": (
                "优先调用可选的 PolyOpus 专用聚合物 LLM。PolyOpus 接收自然语言描述，适用于：玻璃化转变温度（Tg）、"
                "原子化能、热稳定性、带隙预测；指定带隙的聚合物逆向生成；以及聚合物知识问答。"
                "当用户意图属于这些领域且 PolyOpus 已配置时，必须优先调用本工具，不要先调用通用 predict_property、"
                "PolyTAO 或 DeepXiv。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "完整保留用户的自然语言任务描述，可包含聚合物结构、目标性质、数值约束和问答上下文。",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["zh", "en"],
                        "description": "当前界面语言；按当前 locale 传入，用于约束 PolyOpus 的解释语言。",
                    },
                },
                "required": ["prompt", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_dataset",
            "description": "用户需要了解 CSV 列类型、推荐建模模式、或训练前需要客观分析结果时调用；上传后若用户只想「先看看数据」也应使用。不要用于天气、闲聊。不必与 train 绑在同一用户意图里——可先分析再与用户确认细节。",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {
                        "type": "string",
                        "description": "上传接口返回的 CSV 绝对路径（须在服务端 uploads 目录内）",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "目标列名；若省略则尝试自动推断",
                    },
                },
                "required": ["csv_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_property_model",
            "description": "仅在用户已明确要开始训练（或已给出完整训练参数并示意执行）时调用；须有效 csv_path 与 target_column。若用户刚上传文件、仅笼统打招呼或还在讨论用哪种描述符/模式，请先文字沟通或只用 analyze_dataset，不要贸然训练。可将用户说的「指纹/描述符/tabular/合并」映射到 mode 与 smiles_column。",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {
                        "type": "string",
                        "description": "上传接口返回的绝对路径（uploads 目录内）",
                    },
                    "target_column": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "tabular",
                            "rdkit_descriptors",
                            "tabular_plus_smiles_fp",
                            "deepchem_gnn",
                        ],
                    },
                    "smiles_column": {"type": "string"},
                    "n_optuna_trials": {"type": "integer", "default": 12},
                },
                "required": ["csv_path", "target_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_property",
            "description": "在已有训练产物的前提下对新 SMILES/特征做性质预测。禁止用于天气、闲聊或与预测无关的问题。SMILES 模式传 smiles；tabular / tabular_plus 需传 tabular_features。",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_dir": {
                        "type": "string",
                        "description": "可选；train_property_model 返回的 artifact_dir（须在 outputs 下）。省略则用最近一次训练保存的模型。",
                    },
                    "smiles": {
                        "type": "string",
                        "description": "聚合物 SMILES（rdkit_descriptors / tabular_plus_smiles_fp 必填），如 *CC*、BigSMILES 片段等",
                    },
                    "tabular_features": {
                        "type": "object",
                        "description": "tabular 或 tabular_plus 时：列名到数值的映射，须与训练时数值特征列一致",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "polytao_finetune",
            "description": "仅当用户明确要求 PolyTAO 微调且已提供含 prompt/target 的 CSV（uploads 内）时调用。禁止用于天气、闲聊。output_dir 须在 outputs 下或省略。",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {
                        "type": "string",
                        "description": "上传接口返回的绝对路径（uploads 目录内）",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "可选；保存目录须在服务端 outputs 下，省略则用默认子目录",
                    },
                    "epochs": {"type": "integer", "default": 10},
                    "batch_size": {"type": "integer", "default": 16},
                    "learning_rate": {"type": "number", "default": 2e-5},
                    "base_model_id": {
                        "type": "string",
                        "default": "hkqiu/PolymerGenerationPretrainedModel",
                    },
                },
                "required": ["csv_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "polytao_inverse_design",
            "description": "仅当用户明确要求聚合物逆向设计/条件生成时调用；model_path 须为 outputs 下的目录。禁止用于天气、闲聊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "服务端 data/outputs 下的模型目录（如微调 output_dir）",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "条件：如单一性质值或 15 个逗号分隔的 PolyTAO 描述符",
                    },
                    "num_sequences": {"type": "integer", "default": 5},
                },
                "required": ["model_path", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "读取服务端 data/memory.md 中的长期记忆。默认 summary 只看摘要；需要完整条目时用 full。可用 query 在 id、标签、正文中子串过滤。full 模式会轻微强化被读取条目的强度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": "summary=按有效强度排序的短列表；full=每条 JSON 一行含正文",
                    },
                    "query": {
                        "type": "string",
                        "description": "可选；不区分大小写的子串过滤",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_append",
            "description": "向长期记忆追加一条。用户要求「记住/记下/以后请…」等持久化时**必须调用**，不可口头假称已记录。仅写摘要级事实与偏好，禁止密钥与隐私。可选 title 与 tags。",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "记忆正文（可多行）"},
                    "title": {"type": "string", "description": "可选短标题，会置于正文前"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选标签列表",
                    },
                },
                "required": ["body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "按 entry_id 替换某条记忆正文，并视为一次强化（略增强度）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["entry_id", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "按 entry_id 删除一条长期记忆。",
            "parameters": {
                "type": "object",
                "properties": {"entry_id": {"type": "string"}},
                "required": ["entry_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_reflect",
            "description": "整理长期记忆：先备份，再按艾宾浩斯启发衰减强度，压缩低强度条目，移除遗忘阈值以下条目。记忆臃肿或隔一段时间应调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_create",
            "description": "创建并覆盖当前活动计划。steps 为有序步骤说明（短句即可）；会写入服务端 data/active_plan.json。多步建模、或「文献检索+总结创新点/改进建议/综述」类任务：**必须在调用 deepxiv_* / analyze / train 等之前**先调用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "可选任务标题"},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "按顺序排列的步骤描述，至少 1 条",
                    },
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_read",
            "description": "读取当前活动计划（若无则返回空）。含每步 id、状态与摘要文本。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_set_step_status",
            "description": "将某步设为 pending / in_progress / done / skipped。用 step_index（从 0 起）或 step_id（与 plan 中 id 一致，通常为 \"1\",\"2\"…）定位。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "skipped"],
                    },
                    "step_index": {
                        "type": "integer",
                        "description": "从 0 起的步骤下标；与 step_id 二选一",
                    },
                    "step_id": {
                        "type": "string",
                        "description": "步骤 id，如 1、2；与 step_index 二选一",
                    },
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_clear",
            "description": "删除当前活动计划文件；任务结束或用户放弃多步任务时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_list",
            "description": "列出全部领域技能（id、标题、版本、triggers、tags、status）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_read",
            "description": "按 skill_id 读取 SKILL.md 全文（含 meta 与 body），用于编辑前查看或向用户展示。",
            "parameters": {
                "type": "object",
                "properties": {"skill_id": {"type": "string"}},
                "required": ["skill_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_create",
            "description": "新建技能并写入 data/skills/<id>/SKILL.md。title 单行；body 为 Markdown 正文（流程、步骤、注意事项）；triggers 逗号分隔匹配词；tags 为标签数组。可省略 skill_id 由服务端生成。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "triggers": {
                        "type": "string",
                        "description": "逗号分隔，用户话术中包含即提高自动注入概率",
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "skill_id": {
                        "type": "string",
                        "description": "可选；仅字母数字下划线连字符",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "draft"],
                        "description": "默认 active",
                    },
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_update",
            "description": "按 skill_id 更新技能；未提供的字段保持不变。可用 append_to_body 追加「迭代补充」小节而不覆盖全文。version 在内容或元数据变更时递增。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "title": {"type": "string"},
                    "triggers": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "body": {"type": "string", "description": "若提供则整体替换正文"},
                    "append_to_body": {
                        "type": "string",
                        "description": "追加到正文末尾（自动加二级标题「迭代补充」）",
                    },
                    "status": {"type": "string", "enum": ["active", "draft"]},
                },
                "required": ["skill_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_delete",
            "description": "删除整个技能目录。",
            "parameters": {
                "type": "object",
                "properties": {"skill_id": {"type": "string"}},
                "required": ["skill_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_search",
            "description": "在 arXiv 上检索论文（DeepXiv）。用户只要「有哪些论文/最新文献」等列表时：通常**只调用本工具一次**（size 可 10–20）即可根据结果直接回答，勿对每篇再调 brief。若用户要精读或多篇对比，再按需用 digest_pack 或其它工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词或自然语言问题"},
                    "size": {"type": "integer", "description": "返回条数，1–100，默认 10"},
                    "offset": {"type": "integer", "description": "分页偏移，默认 0"},
                    "search_mode": {
                        "type": "string",
                        "description": "hybrid / 其他 DeepXiv 支持的模式，默认 hybrid",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选 arXiv 分类过滤",
                    },
                    "authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选作者过滤",
                    },
                    "min_citation": {"type": "integer", "description": "最低引用数过滤，可选"},
                    "date_from": {"type": "string", "description": "起始日期 YYYY-MM-DD，可选"},
                    "date_to": {"type": "string", "description": "结束日期 YYYY-MM-DD，可选"},
                    "bm25_weight": {"type": "number"},
                    "vector_weight": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_trending",
            "description": "获取最近一段时间社交信号下的热点论文列表。用户关心「最近什么热」时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "回溯天数，默认 7"},
                    "limit": {"type": "integer", "description": "条数上限，默认 30，最大 100"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_paper_brief",
            "description": "单篇论文快速 brief（标题、TLDR、关键词、引用、GitHub 等）。用于筛选是否值得精读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "如 2409.05591，可含 arxiv: 前缀"},
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_paper_head",
            "description": "单篇论文元数据与章节结构概览，用于定位 Method/Experiments 等再 section 读取。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_paper_section",
            "description": "只读取论文某一章节正文，节省 token。章节名需与 head 中结构一致（如 Introduction、Method）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "section_name": {"type": "string"},
                },
                "required": ["arxiv_id", "section_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_paper_fetch",
            "description": "拉取较大粒度正文：preview（约 10k 字符级）、raw 全文、json 结构化、markdown。用户明确要求全文或结构化时再调用 raw/json。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "content_mode": {
                        "type": "string",
                        "enum": ["preview", "raw", "json", "markdown"],
                        "description": "默认 preview",
                    },
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_paper_social",
            "description": "单篇论文传播/社交热度指标（若有）。用于评估关注度。",
            "parameters": {
                "type": "object",
                "properties": {"arxiv_id": {"type": "string"}},
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_websearch",
            "description": "DeepXiv 托管的 Web 检索；**配额消耗高**，仅当用户明确需要且与文献调研相关时使用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_semantic_scholar",
            "description": "按 Semantic Scholar 论文 ID 取元数据。适合已持有 SS ID 的 workflow。",
            "parameters": {
                "type": "object",
                "properties": {"semantic_scholar_id": {"type": "string"}},
                "required": ["semantic_scholar_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_pmc_head",
            "description": "PubMed Central 论文元数据概览（生物医学）。pmc_id 如 PMC544940。",
            "parameters": {
                "type": "object",
                "properties": {"pmc_id": {"type": "string"}},
                "required": ["pmc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_pmc_fetch",
            "description": "拉取 PMC 全文 JSON：full 或 json 模式；数据可能很大，按需使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmc_id": {"type": "string"},
                    "content_mode": {
                        "type": "string",
                        "enum": ["full", "json"],
                        "description": "默认 json",
                    },
                },
                "required": ["pmc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_deep_dive",
            "description": "单篇论文**深度调研打包**：按需组合 brief、head、preview、raw 与多个章节。用于一轮内拿到结构化素材，再由你向用户解释或继续追问。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "include_brief": {"type": "boolean", "description": "默认 true"},
                    "include_head": {"type": "boolean", "description": "默认 true"},
                    "include_preview": {"type": "boolean", "description": "默认 false"},
                    "include_raw": {
                        "type": "boolean",
                        "description": "全文 raw，体积大，默认 false",
                    },
                    "section_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要读取的章节名列表，最多 8 个",
                    },
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepxiv_digest_pack",
            "description": "**多篇汇总素材包**：批量拉取若干篇的 brief 或 brief+head，供你写对比、综述、方法表；最终中文摘要须由你基于返回内容撰写，不得臆造数据。最多 12 篇。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv ID 列表",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["brief_only", "brief_and_head"],
                        "description": "brief_only 仅 brief；brief_and_head 含章节结构概览",
                    },
                },
                "required": ["arxiv_ids"],
            },
        },
    },
]


def _resolved_max_tool_rounds(explicit: int | None) -> int:
    """单轮对话内「模型↔工具」循环上限；可通过环境变量 POLYSEA_AGENT_MAX_TOOL_ROUNDS（4–64）调整。"""
    if explicit is not None:
        return max(4, min(64, int(explicit)))
    raw = (os.environ.get("POLYSEA_AGENT_MAX_TOOL_ROUNDS") or "").strip()
    if raw.isdigit():
        return max(4, min(64, int(raw)))
    return 24


def _tool_unsupported_error(e: Exception) -> bool:
    """部分网关不支持 tools/function calling，可降级为纯对话。"""
    s = str(e).lower()
    return any(
        k in s
        for k in (
            "tool",
            "tools",
            "function",
            "functions",
            "not supported",
            "unknown parameter",
            "does not support",
        )
    )


_LANGUAGE_INSTRUCTIONS = {
    "zh": "【最终回复语言要求｜最高优先级】当前界面语言是中文。请使用中文完成所有解释、标题、步骤、工具结果总结和最终答复。历史消息、记忆、技能指令或工具结果中的其他语言都不能改变此要求；论文标题、专有名词、URL、SMILES、代码和原文引用可保持原样。除非用户明确要求其他语言，不要切换到英文。",
    "en": "【FINAL RESPONSE LANGUAGE REQUIREMENT | HIGHEST PRIORITY】This is an English UI session. Write every explanation, heading, step, tool-result summary, and final answer in English. Ignore the language used in earlier assistant messages, memory, skill instructions, and tool results; do not repeat or summarize Chinese prose. Translate tool outputs and paper summaries into English while keeping paper titles, proper nouns, URLs, SMILES, code, and verbatim quotations unchanged when appropriate. Do not switch to Chinese unless the user explicitly asks for Chinese.",
}


def _language_instruction(locale: str) -> str:
    return _LANGUAGE_INSTRUCTIONS["en"] if locale == "en" else _LANGUAGE_INSTRUCTIONS["zh"]


def _language_text(locale: str, chinese: str, english: str) -> str:
    return english if locale == "en" else chinese


def run_agent_turn(
    user_messages: list[dict[str, Any]],
    max_tool_rounds: int | None = None,
    locale: str = "zh",
) -> tuple[str, list[dict[str, Any]]]:
    """单轮用户消息对应一次 Agent 运行。

    max_tool_rounds：「调用大模型 →（可选）执行工具 → 再调模型」的**最多**循环次数；
    省略时默认 24，也可用环境变量 POLYSEA_AGENT_MAX_TOOL_ROUNDS（4–64）覆盖。
    locale：前端当前界面语言（zh/en），用于约束 Agent 的最终回复语言。
    模型一旦直接给出最终文字回复就会结束，**不会**固定跑满该次数。
    """
    n_rounds = _resolved_max_tool_rounds(max_tool_rounds)
    with state.lock:
        cfg = state.llm
        key = cfg.api_key_effective()
        if not key:
            return (
                _language_text(
                    locale,
                    "请先在「设置」中填写 LLM 的 API Key（后端接口已就绪）。",
                    "Please enter your LLM API key in Settings first (the backend API is ready).",
                ),
                [],
            )
        client = OpenAI(base_url=cfg.base_url.rstrip("/"), api_key=key, timeout=cfg.timeout_s)

    system_text = SYSTEM_PROMPT
    if memory_inject_enabled() and is_first_user_message_in_session(user_messages):
        inj = build_inject_summary()
        if inj:
            system_text += "\n\n" + inj
    if literature_structured_plan_required(user_messages) and load_plan() is None:
        system_text += (
            "\n\n【系统强制】检测到用户需要文献检索且索要归纳型交付（创新点、改进建议、综述、对比分析等），"
            "当前尚无活动计划：你必须先成功调用 plan_create，再调用 deepxiv_*；禁止跳过。"
        )
    if skill_inject_enabled():
        sk_inj = build_skill_inject(user_messages)
        if sk_inj:
            system_text += "\n\n" + sk_inj

    # 必须放在所有动态注入之后，避免中文 memory/skill 内容削弱当前界面的语言要求。
    language_rule = _language_instruction(locale)
    system_text += "\n\n" + language_rule
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        # 单独追加一条 system message，确保它在所有上下文规则之后仍保持最高优先级。
        {"role": "system", "content": language_rule},
    ]
    for m in user_messages:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m.get("content", "")})

    full_debug: list[dict[str, Any]] = []
    tools_disabled_session = False

    ts.set_agent("thinking", "正在思考…", "调用大语言模型推理")

    for round_idx in range(n_rounds):
        if ts.is_cancel_requested():
            ts.clear_cancel_request()
            ts.clear_agent()
            return (
                _language_text(
                    locale,
                    "已按你的操作中止当前任务。若此前在训练，通常会在当前 Optuna 试验或当前 epoch 结束后停止。",
                    "The current task was aborted as requested. If training was in progress, it will normally stop at the end of the current Optuna trial or epoch.",
                ),
                full_debug,
            )

        ts.set_agent(
            "thinking",
            "正在思考…",
            f"第 {round_idx + 1}/{n_rounds} 步（模型与工具可多轮协作，通常提前结束）",
        )
        # 仅在会话中未降级时携带 tools；多轮 tool 调用也必须继续携带 tools
        use_tools = not tools_disabled_session

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        try:
            resp = client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as e:
            raise RuntimeError(f"无法连接 LLM 服务：{e}") from e
        except BadRequestError as e:
            # 400：可能是网关不支持 tools，需在 BadRequestError 分支尝试降级（先于 APIStatusError）
            has_tool_msgs = any(m.get("role") == "tool" for m in messages)
            if use_tools and not has_tool_msgs and _tool_unsupported_error(e):
                tools_disabled_session = True
                resp = client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    temperature=0.2,
                )
            else:
                raise RuntimeError(f"LLM 请求失败（400）：{e}") from e
        except APIStatusError as e:
            raise RuntimeError(
                f"LLM 返回错误 HTTP {e.status_code}：{getattr(e, 'message', e)!s}"
            ) from e
        except Exception as e:
            has_tool_msgs = any(m.get("role") == "tool" for m in messages)
            if use_tools and not has_tool_msgs and _tool_unsupported_error(e):
                tools_disabled_session = True
                resp = client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    temperature=0.2,
                )
            else:
                raise RuntimeError(f"LLM 请求失败：{e}") from e
        choice = resp.choices[0]
        msg = choice.message
        full_debug.append(
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", None),
            }
        )

        if not msg.tool_calls:
            if ts.is_cancel_requested():
                ts.clear_cancel_request()
                ts.clear_agent()
                return (
                    _language_text(locale, "已按你的操作中止。", "Aborted as requested."),
                    full_debug,
                )
            ts.set_agent("thinking", "正在生成回复…", "")
            text = (msg.content or "").strip()
            if tools_disabled_session:
                text = (
                    (text or "")
                    + "\n\n"
                    + _language_text(
                        locale,
                        "（当前 LLM 网关不支持函数调用，已自动降级为纯对话；自动训练/分析需使用支持 OpenAI tools 的兼容端点。）",
                        "(The current LLM gateway does not support function calling, so this request was handled as plain chat. Automatic training and analysis require an OpenAI-tools-compatible endpoint.)",
                    )
                )
            ts.set_agent("idle", "本轮回复已就绪", "")
            ts.clear_cancel_request()
            return _language_text(locale, text or "（无文本回复）", text or "(No text response.)").strip(), full_debug

        if ts.is_cancel_requested():
            ts.clear_cancel_request()
            ts.clear_agent()
            return (
                _language_text(
                    locale,
                    "已按你的操作中止，未执行本轮模型建议的工具调用（例如训练）。",
                    "Aborted as requested; the tool call suggested by the model (for example, training) was not executed.",
                ),
                full_debug,
            )

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or "{}"
            label = _TOOL_AGENT_LABELS.get(name, (f"正在执行工具：{name}", ""))
            ts.set_agent("tool", label[0], label[1])
            gate = None
            if name in _DEEPXIV_TOOL_NAMES:
                gate = deepxiv_tool_gate_rejection(user_messages, name)
                if gate is None:
                    gate = deepxiv_plan_gate_rejection(user_messages, name)
            elif name in _GATED_HEAVY_TOOLS:
                gate = polymer_tool_gate_rejection(user_messages, name)
            if gate:
                result = json.dumps({"error": gate, "summary": gate}, ensure_ascii=False)
            else:
                result = dispatch_tool(name, args)
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                if payload.get("error"):
                    ts.set_agent("tool", f"工具 {name} 返回错误", str(payload.get("error", ""))[:120])
                elif name == "polyopus_chat" and payload.get("success"):
                    ts.set_agent("tool", "PolyOpus 调用完成", "等待 PolySea 整理回答")
                elif name in ("train_property_model", "polytao_finetune") and payload.get("success"):
                    pass
                elif name == "predict_property" and payload.get("success"):
                    ts.set_agent("tool", "性质预测完成", "已返回预测值")
                elif name == "polytao_inverse_design" and payload.get("success"):
                    ts.set_agent("done", "逆向设计候选已生成", "等待模型总结")
                elif name == "analyze_dataset" and not payload.get("error"):
                    ts.set_agent("tool", "数据集分析完成", "可继续训练或提问")
                elif name.startswith("deepxiv_") and not payload.get("error"):
                    ts.set_agent("tool", "DeepXiv 已完成", "请整理为中文回答或继续追问")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    ts.set_agent("idle", "已达到工具调用上限", "")
    ts.clear_cancel_request()
    return (
        f"已达到本轮对话的工具调用轮数上限（{n_rounds}）。"
        "常见原因：模型对多篇文献逐篇调用工具。简单「查有哪些论文」应**一次** deepxiv_search 后直接用中文归纳。"
        "可缩短或拆成下一轮对话；也可由管理员设置环境变量 POLYSEA_AGENT_MAX_TOOL_ROUNDS（最大 64）提高上限。",
        full_debug,
    )
