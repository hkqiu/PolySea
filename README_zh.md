<div align="center">

# 🌊 PolySea

**面向聚合物信息学的本地优先（local-first）Web 框架**

性质预测 · PolyTAO 微调与逆向设计 · DeepXiv 文献研究 · OpenAI 兼容工具调用 Agent

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](backend/requirements.txt)
[![Node](https://img.shields.io/badge/node-20%2B-339933.svg)](frontend/package.json)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](backend/app/main.py)
[![React](https://img.shields.io/badge/frontend-React%2018-61DAFB.svg)](frontend/package.json)

**[🇬🇧 English README](README.md)** ｜ 中文文档

</div>

---

PolySea 在本地运行，覆盖聚合物性质预测、PolyTAO 微调与逆向设计、DeepXiv 文献研究，以及带工具调用的 OpenAI 兼容智能 Agent。所有数据和模型都保存在本地文件系统，不上传第三方服务器（除非你主动配置外部 LLM/PolyOpus/DeepXiv 端点）。

---

## 目录

- [核心能力](#核心能力)
- [技术栈](#技术栈)
- [仓库结构](#仓库结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [数据与路径约束](#数据与路径约束)
- [Agent 工具与能力](#agent-工具与能力)
- [HTTP API](#http-api)
- [生产部署](#生产部署)
- [局限与路线图](#局限与路线图)
- [引用](#引用)
- [许可证](#许可证)

---

## 核心能力

| 能力 | 当前实现 |
|------|------|
| 🧪 **性质预测** | 分析 CSV 中的数值列、SMILES 列和目标列；支持 `tabular`、Morgan 指纹 `rdkit_descriptors`、数值列 + Morgan 指纹 `tabular_plus_smiles_fp`。使用 Optuna 调参和 Stacking 集成；回归报告 R²/RMSE，分类报告 accuracy/macro-F1；训练后可保存模型并对新样本推理。 |
| 🧬 **PolyTAO** | 基于 Hugging Face 的 [`hkqiu/PolymerGenerationPretrainedModel`](https://huggingface.co/hkqiu/PolymerGenerationPretrainedModel) 进行 T5 条件生成；支持含 `prompt`/`target` 的 CSV 微调，以及从 `data/outputs` 下的本地模型目录进行逆向设计。 |
| ⚙️ **可选 PolyOpus** | PolyOpus 是基于 [`DeepSeek-LLM-7B-Chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) 在聚合物数据上 SFT 得到的聚合物专用大模型；权重计划发布于 [huggingface.co/hkqiu](https://huggingface.co/hkqiu)。配置 OpenAI-compatible PolyOpus 端点后，可优先处理 Tg、原子化能、热稳定性、带隙预测、指定带隙逆向生成和聚合物知识问答。默认不启用，本仓库不包含权重。 |
| 📚 **文献研究** | 通过 [DeepXiv](https://github.com/DeepXiv/deepxiv_sdk) 工具访问 arXiv、热点、论文 brief/章节/正文、社交信号、Semantic Scholar、PMC、深度调研和多论文 digest。需要对应服务可用及必要的 token。 |
| 🧠 **持久化 Agent 状态** | `memory_*` 管理长期记忆，`plan_*` 管理多步计划，`skill_*` 管理可复用领域技能；这些数据写入 `backend/data`，不是仅存在对话上下文。 |
| 🌐 **双语与训练控制** | 前端支持中英文切换；`/api/chat` 接收 `locale`；前端轮询训练进度，并可请求中止当前对话/训练。 |

> ⚠️ `deepchem_gnn` 目前仅出现在参数枚举和预留代码中，实际会抛出未实现错误；当前可用的是表格和 RDKit Morgan 指纹路径，不要把 DeepChem GNN 描述为已完成能力。

---

## 技术栈

| 层级 | 说明 |
|------|------|
| 后端 | Python 3.10+、FastAPI、scikit-learn、Optuna、RDKit、PyTorch、Transformers、OpenAI SDK、[DeepXiv SDK](https://github.com/DeepXiv/deepxiv_sdk) |
| 前端 | Node.js 20+、React 18、Vite 5、TypeScript、Tailwind CSS、Framer Motion |
| 可选依赖 | `backend/requirements-optional.txt` 目前仅提供 DeepChem 依赖；安装它不会启用 `deepchem_gnn` 训练实现。 |

---

## 📦 仓库结构

```text
PolySea/
├── backend/
│   ├── app/main.py             # FastAPI 路由和静态资源挂载
│   ├── app/state.py            # 进程内 LLM / PolyOpus 配置
│   ├── app/training_status.py  # 训练和 Agent 状态
│   ├── app/services/           # Agent、自动建模、PolyTAO、DeepXiv、记忆/计划/技能
│   ├── data/uploads/           # 上传 CSV（运行时生成）
│   ├── data/outputs/           # 训练模型和 PolyTAO 输出（运行时生成）
│   ├── requirements.txt
│   └── run.py                  # uvicorn 开发入口
├── frontend/                   # React + Vite 前端
├── dataset/                    # 示例数据
├── package.json                # 根目录并发启动脚本
├── README.md
└── README_zh.md
```

运行时的 `backend/data/memory.md`、`active_plan.json`、`skills/` 和模型/上传文件属于本地状态，发布时不应提交。

---

## 🧰 环境要求

- **Python 3.10+**
- **Node.js 20+**（前端开发和构建）
- **GPU** 可选；PolyTAO 和较大的 Optuna 任务通常受益于 GPU

---

## 🚀 快速开始

### 方式一：根目录启动前后端（推荐）

在与 `backend`、`frontend` 同级的项目根目录执行：

```bash
npm install
npm run dev
```

这会同时启动：

- FastAPI：`http://127.0.0.1:8000`
- Vite：`http://127.0.0.1:5173`

浏览器打开 `http://127.0.0.1:5173`，在右上角「设置」中填写 OpenAI-compatible 服务的 Base URL、Model 和 API Key。API Key 只在后端进程内使用，后端重启后需要重新配置。

> 只运行 `frontend` 目录中的 `npm run dev` 而不启动后端时，Vite 代理会连接失败。

### 方式二：仅后端

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
python run.py
```

后端默认监听 `http://127.0.0.1:8000`，交互式 API 文档位于 `http://127.0.0.1:8000/docs`。

### 方式三：仅前端

后端必须已经运行：

```bash
cd frontend
npm install
npm run dev
```

---

## ⚙️ 配置说明

### LLM 主端点

可以在 Web「设置」中配置 `base_url`、`model`、`api_key` 和 `timeout_s`。后端默认不内置服务商地址、模型或密钥，也可在启动前使用环境变量：

```powershell
$env:POLYSEA_BASE_URL="https://api.openai.com/v1"
$env:POLYSEA_MODEL="your-model"
$env:POLYSEA_API_KEY="<your-key>"
```

Agent 依赖 OpenAI-compatible Chat Completions 的 tools/function calling；不支持工具调用的网关会降级为纯对话。API Key 不会由 `GET /api/settings` 原样返回，只返回是否已设置。

### PolyOpus（可选）

PolyOpus 是基于 [`DeepSeek-LLM-7B-Chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) 在聚合物数据上微调（SFT）得到的聚合物专用大模型。权重计划发布于作者 Hugging Face 主页 [huggingface.co/hkqiu](https://huggingface.co/hkqiu)（暂未发布），本仓库不包含、也不分发该模型权重。

```powershell
$env:POLYOPUS_BASE_URL="http://127.0.0.1:9000/v1"
$env:POLYOPUS_MODEL="polyopus"
$env:POLYOPUS_API_KEY="<your-key>"
$env:POLYOPUS_TIMEOUT_S="120"
```

只有 `POLYOPUS_BASE_URL` 和模型配置后才会启用 PolyOpus；本项目不包含权重。配置后，符合其专长的聚合物任务会优先路由到 `polyopus_chat`。

### DeepXiv

文献研究工具基于 [DeepXiv SDK](https://github.com/DeepXiv/deepxiv_sdk)（感谢 DeepXiv 团队提供的检索与文献处理能力）。

- `DEEPXIV_BASE_URL`：可选的 DeepXiv 服务地址，默认使用官方地址。
- `DEEPXIV_TOKEN`：可选 token；代码也会尝试读取当前目录或用户主目录的 `.env`，请勿将其提交到公开目录。

### 高级 Agent 开关

- `POLYSEA_MEMORY_INJECT_ENABLED`、`POLYSEA_MEMORY_INJECT_MAX_CHARS`
- `POLYSEA_SKILL_INJECT_ENABLED`、`POLYSEA_SKILL_INJECT_MAX_CHARS`
- `POLYSEA_PLAN_GATE_LIT`：是否对“文献检索 + 总结/创新点/改进建议”等多步请求强制先创建计划

### 前后端分离构建

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run build
```

Linux/macOS：

```bash
export VITE_API_BASE=http://127.0.0.1:8000
npm run build
```

开发环境不设置时，Vite 将 `/api` 请求代理到 `127.0.0.1:8000`。

---

## 🗂️ 数据与路径约束

### 性质预测 CSV

- 需要 CSV 和目标列名；`analyze_dataset` 可分析列类型并推荐模式。
- `tabular` 使用数值特征；`rdkit_descriptors` 使用 SMILES 的 2048 位 Morgan 指纹；`tabular_plus_smiles_fp` 拼接数值特征和 Morgan 指纹。
- 目标列主要按数值可转换比例判断回归或分类；训练默认 Optuna 12 次试验，可通过工具参数调整。
- 训练模型保存为 `data/outputs` 下的 bundle，`predict_property` 可使用指定 `artifact_dir`，或读取最近一次成功训练的 `LATEST`。

### PolyTAO

微调 CSV 必须包含：

- `prompt`：性质条件文本
- `target`：目标聚合物 SMILES

`output_dir` 和逆向设计的 `model_path` 必须位于服务端 `data/outputs` 下。

### 路径与运行数据

- 上传接口将文件保存到 `backend/data/uploads`，并返回服务器绝对路径。
- Agent 的 CSV 工具只接受 uploads 目录内的文件；输出和模型工具只接受 outputs 目录内的路径。
- 训练状态是进程内单例；记忆、计划和技能会写入 `backend/data`，适合单用户本地使用。

---

## 🛠️ Agent 工具与能力

### 聚合物建模与生成

| 工具 | 作用 |
|------|------|
| `polyopus_chat` | 可选 PolyOpus 自然语言调用；配置后优先处理其支持的聚合物性质、带隙逆向生成和知识问答。 |
| `analyze_dataset` | 分析 CSV 列、SMILES、任务类型和推荐建模模式。 |
| `train_property_model` | Optuna + Stacking 训练性质预测模型，支持 `auto`、`tabular`、`rdkit_descriptors`、`tabular_plus_smiles_fp`；`deepchem_gnn` 当前不可用。 |
| `predict_property` | 读取已保存 bundle，对新 SMILES 或数值特征做单样本预测；不会重新训练。 |
| `polytao_finetune` | 使用 `prompt`/`target` CSV 微调 PolyTAO。 |
| `polytao_inverse_design` | 从 outputs 下的本地 PolyTAO 模型按条件生成候选 SMILES。 |

### 记忆、计划和技能

- `memory_read`、`memory_append`、`memory_update`、`memory_delete`、`memory_reflect`：维护 `data/memory.md`；代码会拒绝常见密钥、密码和身份证模式。
- `plan_create`、`plan_read`、`plan_set_step_status`、`plan_clear`：维护 `data/active_plan.json`。多步建模，以及“检索文献并总结/提炼创新点/提出改进建议”等请求会先创建计划。
- `skill_list`、`skill_read`、`skill_create`、`skill_update`、`skill_delete`：维护 `data/skills/<id>/SKILL.md`，用于复用领域流程；技能内容同样禁止写入敏感凭据。

### DeepXiv 文献工具

`deepxiv_search`、`deepxiv_trending`、`deepxiv_paper_brief`、`deepxiv_paper_head`、`deepxiv_paper_section`、`deepxiv_paper_fetch`、`deepxiv_paper_social`、`deepxiv_websearch`、`deepxiv_semantic_scholar`、`deepxiv_pmc_head`、`deepxiv_pmc_fetch`、`deepxiv_deep_dive`、`deepxiv_digest_pack`。

这些工具覆盖 arXiv 检索、热点论文、渐进式论文阅读、社交信号、网页级检索、Semantic Scholar、PMC、单篇深挖和多篇素材汇总。纯列表需求通常只需一次 `deepxiv_search`；检索后还要总结/写综述时会与结构化计划联动。

---

## 🔌 HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 返回服务、健康检查和 API 文档入口信息。 |
| `GET` | `/api/health` | 健康检查。 |
| `GET` / `POST` | `/api/settings` | 读取/写入 LLM 配置；读取时只返回 `api_key_set`，不返回密钥原文。 |
| `POST` | `/api/upload` | 上传文件，返回 `{ filename, path, size_bytes }`；文件名会取 basename。 |
| `POST` | `/api/chat` | Agent 对话；body 为 `{ "messages": [{"role":"user|assistant", "content":"..."}], "locale":"zh|en" }`。 |
| `GET` | `/api/training/status` | 返回 Agent 阶段、Optuna 进度、PolyTAO epoch、日志和取消状态。 |
| `POST` | `/api/training/cancel` | 请求中止当前任务；训练在下一个 Optuna trial 或 PolyTAO epoch 边界停止，对话等待会由前端断开。 |

API 文档启动后可访问 `http://127.0.0.1:8000/docs`。

---

## 🏭 生产部署

构建前端并由 FastAPI 单端口托管：

```bash
cd frontend
npm install
npm run build
```

将 `frontend/dist` 的全部内容复制到 `backend/static`（Windows）：

```bash
xcopy /E /I frontend\dist backend\static
```

然后运行：

```bash
cd backend
python run.py
```

访问 `http://127.0.0.1:8000`。生产环境请关闭 `run.py` 中的 reload，并使用进程管理器或正式的 uvicorn 参数。部署到公开环境前还应自行增加认证、访问控制、限流和更严格的上传限制；当前 CORS 配置面向本地 MVP。

---

## 🧭 局限与路线图

- **单用户本地 MVP**：训练状态、LLM 配置和取消标志是进程内单例；没有用户认证、多租户隔离或持久化任务队列。
- **DeepChem GNN 未实现**：虽然保留模式和可选依赖，但当前调用会明确报未实现错误；请使用 RDKit 指纹或表格模式。
- **外部服务依赖**：OpenAI-compatible LLM、PolyOpus 和 DeepXiv 的可用性、配额和凭据由部署者负责。
- **科学有效性**：预测和生成结果依赖数据、划分和验证方式，不替代领域审查、实验验证或合规要求。
- **路线图**：可扩展任务调度、真正可复现的图网络路径、分子动力学（如 LAMMPS）等工作流。

---

## 📖 引用

使用 PolyTAO 预训练权重时，请引用：

```bibtex
@article{qiuOndemandReverseDesign2024,
  title = {On-Demand Reverse Design of Polymers with {{PolyTAO}}},
  author = {Qiu, Haoke and Sun, Zhao-Yan},
  year = {2024},
  journal = {npj Computational Materials},
  volume = {10},
  number = {1},
  pages = {273},
  doi = {10.1038/s41524-024-01466-5}
}
```

---

## 📄 许可证

Apache License 2.0（第三方依赖和 PolyTAO 预训练模型仍需分别遵守其许可证和使用条款）。
