<div align="center">

# 🌊 PolySea

**A local-first web framework for polymer informatics**

Property prediction · PolyTAO fine-tuning & inverse design · DeepXiv literature research · OpenAI-compatible tool-calling agent

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](backend/requirements.txt)
[![Node](https://img.shields.io/badge/node-20%2B-339933.svg)](frontend/package.json)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](backend/app/main.py)
[![React](https://img.shields.io/badge/frontend-React%2018-61DAFB.svg)](frontend/package.json)

**[🇨🇳 简体中文 README](README_zh.md)** ｜ English docs

</div>

---

PolySea runs locally and covers polymer property prediction, PolyTAO fine-tuning and inverse design, DeepXiv literature research, and an OpenAI-compatible tool-calling agent. All data and models stay on the local filesystem and are never uploaded to a third party, unless you explicitly configure an external LLM/PolyOpus/DeepXiv endpoint.

---

## Contents

- [Features](#features)
- [Stack](#stack)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Data and path constraints](#data-and-path-constraints)
- [Agent tools and capabilities](#agent-tools-and-capabilities)
- [HTTP API](#http-api)
- [Production deployment](#production-deployment)
- [Citation](#citation)
- [License](#license)

---

## Features

| Area | Current implementation |
|------|------------------------|
| 🧪 **Property prediction** | Inspects numeric columns, SMILES columns, and targets in CSV files. Supports `tabular`, Morgan fingerprints (`rdkit_descriptors`), and numeric + Morgan fingerprint fusion (`tabular_plus_smiles_fp`). Uses Optuna tuning and stacking ensembles; regression reports R²/RMSE, classification reports accuracy/macro-F1. Saved bundles can be used for single-sample inference. |
| 🧬 **PolyTAO** | Fine-tunes the Hugging Face [`hkqiu/PolymerGenerationPretrainedModel`](https://huggingface.co/hkqiu/PolymerGenerationPretrainedModel) T5 model from CSV `prompt`/`target` pairs, and performs inverse design from a local model directory under `data/outputs`. |
| ⚙️ **Optional PolyOpus** | PolyOpus is a polymer-specialized LLM built by SFT of [`DeepSeek-LLM-7B-Chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) on polymer data; weights are planned to be released on [huggingface.co/hkqiu](https://huggingface.co/hkqiu). When an OpenAI-compatible PolyOpus endpoint is configured, it is preferred for Tg, atomization energy, thermal stability, band-gap prediction, specified-band-gap inverse generation, and polymer knowledge QA. Disabled by default; this repository ships no weights. |
| 📚 **Literature research** | [DeepXiv](https://github.com/DeepXiv/deepxiv_sdk) tools cover arXiv search, trending papers, paper briefs/sections/full-text fragments, social signals, Semantic Scholar, PMC, deep dives, and multi-paper digest material. The corresponding service and credentials must be available. |
| 🧠 **Persistent agent state** | `memory_*` manages long-term memory, `plan_*` manages multi-step plans, and `skill_*` manages reusable domain skills. These are written under `backend/data`, not kept only in the chat context. |
| 🌐 **Bilingual UI and cancellation** | The UI supports Chinese and English; `/api/chat` accepts `locale`; the UI polls training progress and can request cancellation of the current chat/training task. |

> ⚠️ `deepchem_gnn` is present as an enum and reserved code path, but currently raises an explicit not-implemented error. The available property modes are tabular and RDKit Morgan-fingerprint paths.

---

## Stack

| Layer | Notes |
|-------|-------|
| Backend | Python 3.10+, FastAPI, scikit-learn, Optuna, RDKit, PyTorch, Transformers, OpenAI SDK, [DeepXiv SDK](https://github.com/DeepXiv/deepxiv_sdk) |
| Frontend | Node.js 20+, React 18, Vite 5, TypeScript, Tailwind CSS, Framer Motion |
| Optional dependency | `backend/requirements-optional.txt` currently provides DeepChem only; installing it does not enable the `deepchem_gnn` training implementation. |

---

## 📦 Repository layout

```text
PolySea/
├── backend/
│   ├── app/main.py             # FastAPI routes and static mount
│   ├── app/state.py            # In-memory LLM / PolyOpus settings
│   ├── app/training_status.py  # Training and Agent snapshots
│   ├── app/services/           # Agent, AutoML, PolyTAO, DeepXiv, memory/plan/skill
│   ├── data/uploads/           # Uploaded CSVs, created at runtime
│   ├── data/outputs/           # Models and PolyTAO outputs, created at runtime
│   ├── requirements.txt
│   └── run.py                  # uvicorn development entry point
├── frontend/                   # React + Vite frontend
├── dataset/                    # Example datasets
├── package.json                # Root concurrent start scripts
├── README.md
└── README_zh.md
```

Runtime `backend/data/memory.md`, `active_plan.json`, `skills/`, uploads, and generated model files are local state and should not be published.

---

## 🧰 Requirements

- **Python 3.10+**
- **Node.js 20+** for frontend development and builds
- **GPU** optional; recommended for PolyTAO and larger Optuna jobs

---

## 🚀 Quick start

### Option A — Start backend and frontend from the repository root

From the directory containing `backend` and `frontend`:

```bash
npm install
npm run dev
```

This starts:

- FastAPI at `http://127.0.0.1:8000`
- Vite at `http://127.0.0.1:5173`

Open `http://127.0.0.1:5173` and enter the OpenAI-compatible Base URL, Model, and API Key in **Settings**. The API key is used in backend process memory and must be entered again after a backend restart.

> Running only `npm run dev` inside `frontend` without the backend will make the Vite proxy fail.

### Option B — Backend only

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

The backend listens on `http://127.0.0.1:8000`; interactive API docs are at `http://127.0.0.1:8000/docs`.

### Option C — Frontend only

The backend must already be running:

```bash
cd frontend
npm install
npm run dev
```

---

## ⚙️ Configuration

### Main LLM endpoint

Set `base_url`, `model`, `api_key`, and `timeout_s` in the Web **Settings** panel. The backend ships without a provider URL, model, or key by default. Environment variables are also supported:

```powershell
$env:POLYSEA_BASE_URL="https://api.openai.com/v1"
$env:POLYSEA_MODEL="your-model"
$env:POLYSEA_API_KEY="<your-key>"
```

The Agent relies on OpenAI-compatible Chat Completions tools/function calling for automated workflows. Gateways without tool support fall back to plain chat. `GET /api/settings` returns only `api_key_set`, never the key itself.

### PolyOpus (optional)

PolyOpus is a polymer-specialized LLM built by supervised fine-tuning (SFT) of [`DeepSeek-LLM-7B-Chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) on polymer data. Weights are planned to be released on the author's Hugging Face page, [huggingface.co/hkqiu](https://huggingface.co/hkqiu) (not yet published); this repository does not include or distribute the model weights.

**Local deployment.** Because PolyOpus keeps the same architecture, tokenizer, and chat template as `DeepSeek-LLM-7B-Chat`, it can be self-hosted with any OpenAI-compatible serving stack that already supports that base model — for example [vLLM](https://github.com/vllm-project/vllm) or [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference). Once the PolyOpus weights are published, a typical local setup looks like:

```bash
# Example: serve with vLLM's OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
  --model hkqiu/PolyOpus \
  --served-model-name polyopus \
  --port 9000
```

Then point PolySea at that local endpoint:

```powershell
$env:POLYOPUS_BASE_URL="http://127.0.0.1:9000/v1"
$env:POLYOPUS_MODEL="polyopus"
$env:POLYOPUS_API_KEY="<your-key>"
$env:POLYOPUS_TIMEOUT_S="120"
```

PolyOpus is enabled only when its base URL and model are configured. This repository contains no PolyOpus weights. Matching polymer tasks are routed to `polyopus_chat` first.

### DeepXiv

Literature research tools are built on the [DeepXiv SDK](https://github.com/DeepXiv/deepxiv_sdk) (credit to the DeepXiv team for the underlying search and literature-processing capabilities).

- `DEEPXIV_BASE_URL`: optional DeepXiv service URL; the code has an official default.
- `DEEPXIV_TOKEN`: optional token. The code may also load `.env` from the current directory or the user home directory; never commit that file or token.

### Advanced Agent switches

- `POLYSEA_MEMORY_INJECT_ENABLED`, `POLYSEA_MEMORY_INJECT_MAX_CHARS`
- `POLYSEA_SKILL_INJECT_ENABLED`, `POLYSEA_SKILL_INJECT_MAX_CHARS`
- `POLYSEA_PLAN_GATE_LIT`: whether literature search plus synthesis/novelty/improvement requests must create a plan first

### Split frontend and API

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run build
```

Linux/macOS:

```bash
export VITE_API_BASE=http://127.0.0.1:8000
npm run build
```

When unset in development, Vite proxies `/api` to `127.0.0.1:8000`.

---

## 🗂️ Data and path constraints

### Property-prediction CSV

- A CSV and target column are required; `analyze_dataset` can inspect columns and recommend a mode.
- `tabular` uses numeric features; `rdkit_descriptors` uses 2048-bit Morgan fingerprints from SMILES; `tabular_plus_smiles_fp` concatenates numeric features and Morgan fingerprints.
- Regression vs classification is inferred mainly from the fraction of numerically convertible target values. Training defaults to 12 Optuna trials and accepts a custom count.
- Training saves a bundle under `data/outputs`. `predict_property` can use an explicit `artifact_dir` or the latest successful model referenced by `LATEST`.

### PolyTAO

Fine-tuning CSV files must contain:

- `prompt`: property-condition text
- `target`: target polymer SMILES

`output_dir` and inverse-design `model_path` must be inside server-side `data/outputs`.

### Paths and runtime state

- Uploads are stored under `backend/data/uploads` and the API returns the server absolute path.
- Agent CSV tools accept only files inside `uploads`; output and model tools accept only paths inside `outputs`.
- Training state is an in-process singleton. Memory, plans, and skills are written under `backend/data`, so this is intended for a single local user.

---

## 🛠️ Agent tools and capabilities

### Polymer modeling and generation

| Tool | Role |
|------|------|
| `polyopus_chat` | Optional PolyOpus natural-language call; preferred for its configured polymer-property, band-gap, and QA domains. |
| `analyze_dataset` | Inspect CSV columns, SMILES, task type, and recommended modeling mode. |
| `train_property_model` | Optuna + stacking property-model training with `auto`, `tabular`, `rdkit_descriptors`, and `tabular_plus_smiles_fp`; `deepchem_gnn` is not available. |
| `predict_property` | Single-sample prediction from a saved bundle using new SMILES or numeric features; does not retrain. |
| `polytao_finetune` | Fine-tune PolyTAO from `prompt`/`target` CSV data. |
| `polytao_inverse_design` | Generate candidate SMILES from a local PolyTAO model under `outputs`. |

### Memory, plans, and skills

- `memory_read`, `memory_append`, `memory_update`, `memory_delete`, `memory_reflect` maintain `data/memory.md`; common keys, passwords, and ID-number patterns are rejected.
- `plan_create`, `plan_read`, `plan_set_step_status`, `plan_clear` maintain `data/active_plan.json`. Multi-step modeling and literature search plus synthesis/novelty/improvement requests use a plan first.
- `skill_list`, `skill_read`, `skill_create`, `skill_update`, `skill_delete` maintain `data/skills/<id>/SKILL.md` for reusable workflows; skill content also rejects sensitive credentials.

### DeepXiv literature tools

`deepxiv_search`, `deepxiv_trending`, `deepxiv_paper_brief`, `deepxiv_paper_head`, `deepxiv_paper_section`, `deepxiv_paper_fetch`, `deepxiv_paper_social`, `deepxiv_websearch`, `deepxiv_semantic_scholar`, `deepxiv_pmc_head`, `deepxiv_pmc_fetch`, `deepxiv_deep_dive`, and `deepxiv_digest_pack`.

They cover arXiv search, trending papers, progressive paper reading, social signals, web search, Semantic Scholar, PMC, single-paper deep dives, and multi-paper digest material. A simple paper-list request normally needs one `deepxiv_search`; synthesis requests are coupled to structured plans.

---

## 🔌 HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Returns service, health, and API-doc entry information. |
| `GET` | `/api/health` | Health check. |
| `GET` / `POST` | `/api/settings` | Read/write LLM settings; reads return `api_key_set`, not the key. |
| `POST` | `/api/upload` | Upload a file and return `{ filename, path, size_bytes }`; the filename is reduced to its basename. |
| `POST` | `/api/chat` | Agent turn; body: `{ "messages": [{"role":"user|assistant", "content":"..."}], "locale":"zh|en" }`. |
| `GET` | `/api/training/status` | Agent phase, Optuna progress, PolyTAO epoch, logs, and cancellation state. |
| `POST` | `/api/training/cancel` | Request cancellation; training stops at the next Optuna trial or PolyTAO epoch boundary, while the frontend disconnects from the chat wait. |

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## 🏭 Production deployment

Build the frontend and serve it from FastAPI on one port:

```bash
cd frontend
npm install
npm run build
```

Copy the complete `frontend/dist` tree into `backend/static` (Windows):

```bash
xcopy /E /I frontend\dist backend\static
```

Then run:

```bash
cd backend
python run.py
```

Open `http://127.0.0.1:8000`. For production, disable reload in `run.py` and use a process manager or proper uvicorn settings. Before exposing it publicly, add authentication, access control, rate limiting, and stricter upload limits; the current CORS configuration targets a local MVP.

---

## 📖 Citation

If you use PolyTAO pretrained weights, please cite:

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

## 📄 License

Apache License 2.0 (third-party dependencies and the PolyTAO pretrained model remain subject to their respective licenses and terms).
