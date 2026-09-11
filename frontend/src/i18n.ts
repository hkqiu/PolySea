export type Locale = "zh" | "en";

const translations = {
  zh: {
    welcome:
      "你好，我是 **PolySea**，面向聚合物科学家的智能研究助手。我可以协助你完成全流程的聚合物任务，包括聚合物性质预测、聚合物逆向设计与聚合物知识问答；同时集成自动化机器学习训练能力，几乎可以扩展至任意聚合物性质的建模。请先打开右上角的 **设置**，填写 LLM 的 API Key，然后上传 CSV 文件或直接描述你的任务。",
    backendHint:
      "无法连接 PolySea 后端（127.0.0.1:8000）。请先启动后端：在仓库 `backend` 目录执行 `python run.py`；或在项目根目录 `PolySea` 下执行 `npm install` 后运行 `npm run dev`（会同时启动 API 与前端）。然后刷新本页。",
    requestTimeout:
      "请求超时（25 秒内无响应）。请确认：1）PolySea 后端已在运行（默认 127.0.0.1:8000）；2）开发时用 Vite 打开 5173 端口以便代理 /api；3）若静态部署，构建时需设置 VITE_API_BASE 指向后端根地址。",
    propertyPrediction: "性质预测",
    connected: "已连接",
    pending: "待配置",
    clear: "清空",
    clearConfirm: "确定清空当前会话记录？（刷新后也不会恢复）",
    settings: "设置",
    abort: "中止",
    abortTraining: "中止训练",
    processing: "正在处理…",
    fitBest: "正在拟合最优模型并在测试集上评估…",
    uploadCsv: "上传 CSV",
    placeholder: "描述任务…（Shift+Enter 换行）",
    send: "发送",
    modelConnection: "模型连接",
    apiHint: "API 密钥仅保存在本机后端内存，不会写入磁盘或仓库。",
    required: "必填",
    cancel: "取消",
    saving: "保存中…",
    done: "完成",
    baseUrl: "Base URL",
    model: "Model",
    apiKey: "API Key",
    switchLanguage: "切换语言",
    languageName: "English",
    saveSuccessWithKey: "已保存。API Key 已写入后端进程内存（重启后端需重新填写）。",
    saveSuccessWithoutKey: "已保存（当前未填写 API Key，对话前请先填写有效密钥）。",
    abortSent: "已发送中止请求（对话会立即结束等待；训练在下一试验/epoch 边界停止）",
    aborted:
      "已中止：前端已断开本次对话请求。若后端仍在训练，性质预测会在当前 Optuna 试验结束后停止，PolyTAO 会在当前 epoch 结束后停止。",
    uploadFailed: "上传失败",
    uploadReceived:
      "已接收文件 `{{filename}}`（服务器路径已附带在你下一条消息中）。\n\n建议你先告诉我：**要预测哪一列**（目标性质）、更倾向 **纯表格特征**、**基于 SMILES 的分子指纹**，还是 **表格 + 指纹**；若有多个 SMILES 列请指明列名；需要的话还可说明 **Optuna 搜索大约跑多少次试验**（默认 12）。你也可以直接说「先分析一下」或「用默认设置开始训练」——我会按你的明确程度配合。",
    responseErrorSuffix: "\n\n（本条由后端错误处理返回，请根据上文排查 LLM 配置。）",
    requestFailed: "请求失败：{{error}}",
    cannotConnect: "无法连接后端：{{error}}。请先在本机 `backend` 目录执行 `python run.py`（默认 8000 端口），或在项目根目录执行 `npm run dev`，再刷新本页。",
    savedError:
      "保存失败：{{error}}\n请确认：1）后端已启动；2）若前端非 Vite 开发服务器，请在构建时设置环境变量 VITE_API_BASE 指向后端，例如 http://127.0.0.1:8000",
  },
  en: {
    welcome:
      "Hello, I’m **PolySea**, an intelligent research assistant for polymer scientists. I support end-to-end polymer workflows, including polymer property prediction, inverse design, and polymer knowledge Q&A. With integrated automated machine-learning training, PolySea can be extended to model nearly any polymer property. Please open **Settings** in the top-right corner to enter your LLM API key, then upload a CSV file or describe your task directly.",
    backendHint:
      "Unable to connect to the PolySea backend (127.0.0.1:8000). Start it first: run `python run.py` in the repository's `backend` directory; or run `npm install` and then `npm run dev` in the project root `PolySea` (this starts both the API and frontend). Then refresh this page.",
    requestTimeout:
      "Request timed out (no response within 25 seconds). Please confirm: 1) the PolySea backend is running (default 127.0.0.1:8000); 2) during development, open port 5173 through Vite so /api can be proxied; 3) for static deployment, set VITE_API_BASE to the backend root URL when building.",
    propertyPrediction: "Property prediction",
    connected: "Connected",
    pending: "Not configured",
    clear: "Clear",
    clearConfirm: "Clear the current conversation? (It cannot be restored after refresh.)",
    settings: "Settings",
    abort: "Abort",
    abortTraining: "Abort training",
    processing: "Processing…",
    fitBest: "Fitting the best model and evaluating it on the test set…",
    uploadCsv: "Upload CSV",
    placeholder: "Describe your task… (Shift+Enter for a new line)",
    send: "Send",
    modelConnection: "Model connection",
    apiHint: "The API key is stored only in the local backend memory and is never written to disk or the repository.",
    required: "Required",
    cancel: "Cancel",
    saving: "Saving…",
    done: "Done",
    baseUrl: "Base URL",
    model: "Model",
    apiKey: "API Key",
    switchLanguage: "Switch language",
    languageName: "中文",
    saveSuccessWithKey: "Saved. The API key is stored in the backend process memory (you must enter it again after a backend restart).",
    saveSuccessWithoutKey: "Saved (no API key is currently configured; enter a valid key before chatting).",
    abortSent: "Abort request sent (the chat wait ends immediately; training stops at the next trial/epoch boundary).",
    aborted:
      "Aborted: the frontend disconnected this chat request. If the backend is still training, property prediction will stop after the current Optuna trial and PolyTAO after the current epoch.",
    uploadFailed: "Upload failed",
    uploadReceived:
      "File `{{filename}}` received (the server path will be attached to your next message).\n\nFirst tell me: **which column to predict** (the target property), whether you prefer **tabular features**, **SMILES-based molecular fingerprints**, or **tabular + fingerprints**. If there are multiple SMILES columns, specify the column name. You may also state roughly **how many Optuna trials to run** (12 by default). You can simply say “analyze it first” or “start training with default settings”—I will adapt to your level of detail.",
    responseErrorSuffix: "\n\n(This response was generated by backend error handling; check the LLM configuration using the information above.)",
    requestFailed: "Request failed: {{error}}",
    cannotConnect: "Unable to connect to the backend: {{error}}. Run `python run.py` in the local `backend` directory (default port 8000), or run `npm run dev` in the project root, then refresh this page.",
    savedError:
      "Save failed: {{error}}\nPlease confirm: 1) the backend is running; 2) if the frontend is not using the Vite development server, set VITE_API_BASE at build time to the backend URL, for example http://127.0.0.1:8000",
  },
} as const;

export type TranslationKey = keyof typeof translations.zh;

export function getTranslations(locale: Locale) {
  return translations[locale];
}

export function getStoredLocale(): Locale {
  try {
    return localStorage.getItem("polysea_locale") === "en" ? "en" : "zh";
  } catch {
    return "zh";
  }
}

export function storeLocale(locale: Locale) {
  try {
    localStorage.setItem("polysea_locale", locale);
  } catch {
    /* ignore */
  }
}
