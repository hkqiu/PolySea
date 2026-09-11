import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { PolymerChainsBackground } from "./PolymerChainsBackground";
import { getStoredLocale, getTranslations, storeLocale, type Locale } from "./i18n";

type Role = "user" | "assistant";

interface Msg {
  role: Role;
  content: string;
}

const STORAGE_KEY = "polysea_messages_v1";
const LEGACY_WELCOME_NOTICES = [
  "\n\n说明：我**不能**查询实时天气或联网检索；与聚合物建模无关的闲聊请直接文字交流，无需训练模型。任务进行中可随时点 **中止任务** 停止等待（训练会在当前试验/轮次边界停下）。",
  "\n\nNote: I **cannot** check live weather or browse the internet. For casual conversation unrelated to polymer modeling, just type a message—no model training is needed. While a task is running, click **Abort** at any time to stop waiting (training stops at the current trial/epoch boundary).",
] as const;

function removeLegacyWelcomeNotice(messages: Msg[]): Msg[] {
  return messages.map((message) => {
    if (message.role !== "assistant") return message;
    const content = LEGACY_WELCOME_NOTICES.reduce(
      (value, notice) => value.replace(notice, ""),
      message.content,
    );
    return content === message.content ? message : { ...message, content };
  });
}

function isLegacyWelcomeMessage(message: Msg): boolean {
  if (message.role !== "assistant") return false;
  return [
    "你好，我是 **PolySea**。",
    "你好，我是 **PolySea**。",
    "Hello, I'm **PolySea**.",
    "Hello, I'm **PolySea**.",
  ].some((prefix) => message.content.startsWith(prefix));
}

function defaultWelcome(locale: Locale): Msg {
  return { role: "assistant", content: getTranslations(locale).welcome };
}

function normalizeStoredMessages(messages: Msg[], locale: Locale): Msg[] {
  return messages.map((message) => {
    if (isLegacyWelcomeMessage(message)) return defaultWelcome(locale);
    return removeLegacyWelcomeNotice([message])[0] ?? message;
  });
}

function loadMessagesFromStorage(locale: Locale): Msg[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw) as unknown;
      if (Array.isArray(p) && p.length > 0) {
        return normalizeStoredMessages(p as Msg[], locale);
      }
    }
  } catch {
    /* ignore */
  }
  return [defaultWelcome(locale)];
}

interface AgentActivity {
  phase: string;
  message: string;
  detail: string;
}

interface ActivitySnapshot {
  agent: AgentActivity;
  active: boolean;
  kind: string;
  phase: string;
  optuna_current: number;
  optuna_total: number;
  optuna_best_value: number | null;
  epoch_current: number;
  epoch_total: number;
  train_loss: number | null;
  valid_loss: number | null;
  message: string;
  logs: string[];
}

/** 开发环境走 Vite 代理；若前后端分离部署，构建前设置 VITE_API_BASE，例如 http://127.0.0.1:8000 */
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

const BACKEND_UNREACHABLE = "backend_unreachable";
const REQUEST_TIMEOUT = "request_timeout";

function agentPhaseClass(phase: string): string {
  switch (phase) {
    case "thinking":
      return "border-[#d2d2d7] bg-[#f5f5f7] text-[#1d1d1f]";
    case "tool":
      return "border-[#e6d4a8] bg-[#faf8f2] text-[#5c4a21]";
    case "training":
      return "border-[#e8c4c4] bg-[#fdf6f6] text-[#6b2d2d]";
    case "optimizing":
      return "border-[#d4cce8] bg-[#f8f6fc] text-[#3d3566]";
    case "done":
      return "border-[#c4e0cc] bg-[#f4faf6] text-[#1f4d2e]";
    default:
      return "border-[#d2d2d7] bg-white text-[#1d1d1f]";
  }
}

const API_TIMEOUT_MS = 25_000;

/** 用户关闭设置或点「取消」时中止保存请求，与超时区分 */
class UserCancelledError extends Error {
  constructor() {
    super("user_cancelled");
    this.name = "UserCancelledError";
  }
}

function mergeTimeoutWithSignal(timeoutMs: number, outer: AbortSignal | undefined) {
  if (timeoutMs <= 0) {
    if (outer) return { signal: outer, clearTimer: () => {} };
    const idle = new AbortController();
    return { signal: idle.signal, clearTimer: () => {} };
  }
  const timeoutCtrl = new AbortController();
  const tid = window.setTimeout(() => timeoutCtrl.abort(), timeoutMs);
  const clearTimer = () => window.clearTimeout(tid);
  if (!outer) {
    return { signal: timeoutCtrl.signal, clearTimer };
  }
  if (typeof AbortSignal.any === "function") {
    return {
      signal: AbortSignal.any([timeoutCtrl.signal, outer]),
      clearTimer,
    };
  }
  return { signal: timeoutCtrl.signal, clearTimer };
}

type ApiRequestInit = RequestInit & { clientTimeoutMs?: number };

async function api<T>(path: string, init?: ApiRequestInit): Promise<T> {
  const { clientTimeoutMs, ...fetchInit } = init ?? {};
  const timeoutMs = clientTimeoutMs === undefined ? API_TIMEOUT_MS : clientTimeoutMs;
  const { signal, clearTimer } = mergeTimeoutWithSignal(
    timeoutMs,
    fetchInit.signal ?? undefined,
  );
  let r: Response;
  try {
    try {
      r = await fetch(apiUrl(path), {
        ...fetchInit,
        signal,
        headers: {
          "Content-Type": "application/json",
          ...(fetchInit.headers as Record<string, string> | undefined),
        },
      });
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") {
        if (fetchInit.signal?.aborted) {
          throw new UserCancelledError();
        }
        throw new Error(REQUEST_TIMEOUT);
      }
      if (e instanceof TypeError) {
        throw new Error(BACKEND_UNREACHABLE);
      }
      throw e instanceof Error ? e : new Error(String(e));
    }
    if (!r.ok) {
      const t = await r.text();
      const looksLikeViteProxy =
        r.status === 500 || r.status === 502 || r.status === 503
          ? /proxy error|ECONNREFUSED|vite|connect/i.test(t) || t.trim().length < 300
          : false;
      if (looksLikeViteProxy) {
        throw new Error(BACKEND_UNREACHABLE);
      }
      throw new Error(t || `${r.status} ${r.statusText}`);
    }
    return (await r.json()) as T;
  } finally {
    clearTimer();
  }
}

export default function App() {
  const [locale, setLocale] = useState<Locale>(getStoredLocale);
  const copy = getTranslations(locale);
  const [messages, setMessages] = useState<Msg[]>(() => loadMessagesFromStorage(locale));
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [baseUrl, setBaseUrl] = useState("");
  const [modelName, setModelName] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [keySet, setKeySet] = useState(false);
  const [uploadPath, setUploadPath] = useState<string | null>(null);
  const [settingsHint, setSettingsHint] = useState<string | null>(null);
  const [savingSettings, setSavingSettings] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<ActivitySnapshot | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const saveSettingsAbortRef = useRef<AbortController | null>(null);
  const chatAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      /* ignore */
    }
  }, [messages]);

  useEffect(() => {
    if (!loading) {
      setTrainingStatus(null);
      return;
    }
    const tick = async () => {
      try {
        const s = await api<ActivitySnapshot>("/api/training/status");
        setTrainingStatus(s);
      } catch {
        /* ignore */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 650);
    return () => window.clearInterval(id);
  }, [loading]);

  const showToast = (msg: string, ms = 4500) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), ms);
  };

  const scrollDown = () => bottomRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    scrollDown();
  }, [messages]);

  useEffect(() => {
    api<{ api_key_set: boolean; base_url: string; model: string }>("/api/settings")
      .then((s) => {
        setKeySet(s.api_key_set);
        setBaseUrl(s.base_url);
        setModelName(s.model);
      })
      .catch((e) => {
        const rawMessage = e instanceof Error ? e.message : String(e);
        const msg =
          rawMessage === BACKEND_UNREACHABLE
            ? copy.backendHint
            : rawMessage === REQUEST_TIMEOUT
              ? copy.requestTimeout
              : copy.cannotConnect.replace("{{error}}", rawMessage);
        setConnectionError(msg);
        setSettingsHint(msg);
      });
  }, [copy.backendHint, copy.cannotConnect, copy.requestTimeout]);

  const saveSettings = useCallback(async () => {
    saveSettingsAbortRef.current?.abort();
    const ac = new AbortController();
    saveSettingsAbortRef.current = ac;
    setSavingSettings(true);
    setSettingsHint(null);
    try {
      await api<{ ok: boolean }>("/api/settings", {
        method: "POST",
        signal: ac.signal,
        body: JSON.stringify({
          base_url: baseUrl,
          model: modelName,
          api_key: apiKey,
        }),
      });
      setKeySet(!!apiKey.trim());
      setConnectionError(null);
      setSettingsOpen(false);
      setSettingsHint(null);
      showToast(
        apiKey.trim() ? copy.saveSuccessWithKey : copy.saveSuccessWithoutKey
      );
    } catch (e) {
      if (e instanceof UserCancelledError) {
        return;
      }
      const rawMessage = e instanceof Error ? e.message : String(e);
      const msg =
        rawMessage === BACKEND_UNREACHABLE
          ? copy.backendHint
          : rawMessage === REQUEST_TIMEOUT
            ? copy.requestTimeout
            : copy.savedError.replace("{{error}}", rawMessage);
      setSettingsHint(msg);
    } finally {
      if (saveSettingsAbortRef.current === ac) {
        saveSettingsAbortRef.current = null;
      }
      setSavingSettings(false);
    }
  }, [apiKey, baseUrl, modelName, copy.backendHint, copy.requestTimeout, copy.saveSuccessWithKey, copy.saveSuccessWithoutKey, copy.savedError]);

  const closeSettingsModal = useCallback(() => {
    saveSettingsAbortRef.current?.abort();
    setSettingsOpen(false);
  }, []);

  const abortCurrentTask = async () => {
    try {
      await api<{ ok: boolean }>("/api/training/cancel", { method: "POST" });
    } catch {
      /* 后端不可达时仍尝试断开本地请求 */
    }
    chatAbortRef.current?.abort();
    showToast(copy.abortSent, 4000);
  };

  const sendChat = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    const userContent =
      uploadPath !== null ? `${text}\n\n（CSV 文件路径: ${uploadPath}）` : text;
    const history: Msg[] = [...messages, { role: "user", content: userContent }];
    setMessages(history);
    chatAbortRef.current?.abort();
    const ac = new AbortController();
    chatAbortRef.current = ac;
    setLoading(true);
    try {
      const payload = {
        locale,
        messages: history.map((m) => ({ role: m.role, content: m.content })),
      };
      const res = await api<{ reply: string; error?: boolean }>("/api/chat", {
        method: "POST",
        clientTimeoutMs: 0,
        signal: ac.signal,
        body: JSON.stringify(payload),
      });
      const reply =
        res.reply +
        (res.error ? copy.responseErrorSuffix : "");
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (e) {
      if (e instanceof UserCancelledError) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: copy.aborted,
          },
        ]);
        return;
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: copy.requestFailed.replace(
            "{{error}}",
            e instanceof Error ? e.message : String(e),
          ),
        },
      ]);
    } finally {
      if (chatAbortRef.current === ac) {
        chatAbortRef.current = null;
      }
      setLoading(false);
    }
  };

  const onFile = async (f: FileList | null) => {
    if (!f?.[0]) return;
    const form = new FormData();
    form.append("file", f[0]);
    const r = await fetch(apiUrl("/api/upload"), { method: "POST", body: form });
    if (!r.ok) {
      alert(copy.uploadFailed);
      return;
    }
    const j = (await r.json()) as { path: string; filename: string };
    setUploadPath(j.path);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: copy.uploadReceived.replace("{{filename}}", j.filename),
      },
    ]);
  };

  const toggleLocale = () => {
    const nextLocale: Locale = locale === "zh" ? "en" : "zh";
    const currentWelcome = getTranslations(locale).welcome;
    setLocale(nextLocale);
    storeLocale(nextLocale);
    setMessages((prev) =>
      prev.length === 1 &&
      prev[0]?.role === "assistant" &&
      (prev[0].content === currentWelcome || isLegacyWelcomeMessage(prev[0]))
        ? [defaultWelcome(nextLocale)]
        : prev,
    );
  };

  return (
    <div className="min-h-screen relative">
      <PolymerChainsBackground />

      {toast && (
        <div className="fixed top-5 left-1/2 z-[200] -translate-x-1/2 max-w-md px-5 py-2.5 rounded-full bg-[#1d1d1f] text-white text-[13px] text-center shadow-lg">
          {toast}
        </div>
      )}

      {connectionError && (
        <div className="relative z-10 max-w-3xl mx-auto px-5 pt-4">
          <div className="text-[13px] text-[#6b2d2d] whitespace-pre-wrap rounded-xl bg-[#fdf2f2] border border-[#f0c4c4] px-4 py-3 leading-relaxed">
            {connectionError}
          </div>
        </div>
      )}

      <header className="sticky top-0 z-40 border-b border-[#d2d2d7]/60 bg-[#fbfbfd]/92 backdrop-blur-md">
        <div className="max-w-4xl mx-auto flex items-center justify-between px-5 h-[52px]">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.35 }}
            className="flex items-center gap-3"
          >
            <div className="h-8 w-8 rounded-lg bg-[#0071e3] flex items-center justify-center">
              <span className="text-[11px] font-semibold text-white tracking-tight">HK</span>
            </div>
            <div>
              <h1 className="text-[17px] font-semibold text-[#1d1d1f] tracking-tight leading-tight">
                PolySea
              </h1>
              <p className="text-[11px] text-[#86868b] leading-tight">AI for Polymer Science</p>
            </div>
          </motion.div>

          <div className="flex items-center gap-2 sm:gap-3">
            {API_BASE ? (
              <span
                className="hidden sm:inline text-[11px] text-[#86868b] max-w-[120px] truncate"
                title={API_BASE}
              >
                {API_BASE}
              </span>
            ) : null}
            <button
              type="button"
              onClick={toggleLocale}
              title={copy.switchLanguage}
              aria-label={copy.switchLanguage}
              className="btn-ghost px-2 py-1 text-[#0071e3] font-medium"
            >
              {copy.languageName}
            </button>
            <span
              className={`text-[11px] px-2.5 py-0.5 rounded-full font-medium ${
                keySet
                  ? "bg-[#e8f4ea] text-[#1f4d2e]"
                  : "bg-[#fff8e6] text-[#8a6116]"
              }`}
            >
              {keySet ? copy.connected : copy.pending}
            </span>
            <button
              type="button"
              onClick={() => {
                if (!window.confirm(copy.clearConfirm)) return;
                setMessages([defaultWelcome(locale)]);
                try {
                  localStorage.removeItem(STORAGE_KEY);
                } catch {
                  /* ignore */
                }
              }}
              className="btn-ghost px-1 py-1 text-[#86868b] hover:text-[#0071e3] no-underline hover:underline"
            >
              {copy.clear}
            </button>
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="btn-secondary !py-2 !px-4 !text-[13px]"
            >
              {copy.settings}
            </button>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-4xl mx-auto px-5 py-8 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.25, 0.1, 0.25, 1] }}
          className="surface-elevated flex flex-col min-h-[calc(100vh-8rem)] overflow-hidden"
        >
          <div className="flex-1 overflow-y-auto min-h-[min(48vh,480px)] px-5 sm:px-8 py-8 space-y-6">
            <AnimatePresence initial={false}>
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[min(100%,36rem)] rounded-[20px] px-4 py-3 sm:px-5 sm:py-3.5 ${
                      m.role === "user"
                        ? "bg-[#e8e8ed] text-[#1d1d1f]"
                        : "bg-white text-[#1d1d1f] border border-[#d2d2d7]/70 shadow-sm"
                    }`}
                  >
                    <div className="whitespace-pre-wrap break-words text-[15px] leading-[1.5] text-[#1d1d1f]">
                      {m.content.split("**").map((part, j) =>
                        j % 2 === 1 ? (
                          <strong key={j} className="font-semibold text-[#0071e3]">
                            {part}
                          </strong>
                        ) : (
                          <span key={j}>{part}</span>
                        )
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={bottomRef} />
          </div>

          {loading ? (
            <div className="shrink-0 border-t border-[#d2d2d7]/60 px-5 sm:px-8 py-4 bg-[#f5f5f7]/80">
              <div
                className={`rounded-2xl border px-4 py-3 ${agentPhaseClass(trainingStatus?.agent?.phase ?? "thinking")}`}
              >
                <div className="flex items-start gap-3">
                  <span className="flex gap-1.5 shrink-0 mt-1.5">
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={i}
                        className="w-1 h-1 rounded-full bg-[#86868b]"
                        animate={{ opacity: [0.25, 1, 0.25] }}
                        transition={{ repeat: Infinity, duration: 1.1, delay: i * 0.15 }}
                      />
                    ))}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="text-[15px] font-medium text-[#1d1d1f] leading-snug">
                      {trainingStatus?.agent?.message || copy.processing}
                    </div>
                    {trainingStatus?.agent?.detail ? (
                      <div className="text-[12px] text-[#6e6e73] mt-1 leading-relaxed">
                        {trainingStatus.agent.detail}
                      </div>
                    ) : null}
                  </div>
                  <button type="button" onClick={() => void abortCurrentTask()} className="btn-danger-outline">
                    {copy.abort}
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {loading && trainingStatus?.active ? (
            <div className="shrink-0 border-t border-[#d2d2d7]/60 bg-[#f5f5f7] px-5 sm:px-8 py-4">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <div className="flex flex-wrap items-center gap-2 min-w-0">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-[#86868b]">
                    {trainingStatus.kind === "polytao" ? "PolyTAO" : copy.propertyPrediction}
                  </span>
                  <span className="text-[13px] text-[#1d1d1f]">{trainingStatus.message}</span>
                </div>
                <button type="button" onClick={() => void abortCurrentTask()} className="btn-danger-outline">
                  {copy.abortTraining}
                </button>
              </div>
              {trainingStatus.kind !== "polytao" &&
              trainingStatus.optuna_total > 0 &&
              trainingStatus.phase !== "fit_best" ? (
                <div className="h-1 rounded-full bg-[#d2d2d7]/80 overflow-hidden mb-2">
                  <div
                    className="h-full bg-[#0071e3] transition-all duration-300 ease-out"
                    style={{
                      width: `${Math.min(100, (trainingStatus.optuna_current / Math.max(1, trainingStatus.optuna_total)) * 100)}%`,
                    }}
                  />
                </div>
              ) : null}
              {trainingStatus.kind === "polytao" && trainingStatus.epoch_total > 0 ? (
                <div className="h-1 rounded-full bg-[#d2d2d7]/80 overflow-hidden mb-2">
                  <div
                    className="h-full bg-[#0071e3] transition-all duration-300 ease-out"
                    style={{
                      width: `${Math.min(100, (trainingStatus.epoch_current / Math.max(1, trainingStatus.epoch_total)) * 100)}%`,
                    }}
                  />
                </div>
              ) : null}
              {trainingStatus.phase === "fit_best" ? (
                <p className="text-[12px] text-[#6e6e73] mb-2">
                  {copy.fitBest}
                </p>
              ) : null}
              <div className="max-h-20 overflow-y-auto rounded-xl bg-white border border-[#d2d2d7]/50 px-3 py-2 font-mono text-[11px] text-[#6e6e73] leading-relaxed">
                {trainingStatus.logs.slice(-12).map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="shrink-0 border-t border-[#d2d2d7]/60 p-5 sm:p-6 bg-white">
            <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-end">
              <label className="shrink-0">
                <input
                  type="file"
                  accept=".csv,text/csv"
                  className="hidden"
                  onChange={(e) => void onFile(e.target.files)}
                />
                <span className="btn-secondary flex h-11 sm:h-12 items-center cursor-pointer">
                  {copy.uploadCsv}
                </span>
              </label>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendChat();
                  }
                }}
                placeholder={copy.placeholder}
                rows={3}
                className="input-apple flex-1 resize-y min-h-[5.5rem] max-h-48 !rounded-2xl !py-3"
              />
              <button
                type="button"
                disabled={loading}
                onClick={() => void sendChat()}
                className="btn-primary h-11 sm:h-12 sm:self-stretch sm:!px-10 !rounded-2xl"
              >
                {copy.send}
              </button>
            </div>
          </div>
        </motion.div>
      </main>

      <AnimatePresence>
        {settingsOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-5 bg-black/32 backdrop-blur-[6px]"
            onClick={closeSettingsModal}
          >
            <motion.div
              initial={{ opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.98 }}
              transition={{ duration: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
              onClick={(e) => e.stopPropagation()}
              className="surface-elevated w-full max-w-[420px] p-8 shadow-[0_16px_48px_rgba(0,0,0,0.12)]"
            >
              <h2 className="text-[21px] font-semibold text-[#1d1d1f] tracking-tight mb-1">{copy.modelConnection}</h2>
              <p className="text-[13px] text-[#86868b] mb-6 leading-relaxed">
                {copy.apiHint}
              </p>
              {settingsHint && (
                <div className="mb-5 text-[13px] text-[#6b4a00] whitespace-pre-wrap rounded-xl bg-[#fffbf0] border border-[#f0e0b2] px-4 py-3 leading-relaxed">
                  {settingsHint}
                </div>
              )}
              <div className="space-y-5">
                <div>
                  <label className="block text-[12px] font-medium text-[#6e6e73] mb-1.5">{copy.baseUrl}</label>
                  <input
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    className="input-apple"
                  />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#6e6e73] mb-1.5">{copy.model}</label>
                  <input
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="input-apple"
                  />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#6e6e73] mb-1.5">{copy.apiKey}</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={copy.required}
                    className="input-apple"
                  />
                </div>
              </div>
              <div className="mt-10 flex gap-3 justify-end">
                <button type="button" onClick={closeSettingsModal} className="btn-secondary !px-5">
                  {copy.cancel}
                </button>
                <button
                  type="button"
                  disabled={savingSettings}
                  onClick={() => void saveSettings()}
                  className="btn-primary !rounded-xl disabled:opacity-40"
                >
                  {savingSettings ? copy.saving : copy.done}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
