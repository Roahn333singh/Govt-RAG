/**
 * Manual AI UP — Light Civic UI
 * Single 9-dot loader + swapping stage label; light/dark themes.
 */

const API_BASE =
  typeof window !== "undefined" && /^https?:$/.test(window.location.protocol)
    ? window.location.origin
    : "http://127.0.0.1:8000";
const CHAT_URL = `${API_BASE}/chat`;
const STREAM_URL = `${API_BASE}/chat/stream`;
const SESSION_URL = `${API_BASE}/ui/session`;

let authToken = null;
let authTokenExpiry = 0;
let sessionId = null;
let isLoading = false;
let chatMessages = [];
let uiLang = localStorage.getItem("manual_ai_lang") || "en";
let activeStreamAbort = null;

const STAGE_LABELS = {
  en: {
    starting: "Preparing query",
    embedding: "Preparing search",
    cache_lookup: "Checking saved answers",
    cached: "Loading saved answer",
    rewriting: "Refining query",
    retrieving: "Retrieving manuals",
    generating: "Writing answer",
    done: "",
  },
  hi: {
    starting: "प्रश्न तैयार हो रहा है",
    embedding: "खोज तैयार हो रही है",
    cache_lookup: "संग्रहित उत्तर जाँच",
    cached: "संग्रहित उत्तर",
    rewriting: "प्रश्न परिष्कृत हो रहा है",
    retrieving: "नियमावली खोज",
    generating: "उत्तर लिखा जा रहा है",
    done: "",
  },
};

const I18N = {
  hi: {
    dept_short: "उत्तर प्रदेश शासन नियमावली सहायक",
    suggested: "सुझाए गए प्रश्न",
    secure_note: "आपका डेटा सुरक्षित है। सार्वजनिक सेवा हेतु निर्मित।",
    greeting: "नमस्ते! आज मैं आपकी कैसे मदद करूँ?",
    greeting_sub: "नियमावलियों, संहिताओं या आधिकारिक प्रक्रियाओं के बारे में पूछें।",
    new_session: "नया चैट",
    hero_blurb: "उत्तर प्रदेश की सरकारी नियमावलियों पर आधारित सहायक — स्पष्ट, प्रयोग योग्य उत्तर।",
    nav_chat: "चैट",
    nav_suggested: "सुझाए गए प्रश्न",
    nav_about: "इस टूल के बारे में",
    nav_pragati: "प्रगति पोर्टल URL",
    nav_igrs: "IGRS अधिकतम फाइल साइज",
    nav_200cr: "₹200 करोड़+ स्वीकृति",
    nav_tender: "निविदा समिति के कार्य",
    nav_ce: "मुख्य अभियंता के वित्तीय अधिकार",
    chip1: "एकल बोली निविदा कब स्वीकार्य?",
    chip2: "मुख्य अभियंता की वित्तीय शक्तियाँ",
    chip3: "जन-सुनवाई जिला स्तर पर कब लागू?",
    chip4: "सरकारी सेवक की सेवा अवधि का आधार",
    ask_placeholder: "नियमावली या प्रक्रिया के बारे में प्रश्न लिखें…",
    disclaimer: "प्रशासनिक मार्गदर्शन हेतु। हस्ताक्षरित अधिसूचना ही अंतिम मान्य है।",
    sources: "स्रोत",
    copy: "प्रतिलिपि",
    copied: "कॉपी हो गया",
    assistant: "Manual AI UP",
    about_title: "Manual AI UP के बारे में",
    about_body: "उत्तर आधिकारिक नियमावलियों से प्राप्त होते हैं और सत्यापन हेतु उद्धृत किए जाते हैं। हस्ताक्षरित आदेशों का विकल्प नहीं।",
    about_close: "बंद करें",
  },
  en: {
    dept_short: "Uttar Pradesh Government Manuals Assistant",
    suggested: "Suggested questions",
    secure_note: "Your data is secure. Built for public service.",
    greeting: "Namaste! How can I help you today?",
    greeting_sub: "Ask about government manuals, codes, or official procedures.",
    new_session: "New Chat",
    hero_blurb: "Your assistant for Uttar Pradesh government manuals — clear, usable answers from official procedures.",
    nav_chat: "Chat",
    nav_suggested: "Suggested Questions",
    nav_about: "About This Tool",
    nav_pragati: "Pragati Portal URL",
    nav_igrs: "IGRS max attachment size",
    nav_200cr: "₹200 cr+ project approval",
    nav_tender: "Tender committee duties",
    nav_ce: "Chief Engineer financial powers",
    chip1: "When can a single-bid tender be accepted?",
    chip2: "Financial powers of a Chief Engineer",
    chip3: "When was Jan-Sunwai rolled out at district level?",
    chip4: "Constitutional basis of a servant's tenure",
    ask_placeholder: "Ask a question about government manuals or procedures…",
    disclaimer: "For administrative guidance. Signed notifications remain authoritative.",
    sources: "Sources",
    copy: "Copy",
    copied: "Copied",
    assistant: "Manual AI UP",
    about_title: "About Manual AI UP",
    about_body: "Answers are retrieved from official UP government manuals and cited for verification. Not a substitute for signed orders.",
    about_close: "Close",
  },
};

function t(key) {
  return (I18N[uiLang] && I18N[uiLang][key]) || I18N.en[key] || key;
}

function stageLabel(stage) {
  const pack = STAGE_LABELS[uiLang] || STAGE_LABELS.en;
  return pack[stage] || pack.starting;
}

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("manual_ai_theme", next);
  const btn = document.getElementById("btn-toggle-theme");
  if (btn) {
    btn.setAttribute("aria-label", next === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }
}

function applyUiLang(lang) {
  uiLang = lang === "hi" ? "hi" : "en";
  localStorage.setItem("manual_ai_lang", uiLang);
  document.documentElement.lang = uiLang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (key && I18N[uiLang][key] != null) el.textContent = I18N[uiLang][key];
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (key && I18N[uiLang][key] != null) el.setAttribute("placeholder", I18N[uiLang][key]);
  });
  const btn = document.getElementById("btn-toggle-lang");
  if (btn) {
    btn.querySelector(".lang-en")?.classList.toggle("lang-active", uiLang === "en");
    btn.querySelector(".lang-hi")?.classList.toggle("lang-active", uiLang === "hi");
  }
}

async function getAuthToken(force = false) {
  if (!force && authToken && Date.now() < authTokenExpiry) return authToken;
  const res = await fetch(SESSION_URL, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`Could not start a session (status ${res.status})`);
  const data = await res.json();
  authToken = data.token;
  authTokenExpiry = Date.now() + Math.max(0, (data.expires_in - 60) * 1000);
  return authToken;
}

async function authedFetch(url, options = {}) {
  const send = async (token) =>
    fetch(url, {
      ...options,
      headers: { ...(options.headers || {}), "X-Session-Token": token },
    });
  let res = await send(await getAuthToken());
  if (res.status === 401) res = await send(await getAuthToken(true));
  return res;
}

const thread = document.getElementById("thread");
const input = document.getElementById("composer-input");
const btnSend = document.getElementById("btn-send");
const form = document.getElementById("composer-form");
const hero = document.getElementById("hero");
const sidebar = document.getElementById("sidebar");
const backdrop = document.getElementById("sidebar-backdrop");
const btnToggleSidebar = document.getElementById("btn-toggle-sidebar");
const btnCloseSidebar = document.getElementById("btn-close-sidebar");
const btnNewChat = document.getElementById("btn-new-chat");
const btnLang = document.getElementById("btn-toggle-lang");
const btnTheme = document.getElementById("btn-toggle-theme");
const aboutDialog = document.getElementById("about-dialog");

function openSidebar() {
  sidebar.classList.add("is-open");
  backdrop.classList.add("is-open");
  backdrop.hidden = false;
  btnToggleSidebar?.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  sidebar.classList.remove("is-open");
  backdrop.classList.remove("is-open");
  backdrop.hidden = true;
  btnToggleSidebar?.setAttribute("aria-expanded", "false");
}

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Drop trailing Sources / स्रोत blocks — citation stays backend-only. */
function stripSourcesBlock(text) {
  return String(text || "")
    .replace(/(?:\r?\n|^)\s*(?:Sources|स्रोत)\s*:\s*[\s\S]*$/i, "")
    .replace(/\s+$/g, "")
    .trim();
}

function isSafeHttpUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

function normalizeHref(rawUrl) {
  const trimmed = String(rawUrl || "").replace(/[.,;:!?)]+$/g, "");
  if (!trimmed) return "";
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (/^www\./i.test(trimmed)) return `https://${trimmed}`;
  return `https://${trimmed}`;
}

function anchorHtml(href, label) {
  const normalized = normalizeHref(href);
  if (!isSafeHttpUrl(normalized)) return escapeHtml(label || href);
  const safeHref = escapeHtml(normalized);
  const safeLabel = escapeHtml(label || href);
  return `<a class="msg-link" href="${safeHref}" target="_blank" rel="noopener noreferrer">${safeLabel}</a>`;
}

function shouldSkipUrlMatch(full, offset) {
  const before = full.slice(Math.max(0, offset - 8), offset);
  return /href\s*=\s*["']?$/i.test(before) || before.endsWith(">") || before.endsWith('"') || before.endsWith("'");
}

/** Inline markdown + clickable URLs (including bare www. hosts). Input must be escaped first. */
function formatInline(escaped) {
  let html = escaped;
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|www\.[^)\s]+)\)/gi, (_, label, url) =>
    anchorHtml(url, label)
  );
  html = html.replace(
    /(?:https?:\/\/|www\.)[^\s<>"'`)\]}]+/gi,
    (match, offset, full) => {
      if (shouldSkipUrlMatch(full, offset)) return match;
      const trimmed = match.replace(/[.,;:!?)]+$/g, "");
      const trailing = match.slice(trimmed.length);
      return anchorHtml(trimmed, trimmed) + trailing;
    }
  );
  return html;
}

function formatBotResponse(text) {
  if (!text) return "";
  const raw = stripSourcesBlock(text);
  const lines = raw.split(/\r?\n/);
  const blocks = [];
  let i = 0;

  const flushParagraph = (buf) => {
    const joined = buf.join("\n").trim();
    if (!joined) return;
    blocks.push(`<p>${formatInline(escapeHtml(joined)).replace(/\n/g, "<br>")}</p>`);
  };

  while (i < lines.length) {
    const line = lines[i];
    const heading = /^(#{2,3})\s+(.+)$/.exec(line);
    if (heading) {
      blocks.push(`<h3>${formatInline(escapeHtml(heading[2].trim()))}</h3>`);
      i += 1;
      continue;
    }

    const ordered = /^\s*\d+\.\s+(.+)$/.exec(line);
    const bullet = /^\s*[-*]\s+(.+)$/.exec(line);
    if (ordered || bullet) {
      const tag = ordered ? "ol" : "ul";
      const items = [];
      while (i < lines.length) {
        const o = /^\s*\d+\.\s+(.+)$/.exec(lines[i]);
        const b = /^\s*[-*]\s+(.+)$/.exec(lines[i]);
        const item = tag === "ol" ? o : b;
        if (!item) break;
        items.push(`<li>${formatInline(escapeHtml(item[1].trim()))}</li>`);
        i += 1;
      }
      blocks.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    if (!line.trim()) {
      i += 1;
      continue;
    }

    const buf = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{2,3})\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i])
    ) {
      buf.push(lines[i]);
      i += 1;
    }
    flushParagraph(buf);
  }

  return blocks.join("") || "";
}

function svgUse(id, className = "ico") {
  return `<svg class="${className}" aria-hidden="true"><use href="#${id}"></use></svg>`;
}

function thumbIcon(up) {
  const d = up
    ? "M7.5 21H5.8A1.8 1.8 0 0 1 4 19.2V11.8A1.8 1.8 0 0 1 5.8 10H7.5m0 11V10m0 11h6.4a2 2 0 0 0 1.95-1.55l1.4-6.1A1.5 1.5 0 0 0 15.8 10H12V6.2A2.2 2.2 0 0 0 9.8 4L7.5 10"
    : "M7.5 3H5.8A1.8 1.8 0 0 0 4 4.8v7.4A1.8 1.8 0 0 0 5.8 14H7.5m0-11v11m0-11h6.4a2 2 0 0 1 1.95 1.55l1.4 6.1A1.5 1.5 0 0 1 15.8 14H12v3.8A2.2 2.2 0 0 1 9.8 20L7.5 14";
  return `<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${d}"/></svg>`;
}

function dot9Html() {
  return `<span class="dot9" aria-hidden="true">${"<span></span>".repeat(9)}</span>`;
}

function createBotShell() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--bot";
  wrap.innerHTML = `
    <img class="msg__avatar" src="./assets/logo.jpg?v=660" width="34" height="34" alt="" decoding="async"/>
    <div class="msg__card">
      <div class="msg__head">
        <span class="msg__name">${escapeHtml(t("assistant"))}</span>
        <span class="stage-badge" data-role="stage">
          ${dot9Html()}
          <span class="stage-badge__label" data-role="stage-label"></span>
        </span>
      </div>
      <div class="msg__body" data-role="body"></div>
      <div class="msg__footer" data-role="footer" hidden>
        <div data-role="sources"></div>
        <div class="msg__actions" data-role="actions"></div>
      </div>
    </div>`;
  return wrap;
}

function setStage(shell, stage) {
  const badge = shell.querySelector('[data-role="stage"]');
  const label = shell.querySelector('[data-role="stage-label"]');
  if (!badge || !label) return;
  const text = stageLabel(stage);
  if (!text || stage === "done") {
    badge.hidden = true;
    label.textContent = "";
    return;
  }
  badge.hidden = false;
  if (label.textContent !== text) label.textContent = text;
}

function attachSourcesAndActions(shell, text) {
  const footer = shell.querySelector('[data-role="footer"]');
  const sourcesBox = shell.querySelector('[data-role="sources"]');
  const actions = shell.querySelector('[data-role="actions"]');
  if (!footer || !actions) return;

  if (sourcesBox) sourcesBox.innerHTML = "";
  actions.innerHTML = "";
  footer.hidden = false;

  const displayText = stripSourcesBlock(text);

  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "btn btn-ghost btn-icon";
  copy.title = t("copy");
  copy.setAttribute("aria-label", t("copy"));
  copy.innerHTML = svgUse("i-copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(displayText || "");
      copy.innerHTML = svgUse("i-check");
      setTimeout(() => {
        copy.innerHTML = svgUse("i-copy");
      }, 1000);
    } catch {
      /* ignore */
    }
  });

  const up = document.createElement("button");
  up.type = "button";
  up.className = "btn btn-ghost btn-icon";
  up.title = "Helpful";
  up.setAttribute("aria-label", "Helpful");
  up.innerHTML = thumbIcon(true);
  const down = document.createElement("button");
  down.type = "button";
  down.className = "btn btn-ghost btn-icon";
  down.title = "Not helpful";
  down.setAttribute("aria-label", "Not helpful");
  down.innerHTML = thumbIcon(false);
  [up, down].forEach((btn) => {
    btn.addEventListener("click", () => {
      up.classList.remove("is-active");
      down.classList.remove("is-active");
      btn.classList.add("is-active");
    });
  });

  actions.append(copy, up, down);
}

function appendUser(text, { persist = true } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--user";
  const col = document.createElement("div");
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = text;
  const meta = document.createElement("div");
  meta.className = "msg__meta";
  meta.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  col.append(bubble, meta);
  wrap.appendChild(col);
  thread.appendChild(wrap);
  if (persist) chatMessages.push({ role: "user", text });
  scrollToBottom();
}

function visibleQuery(btn) {
  const key = btn.getAttribute("data-i18n-query");
  return (key ? t(key) : "").trim();
}

function triggerQuickQuery(query) {
  if (!query || isLoading) return;
  input.value = query;
  input.dispatchEvent(new Event("input"));
  closeSidebar();
  sendMessage();
}

function saveSession() {
  try {
    localStorage.setItem(
      "manual_ai_session",
      JSON.stringify({ sessionId, messages: chatMessages, lang: uiLang })
    );
  } catch {
    /* quota */
  }
}

function restoreSession() {
  try {
    const raw = localStorage.getItem("manual_ai_session");
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.lang) applyUiLang(data.lang);
    if (data.sessionId) sessionId = data.sessionId;
    const messages = Array.isArray(data.messages) ? data.messages : [];
    if (!messages.length) return;
    hero?.remove();
    chatMessages = [];
    messages.forEach((m) => {
      if (m.role === "user") appendUser(m.text || "", { persist: true });
      else if (m.role === "bot") {
        const shell = createBotShell();
        setStage(shell, "done");
        shell.querySelector('[data-role="body"]').innerHTML = formatBotResponse(m.text || "");
        attachSourcesAndActions(shell, m.text || "");
        thread.appendChild(shell);
        chatMessages.push({ role: "bot", text: m.text || "", sources: m.sources || [] });
      }
    });
    scrollToBottom();
  } catch {
    /* ignore */
  }
}

function startNewChat() {
  if (activeStreamAbort) activeStreamAbort.abort();
  sessionId = null;
  chatMessages = [];
  localStorage.removeItem("manual_ai_session");
  thread.innerHTML = "";
  if (hero) thread.appendChild(hero);
  else location.reload();
  closeSidebar();
  input.focus();
}

async function streamAnswer(question) {
  if (activeStreamAbort) activeStreamAbort.abort();
  const controller = new AbortController();
  activeStreamAbort = controller;

  let res;
  try {
    res = await authedFetch(STREAM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message: question, question, session_id: sessionId }),
      signal: controller.signal,
    });
  } catch (err) {
    if (err && err.name === "AbortError") return { ok: false, aborted: true };
    return { ok: false, aborted: false, error: err.message || "Network error" };
  }

  if (!res.ok || !res.body) {
    let detail = `Stream status ${res.status}`;
    try {
      const payload = await res.json();
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      /* ignore */
    }
    return { ok: false, aborted: false, error: detail };
  }

  const shell = createBotShell();
  setStage(shell, "starting");
  thread.appendChild(shell);
  const body = shell.querySelector('[data-role="body"]');
  let text = "";
  let sources = [];
  let sawToken = false;
  let streamError = null;

  const render = () => {
    const caret = sawToken && !streamError ? '<span class="stream-caret" aria-hidden="true"></span>' : "";
    body.innerHTML = formatBotResponse(text) + caret;
    scrollToBottom();
  };

  const applyEvent = (ev) => {
    if (!ev || typeof ev !== "object") return;
    if (ev.type === "status") {
      if (!sawToken) setStage(shell, ev.stage || "starting");
      return;
    }
    if (ev.type === "token") {
      sawToken = true;
      setStage(shell, "done");
      text += ev.text || "";
      render();
      return;
    }
    if (ev.type === "replace") {
      sawToken = true;
      setStage(shell, "done");
      text = ev.text || "";
      render();
      return;
    }
    if (ev.type === "error") {
      streamError = ev.text || "Stream failed";
      text = text ? `${text}\n\n${streamError}` : streamError;
      sawToken = true;
      setStage(shell, "done");
      render();
      return;
    }
    if (ev.type === "done") {
      sources = Array.isArray(ev.sources) ? ev.sources : [];
      if (ev.session_id) sessionId = ev.session_id;
      setStage(shell, "done");
    }
  };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let split;
      while ((split = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, split);
        buf = buf.slice(split + 2);
        const line = block.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        try {
          applyEvent(JSON.parse(line.slice(5).trim()));
        } catch {
          /* bad frame */
        }
      }
    }
  } catch (err) {
    if (err && err.name === "AbortError") return { ok: false, aborted: true };
    throw err;
  } finally {
    if (activeStreamAbort === controller) activeStreamAbort = null;
    body.querySelector(".stream-caret")?.remove();
  }

  if (!text) {
    text = streamError || "Unable to retrieve an answer right now. Please try again.";
    body.innerHTML = formatBotResponse(text);
  } else {
    body.innerHTML = formatBotResponse(text);
  }
  attachSourcesAndActions(shell, text);
  chatMessages.push({ role: "bot", text, sources });
  saveSession();
  return { ok: true, aborted: false };
}

async function sendMessage() {
  const question = (input.value || "").trim();
  if (!question || isLoading) return;

  hero?.remove();
  input.value = "";
  input.style.height = "auto";
  btnSend.disabled = true;
  appendUser(question);
  saveSession();
  isLoading = true;

  try {
    if (!sessionId) sessionId = crypto.randomUUID();
    const result = await streamAnswer(question);
    if (result.aborted || result.ok) return;

    const res = await authedFetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, question, session_id: sessionId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((typeof err.detail === "string" && err.detail) || result.error || `Server ${res.status}`);
    }
    const data = await res.json();
    const answer = typeof data === "string" ? data : data.answer || data.response || "";
    if (data && data.session_id) sessionId = data.session_id;
    const sources = (data && data.sources) || [];
    const shell = createBotShell();
    setStage(shell, "done");
    shell.querySelector('[data-role="body"]').innerHTML = formatBotResponse(answer);
    attachSourcesAndActions(shell, answer);
    thread.appendChild(shell);
    chatMessages.push({ role: "bot", text: answer, sources });
    saveSession();
  } catch (err) {
    const shell = createBotShell();
    setStage(shell, "done");
    shell.querySelector('[data-role="body"]').innerHTML = formatBotResponse(
      `Unable to retrieve government regulations. ${err.message || ""}`
    );
    thread.appendChild(shell);
  } finally {
    isLoading = false;
    btnSend.disabled = !(input.value || "").trim();
    scrollToBottom();
  }
}

function setupListeners() {
  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });
  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input?.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
    btnSend.disabled = !input.value.trim() || isLoading;
  });

  btnNewChat?.addEventListener("click", startNewChat);
  btnLang?.addEventListener("click", () => applyUiLang(uiLang === "hi" ? "en" : "hi"));
  btnTheme?.addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));
  btnToggleSidebar?.addEventListener("click", () => {
    if (sidebar.classList.contains("is-open")) closeSidebar();
    else openSidebar();
  });
  btnCloseSidebar?.addEventListener("click", closeSidebar);
  backdrop?.addEventListener("click", closeSidebar);

  document.querySelectorAll(".suggest-btn, .chip").forEach((btn) => {
    btn.addEventListener("click", () => triggerQuickQuery(visibleQuery(btn)));
  });

  document.querySelector('[data-scroll="suggested"]')?.addEventListener("click", () => {
    document.getElementById("suggested")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (window.matchMedia("(max-width: 960px)").matches) openSidebar();
  });

  document.getElementById("btn-about")?.addEventListener("click", () => {
    aboutDialog?.showModal();
  });

  window.addEventListener("resize", () => {
    if (window.matchMedia("(min-width: 961px)").matches) closeSidebar();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSidebar();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  applyTheme(currentTheme());
  applyUiLang(uiLang);
  setupListeners();
  restoreSession();
  btnSend.disabled = true;
});
