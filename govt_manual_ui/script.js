/**
 * Manual Assistant — Bento-Grid Dashboard Controller
 *
 * Implements:
 * - Direct REST integration with local FastAPI RAG backend
 * - Dashboard sidebar navigation & chapter quick-trigger
 * - Multi-pane responsiveness & toggles
 * - Document Memo Card styling for replies
 * - Interactive Reference Inspector (right pane details)
 * - Session persistence across reloads
 */

// ── API config ─────────────────────────────────────────────────
// Served from FastAPI at /ui (local or Cloud Run) → same origin.
// Opened as file:// → local API on :8000.
// Always same-origin when served over http(s). Never point Cloud Run at localhost.
const API_BASE =
    typeof window !== "undefined" && /^https?:$/.test(window.location.protocol)
        ? window.location.origin
        : "http://127.0.0.1:8000";
const CHAT_URL = `${API_BASE}/chat`;
const STREAM_URL = `${API_BASE}/chat/stream`;
const SESSION_URL = `${API_BASE}/ui/session`;
console.info("[Manual AI] chat endpoint:", CHAT_URL);

// ── Auth: short-lived token issued to this page by /ui/session ──
let authToken = null;
let authTokenExpiry = 0;

async function getAuthToken(force = false) {
    if (!force && authToken && Date.now() < authTokenExpiry) return authToken;
    const res = await fetch(SESSION_URL, { credentials: "same-origin" });
    if (!res.ok) throw new Error(`Could not start a session (status ${res.status})`);
    const data = await res.json();
    authToken = data.token;
    // Refresh a minute early
    authTokenExpiry = Date.now() + Math.max(0, (data.expires_in - 60)) * 1000;
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

// ── State ──────────────────────────────────────────────────────
let sessionId = null;
let isLoading = false;
let chatMessages = []; // {role, text, sources}
// Keep track of active source contents to load into the inspector
let activeSources = [];
let uiLang = localStorage.getItem("manual_ai_lang") || "hi";

const I18N = {
    hi: {
        online: "Online",
        dept: "सिंचाई एवं जल संसाधन विभाग · उत्तर प्रदेश",
        new_session: "नया सत्र",
        new_session_full: "+ नया सत्र",
        suggested: "सुझाए गए प्रश्न",
        nav_pragati: "प्रगति पोर्टल URL",
        nav_igrs: "IGRS अधिकतम फाइल साइज",
        nav_200cr: "₹200 करोड़+ स्वीकृति",
        nav_200cr_meta: "वित्तीय हस्तपुस्तिका",
        nav_tender: "निविदा समिति के कार्य",
        nav_310: "अनुच्छेद ३१० · सेवा अवधि",
        nav_310_meta: "सिविल सेवा नियमावली",
        nav_ce: "मुख्य अभियंता के वित्तीय अधिकार",
        hero_eyebrow: "अभिलेख परामर्श",
        hero_blurb: "नियमावलियों और शासनादेशों से सत्यापित उत्तर। हिंदी या अंग्रेज़ी में पूछें।",
        chip1_tag: "निविदा",
        chip1: "एकल बोली निविदा कब स्वीकार्य?",
        chip2_tag: "वित्त",
        chip2: "मुख्य अभियंता की वित्तीय शक्तियाँ",
        chip3: "जन-सुनवाई जिला स्तर पर कब लागू?",
        chip4_tag: "अधिष्ठान",
        chip4: "सरकारी सेवक की सेवा अवधि का आधार",
        ask_placeholder: "प्रश्न लिखें… (Hindi or English)",
        ask: "पूछें",
        disclaimer: "उत्तर प्रशासनिक परामर्श हेतु। अंतिम निर्णय हेतु अधिकृत अधिसूचना मान्य।",
        source_desk: "स्रोत पटल",
        empty_title: "कोई अंश खुला नहीं",
        empty_body: "उत्तर में स्रोत बैज पर क्लिक करें — मूल नियमावली अंश यहाँ दिखेगा।",
        memo_desk: "शासन नियमावली डेस्क",
        verified_refs: "सत्यापित संदर्भ:",
        copy_memo: "प्रतिलिपि",
        copied: "कॉपी हो गया!",
        doc_ref: "दस्तावेज़ संदर्भ",
        copy_snippet: "अंश कॉपी करें",
        manual_excerpt: "नियमावली अंश",
        general_excerpt: "सामान्य अंश विवरण",
        no_content: "इस स्रोत के लिए पाठ उपलब्ध नहीं।",
    },
    en: {
        online: "Online",
        dept: "Irrigation & Water Resources Dept · Uttar Pradesh",
        new_session: "New session",
        new_session_full: "+ New session",
        suggested: "Suggested questions",
        nav_pragati: "Pragati Portal URL",
        nav_igrs: "IGRS max attachment size",
        nav_200cr: "₹200 cr+ project approval",
        nav_200cr_meta: "Financial Handbook",
        nav_tender: "Tender committee duties",
        nav_310: "Article 310 · tenure",
        nav_310_meta: "Civil Service Rules",
        nav_ce: "Chief Engineer financial powers",
        hero_eyebrow: "Archive desk",
        hero_blurb: "Answers grounded in manuals and government orders. Ask in Hindi or English.",
        chip1_tag: "Tender",
        chip1: "When can a single-bid tender be accepted?",
        chip2_tag: "Finance",
        chip2: "Financial powers of a Chief Engineer",
        chip3: "When was Jan-Sunwai rolled out at district level?",
        chip4_tag: "Establishment",
        chip4: "Constitutional basis of a servant's tenure",
        ask_placeholder: "Ask a question… (Hindi or English)",
        ask: "Ask",
        disclaimer: "Answers are for administrative guidance. Only a signed gazette notification is legally binding.",
        source_desk: "Source desk",
        empty_title: "No excerpt open",
        empty_body: "Click a source badge on an answer to read the matching manual text here.",
        memo_desk: "Govt. Manual Desk",
        verified_refs: "Verified references:",
        copy_memo: "Copy memo",
        copied: "Copied!",
        doc_ref: "Document reference",
        copy_snippet: "Copy snippet",
        manual_excerpt: "Manual excerpt",
        general_excerpt: "General excerpt details",
        no_content: "No text content returned for this source reference.",
    },
};

function t(key) {
    return (I18N[uiLang] && I18N[uiLang][key]) || (I18N.en[key] || key);
}

function applyUiLang(lang) {
    uiLang = lang === "en" ? "en" : "hi";
    localStorage.setItem("manual_ai_lang", uiLang);
    document.documentElement.lang = uiLang === "hi" ? "hi" : "en";

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
        const en = btn.querySelector(".lang-en");
        const hi = btn.querySelector(".lang-hi");
        if (en) en.classList.toggle("lang-active", uiLang === "en");
        if (hi) hi.classList.toggle("lang-active", uiLang === "hi");
    }
}

// ── DOM References ─────────────────────────────────────────────
const thread            = document.getElementById("thread");
const input             = document.getElementById("composer-input");
const btnSend           = document.getElementById("btn-send");
const btnNewChat        = document.getElementById("btn-new-chat");
const hero              = document.getElementById("hero");
const inspector         = document.getElementById("inspector");
const inspectorBody     = document.getElementById("inspector-body");
const btnCloseInspector = document.getElementById("btn-close-inspector");
const sidebar           = document.getElementById("sidebar");

// Mobile toggles
const btnToggleSidebar   = document.getElementById("btn-toggle-sidebar");
const btnToggleInspector = document.getElementById("btn-toggle-inspector");

// ── Initialization ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setupListeners();
    restoreSession();
});

// ── Event Listeners ────────────────────────────────────────────
function setupListeners() {
    btnSend.addEventListener("click", sendMessage);

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 120) + "px";
        btnSend.disabled = !input.value.trim();
    });

    btnNewChat.addEventListener("click", startNewChat);

    // Chapter button click actions
    document.querySelectorAll(".nav-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            triggerQuickQuery(query);
            // Close mobile sidebar if open
            if (sidebar.classList.contains("is-active")) {
                sidebar.classList.remove("is-active");
            }
        });
    });

    // Suggestion chip clicks
    document.querySelectorAll(".chip").forEach((btn) => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            triggerQuickQuery(query);
        });
    });

    // Mobile Panel Toggles
    if (btnToggleSidebar) {
        btnToggleSidebar.addEventListener("click", () => {
            sidebar.classList.toggle("is-active");
            inspector.classList.remove("is-active"); // Close other drawer
        });
    }

    if (btnToggleInspector) {
        btnToggleInspector.addEventListener("click", () => {
            inspector.classList.toggle("is-active");
            sidebar.classList.remove("is-active"); // Close other drawer
        });
    }

    if (btnCloseInspector) {
        btnCloseInspector.addEventListener("click", () => {
            inspector.classList.remove("is-active");
        });
    }

    // Close drawers when clicking in the workspace
    const workspace = document.querySelector(".workspace");
    if (workspace) {
        workspace.addEventListener("click", (e) => {
            if (!e.target.closest(".mobile-toggle")) {
                sidebar.classList.remove("is-active");
                // Only close inspector if we didn't click a source badge
                if (!e.target.closest(".source-badge")) {
                    inspector.classList.remove("is-active");
                }
            }
        });
    }

    // Theme Toggle — light is default; dark-theme class enables night desk
    const btnToggleTheme = document.getElementById("btn-toggle-theme");
    const iconMoon = document.getElementById("theme-icon-moon");
    const iconSun = document.getElementById("theme-icon-sun");
    const syncThemeIcons = () => {
        const dark = document.documentElement.classList.contains("dark-theme");
        if (iconMoon) iconMoon.style.display = dark ? "none" : "block";
        if (iconSun) iconSun.style.display = dark ? "block" : "none";
    };
    if (localStorage.getItem("manual_ai_theme") === "dark") {
        document.documentElement.classList.add("dark-theme");
    }
    syncThemeIcons();
    if (btnToggleTheme) {
        btnToggleTheme.addEventListener("click", () => {
            document.documentElement.classList.toggle("dark-theme");
            document.documentElement.classList.remove("light-theme");
            localStorage.setItem(
                "manual_ai_theme",
                document.documentElement.classList.contains("dark-theme") ? "dark" : "light"
            );
            syncThemeIcons();
        });
    }

    // English / Hindi UI toggle
    applyUiLang(uiLang);
    const btnToggleLang = document.getElementById("btn-toggle-lang");
    if (btnToggleLang) {
        btnToggleLang.addEventListener("click", () => {
            applyUiLang(uiLang === "hi" ? "en" : "hi");
        });
    }
}

// Helper to trigger query
function triggerQuickQuery(query) {
    input.value = query;
    input.dispatchEvent(new Event("input"));
    sendMessage();
}

// ── Streaming answer ──────────────────────────────────────────────
// Returns true once tokens have started, so a mid-stream failure is not
// sent a second time through /chat.
async function streamAnswer(question, shimmer) {
    let res;
    try {
        res = await authedFetch(STREAM_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: question, question, session_id: sessionId }),
        });
    } catch (err) {
        return false;
    }
    if (!res.ok || !res.body) return false;

    removeShimmer(shimmer);
    const live = createMessageElement("bot", "", [], true);
    thread.appendChild(live);
    const body = live.querySelector(".memo-card__body");
    const memo = live.querySelector(".memo-card");
    let text = "";
    let sources = [];
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    const applyEvent = (ev) => {
        if (ev.type === "token") text += ev.text || "";
        else if (ev.type === "replace") text = ev.text || "";
        else if (ev.type === "done") {
            sources = ev.sources || [];
            if (ev.session_id) sessionId = ev.session_id;
            return;
        } else return;
        body.innerHTML = formatBotResponse(text);
        scrollToBottom();
    };

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
            } catch (err) {
                console.warn("Bad stream event", err);
            }
        }
    }
    attachSources(memo, sources);
    chatMessages.push({ role: "bot", text, sources });
    saveSession();
    return true;
}

// ── Sending message ───────────────────────────────────────────────
async function sendMessage() {
    const question = input.value.trim();
    if (!question || isLoading) return;

    // Hide welcome state
    if (hero) hero.style.display = "none";

    // Clear composer area
    input.value = "";
    input.style.height = "auto";
    btnSend.disabled = true;

    // Render User Bubble
    appendMessage("user", question);
    saveSession();

    // Show Shimmer Memo Loader
    const shimmer = showShimmer();
    isLoading = true;

    try {
        if (!sessionId) sessionId = crypto.randomUUID();

        const streamed = await streamAnswer(question, shimmer);
        if (streamed) return;

        const res = await authedFetch(CHAT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: question,
                question,
                session_id: sessionId,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Communication failure" }));
            const detail =
                typeof err.detail === "string"
                    ? err.detail
                    : Array.isArray(err.detail)
                      ? err.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
                      : err.detail || `Server status ${res.status}`;
            throw new Error(detail);
        }

        const data = await res.json();
        // Local /chat returns a plain JSON string; Cloud-style APIs return an object.
        let answer = "";
        let sources = [];
        if (typeof data === "string") {
            answer = data;
        } else if (data && typeof data === "object") {
            answer = data.answer || data.response || data.message || "";
            if (data.session_id || data.sessionId) {
                sessionId = data.session_id || data.sessionId;
            }
            sources = data.sources || [];
        }

        removeShimmer(shimmer);
        appendMessage("bot", answer, sources, true);
        saveSession();
    } catch (err) {
        removeShimmer(shimmer);
        appendMessage(
            "bot",
            `⚠️ **Error Encountered**\n\nUnable to retrieve government regulations. Details: ${err.message}. Please verify connectivity to ${CHAT_URL}.`,
            []
        );
    } finally {
        isLoading = false;
    }
}

// ── Render message ─────────────────────────────────────────────
function attachSources(memo, sources) {
    if (!memo || !sources || sources.length === 0) return;
    const sourcesDiv = document.createElement("div");
    sourcesDiv.classList.add("memo-card__sources");

    const title = document.createElement("span");
    title.classList.add("sources-title");
    title.textContent = t("verified_refs");
    sourcesDiv.appendChild(title);

    const list = document.createElement("div");
    list.classList.add("sources-list");
    sources.forEach((source, index) => {
        const label = buildSourceLabel(source) || `Ref [${index + 1}]`;
        const badge = document.createElement("button");
        badge.classList.add("source-badge");
        badge.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            ${label}
        `;
        badge.addEventListener("click", (e) => {
            e.stopPropagation();
            inspectSource(source);
        });
        list.appendChild(badge);
    });
    sourcesDiv.appendChild(list);
    memo.appendChild(sourcesDiv);
}

function appendMessage(role, text, sources = [], isNew = false) {
    const el = createMessageElement(role, text, sources, isNew);
    thread.appendChild(el);
    scrollToBottom();
    chatMessages.push({ role, text, sources });
}

function createMessageElement(role, text, sources = [], isNew = false) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("msg", `msg--${role}`);

    if (role === "user") {
        // User Pill Bubble
        const bubble = document.createElement("div");
        bubble.classList.add("msg__bubble");
        bubble.textContent = text;
        wrapper.appendChild(bubble);
    } else {
        // Bot Official Memo
        const memo = document.createElement("div");
        memo.classList.add("memo-card");

        // Header Line
        const header = document.createElement("div");
        header.classList.add("memo-card__header");
        
        const origin = document.createElement("div");
        origin.classList.add("memo-card__origin");
        origin.innerHTML = `
            <div class="memo-card__seal">म</div>
            <div class="memo-card__department">${t("memo_desk")}</div>
        `;
        
        // Random ID for Memo styling authenticity
        const randomId = "UP-MANUAL/REG-" + Math.floor(100000 + Math.random() * 900000);
        const meta = document.createElement("div");
        meta.classList.add("memo-card__meta");
        meta.textContent = randomId;

        header.appendChild(origin);
        header.appendChild(meta);
        memo.appendChild(header);

        // Body content
        const body = document.createElement("div");
        body.classList.add("memo-card__body");
        body.innerHTML = formatBotResponse(text);
        memo.appendChild(body);

        attachSources(memo, sources);

        // Action Bar (Copy Text)
        const actions = document.createElement("div");
        actions.classList.add("memo-card__actions");
        
        const copyBtn = document.createElement("button");
        copyBtn.classList.add("btn-memo-action");
        copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="margin-right:4px"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            ${t("copy_memo")}
        `;
        copyBtn.addEventListener("click", () => copyTextContent(text, copyBtn));
        actions.appendChild(copyBtn);
        memo.appendChild(actions);

        wrapper.appendChild(memo);
    }

    return wrapper;
}

// ── Inspector Action ───────────────────────────────────────────
function inspectSource(source) {
    // Open right panel drawer
    inspector.classList.add("is-active");

    // Clean body
    inspectorBody.innerHTML = "";

    const card = document.createElement("div");
    card.classList.add("cite-card");

    const docName = source.source_document || "Government Regulation Document";
    const title = source.chapter ? `Chapter ${source.chapter}` : t("manual_excerpt");
    
    // Coordinates
    let loc = "";
    if (source.section) loc += `Section ${source.section} · `;
    if (source.page_start) {
        if (source.page_end && source.page_end !== source.page_start) {
            loc += `Pages ${source.page_start}–${source.page_end}`;
        } else {
            loc += `Page ${source.page_start}`;
        }
    }
    if (!loc) loc = t("general_excerpt");

    const content = source.content || t("no_content");

    card.innerHTML = `
        <span class="cite-card__doc">${t("doc_ref")}</span>
        <h4 class="cite-card__title">${docName} — ${title}</h4>
        <div class="cite-card__loc">${loc}</div>
        <div class="cite-card__text">${escapeHtml(content).replace(/\n/g, "<br>")}</div>
        <button class="cite-card__copy" id="btn-copy-citation">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            ${t("copy_snippet")}
        </button>
    `;

    inspectorBody.appendChild(card);

    // Bind copy button in citation
    const copyCitBtn = card.querySelector("#btn-copy-citation");
    copyCitBtn.addEventListener("click", () => {
        copyTextContent(content, copyCitBtn);
    });
}

// ── Copy to Clipboard ──────────────────────────────────────────
async function copyTextContent(text, btn) {
    try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("copied");
        const originalHTML = btn.innerHTML;
        btn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="margin-right:4px"><polyline points="20 6 9 17 4 12"/></svg>
            ${t("copied")}
        `;
        setTimeout(() => {
            btn.classList.remove("copied");
            btn.innerHTML = originalHTML;
        }, 2000);
    } catch {
        // Fallback
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
    }
}

// ── Format bot response (Simple Markdown Table parser included) ──
function formatBotResponse(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Headers
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Bullet lists
    html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
    html = html.replace(/<\/ul>\s*<ul>/g, "");

    // Numbered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Line breaks
    html = html.replace(/\n/g, "<br>");

    return html;
}

function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
}

// ── Build Source Label ─────────────────────────────────────────
function buildSourceLabel(source) {
    const parts = [];
    if (source.chapter) parts.push(`Ch. ${source.chapter}`);
    if (source.section) parts.push(`Sec. ${source.section}`);
    if (source.page_start) parts.push(`p. ${source.page_start}`);
    
    let label = parts.join(", ");
    if (!label && source.source_document) {
        label = source.source_document.replace(/\.[^/.]+$/, ""); // strip extension
    }
    return label;
}

// ── Shimmer typing indicator ───────────────────────────────────
function showShimmer() {
    const wrap = document.createElement("div");
    wrap.classList.add("msg", "msg--bot");

    const shimmer = document.createElement("div");
    shimmer.classList.add("shimmer-memo");
    shimmer.innerHTML = `
        <div class="shimmer-line shimmer-line--1"></div>
        <div class="shimmer-line shimmer-line--2"></div>
        <div class="shimmer-line shimmer-line--3"></div>
        <div class="shimmer-line shimmer-line--4"></div>
    `;

    wrap.appendChild(shimmer);
    thread.appendChild(wrap);
    scrollToBottom();

    return wrap;
}

function removeShimmer(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ── New Chat ───────────────────────────────────────────────────
function startNewChat() {
    sessionId = null;
    chatMessages = [];
    clearSession();

    // Clean thread
    thread.querySelectorAll(".msg").forEach((m) => m.remove());

    // Clean inspector
    inspectorBody.innerHTML = `
        <div class="inspector__empty flex flex-col items-center justify-center text-center h-full min-h-[200px] px-4">
            <div class="empty-mark font-display text-3xl text-rule mb-2">॥</div>
            <h4 class="text-sm font-semibold text-ink mb-1" data-i18n="empty_title">${t("empty_title")}</h4>
            <p class="text-xs text-mute leading-relaxed" data-i18n="empty_body">${t("empty_body")}</p>
        </div>
    `;
    inspector.classList.remove("is-active");

    // Re-reveal Hero (Stitch layout is block + space-y, not flex row)
    if (hero) {
        hero.style.display = "";
        hero.style.animation = "none";
        hero.offsetHeight; // force reflow
        hero.style.animation = "";
    }

    input.value = "";
    input.style.height = "auto";
    input.focus();
}

// ── Session persistence ────────────────────────────────────────
const STORAGE_KEY = "rag_bento_chat_session";

function saveSession() {
    try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
            sessionId,
            messages: chatMessages,
            timestamp: Date.now(),
        }));
    } catch { /* ignore */ }
}

function restoreSession() {
    try {
        const raw = sessionStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        if (!data.messages || !data.messages.length) return;

        sessionId = data.sessionId;
        chatMessages = [];

        if (hero) hero.style.display = "none";

        data.messages.forEach((msg) => {
            const el = createMessageElement(msg.role, msg.text, msg.sources || [], false);
            thread.appendChild(el);
            chatMessages.push({ role: msg.role, text: msg.text, sources: msg.sources || [] });
        });

        scrollToBottom();
    } catch {
        clearSession();
    }
}

function clearSession() {
    sessionStorage.removeItem(STORAGE_KEY);
}

// ── Scroll helper ──────────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        thread.scrollTop = thread.scrollHeight;
    });
}
