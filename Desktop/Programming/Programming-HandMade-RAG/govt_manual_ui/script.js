/**
 * Manual Assistant — Bento-Grid Dashboard Controller
 *
 * Implements:
 * - Direct REST integration with manual-assistant backend
 * - Dashboard sidebar navigation & chapter quick-trigger
 * - Multi-pane responsiveness & toggles
 * - Document Memo Card styling for replies
 * - Interactive Reference Inspector (right pane details)
 * - Session persistence across reloads
 */

// ── State ──────────────────────────────────────────────────────
let sessionId = null;
let isLoading = false;
let chatMessages = []; // {role, text, sources}
// Keep track of active source contents to load into the inspector
let activeSources = [];

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
    document.querySelector(".workspace").addEventListener("click", (e) => {
        if (!e.target.closest(".mobile-toggle")) {
            sidebar.classList.remove("is-active");
            // Only close inspector if we didn't click a source badge
            if (!e.target.closest(".source-badge")) {
                inspector.classList.remove("is-active");
            }
        }
    });
}

// Helper to trigger query
function triggerQuickQuery(query) {
    input.value = query;
    input.dispatchEvent(new Event("input"));
    sendMessage();
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
        const res = await fetch("https://manual-assistant-877548514893.asia-south1.run.app/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                session_id: sessionId,
                model: "gemini-2.5-flash",
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Communication failure" }));
            throw new Error(err.detail || `Server status ${res.status}`);
        }

        const data = await res.json();
        sessionId = data.session_id;

        // Strip shimmer card
        removeShimmer(shimmer);

        // Render Bot response memo
        appendMessage("bot", data.answer, data.sources || [], true);
        saveSession();
    } catch (err) {
        removeShimmer(shimmer);
        appendMessage(
            "bot",
            `⚠️ **Error Encountered**\n\nUnable to retrieve government regulations. Details: ${err.message}. Please verify connectivity.`,
            []
        );
    } finally {
        isLoading = false;
    }
}

// ── Render message ─────────────────────────────────────────────
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
            <div class="memo-card__department">Govt. Manual Desk</div>
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

        // Add Sources References
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement("div");
            sourcesDiv.classList.add("memo-card__sources");
            
            const title = document.createElement("span");
            title.classList.add("sources-title");
            title.textContent = "Verified References:";
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

        // Action Bar (Copy Text)
        const actions = document.createElement("div");
        actions.classList.add("memo-card__actions");
        
        const copyBtn = document.createElement("button");
        copyBtn.classList.add("btn-memo-action");
        copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="margin-right:4px"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy Memo
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
    const title = source.chapter ? `Chapter ${source.chapter}` : "Manual Excerpt";
    
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
    if (!loc) loc = "General Excerpt Details";

    const content = source.content || "No text content returned for this source reference.";

    card.innerHTML = `
        <span class="cite-card__doc">DOC REFERENCE</span>
        <h4 class="cite-card__title">${docName} — ${title}</h4>
        <div class="cite-card__loc">${loc}</div>
        <div class="cite-card__text">${escapeHtml(content).replace(/\n/g, "<br>")}</div>
        <button class="cite-card__copy" id="btn-copy-citation">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy Snippet
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
            Copied!
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
        <div class="inspector__empty">
            <div class="empty-icon">📁</div>
            <h4>No Source Selected</h4>
            <p>When you ask a question and get a response, the system generates source citations. Click on any source tag to inspect the exact official manual text here.</p>
        </div>
    `;
    inspector.classList.remove("is-active");

    // Re-reveal Hero
    if (hero) {
        hero.style.display = "flex";
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
