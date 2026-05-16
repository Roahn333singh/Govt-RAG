(() => {
  const DEFAULTS = {
    webhookUrl:
      "https://n8n.srv1168234.hstgr.cloud/webhook/7d548c3a-48fa-42cf-b22c-99fec7116e7e/chat",
    themeColor: "#00446D",
    title: "AI Chat",
    greeting: "Hi there! 👋 How can I help?",
    position: "bottom-right",
    typewriterSpeed: 12,
  };

  const style = (color) => `
    .cw-widget { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", Arial, sans-serif; }
    .cw-toggle {
      position: fixed;
      ${DEFAULTS.position.includes("bottom") ? "bottom: 20px;" : "top: 20px;"}
      ${DEFAULTS.position.includes("right") ? "right: 20px;" : "left: 20px;"}
      z-index: 9999;
      border: none;
      border-radius: 999px;
      padding: 12px 16px;
      background: ${color};
      color: #fff;
      box-shadow: 0 8px 24px rgba(0,0,0,.16);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
    }
    .cw-container {
      position: fixed;
      ${DEFAULTS.position.includes("bottom") ? "bottom: 80px;" : "top: 80px;"}
      ${DEFAULTS.position.includes("right") ? "right: 20px;" : "left: 20px;"}
      z-index: 9999;
      width: 360px;
      max-width: calc(100vw - 40px);
      height: 520px;
      max-height: calc(100vh - 160px);
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 16px 48px rgba(0,0,0,.18);
      overflow: hidden;
      display: none;
      flex-direction: column;
    }
    .cw-header {
      background: ${color};
      color: #fff;
      padding: 14px 16px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .cw-close {
      background: transparent;
      border: none;
      color: #fff;
      font-size: 18px;
      cursor: pointer;
    }
    .cw-fullscreen {
      background: transparent;
      border: none;
      color: #fff;
      font-size: 18px;
      cursor: pointer;
    }
    .cw-body {
      flex: 1;
      background: #fafafa;
      padding: 12px;
      overflow-y: auto;
    }
    .cw-msg {
      max-width: 78%;
      padding: 10px 12px;
      border-radius: 12px;
      margin: 8px 0;
      line-height: 1.4;
      font-size: 14px;
      word-break: break-word;
      white-space: pre-wrap;
    }
    .cw-msg.user {
      background: ${color}1a;
      color: #1f1f1f;
      margin-left: auto;
      border-bottom-right-radius: 4px;
    }
    .cw-msg.bot {
      background: #fff;
      color: #1f1f1f;
      margin-right: auto;
      border-bottom-left-radius: 4px;
      border: 1px solid #eee;
    }
    .cw-footer {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid #eee;
      background: #fff;
    }
    .cw-input {
      flex: 1;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px 12px;
      outline: none;
    }
    .cw-send {
      border: none;
      background: ${color};
      color: #fff;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .cw-container.cw-full {
      top: 0;
      right: 0;
      bottom: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      max-width: none;
      max-height: none;
      border-radius: 0;
      box-shadow: none;
    }
    .cw-typing {
      display: inline-block;
      width: 6px;
      height: 6px;
      margin-right: 3px;
      border-radius: 50%;
      background: ${color};
      animation: cwBounce 1.2s infinite ease-in-out both;
    }
    .cw-typing:nth-child(2) { animation-delay: -1.1s; }
    .cw-typing:nth-child(3) { animation-delay: -1.0s; }
    @keyframes cwBounce {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1); }
    }
  `;

  const html = (title, greeting) => `
    <button class="cw-toggle" aria-label="Open chat">
      🟠 Chat
    </button>
    <div class="cw-container cw-widget" role="dialog" aria-label="${title}" aria-modal="true">
      <div class="cw-header">
        <div>${title}</div>
        <div style="display:flex;gap:8px;align-items:center">
          <button class="cw-fullscreen" aria-label="Fullscreen">⛶</button>
          <button class="cw-close" aria-label="Close">×</button>
        </div>
      </div>
      <div class="cw-body">
        <div class="cw-msg bot">${greeting}</div>
      </div>
      <div class="cw-footer">
        <input class="cw-input" type="text" placeholder="Type a message..." />
        <button class="cw-send">Send</button>
      </div>
    </div>
  `;

  function pickText(data) {
    const keys = [
      "reply",
      "output",
      "text",
      "message",
      "content",
      "answer",
      "result",
      "response",
    ];
    for (const k of keys) {
      const v = data && typeof data === "object" ? data[k] : undefined;
      if (typeof v === "string" && v.trim()) return v;
    }
    const nested =
      (data &&
        data.data &&
        Array.isArray(data.data) &&
        data.data[0] &&
        (data.data[0].text || data.data[0].answer || data.data[0].content)) ||
      (Array.isArray(data) &&
        data[0] &&
        (data[0].text || data[0].answer || data[0].content)) ||
      (data &&
        data.choices &&
        Array.isArray(data.choices) &&
        data.choices[0] &&
        data.choices[0].message &&
        data.choices[0].message.content);
    if (typeof nested === "string" && nested.trim()) return nested;
    return null;
  }

  function safeText(t) {
    return String(t ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function appendTyping(bodyEl, color) {
    const wrap = document.createElement("div");
    wrap.className = "cw-msg bot";
    wrap.innerHTML = `
      <span class="cw-typing" style="background:${color}"></span>
      <span class="cw-typing" style="background:${color}"></span>
      <span class="cw-typing" style="background:${color}"></span>
    `;
    bodyEl.appendChild(wrap);
    bodyEl.scrollTop = bodyEl.scrollHeight;
    return wrap;
  }

  function appendMsg(bodyEl, role, text) {
    const el = document.createElement("div");
    el.className = `cw-msg ${role}`;
    el.innerHTML = safeText(text);
    bodyEl.appendChild(el);
    bodyEl.scrollTop = bodyEl.scrollHeight;
    return el;
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function appendMsgTypewriter(bodyEl, role, text, speed = 12) {
    const el = appendMsg(bodyEl, role, "");
    const content = String(text ?? "");

    if (role !== "bot" || !content) {
      el.innerHTML = safeText(content);
      bodyEl.scrollTop = bodyEl.scrollHeight;
      return el;
    }

    const reduceMotion =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduceMotion) {
      el.innerHTML = safeText(content);
      bodyEl.scrollTop = bodyEl.scrollHeight;
      return el;
    }

    const baseDelay = Math.max(4, Number(speed) || DEFAULTS.typewriterSpeed);
    const step = content.length > 1200 ? 3 : content.length > 600 ? 2 : 1;

    for (let i = 0; i < content.length; i += step) {
      const chunk = content.slice(i, i + step);
      el.innerHTML += safeText(chunk);
      bodyEl.scrollTop = bodyEl.scrollHeight;

      const lastChar = chunk[chunk.length - 1] || "";
      let pause = baseDelay;
      if (/[.!?]/.test(lastChar)) pause = baseDelay * 4;
      else if (/[,\n]/.test(lastChar)) pause = baseDelay * 2;
      await sleep(pause);
    }

    return el;
  }

  async function postMessage(webhookUrl, message, sessionId) {
    try {
      const res = await fetch(webhookUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          text: message,
          question: message,
          prompt: message,
          chatInput: message,
          sessionId,
          source: "widget",
          timestamp: Date.now(),
        }),
        mode: "cors",
      });
      const ct = (res.headers.get("content-type") || "").toLowerCase();
      const isJson = ct.includes("application/json");
      if (!res.ok) {
        if (isJson) {
          const data = await res.json().catch(() => null);
          const text = data ? pickText(data) : null;
          return String(text || "Error in workflow");
        }
        const text = await res.text().catch(() => "");
        return String(text || "Error in workflow");
      }
      if (isJson) {
        const data = await res.json();
        if (typeof data === "string") return data;
        const text = pickText(data);
        if (text) return text;
        return "Received a structured response";
      }
      return await res.text();
    } catch (e) {
      return `Sorry, I couldn't reach the assistant.`;
    }
  }

  function ensureStyle(color) {
    if (document.getElementById("cw-style")) return;
    const s = document.createElement("style");
    s.id = "cw-style";
    s.textContent = style(color);
    document.head.appendChild(s);
  }

  function injectWidget(opts) {
    const root = document.createElement("div");
    root.className = "cw-widget-root";
    root.innerHTML = html(opts.title, opts.greeting);
    document.body.appendChild(root);

    const toggle = root.querySelector(".cw-toggle");
    const container = root.querySelector(".cw-container");
    const bodyEl = root.querySelector(".cw-body");
    const input = root.querySelector(".cw-input");
    const send = root.querySelector(".cw-send");
    const close = root.querySelector(".cw-close");
    const fullscreen = root.querySelector(".cw-fullscreen");
    const sessionId =
      localStorage.getItem("cw-session-id") ||
      (function () {
        const id = Math.random().toString(36).slice(2);
        localStorage.setItem("cw-session-id", id);
        return id;
      })();

    function open() {
      container.style.display = "flex";
      input.focus();
    }
    function hide() {
      container.style.display = "none";
    }

    toggle.addEventListener("click", open);
    close.addEventListener("click", hide);
    fullscreen.addEventListener("click", () => {
      const isFull = container.classList.toggle("cw-full");
      fullscreen.textContent = isFull ? "🗗" : "⛶";
    });

    async function handleSend() {
      if (send.disabled) return;
      const text = input.value.trim();
      if (!text) return;
      send.disabled = true;
      input.disabled = true;
      input.value = "";
      appendMsg(bodyEl, "user", text);
      const typing = appendTyping(bodyEl, opts.themeColor);
      try {
        const reply = await postMessage(opts.webhookUrl, text, sessionId);
        typing.remove();
        await appendMsgTypewriter(bodyEl, "bot", reply, opts.typewriterSpeed);
      } finally {
        typing.remove();
        send.disabled = false;
        input.disabled = false;
        input.focus();
      }
    }

    send.addEventListener("click", handleSend);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });
  }

  window.ChatWidget = {
    init: function (options = {}) {
      const opts = { ...DEFAULTS, ...options };
      ensureStyle(opts.themeColor);
      injectWidget(opts);
    },
  };
})();
