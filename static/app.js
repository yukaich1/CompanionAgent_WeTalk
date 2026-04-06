const state = {
  snapshot: null,
  preview: null,
  previewSelectedKeywords: [],
  avatarEditor: {
    image: null,
    objectUrl: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    startOffsetX: 0,
    startOffsetY: 0,
  },
};

const DEFAULT_AGENT_AVATAR = "/static/default-agent.svg";
const DEFAULT_USER_AVATAR = "/static/default-user.svg";
const AVATAR_VIEWPORT = { size: 280, output: 512 };
const PREVIEW_KEYWORD_SELECTION_LIMIT = 8;

const elements = {
  messages: document.getElementById("messages"),
  messageInput: document.getElementById("message-input"),
  sendButton: document.getElementById("send-button"),
  typingStatus: document.getElementById("typing-status"),
  agentName: document.getElementById("agent-name"),
  chatAgentAvatar: document.getElementById("chat-agent-avatar"),
  chatTitle: document.getElementById("chat-title"),
  agentMood: document.getElementById("agent-mood"),
  agentAffinity: document.getElementById("agent-affinity"),
  personaCount: document.getElementById("persona-count"),
  keywordList: document.getElementById("keyword-list"),
  debugEvidence: document.getElementById("debug-evidence"),
  activityList: document.getElementById("activity-list"),
  agentAvatar: document.getElementById("agent-avatar"),
  toast: document.getElementById("toast"),
  drawer: document.getElementById("settings-drawer"),
  openSettings: document.getElementById("open-settings"),
  closeSettings: document.getElementById("close-settings"),
  nameInput: document.getElementById("name-input"),
  saveName: document.getElementById("save-name"),
  personaWebSearchToggle: document.getElementById("persona-web-search-toggle"),
  avatarInput: document.getElementById("avatar-input"),
  uploadAvatar: document.getElementById("upload-avatar"),
  avatarEditor: document.getElementById("avatar-editor"),
  closeAvatarEditor: document.getElementById("close-avatar-editor"),
  avatarCanvas: document.getElementById("avatar-canvas"),
  avatarScale: document.getElementById("avatar-scale"),
  reselectAvatar: document.getElementById("reselect-avatar"),
  saveAvatarCrop: document.getElementById("save-avatar-crop"),
  previewPanel: document.getElementById("preview-panel"),
  previewTitle: document.getElementById("preview-title"),
  previewBaseTemplate: document.getElementById("preview-base-template"),
  previewKeywordHint: document.getElementById("preview-keyword-hint"),
  previewKeywords: document.getElementById("preview-keywords"),
  previewSnippets: document.getElementById("preview-snippets"),
  confirmPreview: document.getElementById("confirm-preview"),
  personaFile: document.getElementById("persona-file"),
  uploadPersonaFile: document.getElementById("upload-persona-file"),
  personaLabel: document.getElementById("persona-label"),
  personaText: document.getElementById("persona-text"),
  learnPersonaText: document.getElementById("learn-persona-text"),
  clearPersona: document.getElementById("clear-persona"),
  resetAll: document.getElementById("reset-all"),
};

const avatarCanvasContext = elements.avatarCanvas.getContext("2d");
let autoPreviewTimer = null;

function showToast(text) {
  elements.toast.textContent = text;
  elements.toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.add("hidden"), 2200);
}

function setTypingStatus(text) {
  elements.typingStatus.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeMoodLabel(value) {
  const mapping = {
    neutral: "平静",
    exuberant: "雀跃",
    dependent: "依恋",
    relaxed: "放松",
    docile: "温顺",
    bored: "低落",
    anxious: "紧张",
    disdainful: "冷淡",
    hostile: "不悦",
    calm: "平静",
    happy: "愉快",
    concerned: "关切",
    hurt: "受伤",
  };
  return mapping[value] || value || "平静";
}

function splitPreviewSnippets(preview) {
  return Array.isArray(preview?.snippets) ? preview.snippets : [];
}

function previewSourceLabel(source) {
  const mapping = {
    combined: "综合提炼",
    local: "本地资料",
    web_summary: "联网补充摘要",
    web: "联网资料",
  };
  return mapping[source] || source || "资料";
}

function currentAgentName() {
  const inputName = elements.nameInput?.value?.trim();
  if (inputName) return inputName;
  const snapshotName = state.snapshot?.agent?.name?.trim?.();
  if (snapshotName) return snapshotName;
  return elements.chatTitle?.textContent?.trim() || "";
}

function splitBubblesFromText(text) {
  const normalized = (text || "").replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];

  const paragraphs = normalized.split(/\n\n+/).map((part) => part.trim()).filter(Boolean);
  if (paragraphs.length > 1) return paragraphs;

  const lines = normalized.split(/\n+/).map((part) => part.trim()).filter(Boolean);
  if (lines.length > 1) return lines;

  if (normalized.length <= 90) return [normalized];

  const sentences = normalized.split(/(?<=[。！？!?])\s*/).map((part) => part.trim()).filter(Boolean);
  if (sentences.length <= 1) return [normalized];

  const target = normalized.length < 260 ? 90 : 120;
  const bubbles = [];
  let current = [];
  let currentLength = 0;

  for (const sentence of sentences) {
    if (current.length && currentLength + sentence.length > target) {
      bubbles.push(current.join(" ").trim());
      current = [sentence];
      currentLength = sentence.length;
    } else {
      current.push(sentence);
      currentLength += sentence.length;
    }
  }
  if (current.length) bubbles.push(current.join(" ").trim());
  return bubbles.filter(Boolean);
}

function createMessageRow(message) {
  const row = document.createElement("div");
  row.className = `message-row ${message.role}`;

  const avatar = document.createElement("img");
  avatar.className = "avatar";
  avatar.alt = message.role;
  avatar.src = message.role === "assistant"
    ? (state.snapshot?.agent?.avatarUrl || DEFAULT_AGENT_AVATAR)
    : DEFAULT_USER_AVATAR;

  const stack = document.createElement("div");
  stack.className = "bubble-stack";

  if (message.role === "assistant") {
    row.appendChild(avatar);
    row.appendChild(stack);
  } else {
    row.appendChild(stack);
    row.appendChild(avatar);
  }

  return { row, stack };
}

function appendMessage(message) {
  const { row, stack } = createMessageRow(message);
  const bubbles = (message.bubbles && message.bubbles.length ? message.bubbles : splitBubblesFromText(message.content)).filter(Boolean);
  for (const bubbleText of bubbles) {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.role}`;
    bubble.innerHTML = escapeHtml(bubbleText);
    stack.appendChild(bubble);
  }
  elements.messages.appendChild(row);
}

function renderHistory(history) {
  elements.messages.innerHTML = "";
  history.forEach((message) => appendMessage(message));
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderDebugEvidence(debugInfo) {
  if (!elements.debugEvidence) return;
  const route = debugInfo?.routeType;
  const personaEvidence = Array.isArray(debugInfo?.personaEvidence) ? debugInfo.personaEvidence : [];
  const toolEvidence = Array.isArray(debugInfo?.toolEvidence) ? debugInfo.toolEvidence : [];
  const thoughts = Array.isArray(debugInfo?.thoughts) ? debugInfo.thoughts : [];
  const emotionReason = typeof debugInfo?.emotionReason === "string" ? debugInfo.emotionReason.trim() : "";
  const items = [];

  if (route) {
    items.push(`
      <div class="debug-evidence-item">
        <strong>本轮路由</strong>
        <div>${escapeHtml(route)}</div>
      </div>
    `);
  }
  if (personaEvidence.length) {
    items.push(`
      <div class="debug-evidence-item">
        <strong>人设证据</strong>
        <div>${personaEvidence.map((item) => escapeHtml(item)).join("<br>")}</div>
      </div>
    `);
  }
  if (toolEvidence.length) {
    items.push(`
      <div class="debug-evidence-item">
        <strong>工具证据</strong>
        <div>${toolEvidence.map((item) => escapeHtml(item)).join("<br>")}</div>
      </div>
    `);
  }
  if (thoughts.length) {
    items.push(`
      <div class="debug-evidence-item">
        <strong>慢思考</strong>
        <div>${thoughts.map((item) => escapeHtml(item)).join("<br>")}</div>
      </div>
    `);
  }
  if (emotionReason) {
    items.push(`
      <div class="debug-evidence-item">
        <strong>情绪判断</strong>
        <div>${escapeHtml(emotionReason)}</div>
      </div>
    `);
  }

  if (!items.length) {
    elements.debugEvidence.classList.add("hidden");
    elements.debugEvidence.innerHTML = "";
    return;
  }

  elements.debugEvidence.classList.remove("hidden");
  elements.debugEvidence.innerHTML = items.join("");
}

function renderPreviewKeywords(preview) {
  const keywords = Array.isArray(preview?.summary?.display_keywords) ? preview.summary.display_keywords : [];
  if (!keywords.length) {
    state.previewSelectedKeywords = [];
    elements.previewKeywords.innerHTML = `<span class="tag">等待分析师生成关键词</span>`;
    if (elements.previewKeywordHint) {
      elements.previewKeywordHint.textContent = "分析师生成关键词后，这里会出现可勾选的高权重标签。";
    }
    return;
  }

  if (!Array.isArray(state.previewSelectedKeywords) || !state.previewSelectedKeywords.length) {
    state.previewSelectedKeywords = keywords.slice(0, PREVIEW_KEYWORD_SELECTION_LIMIT);
  } else {
    const available = new Set(keywords);
    state.previewSelectedKeywords = state.previewSelectedKeywords.filter((keyword) => available.has(keyword));
    if (!state.previewSelectedKeywords.length) {
      state.previewSelectedKeywords = keywords.slice(0, PREVIEW_KEYWORD_SELECTION_LIMIT);
    }
  }

  if (elements.previewKeywordHint) {
    elements.previewKeywordHint.textContent = `请从分析师生成的关键词中选择 ${PREVIEW_KEYWORD_SELECTION_LIMIT} 个高权重标签（当前 ${state.previewSelectedKeywords.length}/${PREVIEW_KEYWORD_SELECTION_LIMIT}）。`;
  }

  elements.previewKeywords.innerHTML = keywords.map((keyword) => {
    const checked = state.previewSelectedKeywords.includes(keyword) ? "checked" : "";
    const selectedClass = checked ? "is-selected" : "";
    return `
      <label class="tag keyword-choice ${selectedClass}">
        <input type="checkbox" value="${escapeHtml(keyword)}" ${checked}>
        <span>${escapeHtml(keyword)}</span>
      </label>
    `;
  }).join("");

  elements.previewKeywords.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.addEventListener("change", () => {
      const selected = Array.from(
        elements.previewKeywords.querySelectorAll("input[type='checkbox']:checked"),
      )
        .map((element) => element.value)
        .filter(Boolean);
      if (selected.length > PREVIEW_KEYWORD_SELECTION_LIMIT) {
        input.checked = false;
        showToast(`最多只能选择 ${PREVIEW_KEYWORD_SELECTION_LIMIT} 个关键词`);
        return;
      }
      state.previewSelectedKeywords = selected;
      if (elements.previewKeywordHint) {
        elements.previewKeywordHint.textContent = `请从分析师生成的关键词中选择 ${PREVIEW_KEYWORD_SELECTION_LIMIT} 个高权重标签（当前 ${state.previewSelectedKeywords.length}/${PREVIEW_KEYWORD_SELECTION_LIMIT}）。`;
      }
      elements.previewKeywords.querySelectorAll(".keyword-choice").forEach((label) => {
        const checkbox = label.querySelector("input[type='checkbox']");
        label.classList.toggle("is-selected", Boolean(checkbox?.checked));
      });
    });
  });
}

function renderPreview(preview) {
  state.preview = preview || null;
  if (!preview) {
    state.previewSelectedKeywords = [];
    elements.previewPanel.classList.add("hidden");
    elements.previewBaseTemplate.innerHTML = "";
    if (elements.previewKeywordHint) {
      elements.previewKeywordHint.textContent = "";
    }
    elements.previewKeywords.innerHTML = "";
    elements.previewSnippets.innerHTML = "";
    return;
  }

  elements.previewPanel.classList.remove("hidden");
  const previewName = (preview.persona_name || currentAgentName() || "角色").trim();
  elements.previewTitle.textContent = `${previewName}${preview.work_title ? ` · ${preview.work_title}` : ""} 待确认人设预览`;
  elements.previewBaseTemplate.innerHTML = preview.base_template_text
    ? escapeHtml(preview.base_template_text)
    : "等待提炼角色基础模板。";

  renderPreviewKeywords(preview);

  const snippets = splitPreviewSnippets(preview);
  elements.previewSnippets.innerHTML = snippets.length
    ? snippets.map((item) => `
        <div class="preview-snippet">
          <div class="preview-source">${escapeHtml(previewSourceLabel(item.source))}${item.title ? ` · ${escapeHtml(item.title)}` : ""}</div>
          <div class="preview-text">${escapeHtml(item.text)}</div>
        </div>
      `).join("")
    : `<div class="preview-snippet">暂无可展示的补充资料。</div>`;
}

function renderSnapshot(snapshot, options = {}) {
  const preserveHistory = Boolean(options.preserveHistory);
  state.snapshot = snapshot;
  const { agent, history, recentActivity, settings } = snapshot;

  elements.agentName.textContent = agent.name;
  elements.chatTitle.textContent = agent.name;
  elements.agentMood.textContent = normalizeMoodLabel(agent.mood);
  elements.agentAffinity.textContent = agent.affinity;
  elements.personaCount.textContent = `${agent.personaChunks} 条人设`;
  elements.nameInput.value = agent.name;
  elements.personaWebSearchToggle.checked = Boolean(settings?.personaWebSearchEnabled);

  const avatarUrl = agent.avatarUrl || DEFAULT_AGENT_AVATAR;
  elements.agentAvatar.src = avatarUrl;
  elements.chatAgentAvatar.src = avatarUrl;

  elements.keywordList.innerHTML = agent.keywords.length
    ? agent.keywords.map((keyword) => `<span class="tag">${escapeHtml(keyword)}</span>`).join("")
    : `<span class="tag">等待学习</span>`;

  renderDebugEvidence(snapshot.debug);
  elements.activityList.innerHTML = recentActivity.length
    ? recentActivity.map((item) => `
        <div class="activity-item">
          <div class="activity-text">${escapeHtml(item.text)}</div>
          <div class="activity-time">${escapeHtml(item.time)}</div>
        </div>
      `).join("")
    : `<div class="activity-item"><div class="activity-text">最近还没有新的活动记录。</div></div>`;

  if (!preserveHistory) {
    renderHistory(history);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function typingDelayForChar(char) {
  if ("，、；：".includes(char)) return 55;
  if ("。！？…".includes(char)) return 120;
  if (char === " ") return 8;
  return 22;
}

async function streamAssistantMessage(message) {
  const bubbles = (message.bubbles && message.bubbles.length ? message.bubbles : splitBubblesFromText(message.content)).filter(Boolean);
  if (!bubbles.length) return;

  const { row, stack } = createMessageRow({ role: "assistant" });
  elements.messages.appendChild(row);

  for (let i = 0; i < bubbles.length; i += 1) {
    setTypingStatus("正在输入…");
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    stack.appendChild(bubble);
    elements.messages.scrollTop = elements.messages.scrollHeight;

    let rendered = "";
    for (const char of bubbles[i]) {
      rendered += char;
      bubble.innerHTML = escapeHtml(rendered);
      elements.messages.scrollTop = elements.messages.scrollHeight;
      await sleep(typingDelayForChar(char));
    }
    await sleep(i === bubbles.length - 1 ? 80 : 420);
  }
}

async function bootstrap() {
  setTypingStatus("已连接");
  const snapshot = await requestJson("/api/bootstrap");
  renderSnapshot(snapshot);
  setTypingStatus("已连接");
}

async function sendMessage() {
  const message = elements.messageInput.value.trim();
  if (!message) return;

  appendMessage({ role: "user", content: message, bubbles: [message] });
  elements.messageInput.value = "";
  elements.messages.scrollTop = elements.messages.scrollHeight;
  setTypingStatus("正在输入…");

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    state.snapshot = data.snapshot;
    await sleep(180);
    await streamAssistantMessage(data.assistant);
    renderSnapshot(data.snapshot, { preserveHistory: true });
  } catch (error) {
    showToast(error.message);
  } finally {
    setTypingStatus("已连接");
  }
}

async function saveName() {
  try {
    const data = await requestJson("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: elements.nameInput.value.trim() }),
    });
    renderSnapshot(data.snapshot);
    if (state.preview) {
      state.preview.persona_name = data.snapshot?.agent?.name || elements.nameInput.value.trim() || state.preview.persona_name;
      renderPreview(state.preview);
    }
    showToast("名称已更新");
  } catch (error) {
    showToast(error.message);
  }
}

async function savePersonaWebSearchToggle() {
  try {
    const data = await requestJson("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ personaWebSearchEnabled: Boolean(elements.personaWebSearchToggle.checked) }),
    });
    renderSnapshot(data.snapshot);
    showToast(elements.personaWebSearchToggle.checked ? "已开启人设联网补充" : "已关闭人设联网补充");
  } catch (error) {
    elements.personaWebSearchToggle.checked = !elements.personaWebSearchToggle.checked;
    showToast(error.message);
  }
}

function cleanupAvatarObjectUrl() {
  if (state.avatarEditor.objectUrl) {
    URL.revokeObjectURL(state.avatarEditor.objectUrl);
    state.avatarEditor.objectUrl = null;
  }
}

function closeAvatarEditor() {
  state.avatarEditor.dragging = false;
  elements.avatarCanvas.classList.remove("dragging");
  elements.avatarEditor.classList.add("hidden");
}

function openAvatarEditor() {
  elements.avatarEditor.classList.remove("hidden");
}

function getAvatarViewport() {
  return {
    x: (elements.avatarCanvas.width - AVATAR_VIEWPORT.size) / 2,
    y: (elements.avatarCanvas.height - AVATAR_VIEWPORT.size) / 2,
    size: AVATAR_VIEWPORT.size,
  };
}

function computeAvatarTransform() {
  const editor = state.avatarEditor;
  const viewport = getAvatarViewport();
  const baseScale = Math.max(viewport.size / editor.image.width, viewport.size / editor.image.height);
  const scale = baseScale * editor.scale;
  const width = editor.image.width * scale;
  const height = editor.image.height * scale;
  const x = elements.avatarCanvas.width / 2 - width / 2 + editor.offsetX;
  const y = elements.avatarCanvas.height / 2 - height / 2 + editor.offsetY;
  return { x, y, width, height, viewport };
}

function renderAvatarEditor() {
  if (!state.avatarEditor.image) {
    avatarCanvasContext.clearRect(0, 0, elements.avatarCanvas.width, elements.avatarCanvas.height);
    return;
  }
  const { x, y, width, height, viewport } = computeAvatarTransform();
  avatarCanvasContext.clearRect(0, 0, elements.avatarCanvas.width, elements.avatarCanvas.height);
  avatarCanvasContext.drawImage(state.avatarEditor.image, x, y, width, height);

  avatarCanvasContext.save();
  avatarCanvasContext.fillStyle = "rgba(8, 10, 14, 0.54)";
  avatarCanvasContext.beginPath();
  avatarCanvasContext.rect(0, 0, elements.avatarCanvas.width, elements.avatarCanvas.height);
  avatarCanvasContext.arc(viewport.x + viewport.size / 2, viewport.y + viewport.size / 2, viewport.size / 2, 0, Math.PI * 2, true);
  avatarCanvasContext.fill("evenodd");
  avatarCanvasContext.restore();

  avatarCanvasContext.save();
  avatarCanvasContext.beginPath();
  avatarCanvasContext.arc(viewport.x + viewport.size / 2, viewport.y + viewport.size / 2, viewport.size / 2, 0, Math.PI * 2);
  avatarCanvasContext.strokeStyle = "rgba(255,255,255,0.96)";
  avatarCanvasContext.lineWidth = 3;
  avatarCanvasContext.stroke();
  avatarCanvasContext.restore();
}

function initializeAvatarEditor(image) {
  state.avatarEditor.image = image;
  state.avatarEditor.scale = 1;
  state.avatarEditor.offsetX = 0;
  state.avatarEditor.offsetY = 0;
  elements.avatarScale.min = "1";
  elements.avatarScale.max = "4";
  elements.avatarScale.value = "1";
  openAvatarEditor();
  renderAvatarEditor();
}

function prepareAvatarEditor(file) {
  if (!file) return;
  cleanupAvatarObjectUrl();
  const objectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    state.avatarEditor.objectUrl = objectUrl;
    initializeAvatarEditor(image);
  };
  image.onerror = () => {
    cleanupAvatarObjectUrl();
    showToast("头像图片加载失败");
  };
  image.src = objectUrl;
}

function getCanvasPointer(event) {
  const rect = elements.avatarCanvas.getBoundingClientRect();
  const scaleX = elements.avatarCanvas.width / rect.width;
  const scaleY = elements.avatarCanvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

async function uploadAvatar() {
  if (!elements.avatarInput.files.length) {
    elements.avatarInput.click();
    return;
  }
  prepareAvatarEditor(elements.avatarInput.files[0]);
}

async function saveAvatarCrop() {
  if (!state.avatarEditor.image) {
    showToast("请先选择头像图片");
    return;
  }
  const { x, y, width, height, viewport } = computeAvatarTransform();
  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = AVATAR_VIEWPORT.output;
  outputCanvas.height = AVATAR_VIEWPORT.output;
  const outputContext = outputCanvas.getContext("2d");
  const scaleFactor = AVATAR_VIEWPORT.output / viewport.size;
  outputContext.drawImage(
    state.avatarEditor.image,
    (x - viewport.x) * scaleFactor,
    (y - viewport.y) * scaleFactor,
    width * scaleFactor,
    height * scaleFactor,
  );
  const blob = await new Promise((resolve) => outputCanvas.toBlob(resolve, "image/png"));
  if (!blob) {
    showToast("头像裁切失败");
    return;
  }

  const form = new FormData();
  form.append("file", new File([blob], "avatar.png", { type: "image/png" }));
  try {
    const data = await requestJson("/api/avatar", { method: "POST", body: form });
    renderSnapshot(data.snapshot);
    closeAvatarEditor();
    showToast("头像已更新");
  } catch (error) {
    showToast(error.message);
  }
}

async function uploadPersonaFile() {
  if (!elements.personaFile.files.length) {
    showToast("请先选择一个资料文件。");
    return;
  }
  setTypingStatus("正在整理资料并补充信息…");
  const form = new FormData();
  Array.from(elements.personaFile.files).forEach((file) => form.append("file", file));
  form.append("personaName", currentAgentName());
  form.append("workTitle", "");
  try {
    const data = await requestJson("/api/persona/file", { method: "POST", body: form });
    renderSnapshot(data.snapshot);
    renderPreview(data.preview);
    showToast("已生成待确认的人设预览");
  } catch (error) {
    showToast(error.message);
  } finally {
    setTypingStatus("已连接");
  }
}

async function previewPersona(options = {}) {
  const text = options.text ?? "";
  const label = options.label ?? "";
  if (!text) return;

  setTypingStatus("正在生成人设预览…");
  try {
    const data = await requestJson("/api/persona/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        personaName: currentAgentName(),
        workTitle: "",
        text,
        label,
      }),
    });
    renderSnapshot(data.snapshot);
    renderPreview(data.preview);
    if (!options.silent) showToast("已生成待确认的人设预览");
  } catch (error) {
    if (!options.silent) showToast(error.message);
  } finally {
    setTypingStatus("已连接");
  }
}

async function confirmPreview() {
  const previewId = state.preview?.preview_id || state.preview?.previewId || "";
  if (!previewId) {
    showToast("还没有可确认的预览");
    return;
  }
  setTypingStatus("正在写入人设库…");
  try {
      console.debug("[persona_confirm] submit", {
        previewId,
        selectedKeywords: Array.isArray(state.previewSelectedKeywords) ? state.previewSelectedKeywords : [],
      });
      const data = await requestJson("/api/persona/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          previewId,
          selectedKeywords: Array.isArray(state.previewSelectedKeywords) ? state.previewSelectedKeywords : [],
        }),
      });
    renderSnapshot(data.snapshot);
    renderPreview(null);
    elements.personaText.value = "";
    elements.personaLabel.value = "";
    elements.personaFile.value = "";
    showToast(`已写入 ${data.count} 条人设内容`);
  } catch (error) {
    showToast(error.message);
  } finally {
    setTypingStatus("已连接");
  }
}

async function learnPersonaText() {
  const text = elements.personaText.value.trim();
  if (!text) {
    showToast("请先输入一段设定资料。");
    return;
  }
  await previewPersona({ text, label: elements.personaLabel.value.trim() });
}

async function clearPersona() {
  try {
    const data = await requestJson("/api/persona/clear", { method: "POST" });
    renderSnapshot(data.snapshot);
    renderPreview(null);
    showToast("已清空人设");
  } catch (error) {
    showToast(error.message);
  }
}

async function resetAll() {
  if (!window.confirm("这会删除全部存档、头像和人设资料，确定继续吗？")) return;
  try {
    const data = await requestJson("/api/reset", { method: "POST" });
    renderSnapshot(data.snapshot);
    renderPreview(null);
    showToast("已删除全部存档");
  } catch (error) {
    showToast(error.message);
  }
}

function triggerAutoPreview() {
  clearTimeout(autoPreviewTimer);
  autoPreviewTimer = setTimeout(() => {
    previewPersona({
      silent: true,
      text: elements.personaText.value.trim(),
      label: elements.personaLabel.value.trim(),
    });
  }, 500);
}

elements.sendButton.addEventListener("click", sendMessage);
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

elements.openSettings.addEventListener("click", () => elements.drawer.classList.remove("hidden"));
elements.closeSettings.addEventListener("click", () => elements.drawer.classList.add("hidden"));
elements.drawer.querySelector(".drawer-backdrop").addEventListener("click", () => elements.drawer.classList.add("hidden"));

elements.saveName.addEventListener("click", saveName);
elements.personaWebSearchToggle.addEventListener("change", savePersonaWebSearchToggle);
elements.uploadAvatar.addEventListener("click", uploadAvatar);
elements.avatarInput.addEventListener("change", () => {
  if (elements.avatarInput.files.length) {
    prepareAvatarEditor(elements.avatarInput.files[0]);
  }
});

elements.closeAvatarEditor.addEventListener("click", closeAvatarEditor);
elements.avatarEditor.querySelector(".modal-backdrop").addEventListener("click", closeAvatarEditor);
elements.avatarScale.addEventListener("input", () => {
  state.avatarEditor.scale = Number(elements.avatarScale.value);
  renderAvatarEditor();
});
elements.reselectAvatar.addEventListener("click", () => elements.avatarInput.click());
elements.saveAvatarCrop.addEventListener("click", saveAvatarCrop);

elements.avatarCanvas.addEventListener("pointerdown", (event) => {
  if (!state.avatarEditor.image) return;
  const point = getCanvasPointer(event);
  state.avatarEditor.dragging = true;
  state.avatarEditor.dragStartX = point.x;
  state.avatarEditor.dragStartY = point.y;
  state.avatarEditor.startOffsetX = state.avatarEditor.offsetX;
  state.avatarEditor.startOffsetY = state.avatarEditor.offsetY;
  elements.avatarCanvas.classList.add("dragging");
  elements.avatarCanvas.setPointerCapture(event.pointerId);
});

elements.avatarCanvas.addEventListener("pointermove", (event) => {
  if (!state.avatarEditor.dragging) return;
  const point = getCanvasPointer(event);
  state.avatarEditor.offsetX = state.avatarEditor.startOffsetX + (point.x - state.avatarEditor.dragStartX);
  state.avatarEditor.offsetY = state.avatarEditor.startOffsetY + (point.y - state.avatarEditor.dragStartY);
  renderAvatarEditor();
});

elements.avatarCanvas.addEventListener("pointerup", (event) => {
  state.avatarEditor.dragging = false;
  elements.avatarCanvas.classList.remove("dragging");
  if (elements.avatarCanvas.hasPointerCapture(event.pointerId)) {
    elements.avatarCanvas.releasePointerCapture(event.pointerId);
  }
});

elements.avatarCanvas.addEventListener("pointerleave", () => {
  state.avatarEditor.dragging = false;
  elements.avatarCanvas.classList.remove("dragging");
});

elements.confirmPreview.addEventListener("click", confirmPreview);
elements.uploadPersonaFile.addEventListener("click", uploadPersonaFile);
elements.learnPersonaText.addEventListener("click", learnPersonaText);
elements.personaText.addEventListener("input", triggerAutoPreview);
elements.personaText.addEventListener("blur", () => previewPersona({
  silent: true,
  text: elements.personaText.value.trim(),
  label: elements.personaLabel.value.trim(),
}));
elements.personaLabel.addEventListener("input", triggerAutoPreview);
elements.clearPersona.addEventListener("click", clearPersona);
elements.resetAll.addEventListener("click", resetAll);

bootstrap().catch((error) => {
  setTypingStatus("连接失败");
  showToast(error.message);
});
