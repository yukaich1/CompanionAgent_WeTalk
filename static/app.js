const state = {
  snapshot: null,
  preview: null,
  selectedPreviewKeywords: [],
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
  activityList: document.getElementById("activity-list"),
  agentAvatar: document.getElementById("agent-avatar"),
  toast: document.getElementById("toast"),
  drawer: document.getElementById("settings-drawer"),
  openSettings: document.getElementById("open-settings"),
  closeSettings: document.getElementById("close-settings"),
  nameInput: document.getElementById("name-input"),
  saveName: document.getElementById("save-name"),
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

function splitPreviewSnippets(preview) {
  return Array.isArray(preview?.snippets) ? preview.snippets : [];
}

function previewSourceLabel(source) {
  const mapping = {
    local: "本地资料",
    web_summary: "联网补充摘要",
    web: "联网资料",
  };
  return mapping[source] || source || "资料";
}

function normalizeKeywordSelection(preview) {
  const selected = Array.isArray(preview?.selected_keywords) ? preview.selected_keywords : [];
  const fallback = Array.isArray(preview?.summary?.display_keywords) ? preview.summary.display_keywords : [];
  const merged = [...selected, ...fallback].filter(Boolean);
  return [...new Set(merged)].slice(0, 8);
}

function renderPreviewKeywords(preview) {
  const options = Array.isArray(preview?.keyword_options) ? preview.keyword_options : [];
  state.selectedPreviewKeywords = normalizeKeywordSelection(preview);
  if (!options.length) {
    const keywords = preview.summary?.display_keywords || [];
    elements.previewKeywords.innerHTML = keywords.length
      ? keywords.map((keyword) => `<span class="tag">${escapeHtml(keyword)}</span>`).join("")
      : `<span class="tag">等待提炼关键词</span>`;
    return;
  }

  elements.previewKeywords.innerHTML = `
    <div class="keyword-picker-head">
      <strong>选择 8 个高权重关键词</strong>
      <span id="preview-keyword-count" class="inline-meta">${state.selectedPreviewKeywords.length}/8</span>
    </div>
    <div class="keyword-groups">
      ${options.map((option, optionIndex) => `
        <div class="keyword-group">
          <div class="preview-source">${escapeHtml(previewSourceLabel(option.source))}${option.title ? ` · ${escapeHtml(option.title)}` : ""}</div>
          <div class="keyword-choice-list">
            ${option.keywords.map((keyword, keywordIndex) => {
              const checked = state.selectedPreviewKeywords.includes(keyword);
              return `
                <label class="keyword-choice ${checked ? "selected" : ""}">
                  <input
                    type="checkbox"
                    data-option-index="${optionIndex}"
                    data-keyword-index="${keywordIndex}"
                    value="${escapeHtml(keyword)}"
                    ${checked ? "checked" : ""}
                  >
                  <span>${escapeHtml(keyword)}</span>
                </label>
              `;
            }).join("")}
          </div>
        </div>
      `).join("")}
    </div>
  `;

  const countNode = document.getElementById("preview-keyword-count");
  elements.previewKeywords.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.addEventListener("change", (event) => {
      const keyword = event.target.value;
      if (event.target.checked) {
        if (state.selectedPreviewKeywords.length >= 8) {
          event.target.checked = false;
          showToast("最多只能选择 8 个关键词");
          return;
        }
        state.selectedPreviewKeywords = [...new Set([...state.selectedPreviewKeywords, keyword])];
      } else {
        state.selectedPreviewKeywords = state.selectedPreviewKeywords.filter((item) => item !== keyword);
      }
      countNode.textContent = `${state.selectedPreviewKeywords.length}/8`;
      elements.previewKeywords.querySelectorAll(".keyword-choice").forEach((choice) => {
        const inputNode = choice.querySelector("input");
        choice.classList.toggle("selected", inputNode.checked);
      });
    });
  });
}

function renderPreview(preview) {
  state.preview = preview || null;
  if (!preview) {
    elements.previewPanel.classList.add("hidden");
    elements.previewKeywords.innerHTML = "";
    elements.previewSnippets.innerHTML = "";
    state.selectedPreviewKeywords = [];
    return;
  }

  elements.previewPanel.classList.remove("hidden");
  elements.previewTitle.textContent = `${preview.persona_name}${preview.work_title ? ` · ${preview.work_title}` : ""} 待确认人设预览`;
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

function renderHistory(history) {
  elements.messages.innerHTML = "";
  history.forEach((message) => appendMessage(message));
  elements.messages.scrollTop = elements.messages.scrollHeight;
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
  const bubbles = (message.bubbles && message.bubbles.length ? message.bubbles : [message.content]).filter(Boolean);
  bubbles.forEach((bubbleText) => {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.role}`;
    bubble.innerHTML = escapeHtml(bubbleText);
    stack.appendChild(bubble);
  });
  elements.messages.appendChild(row);
}

function typingDelayForChar(char) {
  if ("，、；：".includes(char)) return 55;
  if ("。！？…".includes(char)) return 120;
  if (" ".includes(char)) return 8;
  return 22;
}

async function streamAssistantMessage(message) {
  const bubbles = (message.bubbles && message.bubbles.length ? message.bubbles : [message.content]).filter(Boolean);
  if (!bubbles.length) return;

  const { row, stack } = createMessageRow({ role: "assistant" });
  elements.messages.appendChild(row);

  for (let i = 0; i < bubbles.length; i += 1) {
    setTypingStatus("正在输入…");
    const bubbleText = bubbles[i];
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    bubble.innerHTML = "";
    stack.appendChild(bubble);
    elements.messages.scrollTop = elements.messages.scrollHeight;

    let rendered = "";
    for (const char of bubbleText) {
      rendered += char;
      bubble.innerHTML = escapeHtml(rendered);
      elements.messages.scrollTop = elements.messages.scrollHeight;
      await sleep(typingDelayForChar(char));
    }

    await sleep(i === bubbles.length - 1 ? 80 : 420);
  }
}

function renderSnapshot(snapshot, options = {}) {
  const preserveHistory = Boolean(options.preserveHistory);
  state.snapshot = snapshot;
  const { agent, history, recentActivity } = snapshot;

  elements.agentName.textContent = agent.name;
  elements.chatTitle.textContent = agent.name;
  elements.agentMood.textContent = agent.mood;
  elements.agentAffinity.textContent = agent.affinity;
  elements.personaCount.textContent = `${agent.personaChunks} 条人设`;
  elements.nameInput.value = agent.name;

  const avatarUrl = agent.avatarUrl || DEFAULT_AGENT_AVATAR;
  elements.agentAvatar.src = avatarUrl;
  elements.chatAgentAvatar.src = avatarUrl;

  elements.keywordList.innerHTML = agent.keywords.length
    ? agent.keywords.map((keyword) => `<span class="tag">${escapeHtml(keyword)}</span>`).join("")
    : `<span class="tag">等待学习</span>`;

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
    showToast("名称已更新");
  } catch (error) {
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
  const canvas = elements.avatarCanvas;
  return {
    x: (canvas.width - AVATAR_VIEWPORT.size) / 2,
    y: (canvas.height - AVATAR_VIEWPORT.size) / 2,
    size: AVATAR_VIEWPORT.size,
  };
}

function computeAvatarTransform() {
  const editor = state.avatarEditor;
  const image = editor.image;
  const viewport = getAvatarViewport();
  const baseScale = Math.max(viewport.size / image.width, viewport.size / image.height);
  const scale = baseScale * editor.scale;
  const width = image.width * scale;
  const height = image.height * scale;
  const x = elements.avatarCanvas.width / 2 - width / 2 + editor.offsetX;
  const y = elements.avatarCanvas.height / 2 - height / 2 + editor.offsetY;
  return { x, y, width, height, viewport };
}

function renderAvatarEditor() {
  const editor = state.avatarEditor;
  if (!editor.image) {
    avatarCanvasContext.clearRect(0, 0, elements.avatarCanvas.width, elements.avatarCanvas.height);
    return;
  }
  const { x, y, width, height, viewport } = computeAvatarTransform();
  avatarCanvasContext.clearRect(0, 0, elements.avatarCanvas.width, elements.avatarCanvas.height);
  avatarCanvasContext.drawImage(editor.image, x, y, width, height);

  avatarCanvasContext.save();
  avatarCanvasContext.fillStyle = "rgba(8, 10, 14, 0.54)";
  avatarCanvasContext.beginPath();
  avatarCanvasContext.rect(0, 0, elements.avatarCanvas.width, elements.avatarCanvas.height);
  avatarCanvasContext.arc(
    viewport.x + viewport.size / 2,
    viewport.y + viewport.size / 2,
    viewport.size / 2,
    0,
    Math.PI * 2,
    true,
  );
  avatarCanvasContext.fill("evenodd");
  avatarCanvasContext.restore();

  avatarCanvasContext.save();
  avatarCanvasContext.beginPath();
  avatarCanvasContext.arc(
    viewport.x + viewport.size / 2,
    viewport.y + viewport.size / 2,
    viewport.size / 2,
    0,
    Math.PI * 2,
  );
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
  const editor = state.avatarEditor;
  if (!editor.image) {
    showToast("先选择头像图片");
    return;
  }
  const { x, y, width, height, viewport } = computeAvatarTransform();
  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = AVATAR_VIEWPORT.output;
  outputCanvas.height = AVATAR_VIEWPORT.output;
  const outputContext = outputCanvas.getContext("2d");
  const scaleFactor = AVATAR_VIEWPORT.output / viewport.size;
  outputContext.drawImage(
    editor.image,
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
    showToast("先选择一个资料文件。");
    return;
  }
  setTypingStatus("正在整理资料并补充信息…");
  const form = new FormData();
  Array.from(elements.personaFile.files).forEach((file) => form.append("file", file));
  form.append("personaName", state.snapshot?.agent?.name || "");
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
  const personaName = state.snapshot?.agent?.name || "";
  const workTitle = "";
  const text = options.text ?? "";
  const label = options.label ?? "";
  if (!text) return;

  setTypingStatus("正在生成人设预览…");
  try {
    const data = await requestJson("/api/persona/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ personaName, workTitle, text, label }),
    });
    renderSnapshot(data.snapshot);
    renderPreview(data.preview);
    if (!options.silent) {
      showToast("已生成待确认的人设预览");
    }
  } catch (error) {
    if (!options.silent) {
      showToast(error.message);
    }
  } finally {
    setTypingStatus("已连接");
  }
}

async function confirmPreview() {
  if (!state.preview?.preview_id) {
    showToast("还没有可确认的预览");
    return;
  }
  setTypingStatus("正在写入人设库…");
  try {
    const data = await requestJson("/api/persona/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        previewId: state.preview.preview_id,
        selectedKeywords: state.selectedPreviewKeywords,
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
    showToast("先输入一段设定资料。");
    return;
  }
  await previewPersona({
    text,
    label: elements.personaLabel.value.trim(),
  });
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
