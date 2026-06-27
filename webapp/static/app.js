const state = {
  jobs: [],
  socket: null,
};

const healthElements = {
  milvusStatus: document.getElementById("milvusStatus"),
  collectionName: document.getElementById("collectionName"),
  ocrStatus: document.getElementById("ocrStatus"),
  milvusDetail: document.getElementById("milvusDetail"),
  uploadDir: document.getElementById("uploadDir"),
  jobState: document.getElementById("jobState"),
};

const chatStatus = document.getElementById("chatStatus");
const conversation = document.getElementById("conversation");
const traceOutput = document.getElementById("traceOutput");
const jobList = document.getElementById("jobList");
const chatSubmitButton = document.querySelector('#chatForm button[type="submit"]');

document.getElementById("refreshHealthButton").addEventListener("click", () => {
  void refreshHealth();
});
document.getElementById("refreshJobsButton").addEventListener("click", () => {
  void refreshJobs();
});
document.getElementById("pathIngestForm").addEventListener("submit", (event) => {
  event.preventDefault();
  void submitPathIngest();
});
document.getElementById("uploadIngestForm").addEventListener("submit", (event) => {
  event.preventDefault();
  void submitUploadIngest();
});
document.getElementById("chatForm").addEventListener("submit", (event) => {
  event.preventDefault();
  void startChat();
});

void bootstrap();

async function bootstrap() {
  await Promise.all([refreshHealth(), refreshJobs()]);
  window.setInterval(() => {
    void refreshHealth();
    void refreshJobs();
  }, 5000);
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthElements.milvusStatus.textContent = data.milvus_status;
    healthElements.collectionName.textContent = data.collection_name;
    healthElements.ocrStatus.textContent = data.ocr_enabled ? "enabled" : "disabled";
    healthElements.milvusDetail.textContent = data.milvus_detail;
    healthElements.uploadDir.textContent = data.upload_dir;
    healthElements.jobState.textContent = data.active_job ? "有任务在执行" : "空闲";
    applyBadgeState(healthElements.milvusStatus, data.milvus_status === "ok" ? "completed" : "failed");
  } catch (error) {
    healthElements.milvusStatus.textContent = "error";
    healthElements.milvusDetail.textContent = String(error);
    applyBadgeState(healthElements.milvusStatus, "failed");
  }
}

async function refreshJobs() {
  try {
    const response = await fetch("/api/jobs");
    const data = await response.json();
    state.jobs = data.jobs || [];
    renderJobs();
  } catch (error) {
    jobList.innerHTML = "";
    jobList.appendChild(buildEmptyState(`任务列表获取失败：${String(error)}`));
  }
}

async function submitPathIngest() {
  const path = document.getElementById("pathInput").value.trim();
  const confirmReset = document.getElementById("pathConfirmReset").checked;
  if (!path) {
    showTransientTrace("请输入要入库的目录路径。", "failed");
    return;
  }

  await submitJson("/api/ingest/path", {
    path,
    confirm_reset: confirmReset,
  });
}

async function submitUploadIngest() {
  const fileInput = document.getElementById("uploadFiles");
  const confirmReset = document.getElementById("uploadConfirmReset").checked;
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    showTransientTrace("至少选择一个 .md 或 .pdf 文件。", "failed");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("confirm_reset", String(confirmReset));

  try {
    const response = await fetch("/api/ingest/upload", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "上传入库失败");
    }
    fileInput.value = "";
    await refreshJobs();
    await refreshHealth();
    showTransientTrace(`任务 ${payload.id} 已创建。`, "completed");
  } catch (error) {
    showTransientTrace(String(error), "failed");
  }
}

async function submitJson(url, payload) {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求失败");
    }
    await refreshJobs();
    await refreshHealth();
    showTransientTrace(`任务 ${data.id} 已创建。`, "completed");
  } catch (error) {
    showTransientTrace(String(error), "failed");
  }
}

function renderJobs() {
  jobList.innerHTML = "";
  if (state.jobs.length === 0) {
    jobList.appendChild(buildEmptyState("还没有任务记录。"));
    return;
  }

  const template = document.getElementById("jobCardTemplate");
  for (const job of state.jobs) {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".job-id").textContent = job.id;
    node.querySelector(".job-source").textContent = job.source;
    node.querySelector(".job-message").textContent = job.error || job.message;
    node.querySelector(".job-meta").textContent = `${job.mode} · ${job.file_count} files · ${job.updated_at}`;
    const badge = node.querySelector(".job-status");
    badge.textContent = job.status;
    applyBadgeState(badge, job.status);
    jobList.appendChild(node);
  }
}

async function startChat() {
  const questionInput = document.getElementById("questionInput");
  const question = questionInput.value.trim();
  if (!question) {
    appendConversationBubble("error", "问题不能为空");
    return;
  }

  resetChatPanels();
  appendConversationBubble("question", question);
  setChatStatus("connecting", "queued");
  setChatPending(true);
  showTransientTrace("问题已发送，正在连接问答工作流...", "running");

  if (state.socket) {
    state.socket.close();
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/chat`);
  state.socket = socket;

  socket.addEventListener("open", () => {
    setChatStatus("streaming", "running");
    showTransientTrace("已经连上后端，正在检索知识库并生成答案...", "running");
    socket.send(JSON.stringify({ question }));
  });

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    handleChatEvent(payload);
  });

  socket.addEventListener("close", () => {
    if (state.socket === socket) {
      state.socket = null;
    }
    setChatPending(false);
    if (chatStatus.textContent === "completed" || chatStatus.textContent === "error") {
      return;
    }
    setChatStatus("idle", "");
  });

  socket.addEventListener("error", () => {
    appendConversationBubble("error", "WebSocket 连接失败");
    setChatStatus("error", "failed");
    setChatPending(false);
  });
}

function handleChatEvent(event) {
  if (event.type === "error") {
    clearTransientTrace();
    appendConversationBubble("error", event.message);
    setChatStatus("error", "failed");
    setChatPending(false);
    return;
  }

  if (event.type === "final") {
    clearTransientTrace();
    appendConversationBubble("answer", event.answer || "没有生成答案");
    setChatStatus("completed", "completed");
    setChatPending(false);
    return;
  }

  if (event.type !== "node") {
    return;
  }

  appendTraceCard(event);
}

function appendTraceCard(event) {
  clearEmptyState(traceOutput);
  clearTransientTrace();
  const template = document.getElementById("traceCardTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".trace-node").textContent = event.node;
  const documents = event.documents || [];
  node.querySelector(".trace-doc-count").textContent = documents.length ? `${documents.length} docs` : "no docs";
  const body = node.querySelector(".trace-body");

  if (event.generation) {
    const preview = document.createElement("div");
    preview.className = "generation-preview";
    preview.textContent = event.generation;
    body.appendChild(preview);
  }

  if (documents.length > 0) {
    for (const doc of documents.slice(0, 4)) {
      body.appendChild(buildDocumentCard(doc));
    }
  }

  traceOutput.prepend(node);
}

function buildDocumentCard(doc) {
  const wrapper = document.createElement("details");
  wrapper.className = "doc-card";
  const summary = document.createElement("summary");
  const metadata = doc.metadata || {};
  const sourceName = metadata.filename || metadata.source || "document";
  const pageInfo = metadata.page_number ? ` · page ${metadata.page_number}` : "";
  const elementInfo = metadata.element_type ? ` · ${metadata.element_type}` : "";
  summary.textContent = `${sourceName}${pageInfo}${elementInfo}`;
  wrapper.appendChild(summary);

  const meta = document.createElement("div");
  meta.className = "doc-meta";
  meta.textContent = `title: ${metadata.title || "-"}${metadata.asset_path ? ` · asset: ${metadata.asset_path}` : ""}`;
  wrapper.appendChild(meta);

  const content = document.createElement("pre");
  content.textContent = doc.content || "";
  wrapper.appendChild(content);
  return wrapper;
}

function appendConversationBubble(type, text) {
  clearEmptyState(conversation);
  const bubble = document.createElement("article");
  bubble.className = `bubble ${type}`;
  const label = document.createElement("span");
  label.className = "bubble-label";
  label.textContent =
    type === "question" ? "问题" :
    type === "answer" ? "答案" :
    "错误";
  const content = document.createElement("div");
  content.textContent = text;
  bubble.append(label, content);
  conversation.appendChild(bubble);
}

function setChatStatus(text, variant) {
  chatStatus.textContent = text;
  chatStatus.className = "badge";
  if (variant) {
    chatStatus.classList.add(variant);
  }
}

function setChatPending(isPending) {
  if (!chatSubmitButton) {
    return;
  }
  chatSubmitButton.disabled = isPending;
  chatSubmitButton.textContent = isPending ? "提问中..." : "开始提问";
}

function resetChatPanels() {
  conversation.innerHTML = "";
  traceOutput.innerHTML = "";
  traceOutput.appendChild(buildEmptyState("等待工作流事件..."));
}

function showTransientTrace(message, variant) {
  clearEmptyState(traceOutput);
  const bubble = document.createElement("div");
  bubble.className = "generation-preview";
  bubble.classList.add("transient-trace");
  bubble.textContent = message;
  if (variant === "failed") {
    bubble.style.background = "rgba(180, 35, 24, 0.1)";
  }
  traceOutput.prepend(bubble);
}

function clearTransientTrace() {
  for (const node of traceOutput.querySelectorAll(".transient-trace")) {
    node.remove();
  }
}

function buildEmptyState(message) {
  const node = document.createElement("div");
  node.className = "empty-state compact";
  const p = document.createElement("p");
  p.textContent = message;
  node.appendChild(p);
  return node;
}

function clearEmptyState(container) {
  const empty = container.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function applyBadgeState(element, status) {
  element.className = "badge";
  if (status) {
    element.classList.add(status);
  }
}
