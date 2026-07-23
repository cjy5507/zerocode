import {
  applyToolApprovalRequest,
  applyToolApprovalResolution,
  basePath,
  drainMessagesForEpoch,
  handleCommandAckTimeout,
  isSessionRevokedClose,
  pairingPollDeadline,
  pausePendingCommands,
  pushSupportState,
  replayPendingCommands,
  replacePendingToolApprovals,
  shouldDismissKeyboardFromTouch,
  shouldRetryPairing,
  shouldShowInstallHint,
  socketAttemptIsCurrent,
  tokenizeMarkdown,
  urlBase64ToUint8Array,
} from "./remote-state.js";

const $ = (selector) => document.querySelector(selector);
const MAX_FRAMES = 512;
const MAX_RENDERED_ITEMS = 240;
const PROTOCOL_VERSION = 1;
const KEYBOARD_STABLE_MS = 300;
const KEYBOARD_SETTLE_MS = 220;
const INSTALL_HINT_KEY = "zo-remote-install-hint";
const BASE_PATH = basePath();
const WS_PROTOCOL = location.protocol === "http:" ? "ws" : "wss";
let viewportFrame = 0;

const ui = {
  pairView: $("#pair-view"),
  pairForm: $("#pair-form"),
  pairButton: $("#pair-button"),
  pairError: $("#pair-error"),
  deviceName: $("#device-name"),
  approvalCard: $("#approval-card"),
  comparisonCode: $("#comparison-code"),
  approvalCommand: $("#approval-command"),
  installAffordances: $("#install-affordances"),
  installButton: $("#install-button"),
  iosInstallHint: $("#ios-install-hint"),
  dismissInstallHint: $("#dismiss-install-hint"),
  remoteView: $("#remote-view"),
  sessionTitle: $("#session-title"),
  connectionLabel: $("#connection-label"),
  connectionDot: $("#connection-dot"),
  connectionBanner: $("#connection-banner"),
  connectionMessage: $("#connection-message"),
  roleBadge: $("#role-badge"),
  pushToggle: $("#push-toggle"),
  requestControl: $("#request-control"),
  observerPanel: $("#observer-panel"),
  observerControl: $("#observer-control"),
  composerWrap: $(".composer-wrap"),
  transcript: $("#transcript"),
  newItems: $("#new-items"),
  newItemsLabel: $(".new-items-label"),
  composer: $("#composer"),
  prompt: $("#prompt"),
  sendButton: $("#send-button"),
  modeSwitch: $("#mode-switch"),
  modeOptions: [...document.querySelectorAll(".mode-option")],
  composerHint: $("#composer-hint"),
  cancelTurn: $("#cancel-turn"),
  cancelDialog: $("#cancel-dialog"),
  announcer: $("#announcer"),
};

const model = {
  socket: null,
  socketEpoch: 0,
  frames: [],
  turn: "idle",
  role: "observer",
  lastSeq: 0,
  renderQueue: [],
  renderScheduled: false,
  newItems: 0,
  selectedMode: "queue",
  reconnectAttempt: 0,
  reconnectTimer: null,
  everConnected: false,
  pairingId: null,
  pairingTimer: null,
  pairingDeadline: 0,
  secret: location.hash.length > 1 ? location.hash.slice(1) : "",
  pending: new Map(),
  approvals: new Map(),
  sessionId: null,
  statusTimer: null,
};

const installState = {
  prompt: null,
  dismissed: installHintWasDismissed(),
};

const pushState = {
  available: false,
  busy: false,
  configRequested: false,
  registration: null,
  serverKey: "",
  state: "hidden",
};

const keyboard = {
  baselineBottom: 0,
  frame: 0,
  lastInset: null,
  layoutBottom: 0,
  pinned: false,
  settling: false,
  stableSince: 0,
  suppressScroll: false,
  touchY: [],
  usesVirtualKeyboard: Boolean(navigator.virtualKeyboard),
};

history.replaceState(null, "", `${location.pathname}${location.search}`);
ui.prompt.value = sessionStorage.getItem("zo.remote.draft") || "";
if (navigator.virtualKeyboard) {
  try {
    navigator.virtualKeyboard.overlaysContent = true;
    ui.remoteView.classList.add("virtual-keyboard-overlay");
  } catch {
    keyboard.usesVirtualKeyboard = false;
  }
}
syncViewport();
resizeComposer();

ui.pairForm.addEventListener("submit", beginPairing);
ui.installButton.addEventListener("click", promptInstall);
ui.dismissInstallHint.addEventListener("click", dismissInstallHint);
ui.pushToggle.addEventListener("click", togglePush);
ui.composer.addEventListener("submit", submitPrompt);
ui.prompt.addEventListener("input", () => {
  sessionStorage.setItem("zo.remote.draft", ui.prompt.value);
  resizeComposer();
  updateComposer();
});
ui.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    ui.composer.requestSubmit();
  }
});
ui.sendButton.addEventListener("pointerdown", preservePromptFocus);
ui.sendButton.addEventListener("touchend", submitFromTouch, { passive: false });
ui.sendButton.addEventListener("mousedown", preservePromptFocus);
ui.sendButton.addEventListener("click", submitFromSendButton);
ui.modeOptions.forEach((button) => button.addEventListener("click", () => selectMode(button.dataset.mode)));
ui.requestControl.addEventListener("click", requestControl);
ui.observerControl.addEventListener("click", requestControl);
ui.cancelTurn.addEventListener("click", () => ui.cancelDialog.showModal());
ui.cancelDialog.addEventListener("close", () => {
  if (ui.cancelDialog.returnValue === "confirm") sendCancel();
});
ui.newItems.addEventListener("click", scrollToLatest);
ui.transcript.addEventListener("click", handleToolApprovalClick);
ui.transcript.addEventListener("scroll", () => {
  if (keyboard.suppressScroll) return;
  const atBottom = nearBottom();
  if (document.activeElement === ui.prompt) keyboard.pinned = atBottom;
  if (atBottom) {
    model.newItems = 0;
    ui.newItems.hidden = true;
  }
}, { passive: true });
ui.transcript.addEventListener("touchstart", beginTranscriptDrag, { passive: true });
ui.transcript.addEventListener("touchmove", continueTranscriptDrag, { passive: true });
ui.transcript.addEventListener("touchend", endTranscriptDrag, { passive: true });
ui.transcript.addEventListener("touchcancel", endTranscriptDrag, { passive: true });
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installState.prompt = event;
  updateInstallAffordances();
});
window.addEventListener("appinstalled", () => {
  installState.prompt = null;
  updateInstallAffordances();
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") reconnectFromForeground();
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted) reconnectFromForeground();
});
window.addEventListener("online", reconnectNow);
window.addEventListener("offline", () => setConnection("offline", "Phone is offline"));
window.addEventListener("resize", scheduleViewportSync, { passive: true });
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", scheduleViewportSync, { passive: true });
  window.visualViewport.addEventListener("scroll", scheduleViewportSync, { passive: true });
}
if (navigator.virtualKeyboard?.addEventListener) {
  navigator.virtualKeyboard.addEventListener("geometrychange", () => wakeKeyboardTracker(true));
}
if (window.ResizeObserver) {
  new ResizeObserver(syncComposerHeight).observe(ui.composerWrap);
}
ui.prompt.addEventListener("focus", handlePromptFocus);
ui.prompt.addEventListener("blur", handlePromptBlur);

if (model.secret) {
  showPairView();
} else {
  showRemoteView();
  connectSocket(false);
}

async function beginPairing(event) {
  event.preventDefault();
  if (!model.secret) {
    showPairError("This QR code is missing or expired. Run /remote qr and scan it again.");
    return;
  }
  ui.pairButton.disabled = true;
  ui.pairError.hidden = true;
  try {
    const response = await fetch(`${BASE_PATH}/api/pair`, {
      method: "POST",
      credentials: "same-origin",
      cache: "no-store",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ secret: model.secret, device_name: ui.deviceName.value.trim() }),
    });
    model.secret = "";
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || "Pairing failed.");
    model.pairingId = body.pairing_id;
    model.pairingDeadline = pairingPollDeadline(Date.now(), body.poll_expires_in_seconds);
    ui.pairForm.hidden = true;
    ui.approvalCard.hidden = false;
    ui.comparisonCode.textContent = body.comparison_code;
    ui.approvalCommand.textContent = `/remote approve ${body.comparison_code}`;
    pollPairing();
  } catch (error) {
    ui.pairButton.disabled = false;
    showPairError(error.message || "Could not reach Zo.");
  }
}

async function pollPairing() {
  if (!model.pairingId) return;
  try {
    const response = await fetch(`${BASE_PATH}/api/pair/${encodeURIComponent(model.pairingId)}`, {
      credentials: "same-origin",
      cache: "no-store",
    });
    const body = await response.json();
    if (body.status === "approved") {
      clearTimeout(model.pairingTimer);
      model.pairingId = null;
      model.pairingDeadline = 0;
      showRemoteView();
      connectSocket(true);
      return;
    }
    if (body.status === "denied" || body.status === "expired" || response.status === 410) {
      model.pairingId = null;
      model.pairingDeadline = 0;
      throw new Error(body.status === "denied" ? "The connection was denied in the terminal." : "The pairing request expired. Run /remote qr and scan again.");
    }
    model.pairingTimer = setTimeout(pollPairing, 900);
  } catch (error) {
    if (shouldRetryPairing(model.pairingId, model.pairingDeadline)) {
      announce("Connection interrupted while waiting for approval. Retrying.");
      model.pairingTimer = setTimeout(pollPairing, 1_200);
      return;
    }
    model.pairingId = null;
    model.pairingDeadline = 0;
    ui.approvalCard.hidden = true;
    ui.pairForm.hidden = false;
    ui.pairButton.disabled = false;
    showPairError(error.message || "Pairing stopped.");
  }
}

function connectSocket(retry) {
  clearTimeout(model.reconnectTimer);
  if (!navigator.onLine) {
    scheduleReconnect();
    return;
  }
  setConnection("connecting", retry ? "Reconnecting to Zo…" : "Connecting to Zo…");
  const epoch = model.socketEpoch + 1;
  model.socketEpoch = epoch;
  const socket = new WebSocket(`${WS_PROTOCOL}://${location.host}${BASE_PATH}/ws`, "zo.remote.v1");
  model.socket = socket;
  const isCurrent = () => socketAttemptIsCurrent(model.socketEpoch, epoch) && model.socket === socket;

  socket.addEventListener("open", () => {
    if (!isCurrent()) return;
    model.everConnected = true;
    model.reconnectAttempt = 0;
    setConnection("online", "Connected through Tailscale");
    send({ type: "hello", version: PROTOCOL_VERSION, last_seq: model.lastSeq });
    replayPendingCommands(model.pending, send, armPendingTimeout);
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
  });
  socket.addEventListener("message", (event) => {
    if (!isCurrent()) return;
    try {
      enqueueMessage(JSON.parse(event.data), epoch);
    } catch {
      setConnection("offline", "Zo sent an unreadable event");
    }
  });
  socket.addEventListener("close", (event) => {
    if (!isCurrent()) return;
    model.socket = null;
    pausePendingCommands(model.pending);
    if (isSessionRevokedClose(event.code) || !model.everConnected) {
      returnToPairing("This remote credential is no longer active. Run /remote qr and scan the current code.");
      return;
    }
    scheduleReconnect();
  });
  socket.addEventListener("error", () => {
    if (isCurrent()) socket.close();
  });
}

function scheduleReconnect() {
  setConnection(navigator.onLine ? "connecting" : "offline", navigator.onLine ? "Reconnecting to Zo…" : "Phone is offline");
  const delay = Math.min(10_000, 500 * (2 ** model.reconnectAttempt));
  model.reconnectAttempt += 1;
  model.reconnectTimer = setTimeout(() => connectSocket(true), delay);
}

function reconnectNow() {
  if (!model.socket || model.socket.readyState > WebSocket.OPEN) connectSocket(true);
}

function reconnectFromForeground() {
  if (pushState.available) refreshPushState().catch(() => {});
  if (model.socket?.readyState === WebSocket.OPEN) return;
  model.reconnectAttempt = 0;
  clearTimeout(model.reconnectTimer);
  model.reconnectTimer = null;
  reconnectNow();
}

function installHintWasDismissed() {
  try {
    return localStorage.getItem(INSTALL_HINT_KEY) === "dismissed";
  } catch {
    return false;
  }
}

function isIOSDevice() {
  return /iPad|iPhone|iPod/i.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

function isStandalone() {
  return navigator.standalone === true || window.matchMedia("(display-mode: standalone)").matches;
}

function updateInstallAffordances() {
  const showPrompt = Boolean(installState.prompt);
  const showHint = shouldShowInstallHint({
    isIOS: isIOSDevice(),
    isStandalone: isStandalone(),
    dismissed: installState.dismissed,
  });
  ui.installButton.hidden = !showPrompt;
  ui.iosInstallHint.hidden = !showHint;
  ui.installAffordances.hidden = !showPrompt && !showHint;
}

async function promptInstall() {
  const promptEvent = installState.prompt;
  if (!promptEvent) return;
  installState.prompt = null;
  updateInstallAffordances();
  try {
    await promptEvent.prompt();
    await promptEvent.userChoice;
  } catch {
    showPairError("Could not open the app install prompt.");
  }
}

function dismissInstallHint() {
  installState.dismissed = true;
  try {
    localStorage.setItem(INSTALL_HINT_KEY, "dismissed");
  } catch {
    // Keep the dismissal for this page even when storage is unavailable.
  }
  updateInstallAffordances();
}

function browserPushSupportState() {
  return pushSupportState({
    isSecure: globalThis.isSecureContext === true,
    hasServiceWorker: "serviceWorker" in navigator,
    hasPushManager: "PushManager" in globalThis,
    hasNotification: "Notification" in globalThis,
    isStandalone: isStandalone(),
    isIOS: isIOSDevice(),
  });
}

async function loadPushConfigOnce() {
  if (pushState.configRequested) return;
  pushState.configRequested = true;
  try {
    const response = await fetch(`${BASE_PATH}/api/push/config`, {
      credentials: "same-origin",
      cache: "no-store",
    });
    if (response.status === 401 || response.status === 404) {
      setPushToggleState("hidden");
      return;
    }
    if (!response.ok) throw new Error("Could not load notification settings.");
    const body = await response.json();
    if (body?.push === null) {
      setPushToggleState("hidden");
      return;
    }
    const serverKey = body?.push?.server_key;
    if (typeof serverKey !== "string" || !serverKey) {
      throw new Error("Zo returned an invalid notification configuration.");
    }
    pushState.available = true;
    pushState.serverKey = serverKey;
    const support = browserPushSupportState();
    if (support === "unsupported") {
      setPushToggleState("hidden");
      return;
    }
    if (support === "needs_install") {
      setPushToggleState("needs_install");
      return;
    }
    pushState.registration = await ensureServiceWorkerRegistration();
    await refreshPushState();
  } catch (error) {
    pushState.available = false;
    setPushToggleState("hidden");
    showRemoteStatus(error.message || "Could not load notification settings.");
  }
}

async function ensureServiceWorkerRegistration() {
  if (!pushState.registration) {
    pushState.registration = await navigator.serviceWorker.register("./sw.js");
  }
  return pushState.registration;
}

async function refreshPushState() {
  if (!pushState.available) {
    setPushToggleState("hidden");
    return;
  }
  const support = browserPushSupportState();
  if (support === "unsupported") {
    setPushToggleState("hidden");
    return;
  }
  if (support === "needs_install") {
    setPushToggleState("needs_install");
    return;
  }
  const registration = await ensureServiceWorkerRegistration();
  const subscription = await registration.pushManager.getSubscription();
  if (Notification.permission === "denied") {
    setPushToggleState("denied");
  } else if (Notification.permission === "granted" && subscription) {
    setPushToggleState("on");
  } else {
    setPushToggleState("off");
  }
}

function setPushToggleState(state) {
  pushState.state = state;
  ui.pushToggle.hidden = state === "hidden";
  ui.pushToggle.dataset.state = state;
  ui.pushToggle.disabled = pushState.busy;
  ui.pushToggle.setAttribute("aria-busy", String(pushState.busy));
  ui.pushToggle.setAttribute("aria-pressed", String(state === "on"));
  const label = pushState.busy
    ? "Updating notification settings"
    : state === "on"
      ? "Disable notifications"
      : state === "denied"
        ? "Notifications blocked in browser settings"
        : state === "needs_install"
          ? "Install this app to enable notifications"
          : "Enable notifications";
  ui.pushToggle.setAttribute("aria-label", label);
  ui.pushToggle.title = label;
}

function togglePush() {
  if (pushState.busy || !pushState.available) return;
  if (pushState.state === "needs_install") {
    showRemoteStatus("Install Zo Remote from the Share menu to enable notifications.");
    return;
  }
  if (pushState.state === "denied") {
    showRemoteStatus("Notifications are blocked. Allow them in your browser settings.");
    return;
  }
  if (pushState.state === "on") disablePush();
  else if (pushState.state === "off") enablePush();
}

async function enablePush() {
  pushState.busy = true;
  setPushToggleState(pushState.state);
  let subscription = null;
  try {
    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
      await refreshPushState();
      showRemoteStatus(permission === "denied"
        ? "Notifications are blocked. Allow them in your browser settings."
        : "Notifications were not enabled.");
      return;
    }
    const registration = await ensureServiceWorkerRegistration();
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(pushState.serverKey),
    });
    const serialized = subscription.toJSON();
    if (!serialized.endpoint || !serialized.keys?.p256dh || !serialized.keys?.auth) {
      throw new Error("The browser returned an invalid notification subscription.");
    }
    const response = await fetch(`${BASE_PATH}/api/push/subscription`, {
      method: "PUT",
      credentials: "same-origin",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        endpoint: serialized.endpoint,
        keys: {
          p256dh: serialized.keys.p256dh,
          auth: serialized.keys.auth,
        },
      }),
    });
    if (!response.ok) throw new Error("Zo could not save this notification subscription.");
    await refreshPushState();
    showRemoteStatus("Notifications enabled.");
  } catch (error) {
    if (subscription) await subscription.unsubscribe().catch(() => {});
    await refreshPushState().catch(() => setPushToggleState("off"));
    showRemoteStatus(error.message || "Could not enable notifications.");
  } finally {
    pushState.busy = false;
    setPushToggleState(pushState.state);
  }
}

async function disablePush() {
  pushState.busy = true;
  setPushToggleState(pushState.state);
  let unsubscribeFailed = false;
  let deleteFailed = false;
  try {
    try {
      const registration = await ensureServiceWorkerRegistration();
      const subscription = await registration.pushManager.getSubscription();
      if (subscription) await subscription.unsubscribe();
    } catch {
      unsubscribeFailed = true;
    }
    try {
      const response = await fetch(`${BASE_PATH}/api/push/subscription`, {
        method: "DELETE",
        credentials: "same-origin",
      });
      deleteFailed = !response.ok;
    } catch {
      deleteFailed = true;
    }
    await refreshPushState();
    if (deleteFailed) throw new Error("Zo could not remove this notification subscription.");
    if (unsubscribeFailed) throw new Error("The browser could not disable notifications.");
    showRemoteStatus("Notifications disabled.");
  } catch (error) {
    await refreshPushState().catch(() => {});
    showRemoteStatus(error.message || "Could not disable notifications.");
  } finally {
    pushState.busy = false;
    setPushToggleState(pushState.state);
  }
}

function enqueueMessage(message, epoch) {
  model.renderQueue.push({ epoch, message });
  if (model.renderScheduled) return;
  model.renderScheduled = true;
  requestAnimationFrame(flushMessages);
}

function flushMessages() {
  model.renderScheduled = false;
  const wasNearBottom = keyboard.suppressScroll ? keyboard.pinned : nearBottom();
  let transcriptChanged = false;
  for (const message of drainMessagesForEpoch(model.renderQueue, model.socketEpoch)) {
    switch (message.type) {
      case "snapshot":
        if (message.replace !== false) model.frames = [];
        if (model.sessionId && model.sessionId !== message.session?.id) model.approvals.clear();
        model.sessionId = message.session?.id || null;
        replacePendingToolApprovals(model.approvals, message.approvals || []);
        mergeFrames(message.frames || []);
        model.turn = message.turn;
        model.role = message.role;
        model.lastSeq = Math.max(0, Number(message.next_seq || 1) - 1);
        ui.sessionTitle.textContent = message.session?.title || "Zo session";
        transcriptChanged = true;
        break;
      case "frame":
        mergeFrames([message.frame]);
        transcriptChanged = true;
        break;
      case "turn_state":
        if (model.turn === "running" && message.turn === "idle") announce("Zo finished the turn.");
        model.turn = message.turn;
        break;
      case "control_state":
        model.role = message.role;
        if (model.approvals.size > 0) transcriptChanged = true;
        break;
      case "command_accepted":
        acceptCommand(message.command_id);
        break;
      case "command_rejected":
        rejectCommand(message.command_id, message.message);
        break;
      case "resync_required":
        model.frames = [];
        model.lastSeq = 0;
        transcriptChanged = true;
        send({ type: "hello", version: PROTOCOL_VERSION, last_seq: 0 });
        break;
      case "tool_approval_request":
        if (applyToolApprovalRequest(model.approvals, message.approval)) {
          transcriptChanged = true;
          announce(`Permission required for ${message.approval?.tool_name || "a tool"}.`);
        }
        break;
      case "tool_approval_resolved":
        if (applyToolApprovalResolution(model.approvals, message)) {
          trimResolvedApprovals();
          transcriptChanged = true;
          announce(message.source === "tui" ? "Permission resolved in the terminal." : "Permission resolved remotely.");
        }
        break;
      case "error":
        if (!message.recoverable) {
          returnToPairing(message.message || "This remote credential is no longer active. Run /remote qr and scan the current code.");
        } else {
          ui.composerHint.textContent = message.message;
        }
        break;
    }
  }
  if (transcriptChanged) {
    renderTranscript();
    if (wasNearBottom) scrollToLatest(false);
    else {
      model.newItems += 1;
      ui.newItemsLabel.textContent = `${model.newItems} new ${model.newItems === 1 ? "update" : "updates"}`;
      ui.newItems.hidden = false;
    }
  }
  updateWorkingIndicator();
  updateSessionControls();
}

function mergeFrames(frames) {
  const bySeq = new Map(model.frames.map((frame) => [Number(frame.seq), frame]));
  for (const frame of frames) {
    const seq = Number(frame.seq);
    if (!Number.isFinite(seq)) continue;
    bySeq.set(seq, frame);
    model.lastSeq = Math.max(model.lastSeq, seq);
  }
  model.frames = [...bySeq.values()].sort((a, b) => a.seq - b.seq).slice(-MAX_FRAMES);
}

function renderTranscript() {
  const items = materialize(model.frames).slice(-MAX_RENDERED_ITEMS);
  const fragment = document.createDocumentFragment();
  if (!items.length && model.approvals.size === 0) {
    const empty = document.createElement("div");
    empty.className = "transcript-empty";
    const glyph = document.createElement("span");
    glyph.className = "transcript-empty-glyph";
    glyph.setAttribute("aria-hidden", "true");
    glyph.textContent = "⬡";
    empty.append(glyph, document.createTextNode("Transcript will appear here"));
    fragment.append(empty);
  } else {
    for (const item of items) fragment.append(renderItem(item));
  }
  for (const approval of model.approvals.values()) {
    fragment.append(renderToolApprovalCard(approval));
  }
  ui.transcript.replaceChildren(fragment);
  updateWorkingIndicator();
}

function updateWorkingIndicator() {
  const existing = ui.transcript.querySelector(".working-indicator");
  if (model.turn !== "running") {
    existing?.remove();
    return;
  }
  if (existing) return;
  const indicator = document.createElement("div");
  indicator.className = "working-indicator";
  indicator.setAttribute("role", "status");
  const dot = document.createElement("span");
  dot.className = "working-dot";
  dot.setAttribute("aria-hidden", "true");
  indicator.append(dot, document.createTextNode("working…"));
  ui.transcript.append(indicator);
}

function materialize(frames) {
  const items = [];
  const textById = new Map();
  const reasoningById = new Map();
  const tools = new Map();
  for (const frame of frames) {
    const block = frame.block || {};
    const id = `${block.type}:${block.id ?? frame.seq}`;
    if (block.type === "text_delta") {
      let item = textById.get(id);
      if (!item) {
        item = { kind: "assistant", text: "" };
        textById.set(id, item);
        items.push(item);
      }
      item.text += block.text || "";
    } else if (block.type === "reasoning") {
      let item = reasoningById.get(id);
      if (!item) {
        item = { kind: "reasoning", text: "" };
        reasoningById.set(id, item);
        items.push(item);
      }
      item.text += block.text || "";
    } else if (block.type === "tool_call") {
      const key = block.tool_call_id || id;
      let item = tools.get(key);
      if (!item) {
        item = { kind: "tool", name: block.name || "Tool", summary: "", content: "", status: "pending" };
        tools.set(key, item);
        items.push(item);
      }
      item.name = block.name || item.name;
      item.summary = block.summary || item.summary;
      item.status = block.status || item.status;
    } else if (block.type === "tool_result") {
      const key = block.tool_call_id || id;
      let item = tools.get(key);
      if (!item) {
        item = { kind: "tool", name: "Tool result", summary: "", content: "", status: "completed" };
        tools.set(key, item);
        items.push(item);
      }
      item.content = block.content || "";
      item.status = block.is_error ? "error" : "completed";
    } else if (block.type === "system") {
      items.push({ kind: block.level === "user" ? "user" : "system", level: block.level, text: block.text || "" });
    } else if (block.type === "agent_result") {
      items.push({ kind: "tool", name: block.label || "Agent", summary: block.status || "completed", content: block.body || "", status: block.status || "completed" });
    } else if (block.type === "permission_prompt") {
      items.push({ kind: "system", level: "warn", text: `Permission required in the local terminal: ${block.tool_name || "tool"}\n${block.reasoning || ""}` });
    } else if (block.type === "user_question_prompt") {
      items.push({ kind: "system", level: "warn", text: `Zo is waiting for a local answer: ${block.question || "Question"}` });
    } else if (block.type === "usage") {
      items.push({ kind: "usage", text: `${block.ctx_tokens || 0} context tokens · ${block.output_tokens || 0} output` });
    } else if (block.type === "image") {
      items.push({ kind: "system", level: "info", text: `Image attachment · ${block.media_type || "image"} · ${block.byte_len || 0} bytes` });
    }
  }
  return items;
}

function renderItem(item) {
  if (item.kind === "tool" || item.kind === "reasoning") {
    const details = document.createElement("details");
    details.className = `event event-tool${item.kind === "reasoning" ? " event-reasoning" : ""}`;
    const summary = document.createElement("summary");
    const label = document.createElement("span");
    label.className = "tool-label";
    const name = document.createElement("span");
    name.className = "tool-name";
    name.textContent = item.kind === "reasoning" ? "Reasoning" : item.name;
    label.append(name);
    if (item.kind === "tool" && item.summary) {
      const description = document.createElement("span");
      description.className = "tool-summary";
      description.textContent = ` · ${item.summary}`;
      label.append(description);
    }
    summary.append(label);
    if (item.kind === "tool") {
      const status = document.createElement("span");
      status.className = "tool-status";
      status.textContent = item.status;
      summary.append(status);
    }
    const content = document.createElement("pre");
    content.textContent = item.text || item.content || "No output yet.";
    details.append(summary, content);
    return details;
  }
  const article = document.createElement("article");
  article.className = `event event-${item.kind}`;
  if (item.level) article.dataset.level = item.level;
  if (item.kind === "assistant") {
    article.append(renderMarkdown(item.text));
    return article;
  }
  const text = document.createElement("p");
  text.textContent = item.text;
  article.append(text);
  return article;
}

function renderToolApprovalCard(approval) {
  const card = document.createElement("article");
  card.className = `tool-approval-card${approval.status === "resolved" ? " is-resolved" : ""}`;
  card.dataset.requestId = approval.request_id;
  card.setAttribute("role", "region");

  const heading = document.createElement("div");
  heading.className = "tool-approval-heading";
  const title = document.createElement("h2");
  title.textContent = approval.status === "resolved" ? "Permission resolved" : "Permission required";
  const tool = document.createElement("span");
  tool.className = "tool-approval-tool";
  tool.textContent = approval.tool_name || "Tool";
  heading.append(title, tool);

  const summary = document.createElement("code");
  summary.className = "tool-approval-summary";
  summary.textContent = approval.input_summary || "Input details hidden";

  const hash = document.createElement("p");
  hash.className = "tool-approval-hash";
  const fullHash = String(approval.input_hash || "");
  hash.textContent = fullHash ? `Input hash ${fullHash.slice(0, 12)}…` : "Input hash unavailable";
  hash.title = fullHash;

  const actions = document.createElement("div");
  actions.className = "tool-approval-actions";
  const connected = model.socket?.readyState === WebSocket.OPEN;
  const canAnswer = approval.status !== "resolved"
    && model.role === "controller"
    && connected
    && model.pending.size === 0
    && !approval.answerPending;
  for (const choice of approval.choices || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `button approval-choice${String(choice.decision).startsWith("deny") ? " approval-deny" : ""}`;
    button.dataset.approvalId = approval.request_id;
    button.dataset.decision = choice.decision;
    button.textContent = choice.label;
    button.disabled = !canAnswer;
    if (approval.status === "resolved" && approval.decision === choice.decision) {
      button.classList.add("is-selected");
    }
    actions.append(button);
  }

  const status = document.createElement("p");
  status.className = "tool-approval-status";
  if (approval.status === "resolved") {
    const choice = (approval.choices || []).find((item) => item.decision === approval.decision);
    const surface = approval.source === "tui" ? "terminal" : "remote";
    status.textContent = `${choice?.label || approval.decision || "Resolved"} · answered from ${surface}`;
  } else if (approval.answerPending) {
    status.textContent = "Sending decision…";
  } else if (model.role !== "controller") {
    status.textContent = "Request control to answer.";
  } else if (!connected) {
    status.textContent = "Reconnect to answer; the terminal prompt is still active.";
  } else {
    status.textContent = "First answer wins between this device and the terminal.";
  }

  card.append(heading, summary, hash, actions, status);
  return card;
}

function handleToolApprovalClick(event) {
  const button = event.target.closest("button[data-approval-id][data-decision]");
  if (!button || !ui.transcript.contains(button) || button.disabled) return;
  submitToolApproval(button.dataset.approvalId, button.dataset.decision);
}

function submitToolApproval(requestId, decision) {
  const approval = model.approvals.get(requestId);
  if (
    !approval
    || approval.status === "resolved"
    || approval.answerPending
    || model.role !== "controller"
    || model.socket?.readyState !== WebSocket.OPEN
    || !(approval.choices || []).some((choice) => choice.decision === decision)
  ) return;
  const id = commandId();
  const wire = {
    type: "tool_approval_respond",
    command_id: id,
    request_id: requestId,
    decision,
  };
  approval.answerPending = true;
  model.pending.set(id, { kind: "approval", approvalId: requestId, wire, timeout: null });
  renderTranscript();
  send(wire);
  armPendingTimeout(id);
}

function trimResolvedApprovals() {
  const resolved = [...model.approvals]
    .filter(([, approval]) => approval.status === "resolved");
  for (const [id] of resolved.slice(0, -8)) model.approvals.delete(id);
}

function renderMarkdown(source) {
  const fragment = document.createDocumentFragment();
  for (const block of tokenizeMarkdown(source)) {
    if (block.type === "code_block") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.append(document.createTextNode(block.text));
      pre.append(code);
      fragment.append(pre);
      continue;
    }

    if (block.type === "list") {
      fragment.append(renderMarkdownList(block));
      continue;
    }

    const element = document.createElement(block.type === "heading"
      ? `h${block.level}`
      : block.type === "blockquote" ? "blockquote" : "p");
    appendInlineMarkdown(element, block.children);
    fragment.append(element);
  }
  return fragment;
}

function renderMarkdownList(block) {
  const list = document.createElement(block.ordered ? "ol" : "ul");
  for (const item of block.items) {
    const listItem = document.createElement("li");
    appendInlineMarkdown(listItem, item.children);
    for (const nested of item.lists) listItem.append(renderMarkdownList(nested));
    list.append(listItem);
  }
  return list;
}

function appendInlineMarkdown(parent, tokens) {
  for (const token of tokens) {
    if (token.type === "text") {
      parent.append(document.createTextNode(token.text));
      continue;
    }
    if (token.type === "link") {
      const anchor = document.createElement("a");
      anchor.href = token.href;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      anchor.append(document.createTextNode(token.text));
      parent.append(anchor);
      continue;
    }
    const element = document.createElement(token.type === "strong"
      ? "strong"
      : token.type === "emphasis" ? "em" : "code");
    if (token.children) appendInlineMarkdown(element, token.children);
    else element.append(document.createTextNode(token.text));
    parent.append(element);
  }
}

function submitPrompt(event) {
  event.preventDefault();
  const text = ui.prompt.value.trim();
  if (!text || model.pending.size > 0 || model.role !== "controller" || model.socket?.readyState !== WebSocket.OPEN) return;
  const mode = model.turn === "running" ? model.selectedMode : "new";
  const id = commandId();
  const wire = { type: "prompt_submit", command_id: id, text, mode };
  model.pending.set(id, { kind: "prompt", text, wire, timeout: null });
  ui.sendButton.disabled = true;
  ui.composerHint.textContent = mode === "queue" ? "Adding to the next turn…" : mode === "steer" ? "Steering the current turn…" : "Sending…";
  send(wire);
  armPendingTimeout(id);
}

function preservePromptFocus(event) {
  event.preventDefault();
}

// iOS WebKit cancels the synthesized click when touchstart is prevented, so a
// click-only submit path never fires on touch. Submit from touchend directly;
// preventDefault here both keeps the textarea focused and suppresses the
// synthetic click (no double submit).
function submitFromTouch(event) {
  if (ui.sendButton.disabled || !event.cancelable) return;
  const touch = event.changedTouches && event.changedTouches[0];
  if (touch) {
    const rect = ui.sendButton.getBoundingClientRect();
    if (
      touch.clientX < rect.left || touch.clientX > rect.right ||
      touch.clientY < rect.top || touch.clientY > rect.bottom
    ) {
      return;
    }
  }
  event.preventDefault();
  submitFromSendButton(event);
}

function submitFromSendButton(event) {
  event.preventDefault();
  ui.composer.requestSubmit();
  ui.prompt.focus({ preventScroll: true });
  scrollToLatest(false);
  keyboard.pinned = true;
  wakeKeyboardTracker();
}

function beginTranscriptDrag(event) {
  keyboard.touchY = document.activeElement === ui.prompt && event.touches.length === 1
    ? [event.touches[0].clientY]
    : [];
}

function continueTranscriptDrag(event) {
  if (!keyboard.touchY.length || event.touches.length !== 1) return;
  keyboard.touchY.push(event.touches[0].clientY);
  if (shouldDismissKeyboardFromTouch(keyboard.touchY, document.activeElement === ui.prompt)) {
    keyboard.touchY = [];
    ui.prompt.blur();
  }
}

function endTranscriptDrag() {
  keyboard.touchY = [];
}

function requestControl() {
  if (model.pending.size > 0 || model.socket?.readyState !== WebSocket.OPEN) return;
  const id = commandId();
  const wire = { type: "control_request", command_id: id };
  model.pending.set(id, { kind: "control", wire, timeout: null });
  send(wire);
  ui.connectionLabel.textContent = "Requesting control";
  updateSessionControls();
  armPendingTimeout(id);
}

function sendCancel() {
  if (model.pending.size > 0 || model.socket?.readyState !== WebSocket.OPEN) return;
  const id = commandId();
  const wire = { type: "turn_cancel", command_id: id };
  model.pending.set(id, { kind: "cancel", wire, timeout: null });
  send(wire);
  ui.cancelTurn.disabled = true;
  armPendingTimeout(id);
}

function acceptCommand(id) {
  const pending = model.pending.get(id);
  if (!pending) return;
  clearTimeout(pending.timeout);
  model.pending.delete(id);
  if (pending.kind === "prompt") {
    ui.prompt.value = "";
    sessionStorage.removeItem("zo.remote.draft");
    resizeComposer();
    if (document.activeElement === ui.prompt) {
      keyboard.pinned = true;
      scrollToLatest(false);
      wakeKeyboardTracker();
    }
  }
  if (pending.kind === "control") model.role = "controller";
  if (pending.kind === "approval") {
    const approval = model.approvals.get(pending.approvalId);
    if (approval) approval.answerPending = approval.status !== "resolved";
    renderTranscript();
  }
  ui.sendButton.disabled = false;
  ui.cancelTurn.disabled = false;
  updateSessionControls();
}

function rejectCommand(id, message) {
  const pending = model.pending.get(id);
  if (!pending) return;
  clearTimeout(pending.timeout);
  model.pending.delete(id);
  if (pending.kind === "approval") {
    const approval = model.approvals.get(pending.approvalId);
    if (approval) approval.answerPending = false;
    renderTranscript();
  }
  ui.sendButton.disabled = false;
  ui.cancelTurn.disabled = false;
  ui.composerHint.textContent = message || "Zo rejected the command.";
  announce(message || "Command rejected.");
  updateSessionControls();
}

function armPendingTimeout(id) {
  const pending = model.pending.get(id);
  if (!pending) return;
  clearTimeout(pending.timeout);
  pending.timeout = setTimeout(() => {
    const socket = model.socket;
    if (socket?.readyState !== WebSocket.OPEN) return;
    handleCommandAckTimeout(model.pending, id, () => {
      setConnection("connecting", "Reconnecting to confirm the pending command…");
      socket.close();
    });
  }, 8_000);
}

function releasePending(message) {
  for (const id of [...model.pending.keys()]) rejectCommand(id, message);
}

function send(message) {
  if (model.socket?.readyState === WebSocket.OPEN) model.socket.send(JSON.stringify(message));
}

function selectMode(mode) {
  model.selectedMode = mode === "steer" ? "steer" : "queue";
  for (const button of ui.modeOptions) {
    const active = button.dataset.mode === model.selectedMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  updateComposer();
}

function updateSessionControls() {
  const controller = model.role === "controller";
  const connected = model.socket?.readyState === WebSocket.OPEN;
  const commandPending = model.pending.size > 0;
  ui.roleBadge.textContent = controller ? "Controller" : "Observer";
  ui.roleBadge.classList.toggle("is-controller", controller);
  ui.requestControl.hidden = controller;
  ui.requestControl.disabled = !connected || commandPending;
  ui.observerPanel.hidden = controller;
  ui.observerControl.disabled = !connected || commandPending;
  ui.composer.hidden = !controller;
  ui.composer.setAttribute("aria-busy", String(commandPending));
  ui.cancelTurn.hidden = !(controller && model.turn === "running");
  ui.cancelTurn.disabled = !connected || commandPending;
  ui.modeSwitch.hidden = !(controller && model.turn === "running");
  ui.prompt.disabled = !connected;
  ui.sendButton.disabled = !connected || commandPending || !ui.prompt.value.trim();
  updateComposer();
}

function updateComposer() {
  const connected = model.socket?.readyState === WebSocket.OPEN;
  const commandPending = model.pending.size > 0;
  ui.prompt.placeholder = "Message Zo…";
  if (commandPending) {
    ui.composerHint.textContent = connected ? "Waiting for Zo to acknowledge…" : "Reconnecting to confirm the pending command…";
  } else if (model.turn === "running") {
    ui.composerHint.textContent = model.selectedMode === "steer" ? "Guide the current response immediately" : "Run after the current turn finishes";
  } else {
    ui.composerHint.textContent = "Send a new request";
  }
  ui.sendButton.disabled = model.role !== "controller" || !connected || commandPending || !ui.prompt.value.trim();
}

function setConnection(state, label) {
  clearTimeout(model.statusTimer);
  model.statusTimer = null;
  ui.connectionLabel.textContent = label;
  ui.connectionDot.classList.toggle("is-online", state === "online");
  ui.connectionDot.classList.toggle("is-offline", state === "offline");
  ui.connectionBanner.hidden = state === "online";
  if (model.approvals.size > 0) renderTranscript();
  ui.connectionMessage.textContent = label;
  updateSessionControls();
}

function showRemoteStatus(message) {
  clearTimeout(model.statusTimer);
  ui.connectionMessage.textContent = message;
  ui.connectionBanner.hidden = false;
  announce(message);
  model.statusTimer = setTimeout(() => {
    model.statusTimer = null;
    ui.connectionMessage.textContent = ui.connectionLabel.textContent;
    ui.connectionBanner.hidden = model.socket?.readyState === WebSocket.OPEN;
  }, 4_000);
}

function returnToPairing(message) {
  const socket = model.socket;
  model.socketEpoch += 1;
  model.socket = null;
  model.everConnected = false;
  model.reconnectAttempt = 0;
  clearTimeout(model.reconnectTimer);
  model.reconnectTimer = null;
  clearTimeout(model.pairingTimer);
  model.pairingTimer = null;
  model.pairingId = null;
  model.pairingDeadline = 0;
  model.renderQueue.splice(0);
  model.approvals.clear();
  model.sessionId = null;
  pausePendingCommands(model.pending);
  releasePending(message);
  socket?.close();
  ui.approvalCard.hidden = true;
  ui.pairForm.hidden = false;
  ui.pairButton.disabled = false;
  showPairView();
  showPairError(message);
}

function showPairView() {
  ui.remoteView.hidden = true;
  ui.pairView.hidden = false;
  updateInstallAffordances();
}

function showRemoteView() {
  ui.pairView.hidden = true;
  ui.remoteView.hidden = false;
  renderTranscript();
  loadPushConfigOnce();
  requestAnimationFrame(syncComposerHeight);
}

function showPairError(message) {
  ui.pairError.textContent = message;
  ui.pairError.hidden = false;
  announce(message);
}

function nearBottom() {
  return ui.transcript.scrollHeight - ui.transcript.scrollTop - ui.transcript.clientHeight < 96;
}

function scrollToLatest(smooth = true) {
  ui.transcript.scrollTo({ top: ui.transcript.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  model.newItems = 0;
  ui.newItems.hidden = true;
}

function resizeComposer() {
  ui.prompt.style.height = "auto";
  ui.prompt.style.height = `${Math.min(140, ui.prompt.scrollHeight)}px`;
}

function syncComposerHeight() {
  ui.remoteView.style.setProperty("--composer-height", `${ui.composerWrap.offsetHeight}px`);
}

function scheduleViewportSync() {
  if (window.visualViewport && (document.activeElement === ui.prompt || keyboard.settling)) {
    if (!keyboard.settling) wakeKeyboardTracker();
    return;
  }
  cancelAnimationFrame(viewportFrame);
  viewportFrame = requestAnimationFrame(syncViewport);
}

function syncViewport() {
  viewportFrame = 0;
  const viewport = window.visualViewport;
  const height = Math.round(viewport?.height || window.innerHeight);
  const offsetTop = Math.round(viewport?.offsetTop || 0);
  document.documentElement.style.setProperty("--app-height", `${height}px`);
  document.documentElement.style.setProperty("--app-top", `${offsetTop}px`);
  if (viewport && document.activeElement !== ui.prompt && !keyboard.settling) {
    keyboard.layoutBottom = height + offsetTop;
  }
}

function handlePromptFocus() {
  if (!window.visualViewport) {
    scheduleViewportSync();
    setTimeout(scheduleViewportSync, 250);
    return;
  }

  cancelAnimationFrame(viewportFrame);
  viewportFrame = 0;
  cancelAnimationFrame(keyboard.frame);
  keyboard.frame = 0;
  keyboard.settling = false;
  keyboard.suppressScroll = true;
  keyboard.pinned = nearBottom();
  keyboard.lastInset = null;
  keyboard.stableSince = 0;
  const visibleBottom = window.visualViewport.height + window.visualViewport.offsetTop;
  keyboard.baselineBottom = Math.max(keyboard.layoutBottom, visibleBottom);
  ui.remoteView.classList.remove("kb-settling");
  wakeKeyboardTracker(true);
}

function handlePromptBlur() {
  if (!window.visualViewport) return;

  cancelAnimationFrame(keyboard.frame);
  keyboard.frame = 0;
  keyboard.settling = true;
  keyboard.suppressScroll = true;
  ui.remoteView.classList.remove("kb-tracking");
  if (!keyboard.usesVirtualKeyboard) ui.remoteView.classList.add("kb-settling");
  keyboard.frame = requestAnimationFrame(beginKeyboardSettle);
}

function wakeKeyboardTracker(resetStability = false) {
  if (!window.visualViewport || document.activeElement !== ui.prompt || keyboard.settling) return;
  if (resetStability) keyboard.stableSince = performance.now();
  if (keyboard.frame) return;
  keyboard.suppressScroll = true;
  ui.remoteView.classList.remove("kb-settling");
  ui.remoteView.classList.add("kb-tracking");
  if (!keyboard.stableSince) keyboard.stableSince = performance.now();
  keyboard.frame = requestAnimationFrame(trackKeyboard);
}

function trackKeyboard(timestamp) {
  keyboard.frame = 0;
  if (document.activeElement !== ui.prompt || keyboard.settling) return;

  window.scrollTo(0, 0);
  const visibleBottom = window.visualViewport.height + window.visualViewport.offsetTop;
  const inset = Math.max(0, Math.round(keyboard.baselineBottom - visibleBottom));
  if (keyboard.lastInset === null || Math.abs(inset - keyboard.lastInset) > 0.5) {
    keyboard.lastInset = inset;
    keyboard.stableSince = timestamp;
  }
  if (!keyboard.usesVirtualKeyboard) {
    ui.remoteView.style.setProperty("--kb-inset", `${inset}px`);
  }
  if (keyboard.pinned) scrollToLatest(false);

  if (timestamp - keyboard.stableSince < KEYBOARD_STABLE_MS) {
    keyboard.frame = requestAnimationFrame(trackKeyboard);
    return;
  }
  keyboard.suppressScroll = false;
  ui.remoteView.classList.remove("kb-tracking");
}

function beginKeyboardSettle(timestamp) {
  keyboard.frame = 0;
  if (document.activeElement === ui.prompt) {
    handlePromptFocus();
    return;
  }
  window.scrollTo(0, 0);
  if (!keyboard.usesVirtualKeyboard) ui.remoteView.style.setProperty("--kb-inset", "0px");
  if (keyboard.pinned) scrollToLatest(false);
  keyboard.stableSince = timestamp;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    finishKeyboardSettle();
    return;
  }
  keyboard.frame = requestAnimationFrame(trackKeyboardSettle);
}

function trackKeyboardSettle(timestamp) {
  keyboard.frame = 0;
  window.scrollTo(0, 0);
  if (keyboard.pinned) scrollToLatest(false);
  if (timestamp - keyboard.stableSince < KEYBOARD_SETTLE_MS) {
    keyboard.frame = requestAnimationFrame(trackKeyboardSettle);
    return;
  }
  finishKeyboardSettle();
}

function finishKeyboardSettle() {
  keyboard.frame = 0;
  keyboard.pinned = false;
  keyboard.settling = false;
  keyboard.suppressScroll = false;
  ui.remoteView.classList.remove("kb-settling", "kb-tracking");
  scheduleViewportSync();
}

function commandId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function announce(message) {
  ui.announcer.textContent = "";
  requestAnimationFrame(() => { ui.announcer.textContent = message; });
}
