export const SESSION_REVOKED_CLOSE_CODE = 4001;

export function basePath(pathname = globalThis.location?.pathname || "/") {
  const path = String(pathname || "/");
  if (path === "/") return "";
  return path.endsWith("/") ? path.slice(0, -1) : path;
}

export function socketAttemptIsCurrent(currentEpoch, attemptEpoch) {
  return currentEpoch === attemptEpoch;
}

export function isSessionRevokedClose(code) {
  return code === SESSION_REVOKED_CLOSE_CODE;
}

export function drainMessagesForEpoch(queue, currentEpoch) {
  const messages = [];
  for (const entry of queue.splice(0)) {
    if (entry.epoch === currentEpoch) messages.push(entry.message);
  }
  return messages;
}

export function applyToolApprovalRequest(approvals, approval) {
  const id = String(approval?.request_id || "");
  if (!id) return false;
  const current = approvals.get(id);
  if (current?.status === "resolved") return false;
  approvals.set(id, {
    ...approval,
    request_id: id,
    choices: Array.isArray(approval.choices) ? approval.choices : [],
    status: "pending",
    answerPending: current?.answerPending || false,
  });
  return true;
}

export function applyToolApprovalResolution(approvals, resolution) {
  const id = String(resolution?.request_id || "");
  if (!id) return false;
  const current = approvals.get(id);
  if (current?.status === "resolved") return false;
  approvals.set(id, {
    ...(current || { request_id: id, choices: [] }),
    status: "resolved",
    decision: resolution.decision,
    source: resolution.source,
    answerPending: false,
  });
  return true;
}

export function replacePendingToolApprovals(approvals, incoming) {
  for (const [id, approval] of approvals) {
    if (approval.status !== "resolved") approvals.delete(id);
  }
  for (const approval of incoming || []) applyToolApprovalRequest(approvals, approval);
}

export function handleCommandAckTimeout(pending, id, reconnect) {
  const command = pending.get(id);
  if (!command) return false;
  command.timeout = null;
  reconnect();
  return true;
}

export function shouldRetryPairing(pairingId, pairingDeadline, now = Date.now()) {
  return Boolean(pairingId) && Number.isFinite(pairingDeadline) && now < pairingDeadline;
}

export function pairingPollDeadline(startedAt, pollExpiresInSeconds) {
  const seconds = Number(pollExpiresInSeconds);
  return startedAt + (Number.isFinite(seconds) && seconds > 0 ? seconds : 210) * 1_000;
}

export function replayPendingCommands(pending, send, armTimeout) {
  for (const [id, command] of pending) {
    send(command.wire);
    armTimeout(id);
  }
}

export function pausePendingCommands(pending, clearTimer = clearTimeout) {
  for (const command of pending.values()) {
    clearTimer(command.timeout);
    command.timeout = null;
  }
}

export function shouldDismissKeyboardFromTouch(yPositions, hasFocus, threshold = 28) {
  if (!hasFocus || !Array.isArray(yPositions) || yPositions.length < 2) return false;
  const start = Number(yPositions[0]);
  if (!Number.isFinite(start)) return false;
  return yPositions.slice(1).some((position) => {
    const y = Number(position);
    return Number.isFinite(y) && y - start > threshold;
  });
}

export function urlBase64ToUint8Array(value) {
  const encoded = String(value || "").trim();
  const padding = "=".repeat((4 - (encoded.length % 4)) % 4);
  const base64 = `${encoded}${padding}`.replace(/-/g, "+").replace(/_/g, "/");
  const decoded = globalThis.atob(base64);
  return Uint8Array.from(decoded, (character) => character.charCodeAt(0));
}

export function pushSupportState({
  isSecure,
  hasServiceWorker,
  hasPushManager,
  hasNotification,
  isStandalone,
  isIOS,
} = {}) {
  if (!isSecure || !hasServiceWorker || !hasPushManager || !hasNotification) return "unsupported";
  if (isIOS && !isStandalone) return "needs_install";
  return "ready";
}

export function shouldShowInstallHint({ isIOS, isStandalone, dismissed } = {}) {
  return Boolean(isIOS && !isStandalone && !dismissed);
}

export function notificationContent(reason) {
  if (reason === "approval") {
    return { title: "Zo Remote", body: "Tool approval waiting" };
  }
  if (reason === "turn_idle") {
    return { title: "Zo Remote", body: "Turn finished" };
  }
  return { title: "Zo Remote", body: "Session update available" };
}

function pushText(tokens, text) {
  if (!text) return;
  const previous = tokens[tokens.length - 1];
  if (previous?.type === "text") previous.text += text;
  else tokens.push({ type: "text", text });
}

function safeHttpUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

export function tokenizeInlineMarkdown(text) {
  const tokens = [];
  let index = 0;
  while (index < text.length) {
    if (text[index] === "`") {
      const end = text.indexOf("`", index + 1);
      if (end > index + 1) {
        tokens.push({ type: "code", text: text.slice(index + 1, end) });
        index = end + 1;
        continue;
      }
    }

    if (text[index] === "[") {
      const labelEnd = text.indexOf("]", index + 1);
      const urlStart = labelEnd + 1;
      if (labelEnd > index + 1 && text[urlStart] === "(") {
        const urlEnd = text.indexOf(")", urlStart + 1);
        if (urlEnd > urlStart + 1) {
          const source = text.slice(index, urlEnd + 1);
          const href = text.slice(urlStart + 1, urlEnd).trim();
          if (safeHttpUrl(href)) {
            tokens.push({ type: "link", text: text.slice(index + 1, labelEnd), href });
          } else {
            pushText(tokens, source);
          }
          index = urlEnd + 1;
          continue;
        }
      }
    }

    if (text.startsWith("**", index)) {
      const end = text.indexOf("**", index + 2);
      if (end > index + 2) {
        tokens.push({
          type: "strong",
          children: tokenizeInlineMarkdown(text.slice(index + 2, end)),
        });
        index = end + 2;
        continue;
      }
    }

    if (text[index] === "*") {
      const end = text.indexOf("*", index + 1);
      if (end > index + 1) {
        tokens.push({
          type: "emphasis",
          children: tokenizeInlineMarkdown(text.slice(index + 1, end)),
        });
        index = end + 1;
        continue;
      }
    }

    pushText(tokens, text[index]);
    index += 1;
  }
  return tokens;
}

function listMatch(line) {
  const match = /^(\s*)([-+*]|\d+\.)\s+(.*)$/.exec(line);
  if (!match) return null;
  return {
    indent: match[1].length,
    ordered: /\d/.test(match[2][0]),
    text: match[3],
  };
}

function startsBlock(line) {
  return /^\s*$/.test(line)
    || /^ {0,3}```/.test(line)
    || /^ {0,3}#{1,3}\s+/.test(line)
    || /^ {0,3}>/.test(line)
    || Boolean(listMatch(line));
}

function tokenizeList(lines, start) {
  const first = listMatch(lines[start]);
  const block = { type: "list", ordered: first.ordered, items: [] };
  let index = start;
  while (index < lines.length) {
    const match = listMatch(lines[index]);
    if (!match || match.indent < first.indent) break;
    if (match.indent === first.indent) {
      if (match.ordered !== first.ordered) break;
      block.items.push({ children: tokenizeInlineMarkdown(match.text), lists: [] });
      index += 1;
      continue;
    }
    const item = block.items[block.items.length - 1];
    if (!item) break;
    const nested = tokenizeList(lines, index);
    item.lists.push(nested.block);
    index = nested.next;
  }
  return { block, next: index };
}

export function tokenizeMarkdown(source) {
  const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = /^ {0,3}```\s*([^\s`]*)\s*$/.exec(line);
    if (fence) {
      const code = [];
      index += 1;
      while (index < lines.length && !/^ {0,3}```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push({ type: "code_block", language: fence[1], text: code.join("\n") });
      continue;
    }

    const heading = /^ {0,3}(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1].length,
        children: tokenizeInlineMarkdown(heading[2]),
      });
      index += 1;
      continue;
    }

    if (/^ {0,3}>/.test(line)) {
      const quote = [];
      while (index < lines.length) {
        const match = /^ {0,3}>\s?(.*)$/.exec(lines[index]);
        if (!match) break;
        quote.push(match[1]);
        index += 1;
      }
      blocks.push({ type: "blockquote", children: tokenizeInlineMarkdown(quote.join("\n")) });
      continue;
    }

    if (listMatch(line)) {
      const list = tokenizeList(lines, index);
      blocks.push(list.block);
      index = list.next;
      continue;
    }

    const paragraph = [line];
    index += 1;
    while (index < lines.length && !startsBlock(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push({ type: "paragraph", children: tokenizeInlineMarkdown(paragraph.join("\n")) });
  }
  return blocks;
}
