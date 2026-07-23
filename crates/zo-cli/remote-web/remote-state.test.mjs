import assert from "node:assert/strict";
import test from "node:test";

import {
  SESSION_REVOKED_CLOSE_CODE,
  applyToolApprovalRequest,
  applyToolApprovalResolution,
  basePath,
  drainMessagesForEpoch,
  handleCommandAckTimeout,
  isSessionRevokedClose,
  notificationContent,
  pairingPollDeadline,
  pausePendingCommands,
  pushSupportState,
  replayPendingCommands,
  replacePendingToolApprovals,
  shouldDismissKeyboardFromTouch,
  shouldRetryPairing,
  shouldShowInstallHint,
  socketAttemptIsCurrent,
  tokenizeInlineMarkdown,
  tokenizeMarkdown,
  urlBase64ToUint8Array,
} from "./remote-state.js";

test("base path follows the mounted document path", () => {
  assert.equal(basePath("/"), "");
  assert.equal(basePath("/s8789/"), "/s8789");
  assert.equal(basePath("/s8790/"), "/s8790");
  assert.equal(basePath("/s8790"), "/s8790");
});

test("URL-safe base64 public keys decode to bytes", () => {
  assert.deepEqual([...urlBase64ToUint8Array("AQIDBA")], [1, 2, 3, 4]);
  assert.deepEqual([...urlBase64ToUint8Array("-_8")], [251, 255]);
  assert.deepEqual([...urlBase64ToUint8Array("AQIDBA==")], [1, 2, 3, 4]);
});

test("push support requires secure browser APIs and standalone mode on iOS", () => {
  const supported = {
    isSecure: true,
    hasServiceWorker: true,
    hasPushManager: true,
    hasNotification: true,
    isStandalone: false,
    isIOS: false,
  };
  assert.equal(pushSupportState(supported), "ready");
  assert.equal(pushSupportState({ ...supported, isIOS: true }), "needs_install");
  assert.equal(pushSupportState({ ...supported, isIOS: true, isStandalone: true }), "ready");
  for (const missing of ["isSecure", "hasServiceWorker", "hasPushManager", "hasNotification"]) {
    assert.equal(pushSupportState({ ...supported, [missing]: false }), "unsupported");
  }
});

test("the iOS install hint is shown only before standalone install and dismissal", () => {
  assert.equal(shouldShowInstallHint({ isIOS: true, isStandalone: false, dismissed: false }), true);
  assert.equal(shouldShowInstallHint({ isIOS: true, isStandalone: true, dismissed: false }), false);
  assert.equal(shouldShowInstallHint({ isIOS: true, isStandalone: false, dismissed: true }), false);
  assert.equal(shouldShowInstallHint({ isIOS: false, isStandalone: false, dismissed: false }), false);
});

test("push reasons map to generic notification copy", () => {
  assert.deepEqual(notificationContent("approval"), {
    title: "Zo Remote",
    body: "Tool approval waiting",
  });
  assert.deepEqual(notificationContent("turn_idle"), {
    title: "Zo Remote",
    body: "Turn finished",
  });
  assert.deepEqual(notificationContent("unexpected"), {
    title: "Zo Remote",
    body: "Session update available",
  });
  assert.deepEqual(notificationContent(), {
    title: "Zo Remote",
    body: "Session update available",
  });
});

test("keyboard drag dismisses only after a deliberate downward move", () => {
  assert.equal(shouldDismissKeyboardFromTouch([100, 110, 121, 129], true), true);
  assert.equal(shouldDismissKeyboardFromTouch([100, 110, 121, 128], true), false);
  assert.equal(shouldDismissKeyboardFromTouch([100, 116, 129], false), false);
});

test("keyboard drag tolerates small direction jitter", () => {
  assert.equal(shouldDismissKeyboardFromTouch([100, 112, 109, 121, 130], true), true);
  assert.equal(shouldDismissKeyboardFromTouch([100, 104, 99, 105, 101, 106], true), false);
});

test("upward keyboard drags never dismiss", () => {
  assert.equal(shouldDismissKeyboardFromTouch([150, 138, 143, 126, 121], true), false);
  assert.equal(shouldDismissKeyboardFromTouch([150, 121], true), false);
});

test("pairing polling retries only while the same request is live", () => {
  assert.equal(shouldRetryPairing("pair-1", 2_000, 1_000), true);
  assert.equal(shouldRetryPairing("pair-1", 2_000, 2_000), false);
  assert.equal(shouldRetryPairing(null, 2_000, 1_000), false);
});

test("pairing polling spans the offer lifetime and approved-result grace", () => {
  const startedAt = 1_000;
  const offerDeadline = startedAt + 90_000;
  const pollingDeadline = pairingPollDeadline(startedAt, 210);

  assert.equal(shouldRetryPairing("pair-1", pollingDeadline, offerDeadline + 1), true);
  assert.equal(shouldRetryPairing("pair-1", pollingDeadline, pollingDeadline), false);
});

test("pending commands pause and replay with the original command id and payload", () => {
  const wire = {
    type: "prompt_submit",
    command_id: "command-1",
    text: "continue",
    mode: "queue",
  };
  const pending = new Map([
    ["command-1", { kind: "prompt", wire, timeout: 42 }],
  ]);
  const cleared = [];
  pausePendingCommands(pending, (timer) => cleared.push(timer));
  assert.deepEqual(cleared, [42]);
  assert.equal(pending.get("command-1").timeout, null);

  const sent = [];
  const armed = [];
  replayPendingCommands(pending, (message) => sent.push(message), (id) => armed.push(id));
  assert.equal(sent[0], wire);
  assert.equal(sent[0].command_id, "command-1");
  assert.deepEqual(armed, ["command-1"]);
});

test("ack timeout preserves the uncertain command for reconnect replay", () => {
  const wire = {
    type: "prompt_submit",
    command_id: "command-1",
    text: "continue",
    mode: "queue",
  };
  const command = { kind: "prompt", wire, timeout: 42 };
  const pending = new Map([["command-1", command]]);
  let reconnects = 0;

  assert.equal(handleCommandAckTimeout(pending, "command-1", () => { reconnects += 1; }), true);
  assert.equal(reconnects, 1);
  assert.equal(pending.size, 1);
  assert.equal(pending.get("command-1"), command);
  assert.equal(command.timeout, null);

  const sent = [];
  replayPendingCommands(pending, (message) => sent.push(message), () => {});
  assert.equal(sent[0], wire);
  assert.equal(sent[0].command_id, "command-1");
});

test("session-revoked close code is terminal", () => {
  assert.equal(SESSION_REVOKED_CLOSE_CODE, 4001);
  assert.equal(isSessionRevokedClose(SESSION_REVOKED_CLOSE_CODE), true);
  assert.equal(isSessionRevokedClose(1000), false);
  assert.equal(isSessionRevokedClose(1006), false);
});

test("only the current socket attempt can mutate connection state", () => {
  assert.equal(socketAttemptIsCurrent(7, 7), true);
  assert.equal(socketAttemptIsCurrent(8, 7), false);
});

test("queued messages from stale socket attempts are discarded", () => {
  const queue = [
    { epoch: 6, message: { type: "frame", seq: 1 } },
    { epoch: 7, message: { type: "turn_state", turn: "running" } },
    { epoch: 6, message: { type: "frame", seq: 2 } },
  ];
  assert.deepEqual(drainMessagesForEpoch(queue, 7), [
    { type: "turn_state", turn: "running" },
  ]);
  assert.deepEqual(queue, []);
});

test("tool approval state is first-writer-wins and snapshot-replay safe", () => {
  const approvals = new Map();
  const request = {
    request_id: "approval-1",
    tool_name: "Bash",
    input_summary: "cargo test",
    input_hash: "abc123",
    choices: [{ label: "Allow once", decision: "allow_once" }],
  };
  assert.equal(applyToolApprovalRequest(approvals, request), true);
  assert.equal(applyToolApprovalResolution(approvals, {
    request_id: "approval-1",
    decision: "allow_once",
    source: "remote",
  }), true);
  assert.equal(applyToolApprovalResolution(approvals, {
    request_id: "approval-1",
    decision: "deny",
    source: "tui",
  }), false);
  assert.equal(approvals.get("approval-1").decision, "allow_once");
  assert.equal(approvals.get("approval-1").source, "remote");

  applyToolApprovalRequest(approvals, { ...request, request_id: "stale-pending" });
  replacePendingToolApprovals(approvals, [{ ...request, request_id: "approval-2" }]);
  assert.equal(approvals.has("stale-pending"), false);
  assert.equal(approvals.get("approval-1").status, "resolved");
  assert.equal(approvals.get("approval-2").status, "pending");
});

test("markdown tokenizes supported block and inline structure", () => {
  const blocks = tokenizeMarkdown([
    "# Heading",
    "",
    "A **bold** and *quiet* [link](https://example.com/path).",
    "",
    "> quoted `code`",
    "",
    "- first",
    "  - nested",
    "- second",
    "",
    "1. one",
    "2. two",
    "",
    "```js",
    "const value = 1;",
    "```",
  ].join("\n"));

  assert.deepEqual(blocks.map((block) => block.type), [
    "heading", "paragraph", "blockquote", "list", "list", "code_block",
  ]);
  assert.deepEqual(blocks[1].children.map((token) => token.type), [
    "text", "strong", "text", "emphasis", "text", "link", "text",
  ]);
  assert.equal(blocks[3].items[0].lists[0].items[0].children[0].text, "nested");
  assert.equal(blocks[4].ordered, true);
  assert.equal(blocks[5].language, "js");
  assert.equal(blocks[5].text, "const value = 1;");
});

test("markdown leaves HTML payloads inert as text tokens", () => {
  const payload = '<img src=x onerror="globalThis.pwned=true">';
  assert.deepEqual(tokenizeInlineMarkdown(payload), [{ type: "text", text: payload }]);
  assert.deepEqual(tokenizeMarkdown(payload), [{
    type: "paragraph",
    children: [{ type: "text", text: payload }],
  }]);
});

test("markdown rejects non-http links without creating link tokens", () => {
  const source = "[x](javascript:alert(1)) and [local](/settings)";
  const tokens = tokenizeInlineMarkdown(source);
  assert.equal(tokens.some((token) => token.type === "link"), false);
  assert.equal(tokens.map((token) => token.text || "").join(""), source);
});

test("markdown permits only http and https links", () => {
  const tokens = tokenizeInlineMarkdown("[web](http://example.com) [safe](https://example.com)");
  assert.deepEqual(tokens.filter((token) => token.type === "link").map((token) => token.href), [
    "http://example.com",
    "https://example.com",
  ]);
});
