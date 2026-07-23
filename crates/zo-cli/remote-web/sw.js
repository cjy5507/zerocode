const CACHE = "zo-remote-shell-v7";
const SHELL = [
  "./",
  "./styles.css",
  "./app.js",
  "./remote-state.js",
  "./manifest.webmanifest",
  "./icon.svg",
  "./apple-touch-icon.png",
];
const API_PATH = new URL("./api/", self.registration.scope).pathname;
const WS_PATH = new URL("./ws", self.registration.scope).pathname;

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin || url.pathname.startsWith(API_PATH) || url.pathname === WS_PATH) {
    return;
  }
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});

self.addEventListener("push", (event) => {
  let payload = null;
  try {
    payload = event.data?.json() || null;
  } catch {
    payload = null;
  }
  const reason = payload?.reason === "approval" || payload?.reason === "turn_idle"
    ? payload.reason
    : "update";
  // Keep this tiny copy map in sync with notificationContent in remote-state.js,
  // which is the tested source of truth. This service worker must stay classic.
  const content = reason === "approval"
    ? { title: "Zo Remote", body: "Tool approval waiting" }
    : reason === "turn_idle"
      ? { title: "Zo Remote", body: "Turn finished" }
      : { title: "Zo Remote", body: "Session update available" };
  event.waitUntil(self.registration.showNotification(content.title, {
    body: content.body,
    tag: `zo-remote-${reason}`,
    icon: "./apple-touch-icon.png",
  }));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((windows) => {
    const client = windows.find((candidate) => candidate.url.startsWith(self.registration.scope));
    return client ? client.focus() : self.clients.openWindow(self.registration.scope);
  }));
});
