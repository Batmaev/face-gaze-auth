export interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

let deferredPrompt: BeforeInstallPromptEvent | null = null;
const listeners = new Set<() => void>();

// Событие приходит до монтирования React, поэтому ловим его сразу при загрузке модуля.
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredPrompt = event as BeforeInstallPromptEvent;
  listeners.forEach((listener) => listener());
});

window.addEventListener("appinstalled", () => {
  deferredPrompt = null;
  listeners.forEach((listener) => listener());
});

export function getInstallPrompt() {
  return deferredPrompt;
}

export function onInstallPromptChange(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function isStandalone(): boolean {
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    window.matchMedia("(display-mode: fullscreen)").matches ||
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

export function isIOS(): boolean {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

export function isMobile(): boolean {
  return isIOS() || /Android|Mobi/i.test(navigator.userAgent);
}

/** Снимает service worker и Cache Storage у уже установленных клиентов. Новый воркер не регистрируется. */
export function clearServiceWorkers() {
  if ("serviceWorker" in navigator) {
    void navigator.serviceWorker.getRegistrations().then((registrations) => {
      for (const registration of registrations) void registration.unregister();
    });
  }
  if ("caches" in window) {
    void caches.keys().then((keys) => Promise.all(keys.map((key) => caches.delete(key))));
  }
}
