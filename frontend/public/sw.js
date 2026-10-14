// Раньше этот воркер кэшировал оболочку. Файл остаётся, чтобы уже установленные
// клиенты получили обновление, сбросили кэш и сняли регистрацию. Новым пользователям
// он не ставится: для PWA достаточно манифеста.
self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
      await self.registration.unregister();
    })(),
  );
});
