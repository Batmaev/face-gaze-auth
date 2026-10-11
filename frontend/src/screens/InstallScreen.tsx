import { useEffect, useState } from "react";
import { getInstallPrompt, isIOS, onInstallPromptChange } from "../pwa";

export default function InstallScreen({ onSkip }: { onSkip: () => void }) {
  const [prompt, setPrompt] = useState(getInstallPrompt);
  const [installed, setInstalled] = useState(false);

  useEffect(() => onInstallPromptChange(() => setPrompt(getInstallPrompt())), []);

  const install = async () => {
    if (!prompt) return;
    await prompt.prompt();
    const { outcome } = await prompt.userChoice;
    if (outcome === "accepted") setInstalled(true);
  };

  return (
    <div className="screen">
      <div className="content center">
        <img src="/icon.svg" alt="" className="app-icon" />
        <h2>Установите приложение</h2>
        {installed ? (
          <p className="muted">Готово! Откройте Liveness с главного экрана.</p>
        ) : prompt ? (
          <>
            <p className="muted">Так проверка пройдёт на весь экран, без панелей браузера.</p>
            <button className="primary" onClick={install}>
              Установить
            </button>
          </>
        ) : isIOS() ? (
          <ol className="steps">
            <li>
              Нажмите «Поделиться» <ShareIcon /> внизу экрана Safari
            </li>
            <li>Выберите «На экран „Домой“»</li>
            <li>Откройте Liveness с главного экрана</li>
          </ol>
        ) : (
          <ol className="steps">
            <li>Откройте меню браузера ⋮</li>
            <li>Выберите «Установить приложение» или «Добавить на главный экран»</li>
            <li>Откройте Liveness с главного экрана</li>
          </ol>
        )}
      </div>
      <div className="footer">
        <button className="link" onClick={onSkip}>
          Пропустить
        </button>
      </div>
    </div>
  );
}

function ShareIcon() {
  return (
    <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M12 3v12M7 8l5-5 5 5" />
      <path d="M5 12v7a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-7" />
    </svg>
  );
}
