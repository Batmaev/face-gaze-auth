import { useEffect, useRef } from "react";
import type { User } from "../storage";

interface Props {
  user: User;
  stream: MediaStream | null;
  cameraError: string | null;
  onRetryCamera: () => void;
  onSettings: () => void;
  onNext: () => void;
}

export default function InstructionScreen({ user, stream, cameraError, onRetryCamera, onSettings, onNext }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = stream;
  }, [stream]);

  return (
    <div className="screen">
      <div className="topbar end">
        <button className="icon-button" onClick={onSettings} aria-label="Настройки">
          <GearIcon />
        </button>
      </div>
      <div className="content center">
        <div className="preview">
          {stream ? (
            <video ref={videoRef} autoPlay playsInline muted />
          ) : (
            <span className="muted small">{cameraError ? "Нет доступа к камере" : "Запрашиваем камеру…"}</span>
          )}
        </div>
        <h2>Вам предстоит следить за движущейся точкой</h2>
        {/* <p className="muted">
          Держите телефон перед лицом, чтобы лицо было хорошо освещено и целиком в кадре. Двигайте только глазами.
        </p> */}
        {cameraError && (
          <>
            <p className="error">{cameraError}</p>
            <button className="secondary" onClick={onRetryCamera}>
              Разрешить доступ к камере
            </button>
          </>
        )}
      </div>
      <div className="footer">
        <button className="primary" onClick={onNext} disabled={!stream}>
          Далее
        </button>
        <div className="user-meta">
          {user.name} · {user.id}
        </div>
      </div>
    </div>
  );
}

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}
