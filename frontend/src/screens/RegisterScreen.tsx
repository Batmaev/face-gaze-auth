import { useState, type FormEvent } from "react";
import { register } from "../api";
import type { User } from "../storage";

interface Props {
  onBack?: () => void;
  onRegistered: (user: User) => void;
  onShowUsers: () => void;
}

export default function RegisterScreen({ onBack, onRegistered, onShowUsers }: Props) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      const id = await register(trimmed);
      onRegistered({ id, name: trimmed });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <form className="screen" onSubmit={submit}>
      {onBack && (
        <div className="topbar">
          <button type="button" className="icon-button" onClick={onBack} aria-label="Назад">
            ←
          </button>
        </div>
      )}
      <div className="content center">
        <h2>Как вас зовут?</h2>
        <input
          className="text-input"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Имя"
          autoComplete="name"
          autoFocus
          maxLength={100}
        />
        {error && <p className="error">{error}</p>}
        <button className="primary" type="submit" disabled={busy || !name.trim()}>
          {busy ? "Регистрация…" : "Продолжить"}
        </button>
      </div>
      <div className="footer">
        <button type="button" className="link small" onClick={onShowUsers}>
          Посмотреть существующих пользователей
        </button>
      </div>
    </form>
  );
}
