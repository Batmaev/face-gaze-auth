import { useEffect, useState } from "react";
import { getUsers } from "../api";
import type { User } from "../storage";

interface Props {
  currentUserId?: string;
  onBack: () => void;
  onSelect: (user: User) => void;
}

export default function UsersScreen({ currentUserId, onBack, onSelect }: Props) {
  const [users, setUsers] = useState<User[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getUsers()
      .then((map) => setUsers(Object.entries(map).map(([id, name]) => ({ id, name }))))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="screen">
      <div className="topbar">
        <button className="icon-button" onClick={onBack} aria-label="Назад">
          ←
        </button>
        <h3>Пользователи</h3>
      </div>
      <div className="content">
        {error && <p className="error">{error}</p>}
        {!users && !error && <p className="muted">Загрузка…</p>}
        {users?.length === 0 && <p className="muted">Пока никого нет</p>}
        <ul className="user-list">
          {users?.map((user) => (
            <li key={user.id}>
              <button className={user.id === currentUserId ? "user-row current" : "user-row"} onClick={() => onSelect(user)}>
                <span className="user-meta">{user.name}</span>
                <span className="user-meta">{user.id}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
