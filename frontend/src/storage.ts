export interface User {
  id: string;
  name: string;
}

const USER_KEY = "liveness.user";
const PWA_SKIP_KEY = "liveness.pwaSkipped";

export function loadUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    const user = JSON.parse(raw);
    return typeof user?.id === "string" && typeof user?.name === "string" ? user : null;
  } catch {
    return null;
  }
}

export function saveUser(user: User) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function isPwaSkipped(): boolean {
  return sessionStorage.getItem(PWA_SKIP_KEY) === "1";
}

export function skipPwa() {
  sessionStorage.setItem(PWA_SKIP_KEY, "1");
}
