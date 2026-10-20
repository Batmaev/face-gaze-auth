// /api срезают Caddy в проде и Vite в dev, см. frontend/Caddyfile и vite.config.ts
const API_URL = "/api";

export interface Stimulus {
  type: "moving-dot";
  fps: number;
  duration_sec: number;
  token: string;
  x: number[];
  y: number[];
  /** Контрольные точки сплайна, (n, 2). Сервер по ним восстановит траекторию, если токена нет в кэше воркера. */
  ctrl: number[][];
}

export interface LivenessResult {
  is_live: boolean;
  /** 1 − p, p — конформный p-value. Живой, если score >= 0.96. */
  score: number;
  /** Взгляд после min-max на сравниваемом окне. null — лица на кадре не было. */
  gaze_x: (number | null)[];
  gaze_y: (number | null)[];
  /** Кадр ближе 0.15 к стимулу по обеим осям. Моргания — false. */
  inlier: boolean[];
  /** Частота gaze_x / gaze_y / inlier после прореживания видео на сервере. */
  fps: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // тело ответа не JSON
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function getUsers(): Promise<Record<string, string>> {
  return request("/users");
}

export async function register(name: string): Promise<string> {
  const form = new FormData();
  form.append("name", name);
  const { user_id } = await request<{ user_id: string }>("/register", { method: "POST", body: form });
  return user_id;
}

export function getStimulus(): Promise<Stimulus> {
  return request("/stim");
}

/** XHR вместо fetch: нужен момент окончания отправки и прогресс загрузки. */
export function checkLivenessVideo(params: {
  userId: string;
  token: string;
  ctrl: number[][];
  video: Blob;
  filename: string;
  metadata: Record<string, unknown>;
  onUploadProgress?: (fraction: number) => void;
  onUploaded?: () => void;
}): Promise<LivenessResult> {
  const form = new FormData();
  form.append("user_id", params.userId);
  form.append("token", params.token);
  form.append("ctrl", JSON.stringify(params.ctrl));
  form.append("metadata", JSON.stringify(params.metadata));
  form.append("file", params.video, params.filename);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/islive_video`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) params.onUploadProgress?.(event.loaded / event.total);
    };
    xhr.upload.onload = () => params.onUploaded?.();
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300 && body) {
        resolve(body as LivenessResult);
      } else {
        const detail = body?.detail;
        reject(new Error(detail ? (typeof detail === "string" ? detail : JSON.stringify(detail)) : `${xhr.status} ${xhr.statusText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Сетевая ошибка"));
    xhr.send(form);
  });
}
