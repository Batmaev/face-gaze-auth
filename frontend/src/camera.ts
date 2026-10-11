const MIME_CANDIDATES = [
  "video/mp4;codecs=avc1",
  "video/mp4",
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
];

export function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

export function extensionFor(mimeType: string): string {
  return mimeType.startsWith("video/mp4") ? "mp4" : "webm";
}

const VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  facingMode: "user",
  width: { ideal: 1280 },
  height: { ideal: 720 },
};

/** Камера с максимальной частотой кадров, которую она поддерживает (для датасета). */
export async function openCamera(): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { ...VIDEO_CONSTRAINTS, frameRate: { ideal: 60 } },
  });
  const track = stream.getVideoTracks()[0];
  const maxFps = track?.getCapabilities?.().frameRate?.max;
  if (track && maxFps && maxFps > (track.getSettings().frameRate ?? 0)) {
    try {
      await track.applyConstraints({ ...VIDEO_CONSTRAINTS, frameRate: { ideal: maxFps } });
    } catch (error) {
      console.warn("Не удалось поднять fps камеры", error);
    }
  }
  return stream;
}

export interface Recording {
  blob: Blob;
  mimeType: string;
  startedAt: number;
  stoppedAt: number;
}

export class VideoRecorder {
  private recorder: MediaRecorder;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private stopped: Promise<Recording>;

  constructor(stream: MediaStream) {
    const mimeType = pickMimeType();
    this.recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 2_500_000 } : undefined);
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    this.stopped = new Promise((resolve, reject) => {
      this.recorder.onstop = () => {
        const type = this.recorder.mimeType || mimeType || "video/webm";
        resolve({
          blob: new Blob(this.chunks, { type }),
          mimeType: type,
          startedAt: this.startedAt,
          stoppedAt: performance.now(),
        });
      };
      this.recorder.onerror = (event) => reject((event as ErrorEvent).error ?? new Error("Ошибка записи видео"));
    });
  }

  /** Резолвится, когда запись реально пошла (событие start), а не в момент вызова. */
  start(): Promise<number> {
    return new Promise((resolve) => {
      this.recorder.onstart = () => {
        this.startedAt = performance.now();
        resolve(this.startedAt);
      };
      this.recorder.start();
    });
  }

  stop(): Promise<Recording> {
    if (this.recorder.state !== "inactive") this.recorder.stop();
    return this.stopped;
  }

  cancel() {
    this.recorder.ondataavailable = null;
    if (this.recorder.state !== "inactive") this.recorder.stop();
  }
}
