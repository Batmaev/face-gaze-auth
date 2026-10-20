import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { StimulusResponse } from "../api";
import { VideoRecorder, type Recording } from "../camera";

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface StimulusRun {
  startedAt: number;
  finishedAt: number;
  renderedFrames: number;
  /** Область, в которую отображён квадрат [0, 1]×[0, 1] стимула, в CSS-пикселях от левого верхнего угла окна. */
  area: Rect;
  viewport: { width: number; height: number };
}

interface Props {
  stimulus: StimulusResponse;
  stream: MediaStream;
  onFinished: (recording: Recording, run: StimulusRun) => void;
  onError: (message: string) => void;
}

// Пауза перед стартом, чтобы взгляд успел найти точку. Запись в это время ещё не идёт.
const READY_DELAY_MS = 1500;
const MARKER_COLOR = "#9b111e";

export default function StimulusScreen({ stimulus, stream, onFinished, onError }: Props) {
  const [layout, setLayout] = useState<{ area: Rect; viewport: StimulusRun["viewport"] } | null>(null);
  const [phase, setPhase] = useState<"ready" | "running">("ready");
  const rootRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLDivElement>(null);
  const markerRef = useRef<SVGGElement>(null);
  const callbacks = useRef({ onFinished, onError });
  callbacks.current = { onFinished, onError };

  // Отступы (safe-area + запас) задаются в CSS, здесь только снимаем получившийся прямоугольник.
  useLayoutEffect(() => {
    const root = rootRef.current!.getBoundingClientRect();
    const rect = areaRef.current!.getBoundingClientRect();
    setLayout({
      area: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
      viewport: { width: root.width, height: root.height },
    });
  }, []);

  const points = useMemo(
    () =>
      layout
        ? stimulus.x.map((x, i) => [layout.area.left + x * layout.area.width, layout.area.top + stimulus.y[i] * layout.area.height] as const)
        : [],
    [stimulus, layout],
  );

  const pathD = useMemo(
    () => points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" "),
    [points],
  );

  useEffect(() => {
    if (!layout || points.length === 0) return;

    const moveMarker = (index: number) => {
      const [x, y] = points[index];
      markerRef.current?.setAttribute("transform", `translate(${x} ${y})`);
    };
    moveMarker(0);

    let cancelled = false;
    let frame = 0;
    let recorder: VideoRecorder | null = null;
    const fps = stimulus.stim.fps;
    const durationMs = (points.length / fps) * 1000;

    const timer = window.setTimeout(async () => {
      try {
        recorder = new VideoRecorder(stream);
        await recorder.start();
      } catch (error) {
        callbacks.current.onError(`Не удалось начать запись: ${error instanceof Error ? error.message : String(error)}`);
        return;
      }
      if (cancelled) return;

      // Отсчёт стимула идёт от фактического старта записи, чтобы видео и траектория совпадали по времени.
      const startedAt = performance.now();
      let renderedFrames = 0;
      setPhase("running");

      const tick = async (now: number) => {
        if (cancelled) return;
        const elapsed = now - startedAt;
        if (elapsed >= durationMs) {
          moveMarker(points.length - 1);
          const finishedAt = performance.now();
          try {
            const recording = await recorder!.stop();
            if (!cancelled) callbacks.current.onFinished(recording, { startedAt, finishedAt, renderedFrames, ...layout });
          } catch (error) {
            if (!cancelled) callbacks.current.onError(`Ошибка записи: ${error instanceof Error ? error.message : String(error)}`);
          }
          return;
        }
        moveMarker(Math.min(points.length - 1, Math.floor((elapsed / 1000) * fps)));
        renderedFrames++;
        frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    }, READY_DELAY_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      cancelAnimationFrame(frame);
      recorder?.cancel();
    };
  }, [points, stimulus.stim.fps, stream, layout]);

  return (
    <div className="stimulus" ref={rootRef}>
      <div className="stimulus-area" ref={areaRef} />
      {layout && (
        <svg width={layout.viewport.width} height={layout.viewport.height}>
          <path d={pathD} fill="none" stroke="#93c5fd" strokeWidth={1.5} />
          <g ref={markerRef}>
            <line x1={-20} y1={0} x2={20} y2={0} stroke={MARKER_COLOR} strokeWidth={1.5} />
            <line x1={0} y1={-20} x2={0} y2={20} stroke={MARKER_COLOR} strokeWidth={1.5} />
            <circle r={4} fill={MARKER_COLOR} className={phase === "ready" ? "marker-dot pulsing" : "marker-dot"} />
          </g>
        </svg>
      )}
      {phase === "ready" && <div className="stimulus-hint">Смотрите на точку</div>}
    </div>
  );
}
