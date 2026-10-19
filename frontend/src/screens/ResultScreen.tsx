import { useMemo } from "react";
import type { LivenessResult, Stimulus } from "../api";
import LineChart from "../components/LineChart";

interface Props {
  stimulus: Stimulus;
  result: LivenessResult;
  uploadMs: number;
  processingMs: number;
  onRetry: () => void;
}

export default function ResultScreen({ stimulus, result, uploadMs, processingMs, onRetry }: Props) {
  const { stimT, gazeT } = useMemo(() => {
    const stimT = stimulus.x.map((_, i) => i / stimulus.fps);
    // Сервер берёт отсчёты стимула в моменты взгляда и stim[-n:], gaze[:n]:
    // взгляд выровнен по концу стимула. При целом шаге это stim[::step].
    const step = stimulus.fps / result.fps;
    const n = result.gaze_x.length;
    const sampled = Math.floor((stimulus.x.length - 1e-9) / step) + 1;
    const offset = sampled - n;
    const gazeT = result.gaze_x.map((_, i) => (offset + i) / result.fps);
    return { stimT, gazeT };
  }, [stimulus, result]);

  return (
    <div className="screen">
      <div className="content scroll">
        <div className={result.is_live ? "verdict ok" : "verdict fail"}>
          <div className="score">{result.score.toFixed(2)}</div>
          <div>Витальность {result.is_live ? "подтверждена" : "не подтверждена"}</div>
        </div>
        <LineChart
          title="x(t)"
          series={[
            { label: "stim_x", t: stimT, v: stimulus.x, color: "#6b7280", width: 1.5 },
            { label: "gaze_x", t: gazeT, v: result.gaze_x, color: "#2563eb", highlight: result.inlier },
          ]}
        />
        <LineChart
          title="y(t)"
          series={[
            { label: "stim_y", t: stimT, v: stimulus.y, color: "#6b7280", width: 1.5 },
            { label: "gaze_y", t: gazeT, v: result.gaze_y, color: "#2563eb", highlight: result.inlier },
          ]}
        />
        <p className="duration">Загрузка видео заняла {formatDuration(uploadMs)} сек</p>
        <p className="duration">Обработка заняла {formatDuration(processingMs)} сек</p>
      </div>
      <div className="footer">
        <button className="primary" onClick={onRetry}>
          Ещё раз
        </button>
      </div>
    </div>
  );
}

/** Две значащие цифры: «1.2», «0.45». */
function formatDuration(durationMs: number): string {
  const seconds = durationMs / 1000;
  if (!Number.isFinite(seconds) || seconds <= 0) return "0";
  return formatSigFigs(seconds, 2);
}

function formatSigFigs(value: number, sig: number): string {
  const rounded = Number(value.toPrecision(sig));
  const decimals = Math.max(0, sig - Math.floor(Math.log10(rounded)) - 1);
  return rounded.toFixed(decimals);
}
