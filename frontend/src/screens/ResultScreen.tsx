import { useMemo } from "react";
import type { LivenessResult, Stimulus } from "../api";
import LineChart from "../components/LineChart";

interface Props {
  stimulus: Stimulus;
  result: LivenessResult;
  durationMs: number;
  onRetry: () => void;
}

export default function ResultScreen({ stimulus, result, durationMs, onRetry }: Props) {
  const { stimT, gazeT } = useMemo(() => {
    const stimT = stimulus.x.map((_, i) => i / stimulus.fps);
    // Сервер берёт stim[::step][-n:] и gaze[:n], то есть взгляд выровнен по концу стимула.
    const step = Math.max(1, Math.round(stimulus.fps / result.fps));
    const n = result.gaze_fit_x.length;
    const offset = Math.ceil(stimulus.x.length / step) - n;
    const gazeT = result.gaze_fit_x.map((_, i) => ((offset + i) * step) / stimulus.fps);
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
            { label: "gaze_fit_x", t: gazeT, v: result.gaze_fit_x, color: "#2563eb", highlight: result.inlier },
          ]}
        />
        <LineChart
          title="y(t)"
          series={[
            { label: "stim_y", t: stimT, v: stimulus.y, color: "#6b7280", width: 1.5 },
            { label: "gaze_fit_y", t: gazeT, v: result.gaze_fit_y, color: "#2563eb", highlight: result.inlier },
          ]}
        />
        <p className="duration">Обработка заняла {formatDuration(durationMs)}</p>
      </div>
      <div className="footer">
        <button className="primary" onClick={onRetry}>
          Ещё раз
        </button>
      </div>
    </div>
  );
}

/** Две значащие цифры: «1.2 секунды», «0.45 секунд». */
function formatDuration(durationMs: number): string {
  const seconds = durationMs / 1000;
  if (!Number.isFinite(seconds) || seconds <= 0) return "0 секунд";
  const text = formatSigFigs(seconds, 2);
  return `${text} ${secondsWord(text)}`;
}

function formatSigFigs(value: number, sig: number): string {
  const rounded = Number(value.toPrecision(sig));
  const decimals = Math.max(0, sig - Math.floor(Math.log10(rounded)) - 1);
  return rounded.toFixed(decimals);
}

function secondsWord(formatted: string): string {
  const [intText, frac] = formatted.split(".");
  if (frac === undefined || Number(frac) === 0) return plural(Math.abs(Number(intText)));
  return plural(Number(frac[frac.length - 1]));
}

function plural(n: number): string {
  const mod100 = n % 100;
  const mod10 = n % 10;
  if (mod10 === 1 && mod100 !== 11) return "секунда";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "секунды";
  return "секунд";
}
