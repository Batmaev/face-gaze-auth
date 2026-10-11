export interface Series {
  label: string;
  t: number[];
  v: number[];
  color: string;
  width?: number;
  /** Точки, где false, дополнительно помечаются серыми кружками (выбросы). */
  highlight?: boolean[];
}

interface Props {
  title: string;
  series: Series[];
}

const W = 600;
const H = 220;
const PAD = { left: 36, right: 12, top: 12, bottom: 26 };

export default function LineChart({ title, series }: Props) {
  const all = series.flatMap((s) => s.v.filter(Number.isFinite));
  const allT = series.flatMap((s) => s.t);
  const tMin = Math.min(0, ...allT);
  const tMax = Math.max(...allT, tMin + 1e-6);
  let vMin = Math.min(...all);
  let vMax = Math.max(...all);
  if (!Number.isFinite(vMin) || !Number.isFinite(vMax)) [vMin, vMax] = [0, 1];
  const vPad = (vMax - vMin || 1) * 0.08;
  vMin -= vPad;
  vMax += vPad;

  const sx = (t: number) => PAD.left + ((t - tMin) / (tMax - tMin)) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - ((v - vMin) / (vMax - vMin)) * (H - PAD.top - PAD.bottom);

  const pathFor = (s: Series) => {
    let d = "";
    let pen = false;
    s.v.forEach((v, i) => {
      if (!Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${sx(s.t[i]).toFixed(1)},${sy(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const tTicks = ticks(tMin, tMax, 5);
  const vTicks = ticks(vMin, vMax, 4);

  return (
    <figure className="chart">
      <figcaption>
        <span>{title}</span>
        <span className="legend">
          {series.map((s) => (
            <span key={s.label}>
              <i style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {vTicks.map((v) => (
          <g key={`v${v}`}>
            <line x1={PAD.left} x2={W - PAD.right} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={PAD.left - 6} y={sy(v)} className="tick" textAnchor="end" dominantBaseline="middle">
              {format(v)}
            </text>
          </g>
        ))}
        {tTicks.map((t) => (
          <text key={`t${t}`} x={sx(t)} y={H - 8} className="tick" textAnchor="middle">
            {format(t)}
          </text>
        ))}
        <text x={W - PAD.right} y={H - 8} className="tick" textAnchor="end">
          t, с
        </text>
        {series.map((s) => (
          <path key={s.label} d={pathFor(s)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} vectorEffect="non-scaling-stroke" />
        ))}
        {series.map((s) =>
          s.highlight?.map((ok, i) =>
            ok || !Number.isFinite(s.v[i]) ? null : (
              <circle key={`${s.label}${i}`} cx={sx(s.t[i])} cy={sy(s.v[i])} r={3} className="outlier" />
            ),
          ),
        )}
      </svg>
    </figure>
  );
}

function ticks(min: number, max: number, count: number): number[] {
  const step = niceStep((max - min) / count);
  const result: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) result.push(+v.toFixed(10));
  return result;
}

function niceStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(raw || 1));
  const n = raw / power;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * power;
}

function format(v: number) {
  return Math.abs(v) >= 10 ? v.toFixed(0) : +v.toFixed(2) + "";
}
