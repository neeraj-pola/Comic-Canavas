"use client";

import { useState } from "react";

/**
 * Task 10.11's pick-rate chart, rebuilt as a real SVG chart per the
 * dataviz skill (the mock draws this on a raw `<canvas>`, with no
 * hover/tooltip and no accessible fallback) — real, live-fetched
 * `pick_rate_series` from `GET /learning`, not synthetic weekly data.
 *
 * **Real mismatch, documented**: the mock's chart is "pick rate by
 * week" (10 fixed weekly points with dashed "checkpoint promoted"
 * markers); the real endpoint reports pick rate *by day*
 * (`pick_rate_series: [{day_id, pick_rate}]`), and `checkpoints` has no
 * "was this promoted" field to key dashed markers off (that decision
 * lives on `training_runs`, not exposed by this endpoint) — so this
 * shows the real daily series without fabricating weekly buckets or
 * promotion markers that don't exist in the data.
 *
 * Ink line + mustard, ink-stroked dots are the frozen design's own
 * established motif (every card/panel in this app uses a 1.5px ink
 * border) — kept as specified rather than swapped for the dataviz
 * skill's generic "surface-ring instead of a stroke" default, since a
 * fixed design system already governs this choice here.
 */
export function PickRateChart({
  series,
  label = "First-pick rate",
  chance,
}: {
  series: { day_id: string; pick_rate: number; taps?: number }[];
  /** What the rate means, for the accessible name and the table header. */
  label?: string;
  /** A dashed reference line: the rate you'd get by picking at random. */
  chance?: number | null;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (series.length === 0) {
    return <p style={{ color: "var(--mute)" }}>No picks recorded yet.</p>;
  }

  const W = 880;
  const H = 320;
  const padL = 44;
  const padR = 20;
  const padT = 20;
  const padB = 36;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const maxRate = Math.max(1, ...series.map((s) => s.pick_rate));
  const x = (i: number) => padL + (series.length === 1 ? 0 : (i * innerW) / (series.length - 1));
  const y = (v: number) => padT + innerH - (v / maxRate) * innerH;

  const linePath = series
    .map((s, i) => `${i ? "L" : "M"} ${x(i).toFixed(1)} ${y(s.pick_rate).toFixed(1)}`)
    .join(" ");
  const gridLines = Array.from({ length: 5 }, (_, i) => (maxRate * i) / 4);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${label} by day`}>
        {gridLines.map((g) => (
          <g key={g}>
            <line
              x1={padL}
              x2={W - padR}
              y1={y(g)}
              y2={y(g)}
              stroke="var(--rule-soft)"
              strokeWidth={1}
            />
            <text
              x={padL - 8}
              y={y(g)}
              fontSize={11}
              fill="var(--mute)"
              textAnchor="end"
              dominantBaseline="middle"
            >
              {Math.round(g * 100)}%
            </text>
          </g>
        ))}
        {chance != null && (
          <g>
            <line
              x1={padL}
              x2={W - padR}
              y1={y(chance)}
              y2={y(chance)}
              stroke="var(--mute)"
              strokeWidth={1.5}
              strokeDasharray="6,4"
            />
            <text x={W - padR} y={y(chance) - 6} fontSize={11} fill="var(--mute)" textAnchor="end">
              chance {Math.round(chance * 100)}%
            </text>
          </g>
        )}
        <path
          d={linePath}
          fill="none"
          stroke="var(--ink)"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {series.map((s, i) => (
          <g key={s.day_id}>
            {/* real 24px+ hit target, per the interaction spec, larger than the painted dot */}
            <circle
              cx={x(i)}
              cy={y(s.pick_rate)}
              r={14}
              fill="transparent"
              onPointerEnter={() => setHoverIndex(i)}
              onPointerLeave={() => setHoverIndex(null)}
              onFocus={() => setHoverIndex(i)}
              onBlur={() => setHoverIndex(null)}
              tabIndex={0}
            />
            <circle
              cx={x(i)}
              cy={y(s.pick_rate)}
              r={hoverIndex === i ? 7 : 6}
              fill="var(--mustard)"
              stroke="var(--ink)"
              strokeWidth={1.5}
              pointerEvents="none"
            />
          </g>
        ))}
        {hoverIndex !== null && (
          <line
            x1={x(hoverIndex)}
            x2={x(hoverIndex)}
            y1={padT}
            y2={H - padB}
            stroke="var(--mute)"
            strokeWidth={1}
            strokeDasharray="3,3"
            pointerEvents="none"
          />
        )}
      </svg>
      {hoverIndex !== null && (
        <div className="mono" style={{ marginTop: -8 }}>
          <b>{Math.round(series[hoverIndex]!.pick_rate * 100)}%</b> · {series[hoverIndex]!.day_id}
          {series[hoverIndex]!.taps ? ` · ${series[hoverIndex]!.taps} taps` : ""}
        </div>
      )}
      <details style={{ marginTop: 8 }}>
        <summary className="mono" style={{ cursor: "pointer" }}>
          View as table
        </summary>
        <table>
          <thead>
            <tr>
              <th>Day</th>
              <th>{label}</th>
            </tr>
          </thead>
          <tbody>
            {series.map((s) => (
              <tr key={s.day_id}>
                <td>{s.day_id}</td>
                <td>{Math.round(s.pick_rate * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
