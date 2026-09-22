"use client";

import { useState } from "react";
import type { KnobModel } from "@/lib/api";
import { AXIS_KEYS, FEATURE_BY_KEY } from "@/lib/preference";
import { Hint } from "./Hint";

/**
 * The knob model, drawn: for each style knob, how much you like each level from
 * -2 to +2 relative to "as usual" (level 0). The line is its best guess, the band
 * is how unsure it is — wide early on, narrowing as you pick. A peak in the
 * middle is a sweet spot ("warm, but not too warm"); a slope says "more is
 * better". The dot marks the default it currently bakes into every prompt.
 *
 * All three knobs share one vertical scale (each on its own would make the
 * least-known knob look as decisive as the best-known one), and the band is
 * clipped to the plot rather than stretching the scale.
 */
const W = 300;
const H = 170;
const padL = 30;
const padR = 10;
const padT = 12;
const padB = 30;
const Y_MAX = 3;

const levelWord = (axis: string, level: number) => {
  const meta = FEATURE_BY_KEY[axis]!;
  if (level === 0) return "as usual";
  const mag = Math.abs(level) >= 2 ? "much " : "";
  return `${mag}${level > 0 ? meta.pos : meta.neg}`;
};

export function KnobCurves({ knobs, lean }: { knobs: KnobModel; lean: Record<string, number> }) {
  const [hover, setHover] = useState<Record<string, number | undefined>>({});
  const x = (level: number) => padL + ((level + 2) / 4) * (W - padL - padR);
  const y = (v: number) => padT + (1 - (v + Y_MAX) / (2 * Y_MAX)) * (H - padT - padB);
  const clampY = (v: number) => Math.max(-Y_MAX, Math.min(Y_MAX, v));

  return (
    <div className="pl-knobs">
      {AXIS_KEYS.map((axis) => {
        const meta = FEATURE_BY_KEY[axis]!;
        const points = knobs.curves?.[axis] ?? [];
        if (points.length === 0) return null;
        const isSeen = (p: { seen?: number }) => p.seen !== 0;
        const segment = (a: (typeof points)[number], b: (typeof points)[number]) =>
          `M ${x(a.level).toFixed(1)} ${y(clampY(a.mean)).toFixed(1)} L ${x(b.level).toFixed(1)} ${y(clampY(b.mean)).toFixed(1)}`;
        const pairs = points.slice(1).map((p, i) => [points[i]!, p] as const);
        const solid = pairs
          .filter(([a, b]) => isSeen(a) && isSeen(b))
          .map(([a, b]) => segment(a, b))
          .join(" ");
        const dashed = pairs
          .filter(([a, b]) => !(isSeen(a) && isSeen(b)))
          .map(([a, b]) => segment(a, b))
          .join(" ");
        const band =
          points
            .map((p) => `${x(p.level).toFixed(1)},${y(clampY(p.mean + p.sd)).toFixed(1)}`)
            .join(" L ") +
          " L " +
          [...points]
            .reverse()
            .map((p) => `${x(p.level).toFixed(1)},${y(clampY(p.mean - p.sd)).toFixed(1)}`)
            .join(" L ");
        const current = lean[axis] ?? 0;
        const best = knobs.best?.[axis] ?? 0;
        const at = hover[axis];
        const shown = at !== undefined ? points.find((p) => p.level === at) : undefined;
        const currentPoint = points.find((p) => p.level === current);
        return (
          <div key={axis} className="pl-knob">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>
                {meta.label}
                <Hint>{meta.desc}</Hint>
              </b>
              <span className="mono">
                default: <b>{levelWord(axis, current)}</b>
              </span>
            </div>
            <svg
              className="pl-chart"
              viewBox={`0 0 ${W} ${H}`}
              width="100%"
              role="img"
              aria-label={`${meta.label}: best guess ${levelWord(axis, best)}, current default ${levelWord(axis, current)}`}
              onPointerLeave={() => setHover((h) => ({ ...h, [axis]: undefined }))}
            >
              <defs>
                <clipPath id={`clip-${axis}`}>
                  <rect x={padL} y={padT} width={W - padL - padR} height={H - padT - padB} />
                </clipPath>
              </defs>
              <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)} stroke="var(--rule)" />
              <text x={padL - 6} y={y(0)} textAnchor="end" dominantBaseline="middle">
                0
              </text>
              <text x={padL - 6} y={y(Y_MAX - 0.4)} textAnchor="end" dominantBaseline="middle">
                +
              </text>
              <text x={padL - 6} y={y(-Y_MAX + 0.4)} textAnchor="end" dominantBaseline="middle">
                −
              </text>
              <g clipPath={`url(#clip-${axis})`}>
                <path d={`M ${band} Z`} fill="var(--mustard)" opacity={0.28} />
                <path d={solid} fill="none" stroke="var(--ink)" strokeWidth={2.2} />
                <path
                  d={dashed}
                  fill="none"
                  stroke="var(--mute)"
                  strokeWidth={1.8}
                  strokeDasharray="4,4"
                />
              </g>
              {points.map((p) => (
                <g key={p.level}>
                  <text x={x(p.level)} y={H - 10} textAnchor="middle">
                    {p.level > 0 ? `+${p.level}` : p.level}
                  </text>
                  {!isSeen(p) && (
                    <circle
                      cx={x(p.level)}
                      cy={y(clampY(p.mean))}
                      r={3.5}
                      fill="var(--card)"
                      stroke="var(--mute)"
                      strokeWidth={1.2}
                      pointerEvents="none"
                    />
                  )}
                  <circle
                    cx={x(p.level)}
                    cy={y(clampY(p.mean))}
                    r={16}
                    fill="transparent"
                    tabIndex={0}
                    onPointerEnter={() => setHover((h) => ({ ...h, [axis]: p.level }))}
                    onFocus={() => setHover((h) => ({ ...h, [axis]: p.level }))}
                  />
                </g>
              ))}
              {currentPoint && (
                <circle
                  cx={x(current)}
                  cy={y(clampY(currentPoint.mean))}
                  r={7}
                  fill="var(--mustard)"
                  stroke="var(--ink)"
                  strokeWidth={1.5}
                  pointerEvents="none"
                />
              )}
              {best !== current && (
                <circle
                  cx={x(best)}
                  cy={y(clampY(points.find((p) => p.level === best)?.mean ?? 0))}
                  r={6}
                  fill="none"
                  stroke="var(--ink)"
                  strokeWidth={1.5}
                  strokeDasharray="3,2"
                  pointerEvents="none"
                />
              )}
            </svg>
            <div className="mono" style={{ minHeight: 32 }}>
              {shown ? (
                <>
                  <b>{levelWord(axis, shown.level)}</b> vs as usual: {shown.mean >= 0 ? "+" : "−"}
                  {Math.abs(shown.mean).toFixed(1)} ± {shown.sd.toFixed(1)}
                  {isSeen(shown) ? "" : " · never shown to you: a guess, not learned"}
                </>
              ) : (
                <>
                  ← {meta.neg} · {meta.pos} → &nbsp; best guess: <b>{levelWord(axis, best)}</b>
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
