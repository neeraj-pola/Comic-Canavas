"use client";

import { useState } from "react";
import type { PreferenceSnapshot } from "@/lib/api";
import { CI_Z, FEATURES, FEATURE_BY_KEY, pct } from "@/lib/preference";
import { Hint } from "./Hint";

const W = 880;
const H = 320;
const PAD = { l: 46, r: 24, t: 30, b: 52 };
const X_MIN = -3.5;
const X_MAX = 3.5;
const PRIOR_SD = 1; // the model's starting belief: N(0, 1) for every weight
const GHOST_TAPS = [5, 12, 20];

const bx = (x: number) => PAD.l + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - PAD.l - PAD.r);
const pdf = (x: number, m: number, s: number) =>
  Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

/** Standard normal CDF (Abramowitz & Stegun 26.2.17, error < 1e-7). */
function Phi(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p =
    d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/** Height = probability density 1/(sd*sqrt(2*pi)). One scale for the whole chart
 * (max over every tap and feature, +12%) so a curve visibly gets taller as the
 * model becomes surer — rescaling per frame would hide exactly that. */
function fixedPeak(snapshots: PreferenceSnapshot[]): number {
  let best = 1 / (PRIOR_SD * Math.sqrt(2 * Math.PI));
  for (const s of snapshots)
    for (const v of s.sd) best = Math.max(best, 1 / (v * Math.sqrt(2 * Math.PI)));
  return best * 1.12;
}

const XS: number[] = [];
for (let x = X_MIN; x <= X_MAX + 1e-9; x += 0.05) XS.push(Number(x.toFixed(3)));

/** Evenly spaced y-axis gridlines from 0 up to (not including) `peak`, at a
 * "nice" step (1/2.5/5 x a power of ten) rather than a fixed list — a very
 * confident feature (small sd) makes a curve much taller than a wide one, so
 * a hardcoded set of gridlines runs out and leaves the top of a tall curve
 * unlabeled. Aims for about 5 gridlines regardless of how tall `peak` is. */
function densityTicks(peak: number): number[] {
  const target = peak / 5;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const residual = target / magnitude;
  const step = (residual < 1.5 ? 1 : residual < 3.5 ? 2.5 : residual < 7.5 ? 5 : 10) * magnitude;
  const ticks: number[] = [];
  for (let v = step; v < peak; v += step) ticks.push(Number(v.toFixed(2)));
  return ticks;
}

/**
 * The bell curve behind each bar: the model's belief about one weight is a
 * normal distribution N(mean, sd²). Narrow and tall = it's sure; wide and flat
 * = it isn't. The yellow area right of zero is the probability you like MORE of
 * the feature. The dashed curve is where it started (before any taps), faint
 * lines are earlier taps, so you can watch the curve narrow as you replay.
 */
export function PosteriorChart({
  snapshots,
  features,
  at,
}: {
  snapshots: PreferenceSnapshot[];
  features: string[];
  at: number;
}) {
  const [feat, setFeat] = useState("warmth");
  const [hoverChip, setHoverChip] = useState<string | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const meta = FEATURE_BY_KEY[feat]!;
  const shownMeta = FEATURE_BY_KEY[hoverChip ?? feat]!;
  const i = features.indexOf(feat);
  const snap = snapshots[at];
  if (!snap || i < 0) return null;

  const m = snap.mu[i]!;
  const sd = snap.sd[i]!;
  const ghosts = GHOST_TAPS.filter((t) => t < snap.tap).map((t) => snapshots[t - 1]!);
  const peak = fixedPeak(snapshots);
  const y = (v: number) => PAD.t + (H - PAD.t - PAD.b) * (1 - v / peak);
  const base = y(0);
  const line = (mm: number, ss: number) =>
    XS.map((x, k) => `${k ? "L" : "M"}${bx(x).toFixed(1)} ${y(pdf(x, mm, ss)).toFixed(1)}`).join(
      " ",
    );
  const area = (lo: number, hi: number) => {
    const p = XS.filter((x) => x >= lo - 1e-9 && x <= hi + 1e-9);
    return (
      `M${bx(p[0]!)} ${base} ` +
      p.map((x) => `L${bx(x).toFixed(1)} ${y(pdf(x, m, sd)).toFixed(1)}`).join(" ") +
      ` L${bx(p[p.length - 1]!)} ${base} Z`
    );
  };
  const lo = Math.max(X_MIN, m - CI_Z * sd);
  const hi = Math.min(X_MAX, m + CI_Z * sd);
  const by2 = base + 34;
  const pLike = Phi(m / sd);
  const conf = Math.max(pLike, 1 - pLike);

  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="pl-fchips" role="group" aria-label="Choose a feature">
        {FEATURES.map((f) => (
          <button
            key={f.key}
            type="button"
            className="pl-fchip"
            aria-pressed={f.key === feat}
            onClick={() => setFeat(f.key)}
            onPointerEnter={() => setHoverChip(f.key)}
            onPointerLeave={() => setHoverChip(null)}
            onFocus={() => setHoverChip(f.key)}
            onBlur={() => setHoverChip(null)}
          >
            {f.label}
          </button>
        ))}
      </div>
      <p className="mono" style={{ color: "var(--mute)", margin: 0 }} aria-live="polite">
        <b style={{ color: "var(--ink)" }}>{shownMeta.label}:</b> {shownMeta.desc}
      </p>
      <div className="pl-bellgrid">
        <div>
          <span className="mono" style={{ color: "var(--mute)" }}>
            y-axis: probability density
            <Hint>
              Not a probability itself: the height of the belief curve at that weight value. A
              tall, narrow curve (very sure) can go well above 1, since it&apos;s the total AREA
              under the whole curve that always adds up to 1, not any single point&apos;s height.
              The gridlines rescale so the tallest curve on this chart always has room to show its
              full peak.
            </Hint>
          </span>
          <div
            style={{ overflowX: "auto" }}
            tabIndex={0}
            role="region"
            aria-label="Bell curve chart (scrolls sideways on a narrow screen)"
          >
            <svg
              className="pl-chart"
              viewBox={`0 0 ${W} ${H}`}
              width="100%"
              style={{ minWidth: 520 }}
              role="img"
              aria-label={`Bell curve of how much you care about ${meta.label.toLowerCase()}, after tap ${snap.tap}: ${pct(conf)} chance you prefer ${pLike >= 0.5 ? meta.pos : meta.neg}`}
            >
              {densityTicks(peak).map((v) => (
                <g key={v}>
                  <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="var(--rule)" />
                  <text x={PAD.l - 6} y={y(v)} textAnchor="end" dominantBaseline="middle">
                    {v.toFixed(2)}
                  </text>
                </g>
              ))}
              <text x={PAD.l} y={PAD.t - 14}>
                density
              </text>
              <line
                x1={PAD.l}
                x2={W - PAD.r}
                y1={base}
                y2={base}
                stroke="var(--ink)"
                strokeWidth={1.5}
              />
              {[-3, -2, -1, 0, 1, 2, 3].map((t) => (
                <g key={t}>
                  <line x1={bx(t)} x2={bx(t)} y1={base} y2={base + 5} stroke="var(--ink)" />
                  <text x={bx(t)} y={base + 18} textAnchor="middle">
                    {t > 0 ? `+${t}` : t}
                  </text>
                </g>
              ))}
              <text x={PAD.l} y={H - 8}>
                ← prefers {meta.neg}
              </text>
              <text x={W - PAD.r} y={H - 8} textAnchor="end">
                prefers {meta.pos} →
              </text>
              <path
                d={line(0, PRIOR_SD)}
                fill="none"
                stroke="var(--mute)"
                strokeWidth={1.5}
                strokeDasharray="4,4"
              />
              <text x={bx(-2.6)} y={y(pdf(-2.6, 0, PRIOR_SD)) - 8} textAnchor="middle">
                before any taps
              </text>
              {ghosts.map((g) => (
                <path
                  key={g.tap}
                  d={line(g.mu[i]!, g.sd[i]!)}
                  fill="none"
                  stroke="var(--ink)"
                  strokeOpacity={0.35}
                  strokeWidth={1.5}
                />
              ))}
              <path d={area(X_MIN, 0)} fill="var(--charcoal)" fillOpacity={0.22} />
              <path d={area(0, X_MAX)} fill="var(--mustard)" fillOpacity={0.85} />
              <path
                d={line(m, sd)}
                fill="none"
                stroke="var(--ink)"
                strokeWidth={2.6}
                strokeLinejoin="round"
              />
              <line
                x1={bx(0)}
                x2={bx(0)}
                y1={PAD.t - 6}
                y2={base}
                stroke="var(--ink)"
                strokeWidth={1.5}
                strokeDasharray="2,3"
              />
              <text x={bx(0) + 6} y={PAD.t}>
                no preference
              </text>
              <line
                x1={bx(m)}
                x2={bx(m)}
                y1={y(pdf(m, m, sd))}
                y2={base}
                stroke="var(--ink)"
                strokeWidth={1.5}
              />
              <circle
                cx={bx(m)}
                cy={y(pdf(m, m, sd))}
                r={5}
                fill="var(--card)"
                stroke="var(--ink)"
                strokeWidth={1.5}
              />
              <line x1={bx(lo)} x2={bx(hi)} y1={by2} y2={by2} stroke="var(--ink)" strokeWidth={2} />
              <line
                x1={bx(lo)}
                x2={bx(lo)}
                y1={by2 - 4}
                y2={by2 + 4}
                stroke="var(--ink)"
                strokeWidth={2}
              />
              <line
                x1={bx(hi)}
                x2={bx(hi)}
                y1={by2 - 4}
                y2={by2 + 4}
                stroke="var(--ink)"
                strokeWidth={2}
              />
              <text className="lab" x={Math.min(bx(hi) + 10, W - PAD.r - 60)} y={by2 + 4}>
                95% range
              </text>
              {hoverX !== null && (
                <line
                  x1={bx(hoverX)}
                  x2={bx(hoverX)}
                  y1={PAD.t}
                  y2={base}
                  stroke="var(--ink)"
                  strokeWidth={1}
                  opacity={0.6}
                  pointerEvents="none"
                />
              )}
              <rect
                x={PAD.l}
                y={PAD.t}
                width={W - PAD.l - PAD.r}
                height={base - PAD.t}
                fill="transparent"
                onPointerMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  setHoverX(X_MIN + ((e.clientX - r.left) / r.width) * (X_MAX - X_MIN));
                }}
                onPointerLeave={() => setHoverX(null)}
              />
            </svg>
          </div>
          <div className="pl-tip" aria-live="polite">
            {hoverX !== null ? (
              <>
                weight{" "}
                <b>
                  {hoverX >= 0 ? "+" : ""}
                  {hoverX.toFixed(2)}
                </b>{" "}
                · chance it&apos;s higher than this: <b>{pct(1 - Phi((hoverX - m) / sd))}</b>
              </>
            ) : (
              <span style={{ color: "var(--mute)" }}>
                Move over the curve. Dashed = before any taps · faint lines = earlier taps · yellow
                = now.
              </span>
            )}
          </div>
        </div>
        <div className="pl-sure">
          <span className="mono">{meta.label}</span>
          <div className="big">{pct(conf)}</div>
          <span>
            chance you prefer <b>{pLike >= 0.5 ? meta.pos : meta.neg}</b>
          </span>
          <span className="mono">
            best guess {m >= 0 ? "+" : ""}
            {m.toFixed(2)} ± {(CI_Z * sd).toFixed(2)}
          </span>
          <span className="mono">
            {sd < 0.45
              ? "a narrow curve: it knows."
              : sd < 0.7
                ? "getting narrower with each tap."
                : "still wide: it needs more taps."}
          </span>
        </div>
      </div>
    </div>
  );
}
