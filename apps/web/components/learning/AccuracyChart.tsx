"use client";

import { useState } from "react";
import type { PreferenceSnapshot } from "@/lib/api";
import { pct, shortDate } from "@/lib/preference";

const W = 880;
const H = 300;
const PAD = { l: 44, r: 96, t: 16, b: 58 };
const WINDOW = 10;

function rolling(values: number[], warmup: number): (number | null)[] {
  return values.map((_, i) => {
    if (i < warmup) return null;
    const slice = values.slice(Math.max(warmup, i - WINDOW + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

/** Same running hit rate but from tap 1, so the warm-up taps can be drawn (faded). */
function running(values: number[]): number[] {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - WINDOW + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

const RESULT = (v: number) =>
  v > 0.5 ? "called it" : v === 0.5 ? "a tie (nothing to go on yet)" : "missed";

/**
 * "Did it predict you?" — each tap is a dot at the bottom (filled = the model
 * called your pick before seeing it), and the lines are the running hit rate
 * over the last 10 taps: the learned model against the hand-set weights it
 * has to beat. Click a dot or the chart to replay that moment.
 */
export function AccuracyChart({
  snapshots,
  selected,
  onSelect,
  minTaps,
  warmup,
}: {
  snapshots: PreferenceSnapshot[];
  selected: number;
  onSelect: (index: number) => void;
  minTaps: number;
  warmup: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const n = snapshots.length;
  if (n < 2) {
    return <p style={{ color: "var(--mute)" }}>Tap a few more pairs to see this fill in.</p>;
  }
  const learned = rolling(
    snapshots.map((s) => s.correct),
    warmup,
  );
  const hand = rolling(
    snapshots.map((s) => s.hand_correct),
    warmup,
  );
  const learnedAll = running(snapshots.map((s) => s.correct));
  const handAll = running(snapshots.map((s) => s.hand_correct));
  const warmEnd = Math.min(n - 1, warmup - 1); // last warm-up tap index
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (n === 1 ? 0 : (i * iw) / (n - 1));
  const y = (v: number) => PAD.t + ih - v * ih;
  const path = (series: (number | null)[]) =>
    series
      .map((v, i) =>
        v === null
          ? null
          : `${i === 0 || series[i - 1] === null ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`,
      )
      .filter(Boolean)
      .join(" ");
  const segment = (series: number[], from: number, to: number) =>
    series
      .slice(from, to + 1)
      .map((v, k) => `${k === 0 ? "M" : "L"} ${x(from + k).toFixed(1)} ${y(v).toFixed(1)}`)
      .join(" ");
  const last = (series: (number | null)[]) => {
    for (let i = series.length - 1; i >= 0; i--)
      if (series[i] !== null) return { i, v: series[i]! };
    return null;
  };
  const endLearned = last(learned);
  const endHand = last(hand);
  const at = hover ?? selected;
  const snap = snapshots[at];

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <svg
          className="pl-chart"
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          style={{ minWidth: 620 }}
          role="group"
          aria-label="Running share of your picks the model predicted, learned model versus hand-set weights"
          onPointerLeave={() => setHover(null)}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((g) => (
            <g key={g}>
              <line
                x1={PAD.l}
                x2={W - PAD.r}
                y1={y(g)}
                y2={y(g)}
                stroke={g === 0.5 ? "var(--mute)" : "var(--rule-soft)"}
                strokeDasharray={g === 0.5 ? "4,4" : undefined}
              />
              <text x={PAD.l - 8} y={y(g)} textAnchor="end" dominantBaseline="middle">
                {Math.round(g * 100)}%
              </text>
            </g>
          ))}
          <text x={W - PAD.r + 8} y={y(0.5)} dominantBaseline="middle">
            random guess
          </text>
          {n > minTaps && (
            <g>
              <line
                x1={x(minTaps - 1)}
                x2={x(minTaps - 1)}
                y1={PAD.t}
                y2={PAD.t + ih}
                stroke="var(--ink)"
                strokeDasharray="2,4"
              />
              <text x={x(minTaps - 1) + 6} y={PAD.t + 10}>
                can take over from here
              </text>
            </g>
          )}
          {/* warm-up: the first taps are the model's practice, so they aren't counted */}
          <rect
            x={x(0) - 8}
            y={PAD.t}
            width={x(warmEnd) - x(0) + 16}
            height={ih}
            fill="var(--mustard-soft)"
            opacity={0.55}
          />
          <text x={x(0) - 2} y={PAD.t + 12}>
            warm-up: not counted
          </text>
          {warmEnd >= 1 && (
            <>
              <path
                d={segment(handAll, 0, warmEnd)}
                fill="none"
                stroke="var(--mute)"
                strokeWidth={1.5}
                strokeDasharray="2,4"
                opacity={0.7}
              />
              <path
                d={segment(learnedAll, 0, warmEnd)}
                fill="none"
                stroke="var(--ink)"
                strokeWidth={2}
                strokeDasharray="2,4"
                opacity={0.7}
              />
            </>
          )}
          <path
            d={path(hand)}
            fill="none"
            stroke="var(--mute)"
            strokeWidth={2}
            strokeDasharray="6,5"
          />
          <path
            d={path(learned)}
            fill="none"
            stroke="var(--ink)"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {endLearned && (
            <>
              <circle
                cx={x(endLearned.i)}
                cy={y(endLearned.v)}
                r={6}
                fill="var(--mustard)"
                stroke="var(--ink)"
                strokeWidth={1.5}
              />
              <text
                className="lab"
                x={x(endLearned.i) + 12}
                y={y(endLearned.v) - (endHand && Math.abs(endHand.v - endLearned.v) < 0.08 ? 8 : 0)}
                dominantBaseline="middle"
              >
                Learned {pct(endLearned.v)}
              </text>
            </>
          )}
          {endHand && (
            <text
              x={x(endHand.i) + 12}
              y={y(endHand.v) + (endLearned && Math.abs(endHand.v - endLearned.v) < 0.08 ? 10 : 0)}
              dominantBaseline="middle"
            >
              Hand-set {pct(endHand.v)}
            </text>
          )}

          {/* the replay cursor */}
          <line
            x1={x(at)}
            x2={x(at)}
            y1={PAD.t}
            y2={PAD.t + ih}
            stroke="var(--ink)"
            strokeWidth={1.5}
            opacity={0.6}
          />

          {/* one dot per tap: filled = it called your pick */}
          <circle
            cx={W - PAD.r + 14}
            cy={H - PAD.b + 14}
            r={5}
            fill="var(--mustard)"
            stroke="var(--ink)"
            strokeWidth={1.5}
          />
          <text x={W - PAD.r + 26} y={H - PAD.b + 14} dominantBaseline="middle">
            called it
          </text>
          <circle
            cx={W - PAD.r + 14}
            cy={H - PAD.b + 32}
            r={5}
            fill="var(--card)"
            stroke="var(--ink)"
            strokeWidth={1.5}
          />
          <text x={W - PAD.r + 26} y={H - PAD.b + 32} dominantBaseline="middle">
            missed
          </text>
          <circle
            cx={W - PAD.r + 14}
            cy={H - PAD.b + 50}
            r={5}
            fill="var(--mustard-soft)"
            stroke="var(--ink)"
            strokeWidth={1.5}
            strokeDasharray="2,2"
          />
          <text x={W - PAD.r + 26} y={H - PAD.b + 50} dominantBaseline="middle">
            tie
          </text>
          {snapshots.map((s, i) => {
            const hit = s.correct > 0.5;
            const tie = s.correct === 0.5;
            return (
              <g key={s.tap}>
                <circle
                  cx={x(i)}
                  cy={H - PAD.b + 22}
                  r={14}
                  fill="transparent"
                  style={{ cursor: "pointer" }}
                  tabIndex={0}
                  role="button"
                  aria-label={`Tap ${s.tap}: ${tie ? "a tie" : hit ? "predicted" : "missed"}`}
                  onPointerEnter={() => setHover(i)}
                  onFocus={() => setHover(i)}
                  onBlur={() => setHover(null)}
                  onClick={() => onSelect(i)}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(i)}
                />
                <circle
                  cx={x(i)}
                  cy={H - PAD.b + 22}
                  r={i === at ? 7 : 5}
                  fill={hit ? "var(--mustard)" : tie ? "var(--mustard-soft)" : "var(--card)"}
                  stroke="var(--ink)"
                  strokeWidth={1.5}
                  strokeDasharray={tie ? "2,2" : undefined}
                  pointerEvents="none"
                />
              </g>
            );
          })}
          <text x={PAD.l} y={H - 8}>
            tap 1
          </text>
          <text x={W - PAD.r} y={H - 8} textAnchor="end">
            tap {n}
          </text>
        </svg>
      </div>
      <div className="pl-tip" aria-live="polite">
        {snap && (
          <>
            <b>Tap {snap.tap}</b>
            {snap.date ? ` · ${shortDate(snap.date)}` : ""} · gave your pick {pct(snap.prob)}:{" "}
            {RESULT(snap.correct)}; hand-set weights {RESULT(snap.hand_correct)}
            {at < warmup ? " (warm-up tap, not counted)" : ""}
          </>
        )}
      </div>
      {n <= warmup + 1 && (
        <p style={{ color: "var(--mute)", maxWidth: "78ch" }}>
          Only {n} taps so far. The first {warmup} are the model&apos;s <b>warm-up</b> (shaded):
          with almost nothing to learn from, its guesses aren&apos;t counted, so those lines are
          dotted. Solid lines start at tap {warmup + 1}. The dashed 50% line is what guessing at
          random would score; the goal is to stay above it.
        </p>
      )}
      <details style={{ marginTop: 8 }}>
        <summary className="mono" style={{ cursor: "pointer" }}>
          View as table
        </summary>
        <table>
          <thead>
            <tr>
              <th>Tap</th>
              <th>Date</th>
              <th>Learned model gave your pick</th>
              <th>Learned</th>
              <th>Hand-set</th>
            </tr>
          </thead>
          <tbody>
            {snapshots.map((s) => (
              <tr key={s.tap}>
                <td>{s.tap}</td>
                <td>{shortDate(s.date)}</td>
                <td>{pct(s.prob)}</td>
                <td>{RESULT(s.correct)}</td>
                <td>{RESULT(s.hand_correct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
