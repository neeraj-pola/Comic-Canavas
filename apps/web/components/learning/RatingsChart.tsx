"use client";

import { useState } from "react";
import { shortDate } from "@/lib/preference";
import "./preference.css";

type Day = { day_id: string; off: number; ok: number; great: number };

/**
 * How good are the images you keep? One bar per day, split into what you rated great, ok and off
 * (the image in your strip is what you usually rate). Bars are shares of that day's ratings, so
 * days with more ratings don't look better; the share rated "great" is the number that should
 * climb as the app learns. Segments carry their counts on hover and in the table, and the legend
 * names them — colour is never the only cue.
 */
export function RatingsChart({ days }: { days: Day[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (days.length === 0) {
    return (
      <p style={{ color: "var(--mute)" }}>
        Nothing rated yet. Under each panel on Today, tap <b>off / ok / great</b> for the image in
        your strip. A rating tells it whether an image is good at all, which a pick among three
        can&apos;t.
      </p>
    );
  }
  const W = 560;
  const H = 190;
  const padL = 34;
  const padB = 28;
  const padT = 8;
  const slot = (W - padL - 8) / days.length;
  const barW = Math.min(34, slot - 6);
  const inner = H - padB - padT;
  const day = hover !== null ? days[hover] : null;
  const labelEvery = Math.ceil(days.length / 10); // keep the date labels legible on long histories
  const total = (d: Day) => d.off + d.ok + d.great;

  return (
    <div>
      <svg
        className="pl-chart"
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Share of each day's ratings that were great, ok and off"
        onPointerLeave={() => setHover(null)}
      >
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line
              x1={padL}
              x2={W - 8}
              y1={padT + inner * (1 - g)}
              y2={padT + inner * (1 - g)}
              stroke="var(--rule-soft)"
            />
            <text
              x={padL - 6}
              y={padT + inner * (1 - g)}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {Math.round(g * 100)}%
            </text>
          </g>
        ))}
        {days.map((d, i) => {
          const n = total(d) || 1;
          const cx = padL + slot * i + slot / 2;
          let top = padT + inner; // stack from the bottom: off, then ok, then great
          const segments = [
            { key: "off", v: d.off, fill: "var(--ink)" },
            { key: "ok", v: d.ok, fill: "var(--mustard-soft)" },
            { key: "great", v: d.great, fill: "var(--mustard)" },
          ];
          return (
            <g key={d.day_id} onPointerEnter={() => setHover(i)}>
              {segments.map((seg) => {
                const h = (seg.v / n) * inner;
                top -= h;
                return h > 0 ? (
                  <rect
                    key={seg.key}
                    x={cx - barW / 2}
                    y={top}
                    width={barW}
                    height={h}
                    fill={seg.fill}
                    stroke="var(--ink)"
                    strokeWidth={1}
                  />
                ) : null;
              })}
              <rect
                x={cx - slot / 2}
                y={padT}
                width={slot}
                height={inner}
                fill="transparent"
                tabIndex={0}
                role="img"
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
                aria-label={`${d.day_id}: ${d.great} great, ${d.ok} ok, ${d.off} off`}
              />
              {i % labelEvery === 0 && (
                <text x={cx} y={H - 8} textAnchor="middle">
                  {shortDate(d.day_id)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
        {(
          [
            ["great", "var(--mustard)"],
            ["ok", "var(--mustard-soft)"],
            ["off", "var(--ink)"],
          ] as const
        ).map(([name, fill]) => (
          <span key={name} className="mono row" style={{ gap: 6, alignItems: "center" }}>
            <i
              aria-hidden="true"
              style={{
                display: "inline-block",
                width: 12,
                height: 12,
                background: fill,
                border: "1.5px solid var(--ink)",
              }}
            />
            {name}
          </span>
        ))}
        {day && (
          <span className="mono">
            <b>{shortDate(day.day_id)}</b>: {day.great} great · {day.ok} ok · {day.off} off
          </span>
        )}
      </div>
      <details style={{ marginTop: 8 }}>
        <summary className="mono" style={{ cursor: "pointer" }}>
          View as table
        </summary>
        <table>
          <thead>
            <tr>
              <th>Day</th>
              <th>Great</th>
              <th>Ok</th>
              <th>Off</th>
            </tr>
          </thead>
          <tbody>
            {days.map((d) => (
              <tr key={d.day_id}>
                <td>{d.day_id}</td>
                <td>{d.great}</td>
                <td>{d.ok}</td>
                <td>{d.off}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
