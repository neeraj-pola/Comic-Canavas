"use client";

import { useState } from "react";
import { pct, shortDate } from "@/lib/preference";

/**
 * How much of each day it tried something different from its favourite. It
 * only explores once the learned model is active, and explores less as it
 * gets surer — so this should fall over time. Bars are share of that day's
 * panels where the pick was an exploratory one.
 */
export function ExploreChart({
  days,
  active,
}: {
  days: { date: string; panels: number; explored: number }[];
  active: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  if (days.length === 0) {
    return <p style={{ color: "var(--mute)" }}>No strips generated with a learned model yet.</p>;
  }
  const W = 420;
  const H = 170;
  const padL = 34;
  const padB = 26;
  const slot = (W - padL - 8) / days.length;
  const barW = Math.min(30, slot - 6);
  const h = (v: number) => (H - padB - 10) * v;
  const day = hover !== null ? days[hover] : null;

  return (
    <div>
      <svg
        className="pl-chart"
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="group"
        aria-label="Share of each day's panels where it tried something different from its favourite"
        onPointerLeave={() => setHover(null)}
      >
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line
              x1={padL}
              x2={W - 8}
              y1={H - padB - h(g)}
              y2={H - padB - h(g)}
              stroke="var(--rule-soft)"
            />
            <text x={padL - 6} y={H - padB - h(g)} textAnchor="end" dominantBaseline="middle">
              {Math.round(g * 100)}%
            </text>
          </g>
        ))}
        {days.map((d, i) => {
          const share = d.panels ? d.explored / d.panels : 0;
          const cx = padL + slot * i + slot / 2;
          return (
            <g key={d.date}>
              <rect
                x={cx - slot / 2}
                y={0}
                width={slot}
                height={H - padB}
                fill="transparent"
                tabIndex={0}
                onPointerEnter={() => setHover(i)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              />
              <rect
                x={cx - barW / 2}
                y={H - padB - Math.max(h(share), 2)}
                width={barW}
                height={Math.max(h(share), 2)}
                rx={4}
                fill={share > 0 ? "var(--mustard)" : "var(--rule)"}
                stroke="var(--ink)"
                strokeWidth={1.5}
                opacity={hover === null || hover === i ? 1 : 0.55}
                pointerEvents="none"
              />
              {(days.length <= 8 || i % 2 === 0) && (
                <text x={cx} y={H - 8} textAnchor="middle">
                  {shortDate(d.date)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="pl-tip" aria-live="polite">
        {day ? (
          <>
            <b>{shortDate(day.date)}</b> · {day.explored} of {day.panels} panels explored (
            {pct(day.panels ? day.explored / day.panels : 0)})
          </>
        ) : !active ? (
          <span style={{ color: "var(--mute)" }}>
            Bars stay empty until the learned model takes over. Until then, the variation comes
            from each panel&apos;s three options.
          </span>
        ) : null}
      </div>
    </div>
  );
}
