"use client";

import { useState } from "react";
import { CI_Z, FEATURES, type FeatureMeta } from "@/lib/preference";
import { Hint } from "./Hint";

const DOMAIN = 3; // weights are shown from -3 to +3 (in "typical difference" units)
const pos = (v: number) => 50 + (Math.max(-DOMAIN, Math.min(DOMAIN, v)) / DOMAIN) * 50;

/** sure ≈ 95% one-sided, leaning ≈ 80% — the same probabilities the bell
 * curves show, so a bar and its curve never disagree. */
function describe(meta: FeatureMeta, mu: number, sd: number): { text: string; level: 0 | 1 | 2 } {
  const z = Math.abs(mu) / sd;
  const level = z >= 1.645 ? 2 : z >= 0.84 ? 1 : 0;
  const word = mu > 0 ? meta.pos : meta.neg;
  if (level === 2) return { text: `prefers ${word}`, level };
  if (level === 1) return { text: `leaning ${word}`, level };
  return { text: "not sure yet", level };
}

/**
 * What the model currently thinks you like, one row per feature: a bar for
 * the learned weight and a whisker for the 95% range it could plausibly be.
 * A bar whose whisker still crosses zero is drawn faded and dashed ("not
 * sure yet") — as taps accumulate the whiskers shrink and bars turn solid.
 * Hover or focus a row for the exact numbers.
 */
export function PreferenceBars({
  features,
  mu,
  sd,
}: {
  features: string[];
  mu: number[];
  sd: number[];
}) {
  const [hover, setHover] = useState<string | null>(null);
  const index = (key: string) => features.indexOf(key);
  const groups: { title: string; items: FeatureMeta[] }[] = [
    { title: "What you like in the picture", items: FEATURES.filter((f) => f.group === "look") },
    { title: "What the critic measures", items: FEATURES.filter((f) => f.group === "critic") },
  ];
  const hovered = hover ? FEATURES.find((f) => f.key === hover) : null;
  const hi = hovered ? index(hovered.key) : -1;

  return (
    <div>
      <div className="pl-scale" aria-hidden="true">
        <div />
        <div>
          <span>← dislikes</span>
          <span>0</span>
          <span>likes →</span>
        </div>
        <div />
      </div>
      {groups.map((group) => (
        <div key={group.title}>
          <div className="pl-group">{group.title}</div>
          {group.items.map((meta) => {
            const i = index(meta.key);
            if (i < 0) return null;
            const m = mu[i] ?? 0;
            const s = sd[i] ?? 1;
            const { text, level } = describe(meta, m, s);
            const left = Math.min(50, pos(m));
            const width = Math.abs(pos(m) - 50);
            const wl = pos(m - CI_Z * s);
            const wr = pos(m + CI_Z * s);
            return (
              <div
                key={meta.key}
                className={`pl-row${hover === meta.key ? " on" : ""}`}
                tabIndex={0}
                onPointerEnter={() => setHover(meta.key)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(meta.key)}
                onBlur={() => setHover(null)}
              >
                <div className="lbl">
                  <b>
                    {meta.label}
                    <Hint>{meta.desc}</Hint>
                  </b>
                  <span>
                    {meta.neg} ↔ {meta.pos}
                  </span>
                </div>
                <div className="pl-plot" role="img" aria-label={`${meta.label}: ${text}`}>
                  {[-2, -1, 1, 2].map((t) => (
                    <span key={t} className="tick" style={{ left: `${pos(t)}%` }} />
                  ))}
                  <span
                    className={`pl-bar ${m >= 0 ? "pos" : "neg"}${level === 2 ? "" : level === 1 ? " lean" : " unsure"}`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                  />
                  <span className="pl-whisker" style={{ left: `${wl}%`, width: `${wr - wl}%` }} />
                </div>
                <div className={`pl-note${level === 0 ? " muted" : ""}`}>{text}</div>
              </div>
            );
          })}
        </div>
      ))}
      <div className="pl-tip" aria-live="polite" style={{ marginTop: 10 }}>
        {hovered && hi >= 0 ? (
          <>
            <b>{hovered.label}</b> · weight {mu[hi]! >= 0 ? "+" : ""}
            {mu[hi]!.toFixed(2)} · 95% range {(mu[hi]! - CI_Z * sd[hi]!).toFixed(2)} to{" "}
            {(mu[hi]! + CI_Z * sd[hi]!).toFixed(2)}
          </>
        ) : (
          <span style={{ color: "var(--mute)" }}>
            Bar = what it thinks you like · whisker = how sure it is. Hover a row for numbers.
          </span>
        )}
      </div>
    </div>
  );
}
