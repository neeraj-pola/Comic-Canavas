"use client";

import { AXIS_KEYS, FEATURE_BY_KEY, pct, shortDate } from "@/lib/preference";
import type { PreferenceSnapshot } from "@/lib/api";
import { Hint } from "./Hint";

/**
 * One tap, explained: the two images, what the model predicted for your pick
 * BEFORE it saw the answer, and the differences that mattered — "you picked
 * the warmer one, so warmth moved up".
 */
export function LastTap({
  snapshot,
  features,
}: {
  snapshot: PreferenceSnapshot;
  features: string[];
}) {
  const called = snapshot.correct > 0.5;
  const passed = snapshot.rejected_urls?.length ? snapshot.rejected_urls : [snapshot.rejected_url];
  const drivers = features
    .map((key, i) => ({ key, d: snapshot.delta[i] ?? 0 }))
    .filter(({ key }) => FEATURE_BY_KEY[key])
    .sort((a, b) => Math.abs(b.d) - Math.abs(a.d))
    .slice(0, 3)
    .filter(({ d }) => Math.abs(d) >= 0.15);
  const axis = snapshot.axis != null ? FEATURE_BY_KEY[AXIS_KEYS[snapshot.axis] ?? ""] : null;

  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="pl-pair" style={{ gridTemplateColumns: `repeat(${passed.length + 1}, 1fr)` }}>
        <figure className="won">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={snapshot.chosen_url} alt="The image you picked" />
          <figcaption>you picked</figcaption>
        </figure>
        {passed.map((url) => (
          <figure key={url}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={url} alt="An image you passed on" />
            <figcaption>you passed</figcaption>
          </figure>
        ))}
      </div>
      <p>
        <b>Tap {snapshot.tap}</b>
        {snapshot.date ? ` · ${shortDate(snapshot.date)}` : ""} · panel {snapshot.panel_id}. Before
        seeing your answer it gave your pick <b>{pct(snapshot.prob)}</b>
        <Hint>
          This is how likely the model thought your actual pick was, just for this one tap. It is
          not the same as the accuracy stat above, which is the average hit rate across many taps.
          A single tap can be a correct call (it called it) at any confidence over 50%, even a
          close one like this.
        </Hint>
        . {called ? "It called it." : "It missed this one, and learned from it."}
      </p>
      {drivers.length > 0 ? (
        <div className="pl-chips">
          {drivers.map(({ key, d }) => {
            const meta = FEATURE_BY_KEY[key]!;
            return (
              <span key={key} className="pill mus" title={`${d > 0 ? "+" : ""}${d.toFixed(2)}`}>
                {d > 0 ? meta.pos : meta.neg}
              </span>
            );
          })}
        </div>
      ) : (
        <p style={{ color: "var(--mute)" }}>These options were very alike: little to learn.</p>
      )}
      {axis && (
        <span className="mono">
          These options differed mainly in <b>{axis.label.toLowerCase()}</b>.
        </span>
      )}
    </div>
  );
}
