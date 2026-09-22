"use client";

import { useState } from "react";
import type { Candidate } from "@/lib/api";
import { AXIS_KEYS, FEATURE_BY_KEY } from "@/lib/preference";

/** Quick picks: best-of-3. Each panel is drawn three ways that differ in one
 * thing — warmth, framing or expression — so a single tap ("this one") teaches
 * a direction, and it teaches it twice as fast as picking between two: the
 * pick beats each of the other two. The dashed outline is what is in your
 * strip right now (the app's choice until you tap).
 *
 * Taps are keyed on candidate `.url` (that is what `image_pairs` and the
 * worker's `apply_pair_choice_job` match a pick back to a stored image
 * with), never `.id`. */

type Option = { candidate: Candidate; label: string; isDefault: boolean };
type Panel = { panelId: number; options: Option[]; current: Candidate };

/** The panel's three options are the candidates generated together: they share a seed, or
 * (the older design) use `base + k` for k = 0..2 — a regenerate uses a fresh base far away —
 * so the options are the ones within 2 seeds of the one currently in your strip. */
function buildPanels(candidates: Candidate[]): Panel[] {
  const byPanel = new Map<number, Candidate[]>();
  for (const c of candidates) {
    const list = byPanel.get(c.panel_id) ?? [];
    list.push(c);
    byPanel.set(c.panel_id, list);
  }
  const panels: Panel[] = [];
  for (const [panelId, list] of byPanel) {
    const current = list.find((c) => c.chosen);
    if (!current) continue;
    const batch = list.filter((c) => Math.abs(c.seed - current.seed) <= 2).slice(0, 3);
    if (batch.length < 2) continue;
    panels.push({ panelId, current, options: describeOptions(batch) });
  }
  return panels.sort((a, b) => a.panelId - b.panelId);
}

const KNOBS = ["warmth", "closeness", "expression"] as const;

/** What each option is, in words, relative to the option that is the app's best guess: the knobs
 * it moves, e.g. "warmer + wider". Options are drawn either as one knob stepped either way, or —
 * once it has enough of your picks — as two independent draws from what it has learned, which can
 * differ from the guess on several knobs. Either way the levels are recorded on the candidate
 * (`level_*`); older candidates only recorded the one axis they stepped. */
function describeOptions(batch: Candidate[]): Option[] {
  const level = (c: Candidate, axis: string): number | undefined => c.scores[`level_${axis}`];
  const isCentre = (c: Candidate) => c.scores.offset === 0 || c.scores.role === 0;
  const centre = batch.find(isCentre);
  const options = batch.map((candidate): Option => {
    if (candidate === centre) return { candidate, label: "our best guess", isDefault: true };
    if (centre && KNOBS.every((k) => level(candidate, k) !== undefined)) {
      const words = KNOBS.flatMap((axis) => {
        const diff = (level(candidate, axis) ?? 0) - (level(centre, axis) ?? 0);
        const meta = FEATURE_BY_KEY[axis];
        if (!meta || diff === 0) return [];
        return [`${Math.abs(diff) >= 2 ? "much " : ""}${diff > 0 ? meta.pos : meta.neg}`];
      });
      return { candidate, label: words.join(" + ") || "option", isDefault: false };
    }
    return describeLegacy(candidate);
  });
  // The best guess in the middle, the others either side (ordered by how far the knobs differ).
  const others = options
    .filter((o) => !o.isDefault)
    .sort((a, b) => (a.candidate.scores.offset ?? 0) - (b.candidate.scores.offset ?? 0));
  const middle = options.filter((o) => o.isDefault);
  const [first, ...rest] = others;
  return [...(first ? [first] : []), ...middle, ...rest];
}

/** An older candidate: only the axis it stepped and the step were recorded. */
function describeLegacy(candidate: Candidate): Option {
  const s = candidate.scores;
  const meta = s.axis_id !== undefined ? FEATURE_BY_KEY[AXIS_KEYS[s.axis_id] ?? ""] : undefined;
  const step = s.offset ?? s.axis_level;
  if (!meta || step === undefined) return { candidate, label: "option", isDefault: false };
  if (Math.abs(step) < 1e-9) return { candidate, label: "our best guess", isDefault: true };
  return { candidate, label: step > 0 ? meta.pos : meta.neg, isDefault: false };
}

const RATINGS = [
  { value: 0, label: "off" },
  { value: 1, label: "ok" },
  { value: 2, label: "great" },
] as const;

export function ABPairs({
  candidates,
  onPick,
  onRate,
  savedPicks,
  savedRatings,
}: {
  candidates: Candidate[];
  onPick: (panelId: number, picked: string, others: string[]) => void;
  onRate?: (panelId: number, url: string, rating: number) => void;
  /** What you already picked / rated on this day (from the server), so coming back to a day
   * shows it instead of an untouched picker. */
  savedPicks?: Record<string, string>;
  savedRatings?: Record<string, number>;
}) {
  const [decided, setDecided] = useState<Map<number, string>>(
    () => new Map(Object.entries(savedPicks ?? {}).map(([panel, url]) => [Number(panel), url])),
  );
  const [rated, setRated] = useState<Map<string, number>>(
    () => new Map(Object.entries(savedRatings ?? {})),
  );
  const panels = buildPanels(candidates);

  if (panels.length === 0) return null;

  function pick(panel: Panel, url: string) {
    // A changed mind is allowed: the latest pick for a panel is the one that counts.
    if (decided.get(panel.panelId) === url) return;
    setDecided((prev) => new Map(prev).set(panel.panelId, url));
    onPick(
      panel.panelId,
      url,
      panel.options.map((o) => o.candidate.url).filter((u) => u !== url),
    );
  }

  return (
    <div className="card s8 stack" id="abCard">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Quick picks</h2>
        <span className="mono">
          picked:{" "}
          <b>
            {
              panels.filter((p) =>
                p.options.some((o) => o.candidate.url === decided.get(p.panelId)),
              ).length
            }
          </b>
          /{panels.length}
        </span>
      </div>
      <p style={{ color: "var(--mute)" }}>
        Each panel is drawn three ways. Tap your favourite, since every pick teaches it your taste.
        The dashed one is what&apos;s in your strip now; the middle one is our best guess for you.
      </p>
      <div className="stack" style={{ gap: 18 }}>
        {panels.map((panel) => {
          const savedUrl = decided.get(panel.panelId);
          // A saved pick only shows if it is still one of this panel's options (a regenerate
          // replaces them).
          const chosenUrl = panel.options.some((o) => o.candidate.url === savedUrl)
            ? savedUrl
            : undefined;
          const done = chosenUrl !== undefined;
          return (
            <div key={panel.panelId} className="stack" style={{ gap: 6 }}>
              <b>Panel {panel.panelId}</b>
              <div className="ab three">
                {panel.options.map(({ candidate, label }) => {
                  const isPick = done && candidate.url === chosenUrl;
                  const classes = [
                    "opt",
                    !done && candidate.id === panel.current.id ? "crit" : "",
                    isPick ? "pick" : "",
                    done && !isPick ? "lose" : "",
                  ]
                    .filter(Boolean)
                    .join(" ");
                  return (
                    <button
                      key={candidate.id}
                      type="button"
                      className={classes}
                      onClick={() => pick(panel, candidate.url)}
                      aria-label={`Panel ${panel.panelId}, ${label}: pick this one`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={candidate.url} alt={`Panel ${panel.panelId}, ${label}`} />
                      <div className="m">{label}</div>
                    </button>
                  );
                })}
              </div>
              {onRate && (
                <RatingRow
                  panelId={panel.panelId}
                  // What is in your strip: your pick if you have tapped, else the app's.
                  url={chosenUrl ?? panel.current.url}
                  rated={rated}
                  onRate={(url, value) => {
                    setRated((prev) => new Map(prev).set(url, value));
                    onRate(panel.panelId, url, value);
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** "How good is the one in your strip?" — a rating says whether an image is good AT ALL, which
 * a pick among three can't (the best of three weak options is still weak). */
function RatingRow({
  panelId,
  url,
  rated,
  onRate,
}: {
  panelId: number;
  url: string;
  rated: Map<string, number>;
  onRate: (url: string, value: number) => void;
}) {
  const current = rated.get(url);
  return (
    <div className="row" style={{ gap: 8, alignItems: "center" }}>
      <span className="mono">the one in your strip:</span>
      <div
        role="group"
        aria-label={`Rate panel ${panelId}'s image`}
        className="row"
        style={{ gap: 6 }}
      >
        {RATINGS.map((r) => (
          <button
            key={r.value}
            type="button"
            className={`btn sm${current === r.value ? " pri" : ""}`}
            aria-pressed={current === r.value}
            onClick={() => onRate(url, r.value)}
          >
            {r.label}
          </button>
        ))}
      </div>
    </div>
  );
}
