"use client";

import { type CSSProperties, useId, useState } from "react";
import type { Candidate } from "@/lib/api";
import { AXIS_KEYS, FEATURE_BY_KEY } from "@/lib/preference";

/**
 * One candidate image. Hover (or focus / tap) shows a small table of everything
 * the critic and the preference model measured, grouped and labelled, instead
 * of a run-on line of `key value · key value` under every image.
 */

type Row = { label: string; value: string };
type Group = { title: string; rows: Row[] };

const CRITIC: [string, string][] = [
  ["identity", "Looks like you"],
  ["style", "Comic style match"],
  ["alignment", "Matches the caption"],
  ["detail", "Background detail"],
  ["reward", "Overall reward"],
];
const MEASURED: [string, string][] = [
  ["warmth", "Color warmth"],
  ["brightness", "Brightness"],
  ["saturation", "Color intensity"],
  ["contrast", "Contrast"],
  ["closeness", "Framing (face size)"],
];
const HANDLED = new Set([
  ...CRITIC.map(([k]) => k),
  ...MEASURED.map(([k]) => k),
  "axis_id",
  "axis_level",
  "expression",
  "explored",
  "text_detected",
]);

const num = (v: number, signed = false) => `${signed && v > 0 ? "+" : ""}${v.toFixed(2)}`;
const yesNo = (v: number) => (v >= 0.5 ? "yes" : "no");

/** "warmer (+1)" / "cooler (−1)" / "baseline" for a variation level on a feature. */
function levelWords(featureKey: string, level: number): string {
  const meta = FEATURE_BY_KEY[featureKey];
  if (!meta || level === 0) return "baseline";
  return `${level > 0 ? meta.pos : meta.neg} (${level > 0 ? "+" : "−"}${Math.abs(level)})`;
}

/** Short word for the variation this candidate carries, shown under the image. */
export function variationWord(c: Candidate): string | null {
  const axisId = c.scores.axis_id;
  const level = c.scores.axis_level;
  if (axisId === undefined || level === undefined) return null;
  const key = AXIS_KEYS[axisId];
  const meta = key ? FEATURE_BY_KEY[key] : undefined;
  if (!meta || level === 0) return "baseline";
  return level > 0 ? meta.pos : meta.neg;
}

export function candidateGroups(c: Candidate): Group[] {
  const s = c.scores;
  const groups: Group[] = [];

  const critic = CRITIC.filter(([k]) => k in s).map(([k, label]) => ({ label, value: num(s[k]!) }));
  if (critic.length) groups.push({ title: "Critic scores", rows: critic });

  const measured = MEASURED.filter(([k]) => k in s).map(([k, label]) => ({
    label,
    value: num(s[k]!, k === "warmth"),
  }));
  if (measured.length) groups.push({ title: "Measured in the image", rows: measured });

  const variation: Row[] = [];
  if (s.axis_id !== undefined && s.axis_level !== undefined) {
    const key = AXIS_KEYS[s.axis_id];
    if (key) {
      variation.push({
        label: "This panel varied",
        value: FEATURE_BY_KEY[key]!.label.toLowerCase(),
      });
      variation.push({ label: "This option was", value: levelWords(key, s.axis_level) });
    }
  }
  if (s.expression && s.axis_id !== 2) {
    variation.push({ label: "Expression asked", value: levelWords("expression", s.expression) });
  }
  if (variation.length) groups.push({ title: "How the options differ", rows: variation });

  const flags: Row[] = [{ label: "Chosen for the strip", value: c.chosen ? "yes" : "no" }];
  if (c.chosen && "explored" in s) {
    flags.push({ label: "Exploratory pick", value: yesNo(s.explored!) });
  }
  if ("text_detected" in s)
    flags.push({ label: "Text found in image", value: yesNo(s.text_detected!) });
  groups.push({ title: "Status", rows: flags });

  const other = Object.entries(s)
    .filter(([k]) => !HANDLED.has(k))
    .map(([k, v]) => ({ label: k, value: num(v) }));
  if (other.length) groups.push({ title: "Other", rows: other });

  return groups;
}

export function CandidateCard({ candidate }: { candidate: Candidate }) {
  const popId = useId();
  const word = variationWord(candidate);
  const [pos, setPos] = useState<{ left: number; top: number; dock: "top" | "bottom" | null }>({
    left: 0,
    top: 0,
    dock: null,
  });

  /** Where the score card opens (measured when it opens, in viewport
   * coordinates, because where a photo sits changes with screen width):
   *  1. to the top-RIGHT of the whole row of options, if there's room;
   *  2. otherwise to the top-LEFT of the row — both spots are empty, so the
   *     card never covers a photo (its top lines up with the hovered image);
   *  3. only when neither side has room (tablets, phones): a compact panel
   *     docked to the top or bottom of the screen — whichever half does NOT
   *     contain the image you're inspecting, so that image stays visible. */
  function place(el: HTMLElement) {
    const pop = el.querySelector<HTMLElement>(".cand-pop");
    if (!pop) return;
    const row = el.closest<HTMLElement>(".cands") ?? el;
    const img = el.getBoundingClientRect();
    // The row container stretches to the card's full width, so measure the
    // photos themselves: the union of every option's box in this row.
    const boxes = [...row.querySelectorAll<HTMLElement>(".cand-img")].map((n) =>
      n.getBoundingClientRect(),
    );
    const rowBox = {
      left: Math.min(...boxes.map((b) => b.left)),
      right: Math.max(...boxes.map((b) => b.right)),
      top: Math.min(...boxes.map((b) => b.top)),
      bottom: Math.max(...boxes.map((b) => b.bottom)),
    };
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const GAP = 14;
    const M = 12;
    const w = Math.min(262, vw - 2 * M);
    const h = pop.offsetHeight;
    const clampY = (y: number) => Math.max(M, Math.min(y, vh - h - M));

    if (rowBox.right + GAP + w + M <= vw) {
      setPos({ left: rowBox.right + GAP, top: clampY(img.top), dock: null });
    } else if (rowBox.left - GAP - w >= M) {
      setPos({ left: rowBox.left - GAP - w, top: clampY(img.top), dock: null });
    } else {
      const imageInLowerHalf = img.top + img.height / 2 > vh / 2;
      setPos({ left: 0, top: 0, dock: imageInLowerHalf ? "top" : "bottom" });
    }
  }

  return (
    <div className={`cand${candidate.chosen ? " chosen" : ""}`}>
      <div
        className="cand-img"
        tabIndex={0}
        aria-describedby={popId}
        style={{ "--pop-left": `${pos.left}px`, "--pop-top": `${pos.top}px` } as CSSProperties}
        onPointerEnter={(e) => place(e.currentTarget)}
        onFocus={(e) => place(e.currentTarget)}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={candidate.url}
          alt={`Candidate${word ? `, ${word}` : ""}${candidate.chosen ? ", chosen" : ""}`}
        />
        <div className="cand-pop" role="tooltip" id={popId} data-dock={pos.dock ?? undefined}>
          {candidateGroups(candidate).map((g) => (
            <table key={g.title}>
              <caption>{g.title}</caption>
              <tbody>
                {g.rows.map((r) => (
                  <tr key={r.label}>
                    <th scope="row">{r.label}</th>
                    <td>{r.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
        </div>
      </div>
      <div className="cand-tags">
        {candidate.chosen && <span className="pill ok">chosen</span>}
        {word && <span className="pill">{word}</span>}
      </div>
    </div>
  );
}
