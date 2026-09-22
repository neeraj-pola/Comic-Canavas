"use client";

import { useEffect, useState } from "react";
import type { KnobModel, PreferenceLearning as PreferenceData } from "@/lib/api";
import { AXIS_KEYS, FEATURES, FEATURE_BY_KEY, pct, shortDate } from "@/lib/preference";
import { AccuracyChart } from "./AccuracyChart";
import { ExploreChart } from "./ExploreChart";
import { Hint } from "./Hint";
import { KnobCurves } from "./KnobCurves";
import { LastTap } from "./LastTap";
import { PickRateChart } from "./PickRateChart";
import { PosteriorChart } from "./PosteriorChart";
import { PreferenceBars } from "./PreferenceBars";
import { TasteNotes } from "./TasteNotes";
import "./preference.css";

/**
 * "How it's learning your taste" — the live view of the per-user preference
 * model. Every A/B tap re-trains a small Bayesian model over the features of
 * the image; this page replays that: what it currently believes (bars, with
 * how sure it is), what the last tap taught it, whether it is actually
 * predicting you better than the hand-set weights, and how much it is still
 * exploring. Scrub or play the slider to watch the beliefs form tap by tap.
 */
export function PreferenceLearning({ data }: { data: PreferenceData }) {
  const { snapshots, n_taps: nTaps, active, accuracy, lean } = data;
  const minTaps = accuracy.min_taps ?? 20;
  const warmup = accuracy.warmup ?? 5;
  const features = data.features.length ? data.features : FEATURES.map((f) => f.key);

  const [picked, setPicked] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const last = snapshots.length - 1;
  const at = picked === null ? last : Math.min(picked, last);
  const snap = at >= 0 ? snapshots[at] : null;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setPicked((p) => {
        const next = (p === null || p >= last ? -1 : p) + 1;
        if (next >= last) setPlaying(false);
        return next;
      });
    }, 450);
    return () => clearInterval(id);
  }, [playing, last]);

  // The person's defaults: the level each style knob is baked in at (-2..2, 0 = plain prompt).
  const leaning = AXIS_KEYS.map((axis) => ({ axis, dir: lean[axis] ?? 0 }))
    .filter(({ dir }) => dir !== 0)
    .map(({ axis, dir }) => {
      const meta = FEATURE_BY_KEY[axis]!;
      return `${Math.abs(dir) >= 2 ? "much " : ""}${dir > 0 ? meta.pos : meta.neg}`;
    });
  const knobs = "curves" in data.knobs ? (data.knobs as KnobModel) : null;
  // Where new panels are centred: the confident defaults plus the softer best guess.
  const guessWords = AXIS_KEYS.map((axis) => ({ axis, dir: knobs?.guess?.[axis] ?? 0 }))
    .filter(({ dir }) => dir !== 0)
    .map(({ axis, dir }) => {
      const meta = FEATURE_BY_KEY[axis]!;
      return `${Math.abs(dir) >= 2 ? "much " : ""}${dir > 0 ? meta.pos : meta.neg}`;
    });
  const defaultPick = accuracy.default_pick ?? null;
  const chance = accuracy.default_pick_chance ?? 1 / 3;

  if (nTaps === 0) {
    return (
      <div className="pl-empty">
        <div className="kick">Waiting for your first tap</div>
        <h3>Nothing to show yet, and that&apos;s the point</h3>
        <p>
          Each panel gets three options that differ in one thing: <b>warmth</b>, <b>framing</b> or{" "}
          <b>expression</b>. Tap your favourite under <i>Today → Quick picks</i>. After every pick
          this page updates: the bars show what it thinks you like, and how sure it is.
        </p>
        <p>
          Around {minTaps} taps in, it starts trying to beat the default ranking on your picks, and
          only takes over if it actually does.
        </p>
      </div>
    );
  }

  /** Rough 95% margin (in fraction) on a hit rate measured over `n` taps. */
  const margin = (p: number, n: number) => 1.96 * Math.sqrt((p * (1 - p)) / Math.max(n, 1));
  const fmtEdge = (e: number) => `${e > 0 ? "+" : e < 0 ? "−" : ""}${Math.abs(e)}`;
  const learnedAcc = accuracy.learned ?? null;
  const handAcc = accuracy.hand ?? null;
  const status = active
    ? "Active: it's choosing for you"
    : nTaps < minTaps
      ? `Learning: ${minTaps - nTaps} more taps until it can take over`
      : "Watching: not clearly ahead of the default yet";

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="pl-stats">
        <div className="pl-stat">
          <span className="mono">
            Taps learned from
            <Hint>
              Every quick pick counts once, even a best-of-three tap, since it teaches the model
              twice: your pick beat each of the other two options. Regenerated panels and caption
              edits aren&apos;t counted here.
            </Hint>
          </span>
          <div className="big">{nTaps}</div>
          <div className="pl-meter" role="img" aria-label={`${nTaps} of ${minTaps} taps`}>
            <i style={{ width: `${Math.min(100, (nTaps / minTaps) * 100)}%` }} />
          </div>
          <span className={`pill ${active ? "ok" : "mus"}`} style={{ width: "fit-content" }}>
            {status}
          </span>
        </div>
        <div className="pl-stat">
          <span className="mono">
            Is it beating the default ranking?
            {accuracy.n_eval ? (
              <Hint>
                <span>
                  learned won {accuracy.learned_wins ?? 0} calls the default lost · default won{" "}
                  {accuracy.hand_wins ?? 0} · net {fmtEdge(accuracy.edge ?? 0)} (needs +
                  {accuracy.min_edge ?? 3} to take over)
                </span>
                <span>
                  ±{Math.round(margin(learnedAcc ?? 0.5, accuracy.n_eval) * 100)} points of noise
                  on this many taps, too few to call it either way
                </span>
                <span>
                  {nTaps} taps total, minus the first {warmup} used only to warm-start it ={" "}
                  {accuracy.n_eval} evaluated
                </span>
              </Hint>
            ) : null}
          </span>
          <div className="big">
            {learnedAcc === null ? "N/A" : pct(learnedAcc)}{" "}
            <span style={{ fontSize: 16, fontWeight: 600 }}>
              vs {handAcc === null ? "N/A" : pct(handAcc)}
            </span>
          </div>
          <span>
            learned model vs default ranking, on <b>{accuracy.n_eval ?? 0}</b> taps it predicted
            before seeing them
          </span>
          {!accuracy.n_eval && <span className="mono">counted after the first {warmup} taps</span>}
        </div>
        <div className="pl-stat">
          <span className="mono">
            You picked the default option
            <Hint>
              The default is the middle, unmodified option: no step in either direction from what
              it currently believes you&apos;d like. Picking it more often than chance means its
              guess is genuinely close to your taste, not just lucky.
            </Hint>
          </span>
          <div className="big">{defaultPick === null ? "N/A" : pct(defaultPick)}</div>
          <span>
            of your <b>{accuracy.default_pick_n ?? 0}</b> picks; picking at random would be{" "}
            {pct(chance)}
          </span>
          <span className="mono">rises as its defaults get closer to your taste</span>
        </div>
        <div className="pl-stat">
          <span className="mono">
            Its defaults for you
            <Hint>
              Once it&apos;s very sure (from a lot of consistent picks, not a lucky streak) it
              nudges a knob&apos;s default one level at a time, and lets it fade back if your picks
              stop supporting it.
            </Hint>
          </span>
          {leaning.length > 0 ? (
            <div className="pl-chips">
              {leaning.map((w) => (
                <span key={w} className="pill mus" style={{ fontSize: 14 }}>
                  {w}
                </span>
              ))}
            </div>
          ) : (
            <div style={{ color: "var(--mute)" }}>Plain prompts: it isn&apos;t sure enough yet.</div>
          )}
          <span className="mono">
            {leaning.length > 0
              ? "baked into every new panel's prompt"
              : `confident defaults start after ${knobs?.min_taps ?? 20} picks`}
          </span>
          <span className="mono">
            middle option of each panel = best guess:{" "}
            <b>{guessWords.length > 0 ? guessWords.join(", ") : "plain"}</b>
          </span>
        </div>
      </div>

      <div className="card stack" style={{ padding: 16 }}>
        <div className="pl-replay">
          <button
            type="button"
            className="btn sm"
            onClick={() => {
              if (!playing && (picked === null || picked >= last)) setPicked(0);
              setPlaying((p) => !p);
            }}
            aria-label={playing ? "Pause replay" : "Replay how it learned"}
          >
            {playing ? "❚❚ Pause" : "▶ Replay"}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(last, 0)}
            value={at}
            onChange={(e) => {
              setPlaying(false);
              setPicked(Number(e.target.value));
            }}
            aria-label="Move through your taps"
          />
          <span className="mono">
            tap <b>{snap?.tap ?? 0}</b> of {snapshots.length}
            {snap?.date ? ` · ${shortDate(snap.date)}` : ""}
          </span>
        </div>
      </div>

      <div className="grid">
        <div className="card s7 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>What it thinks you like</h2>
            <span className="mono">after tap {snap?.tap ?? 0}</span>
          </div>
          <PreferenceBars
            features={features}
            mu={snap?.mu ?? features.map(() => 0)}
            sd={snap?.sd ?? features.map(() => 1)}
          />
        </div>
        <div className="s5" style={{ display: "grid", gap: 18, alignContent: "start" }}>
          <div className="card stack">
            <h2>{at === last ? "Your latest tap" : `Tap ${snap?.tap ?? ""}`}</h2>
            {snap ? <LastTap snapshot={snap} features={features} /> : null}
          </div>
          <div className="card stack">
            <h2>
              Still exploring?
              <Hint>
                Sometimes it deliberately shows an option away from its best guess, to check
                whether that guess is actually right. Without this, it could lock onto a wrong
                belief and never find out.
              </Hint>
            </h2>
            <ExploreChart days={data.explore_by_day} active={active} />
          </div>
        </div>
        {knobs && (
          <div className="card s12 stack">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>
                Your sweet spot, knob by knob
                <Hint>
                  Three small charts, one per style knob it can adjust (warmth, framing,
                  expression). Each is learned only from taps where a panel actually varied that
                  particular knob, so they fill in at different speeds.
                </Hint>
              </h2>
              <span className="mono">learned from {knobs.n_taps} picks</span>
            </div>
            <p style={{ color: "var(--mute)", maxWidth: "78ch" }}>
              For each style knob, how much you like each level compared with <i>as usual</i>. The
              line is its best guess and the band is how unsure it is. A peak in the middle is a
              sweet spot: &ldquo;warm, but not too warm&rdquo;. The yellow dot is the default it
              uses now; a dashed ring is where it thinks you&apos;d like it better but isn&apos;t
              yet sure enough to move.
            </p>
            <KnobCurves knobs={knobs} lean={lean} />
          </div>
        )}
        <div className="card s7 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              Do its defaults suit you?
              <Hint>
                Per day, the share of your picks that were the default (unmodified) option. It
                should climb over time as its defaults get closer to your real taste.
              </Hint>
            </h2>
            <span className="mono">share of picks that were the default option</span>
          </div>
          <PickRateChart
            series={data.default_pick_by_day}
            label="Default option picked"
            chance={chance}
          />
        </div>
        <div className="card s5 stack">
          <h2>What it has written about you</h2>
          <TasteNotes notes={data.notes} />
        </div>
        <div className="card s12 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              The bell curve behind each bar
              <Hint>
                Every bar in &ldquo;What it thinks you like&rdquo; is really one of these: pick any
                feature below to see its own curve. Hover a feature chip for what it measures.
              </Hint>
            </h2>
            <span className="mono">after tap {snap?.tap ?? 0}</span>
          </div>
          <p style={{ color: "var(--mute)", maxWidth: "78ch" }}>
            Every bar is the middle of a bell curve: what the model believes about how much you care
            about a feature. <b style={{ color: "var(--ink)" }}>Narrow and tall</b> means it&apos;s
            sure, <b style={{ color: "var(--ink)" }}>wide and flat</b> means it isn&apos;t. The
            share of the curve right of zero is the chance you like <i>more</i> of it. Every curve
            has the same total area, so a narrower curve has to be{" "}
            <b style={{ color: "var(--ink)" }}>taller</b>. Watch the height grow as you replay.
          </p>
          <PosteriorChart snapshots={snapshots} features={features} at={at} />
        </div>
        <div className="card s12 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              Is it predicting you?
              <Hint>
                A rolling hit rate over your last 10 taps: how often the learned model&apos;s top
                pick matched yours, next to the same for the hand-set default. Only counted after
                the first {warmup} taps, once there&apos;s enough signal to predict from.
              </Hint>
            </h2>
            <span className="mono">running hit rate, last 10 taps</span>
          </div>
          <AccuracyChart
            snapshots={snapshots}
            selected={at}
            onSelect={(i) => {
              setPlaying(false);
              setPicked(i);
            }}
            minTaps={minTaps}
            warmup={warmup}
          />
        </div>
      </div>

      <details>
        <summary className="mono" style={{ cursor: "pointer" }}>
          How this works
        </summary>
        <p style={{ marginTop: 8 }}>
          Each panel&apos;s three options differ on one axis (warmth, framing or expression), so
          your pick says which <i>direction</i> you like, and beats each of the other two. Two small
          Bayesian models (Bradley-Terry with a Gaussian posterior) re-learn after every pick, on
          the server, for free: one over the measured look of the image, which only replaces the
          default ranking once it has beaten it on picks it predicted <i>before</i> seeing them; and
          one over the three style knobs, which moves your <i>defaults</i> one level at a time, only
          when it is very sure, and lets them fade back if your taste stops supporting them. Older
          picks count for less (half as much after about 80 more picks), so a change of taste is
          followed, not outvoted.
        </p>
      </details>
    </div>
  );
}
