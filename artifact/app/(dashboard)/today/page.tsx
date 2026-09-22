"use client";

import { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuth, useUser } from "@/lib/auth";
import {
  createDay,
  getDay,
  getLibrary,
  latestDemoDate,
  regeneratePanel,
  submitCaptionFeedback,
  submitPick,
  submitRating,
  type Day,
} from "@/lib/api";
import { streamJobEvents } from "@/lib/sse";
import { isoWeekDates, isoWeekOf, shiftIsoWeek } from "@/lib/isoWeek";
import { REAL_STAGES, Stages, type StageStatus } from "@/components/today/Stages";
import { Strip } from "@/components/today/Strip";
import { ABPairs } from "@/components/today/ABPairs";
import "@/components/dashboard/bento.css";

/**
 * Ported from `renderToday()`. The stage ticker shows the real 8 graph
 * nodes (see `Stages.tsx`), "Regenerate with a note" is per-panel (the
 * real API has no whole-strip regenerate endpoint), and voice recording
 * has no UI anywhere in this app — ASR isn't built, so a feature with no
 * backend shouldn't have a standing UI affordance.
 */

const dayLabel = (iso: string, opts: Intl.DateTimeFormatOptions) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, opts);

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

export default function TodayPage() {
  const { getToken } = useAuth();
  const { user } = useUser();
  // Public demo: anchored to the latest real generated day, not the visitor's
  // own real calendar date (which this demo's fixed dataset has nothing for),
  // so the very first thing shown is real output, not an empty entry box.
  const today = useMemo(() => latestDemoDate(), []);
  // The strip shows one week at a time; the arrows move it (0 = this week, -1 = last week...).
  const [weekOffset, setWeekOffset] = useState(0);
  const week = useMemo(
    () => isoWeekDates(shiftIsoWeek(isoWeekOf(today), weekOffset)),
    [weekOffset, today],
  );
  // Any day can be written up (to backfill a week for the weekly recap).
  // Generation state is kept per day so the next day can be started while
  // one is still drawing.
  const [date, setDate] = useState(today);
  const dateRef = useRef(today);
  const [day, setDay] = useState<Day | null>(null);
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [text, setText] = useState("");
  const drafts = useRef<Record<string, string>>({});
  const [running, setRunning] = useState<Set<string>>(new Set());
  const [stages, setStages] = useState<Record<string, Record<string, StageStatus>>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [doneDates, setDoneDates] = useState<Set<string>>(new Set());
  const controllers = useRef<Set<AbortController>>(new Set());
  const [regenerating, setRegenerating] = useState<Set<string>>(new Set()); // `${date}:${panelId}`

  const isToday = date === today;
  const isRunning = running.has(date);
  const stageStatuses = stages[date] ?? {};
  const error = errors[date] ?? null;

  const loadDay = useCallback(
    async (d: string) => {
      const token = await getToken();
      if (!token) return;
      const result = await getDay(token, d);
      if (dateRef.current === d) setDay(result); // the person may have moved on meanwhile
      return result;
    },
    [getToken],
  );

  /** Which days of this week already have a comic (the dots on the week strip). */
  const refreshDots = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    const months = [...new Set(week.map((d) => d.slice(0, 7)))];
    const found = new Set<string>();
    for (const m of months) {
      const { days } = await getLibrary(token, m);
      for (const d of days) if (d.strip_url) found.add(d.date);
    }
    setDoneDates(found);
  }, [getToken, week]);

  useEffect(() => {
    void refreshDots();
  }, [refreshDots]);

  useEffect(() => {
    dateRef.current = date;
    setDay(null);
    let cancelled = false;
    loadDay(date)
      .then((result) => {
        if (cancelled) return;
        setText(drafts.current[date] ?? result?.text ?? "");
      })
      .finally(() => !cancelled && setLoadingInitial(false));
    return () => {
      cancelled = true;
    };
  }, [date, loadDay]);

  useEffect(() => {
    const live = controllers.current;
    return () => live.forEach((c) => c.abort());
  }, []);

  function selectDate(next: string) {
    drafts.current[date] = text;
    setDate(next);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const d = date; // this run belongs to the day it was started on
    setErrors((prev) => ({ ...prev, [d]: "" }));
    const token = await getToken();
    if (!token) return;
    setRunning((prev) => new Set(prev).add(d));
    setStages((prev) => ({ ...prev, [d]: {} }));
    const controller = new AbortController();
    controllers.current.add(controller);
    try {
      const { job_id } = await createDay(token, { text, date: d });
      await streamJobEvents(
        job_id,
        token,
        (evt) => {
          if (evt.event === "node") {
            const node = String(evt.data.node);
            setStages((prev) => {
              const next = { ...(prev[d] ?? {}) };
              const idx = REAL_STAGES.indexOf(node as (typeof REAL_STAGES)[number]);
              REAL_STAGES.forEach((s, i) => {
                if (i < idx) next[s] = "done";
              });
              next[node] = "done";
              return { ...prev, [d]: next };
            });
          } else if (evt.event === "done" && evt.data.status === "failed") {
            setErrors((prev) => ({ ...prev, [d]: String(evt.data.error ?? "The job failed.") }));
          }
        },
        controller.signal,
      );
      await loadDay(d);
      void refreshDots();
    } catch (err) {
      setErrors((prev) => ({
        ...prev,
        [d]: err instanceof Error ? err.message : "Something went wrong.",
      }));
    } finally {
      controllers.current.delete(controller);
      setRunning((prev) => {
        const next = new Set(prev);
        next.delete(d);
        return next;
      });
    }
  }

  /** Save a caption. Throws on failure so the panel can show why. The strip
   * file is re-composed by the worker, so refresh the day shortly after. */
  async function handleCaptionSave(panelId: number, original: string, edited: string) {
    const token = await getToken();
    if (!token) throw new Error("You're signed out. Sign in again to save.");
    await submitCaptionFeedback(token, { date, panel_id: panelId, original, edited });
    setDay((d) =>
      d
        ? {
            ...d,
            panels: d.panels.map((p) => (p.id === panelId ? { ...p, caption_a: edited } : p)),
          }
        : d,
    );
    setTimeout(() => void loadDay(date), 2500);
  }

  async function handleRate(panelId: number, url: string, rating: number) {
    const token = await getToken();
    if (!token) return;
    await submitRating(token, { date, panel_id: panelId, url, rating });
  }

  async function handlePairPick(panelId: number, picked: string, others: string[]) {
    const token = await getToken();
    if (!token) return;
    await submitPick(token, { date, panel_id: panelId, picked, others });
    // A tap also re-composes the strip with the tapped candidate
    // (`apply_pair_choice_job`) — no external API call needed on the
    // worker side (the image already exists), so this finishes much
    // faster than a real regenerate; a short delay before reload is
    // enough.
    setTimeout(() => void loadDay(date), 1500);
  }

  /** Regenerate one panel, then WAIT for the new picture. The old version
   * reloaded once after a fixed 4 seconds, but a regenerate draws new candidates
   * and scores them — closer to a minute — so the person saw nothing change. We
   * poll until the panel's chosen image is a different one (new seeds every
   * press, so it always is), with a clear message if it takes too long. */
  async function handleRegenerate(panelId: number, note: string) {
    const d = date;
    const key = `${d}:${panelId}`;
    const token = await getToken();
    if (!token) return;
    const chosenId = (x: Day | null) =>
      x?.candidates.find((c) => c.panel_id === panelId && c.chosen)?.id ?? null;
    const before = chosenId(day);
    setErrors((prev) => ({ ...prev, [d]: "" }));
    setRegenerating((prev) => new Set(prev).add(key));
    try {
      await regeneratePanel(token, d, panelId, note);
      const started = Date.now();
      while (Date.now() - started < 240_000) {
        await new Promise((r) => setTimeout(r, 3000));
        const fresh = await getToken();
        if (!fresh) return;
        const latest = await getDay(fresh, d);
        if (chosenId(latest) && chosenId(latest) !== before) {
          if (dateRef.current === d) setDay(latest);
          void refreshDots();
          return;
        }
      }
      setErrors((prev) => ({
        ...prev,
        [d]: "Regenerating is taking longer than usual. It may still finish; reload in a minute.",
      }));
    } catch (err) {
      setErrors((prev) => ({
        ...prev,
        [d]: err instanceof Error ? err.message : "Could not regenerate that panel.",
      }));
    } finally {
      setRegenerating((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  }

  if (loadingInitial) return null;

  const firstName = user?.firstName ?? "there";
  const done = Boolean(day?.strip_url);

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">{date}</div>
          <h1>
            {isToday
              ? `${greeting()}, ${firstName}.`
              : dayLabel(date, { weekday: "long", month: "long", day: "numeric" })}
          </h1>
        </div>
        <p>
          Tell it about your day in a few lines. Then a few quick taps teach it your taste.
          {!isToday && " You're writing up a past day."}
        </p>
      </div>
      <div className="grid">
        <div className="card s5 stack">
          <h2>
            {isToday
              ? "Today's entry"
              : `Entry for ${dayLabel(date, { month: "short", day: "numeric" })}`}
          </h2>
          <div className="stack" style={{ gap: 6 }}>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
              <span className="mono">
                {weekOffset === 0
                  ? "This week"
                  : `${dayLabel(week[0]!, { month: "short", day: "numeric" })} to ${dayLabel(week[6]!, { month: "short", day: "numeric" })}`}{" "}
                : pick a day (● has a comic)
              </span>
              <div className="row" style={{ gap: 6 }}>
                <button
                  type="button"
                  className="btn sm"
                  aria-label="Previous week"
                  onClick={() => setWeekOffset((o) => o - 1)}
                >
                  ← Earlier
                </button>
                <button
                  type="button"
                  className="btn sm"
                  aria-label="Next week"
                  disabled={weekOffset >= 0}
                  onClick={() => setWeekOffset((o) => o + 1)}
                >
                  Later →
                </button>
              </div>
            </div>
            <div
              role="group"
              aria-label="Pick a day of the shown week"
              style={{ display: "flex", flexWrap: "wrap", gap: 6 }}
            >
              {week.map((d) => (
                <button
                  key={d}
                  type="button"
                  className={`btn sm${d === date ? " pri" : ""}`}
                  aria-pressed={d === date}
                  disabled={d > today}
                  onClick={() => selectDate(d)}
                >
                  {dayLabel(d, { weekday: "short" })} {Number(d.slice(8))}
                  {doneDates.has(d) ? " ●" : running.has(d) ? " …" : ""}
                </button>
              ))}
            </div>
            <label className="f" style={{ maxWidth: 220 }}>
              Or another date
              <input
                type="date"
                max={today}
                value={date}
                onChange={(e) => e.target.value && selectDate(e.target.value)}
              />
            </label>
          </div>
          <form onSubmit={handleSubmit}>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Woke up late, fixed a bug I'd been stuck on, coffee with a friend…"
              disabled={isRunning}
            />
            <div className="row">
              <button type="submit" className="btn pri" disabled={isRunning || !text.trim()}>
                {done
                  ? isToday
                    ? "Redo today's strip"
                    : "Redo this day's strip"
                  : isToday
                    ? "Make today's comic"
                    : "Make this day's comic"}
              </button>
            </div>
          </form>
          {(isRunning || done) && <Stages statuses={stageStatuses} />}
          {error && <div className="err">{error}</div>}
        </div>
        <div className="card s7 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              {done
                ? isToday
                  ? "Today's strip"
                  : "This day's strip"
                : "Your strip will appear here"}
            </h2>
            {day?.mood && <span className="pill mus">mood: {day.mood}</span>}
          </div>
          {done && day ? (
            <>
              <Strip
                panels={day.panels}
                candidates={day.candidates}
                regenerating={
                  new Set(
                    day.panels.filter((p) => regenerating.has(`${date}:${p.id}`)).map((p) => p.id),
                  )
                }
                onCaptionSave={handleCaptionSave}
                onRegenerate={handleRegenerate}
              />
              <p className="mono">Click a caption to edit it, then Save. Edits train the writer.</p>
              {day.strip_url && (
                <a className="btn sm dark" href={day.strip_url} download>
                  Download strip
                </a>
              )}
            </>
          ) : (
            <p style={{ color: "var(--mute)" }}>
              Four panels, in your style, with you in frame. Quiet day? It draws two.
            </p>
          )}
        </div>
        {done && day && (
          <ABPairs
            key={date}
            candidates={day.candidates}
            onPick={handlePairPick}
            onRate={handleRate}
            savedPicks={day.picks}
            savedRatings={day.ratings}
          />
        )}
      </div>
    </>
  );
}
