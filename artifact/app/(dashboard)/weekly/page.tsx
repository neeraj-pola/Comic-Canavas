"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import {
  downloadWeeklyPdf,
  generateWeekly,
  getLibrary,
  getWeekly,
  latestDemoWeek,
  type LibraryDay,
  type WeeklyContext,
} from "@/lib/api";
import { isoWeekDates, shiftIsoWeek } from "@/lib/isoWeek";
import { streamJobEvents } from "@/lib/sse";
import { DownloadIcon } from "@/components/dashboard/icons";
import { WeeklyBook, type WeeklyBookPage } from "@/components/weekly/WeeklyBook";
import "@/components/dashboard/bento.css";
import "@/components/weekly/weekly.css";

/**
 * Ported from `renderWeekly()`, wired to the real `GET /weekly/{iso_week}`
 * (Playwright-rendered PDF). `GET /weekly/{iso_week}.pdf` needs a bearer
 * token too, which a plain `<a href>` download can't send — `downloadWeeklyPdf`
 * fetches with auth and triggers a blob download instead (`lib/api.ts`).
 * The mock's mood sparkline needs 0..1 numeric values per day, but the
 * real API (`mood_series`) returns mood words, not numbers — there's no
 * honest way to plot those as a line chart, so this shows the real words
 * as a row instead of fabricating a numeric scale. `/weeks` (an archive
 * of past weeks) has no backing endpoint in the real API — kept as a
 * minimal, explicitly-labeled placeholder rather than invented data.
 */
export default function WeeklyPage() {
  const { getToken } = useAuth();
  // Public demo: anchored to the one real week this fixed dataset has a
  // recap for, not the visitor's own real current ISO week.
  const [isoWeek, setIsoWeek] = useState(latestDemoWeek);
  const thisWeek = latestDemoWeek();
  const [weekly, setWeekly] = useState<WeeklyContext | null>(null);
  const [bookPages, setBookPages] = useState<WeeklyBookPage[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    setWeekly(await getWeekly(token, isoWeek));
    setLoading(false);
  }, [getToken, isoWeek]);

  // The "flip through your week" book needs each individual day's own
  // real composed strip — `GET /weekly/{iso_week}` only returns the
  // synthesized 6-panel recap, not the 7 individual days. `/library`
  // has exactly this, scoped by calendar month — a week can span two
  // months (e.g. late Dec into early Jan), so this fetches every
  // distinct month the week's 7 real dates touch and keeps only the
  // ones with a real `strip_url`.
  const loadBookPages = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    const dates = isoWeekDates(isoWeek);
    const months = Array.from(new Set(dates.map((d) => d.slice(0, 7))));
    const results = await Promise.all(months.map((month) => getLibrary(token, month)));
    const byDate = new Map<string, LibraryDay>();
    for (const result of results) {
      for (const day of result.days) byDate.set(day.date, day);
    }
    const pages: WeeklyBookPage[] = dates
      .map((date) => byDate.get(date))
      .filter((day): day is LibraryDay => Boolean(day?.strip_url))
      .map((day) => ({ date: day.date, stripUrl: day.strip_url as string, mood: day.mood }));
    setBookPages(pages);
  }, [getToken, isoWeek]);

  useEffect(() => {
    load();
    loadBookPages();
  }, [load, loadBookPages]);

  /** Move to another week. Clears what the last week showed so it never flashes under the new
   * week's heading. */
  function goToWeek(delta: number) {
    setError(null);
    setWeekly(null);
    setBookPages([]);
    setLoading(true);
    setIsoWeek((w) => shiftIsoWeek(w, delta));
  }

  const weekNav = (
    <div className="row" style={{ gap: 6 }}>
      <button type="button" className="btn" onClick={() => goToWeek(-1)} disabled={generating}>
        ← Earlier week
      </button>
      <button
        type="button"
        className="btn"
        onClick={() => goToWeek(1)}
        disabled={generating || isoWeek >= thisWeek}
      >
        Later week →
      </button>
    </div>
  );

  /** Sequential real downloads of every real day's own composed strip
   * this week — `strip_url` is a real, unauthenticated `/media/*` URL
   * (`LocalFileStorage`'s own contract, same as the Today page's
   * "Download strip"), so no auth header juggling is needed here. */
  function handleDownloadAll() {
    for (const page of bookPages) {
      const a = document.createElement("a");
      a.href = page.stripUrl;
      a.download = `comiccanvas-${page.date}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }

  /** Starts the recap and watches the job to completion, so a recap that's still being drawn or
   * that failed doesn't look identical to nothing happening. */
  async function handleGenerate() {
    setError(null);
    const token = await getToken();
    if (!token) return;
    setGenerating(true);
    try {
      const { job_id } = await generateWeekly(token, isoWeek);
      const outcome = { failure: null as string | null };
      await streamJobEvents(job_id, token, (evt) => {
        if (evt.event === "done" && evt.data.status === "failed") {
          outcome.failure = String(evt.data.error ?? "unknown error");
        }
      });
      if (outcome.failure) {
        throw new Error(`The recap couldn't be built: ${outcome.failure.slice(0, 200)}`);
      }
      await load();
      await loadBookPages();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setGenerating(false);
    }
  }

  async function handleDownload() {
    setError(null);
    const token = await getToken();
    if (!token) return;
    try {
      await downloadWeeklyPdf(token, isoWeek);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not download the PDF.");
    }
  }

  if (loading) return null;

  if (!weekly) {
    return (
      <>
        <div className="head">
          <div>
            <div className="kick">
              {isoWeek} · {isoWeekDates(isoWeek)[0]} to {isoWeekDates(isoWeek)[6]}
            </div>
            <h1>No recap yet</h1>
          </div>
          {weekNav}
        </div>
        <button className="btn pri" onClick={handleGenerate} disabled={generating}>
          {generating ? "Generating…" : "Generate this week's recap"}
        </button>
        {generating && (
          <p role="status" style={{ color: "var(--mute)" }}>
            Choosing the week&apos;s highlights from your daily panels. Nothing is drawn, so it
            takes a few seconds.
          </p>
        )}
        {error && <div className="err">{error}</div>}
      </>
    );
  }

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">
            {weekly.week_label} · {weekly.date_range}
          </div>
          <h1>The week in {weekly.panels.length} panels</h1>
        </div>
        <div className="row">
          <button className="btn pri" onClick={handleDownload}>
            <DownloadIcon />
            Download PDF
          </button>
          {bookPages.length > 0 && (
            <button className="btn" onClick={handleDownloadAll}>
              <DownloadIcon />
              Download all {bookPages.length} days
            </button>
          )}
          {weekNav}
        </div>
      </div>
      {error && <div className="err">{error}</div>}
      <WeeklyBook pages={bookPages} dateRange={weekly.date_range} />
      <div className="grid">
        <div className="card s8 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>Sunday recap</h2>
            {weekly.mood && <span className="pill mus">{weekly.mood}</span>}
          </div>
          <div className="strip six">
            {weekly.panels.map((p) => (
              <div className="pnl" key={p.id}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={p.url} alt={p.caption} />
                <div className="c">{p.caption}</div>
              </div>
            ))}
          </div>
          <p className="mono">
            Highlights chosen from this week&apos;s panels: the same pictures you kept in each
            daily strip.
          </p>
        </div>
        <div className="card s4 stack soft">
          <div className="kick">Mood, day by day</div>
          <div className="row">
            {weekly.mood_series.map((m, i) => (
              <span className="pill" key={i}>
                {m}
              </span>
            ))}
          </div>
        </div>
        <div className="card s4 stack">
          <div className="kick">Stats</div>
          <div className="row" style={{ gap: 18 }}>
            <div>
              <div className="big">{weekly.stats.days}</div>
              <span className="mono">days</span>
            </div>
            <div>
              <div className="big">{weekly.stats.panels}</div>
              <span className="mono">panels</span>
            </div>
            <div>
              <div className="big">{weekly.stats.pairs}</div>
              <span className="mono">pairs</span>
            </div>
            <div>
              <div className="big">{weekly.stats.first_pick_pct}%</div>
              <span className="mono">first pick</span>
            </div>
          </div>
        </div>
        <div className="card s4 stack">
          <div className="kick">Cast this week</div>
          {weekly.cast.length === 0 ? (
            <p style={{ color: "var(--mute)" }}>Just you, this week.</p>
          ) : (
            weekly.cast.map((c) => <div key={c.name}>{c.name}</div>)
          )}
        </div>
        <div className="card s4 stack dark">
          <div className="kick">Training run</div>
          {weekly.training_run ? (
            <p>
              {weekly.training_run.kind} · {weekly.training_run.decision} ·{" "}
              {weekly.training_run.created_at}
            </p>
          ) : (
            <p>No training run yet.</p>
          )}
        </div>
      </div>
    </>
  );
}
