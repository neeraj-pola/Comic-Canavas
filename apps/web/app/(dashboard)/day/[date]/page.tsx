"use client";

import { use, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { getDay, type Day } from "@/lib/api";
import { CandidateCard } from "@/components/library/CandidateCard";
import "@/components/dashboard/bento.css";
import "@/components/library/library.css";

/** Day-detail sub-route (`renderDay(d)` in the prototype) — strip, transcript,
 * beats table, candidates with real scores. Scores open in a hover table per
 * image; prompts stay in the backend (`GET /days/{date}` still returns them)
 * but aren't shown here. */
export default function DayDetailPage({ params }: { params: Promise<{ date: string }> }) {
  const { date } = use(params);
  const { getToken } = useAuth();
  const [day, setDay] = useState<Day | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const token = await getToken();
      if (!token || cancelled) return;
      const result = await getDay(token, date);
      if (cancelled) return;
      if (!result) setNotFound(true);
      else setDay(result);
      setLoading(false);
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [getToken, date]);

  if (loading) return null;
  if (notFound || !day) {
    return (
      <div className="head">
        <h1>No day for {date}</h1>
      </div>
    );
  }

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">{day.date}</div>
          <h1>{day.mood ?? "That day"}</h1>
        </div>
      </div>
      <div className="grid">
        <div className="card s8 stack">
          <h2>The strip</h2>
          {day.strip_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={day.strip_url}
              alt="the day's strip"
              style={{ width: "100%", borderRadius: 10 }}
            />
          ) : (
            <p style={{ color: "var(--mute)" }}>No strip generated for this day yet.</p>
          )}
          <p className="mono">
            {day.versions.generator_flag ?? "generator"} · {day.candidates.length} candidates · $
            {day.cost_usd.toFixed(2)}
          </p>
        </div>
        <div className="card s4 stack">
          <h2>What you wrote</h2>
          <p>{day.transcript ?? day.text ?? "None"}</p>
        </div>
        <div className="card s12 stack">
          <h2>Beats the extractor found</h2>
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Where</th>
                <th>What</th>
                <th>Emotion</th>
                <th>Importance</th>
              </tr>
            </thead>
            <tbody>
              {day.beats.map((beat) => (
                <tr key={beat.id}>
                  <td>{beat.time}</td>
                  <td>{beat.place}</td>
                  <td>{beat.event}</td>
                  <td>{beat.emotion}</td>
                  <td>{beat.importance.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card s12 stack">
          <h2>Candidates the critic saw</h2>
          <p style={{ color: "var(--mute)" }}>Hover, tap or focus an image to see its scores.</p>
          {day.panels.map((panel) => {
            const candidates = day.candidates.filter((c) => c.panel_id === panel.id);
            return (
              <div key={panel.id}>
                <h3>Panel {panel.id}</h3>
                <div className="cands">
                  {candidates.map((c) => (
                    <CandidateCard key={c.id} candidate={c} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}
