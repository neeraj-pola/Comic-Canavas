"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { getLibrary, search, type LibraryDay, type SearchHit } from "@/lib/api";
import "@/components/dashboard/bento.css";
import "@/components/library/library.css";

/** Task 10.6: ported from `renderLibrary(q)`. Real `GET /library`/
 * `GET /search` data — thumbnails show each day's own real composed
 * `strip_url` image (the API has no per-panel-thumbnail endpoint, so a
 * 2×2 mock mini-grid of procedural SVGs isn't reconstructable from real
 * data; the real strip image is the honest equivalent). */

function monthLabel(month: string): string {
  const [y, m] = month.split("-").map(Number);
  return new Date(y!, m! - 1, 1).toLocaleDateString("en-US", { month: "long", year: "numeric" });
}

function daysInMonth(month: string): number {
  const [y, m] = month.split("-").map(Number);
  return new Date(y!, m!, 0).getDate();
}

function LibraryContent() {
  const { getToken } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const q = searchParams.get("q") ?? "";

  const [month] = useState(() => new Date().toISOString().slice(0, 7));
  const [days, setDays] = useState<LibraryDay[]>([]);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      const token = await getToken();
      if (!token || cancelled) return;
      const [lib, searchResult] = await Promise.all([
        getLibrary(token, month),
        q ? search(token, q) : Promise.resolve(null),
      ]);
      if (cancelled) return;
      setDays(lib.days);
      setHits(searchResult ? searchResult.hits : null);
      setLoading(false);
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [getToken, month, q]);

  if (loading) return null;

  const todayDate = new Date().getDate();
  const dayByDate = new Map(days.map((d) => [Number(d.date.slice(-2)), d]));

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">Library</div>
          <h1>Every day you&apos;ve drawn</h1>
        </div>
        <p>
          Search speaks your own words back: ask about a person, a place, or a mood and get the
          panels.
        </p>
      </div>
      <div className="grid">
        <div className="card s4 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>{monthLabel(month)}</h2>
            <span className="mono">{days.length} strips</span>
          </div>
          <div className="cal">
            {["M", "T", "W", "T", "F", "S", "S"].map((h, i) => (
              <div className="h" key={i}>
                {h}
              </div>
            ))}
            {Array.from({ length: daysInMonth(month) }, (_, i) => i + 1).map((d) => {
              const entry = dayByDate.get(d);
              return (
                <div
                  key={d}
                  className={`d ${entry ? "has" : ""} ${d === todayDate ? "today" : ""}`}
                  onClick={() => entry && router.push(`/day/${entry.date}`)}
                >
                  {d}
                </div>
              );
            })}
          </div>
        </div>
        <div className="card s8 stack">
          {hits ? (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2>&ldquo;{q}&rdquo;</h2>
                <span className="mono">
                  {hits.length} {hits.length === 1 ? "match" : "matches"} · from your beats
                </span>
              </div>
              {hits.length === 0 ? (
                <p style={{ color: "var(--mute)" }}>No matches yet.</p>
              ) : (
                <div className="thumbgrid">
                  {hits.map((hit, i) => (
                    <div className="thumb" key={i} onClick={() => router.push(`/day/${hit.date}`)}>
                      {hit.panel_urls[0] && (
                        <div className="mini">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={hit.panel_urls[0]} alt={hit.event ?? ""} />
                        </div>
                      )}
                      <div className="t">
                        <span>{hit.date}</span>
                        <span>{hit.place}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2>Recent strips</h2>
                <span className="mono">newest first</span>
              </div>
              <div className="thumbgrid">
                {[...days].reverse().map((d) => (
                  <div className="thumb" key={d.date} onClick={() => router.push(`/day/${d.date}`)}>
                    {d.strip_url && (
                      <div className="mini">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={d.strip_url} alt={d.mood ?? ""} />
                      </div>
                    )}
                    <div className="t">
                      <span>{d.date}</span>
                      <span>{d.mood}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

export default function LibraryPage() {
  return (
    <Suspense fallback={null}>
      <LibraryContent />
    </Suspense>
  );
}
