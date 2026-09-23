"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { preload } from "react-dom";
import "./weekly-book.css";

export type WeeklyBookPage = { date: string; stripUrl: string; mood: string | null };

/** What a sheet's front shows: a plain cover, or one day. */
type Sheet = { kind: "cover" } | { kind: "day"; page: WeeklyBookPage };

const dayName = (iso: string) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
  });

/**
 * "Flip through your week" — reuses `components/landing/LandingBook.tsx`'s
 * own scroll-driven 3D page-flip mechanism, generalized from that
 * component's hardcoded single sheet (`FLIPS = 1`, the cover) to N real
 * sheets, one per day that has a real composed `strip_url` that week. The
 * underlying dwell/per/ease formulas are the same ones
 * (`DWELL = 1/(1+1.5*FLIPS+RUNOFF)`, `PER = 1.5*DWELL`) — the source
 * already parameterized them by `FLIPS`, just hardcoded to 1 in that
 * port; this only generalizes `FLIPS` into a real prop and renders one
 * sheet per page instead of one.
 *
 * Each sheet's front face shows that day's real, already-composed strip
 * image; the back face is decorative paper texture (matching
 * `LandingBook`'s own cover/endpaper convention — the actual "next page"
 * is a separate sheet sitting underneath, not the flipped sheet's own
 * back).
 *
 * `prefers-reduced-motion` gets a genuinely different, simpler static
 * layout — not a shrunk version of the 3D flip, a plain vertical list:
 * every day's real strip, in normal document flow, no transforms, no
 * scroll listeners. Every other visitor, including phones, gets the real
 * animated flip: `visualViewport.height` (not `window.innerHeight`) is
 * what actually makes that reliable on mobile Safari, where innerHeight
 * lags behind the true visible height while the address-bar chrome
 * animates in/out mid-scroll.
 */

const RUNOFF = 0.7;

function pageMath(numPages: number) {
  const dwell = 1 / (1 + 1.5 * numPages + RUNOFF);
  const per = 1.5 * dwell;
  const endStart = dwell + numPages * per;
  return { dwell, per, endStart };
}

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * A plain cover, then one sheet per day (a recap sheet was tried and
 * dropped — the recap lives on the page itself). Each picture is shown
 * whole (`object-fit: contain`) rather than `cover`, which would crop
 * every strip to fill the narrow page and cut off the date header and
 * panel edges.
 */
export function WeeklyBook({
  pages,
  dateRange,
}: {
  pages: WeeklyBookPage[];
  dateRange?: string | null;
}) {
  const sheets = useMemo<Sheet[]>(
    () => [{ kind: "cover" }, ...pages.map((page) => ({ kind: "day", page }) as const)],
    [pages],
  );

  // Only `prefers-reduced-motion` gets the plain static list now — a real
  // accessibility need. The same 3D flip runs at every screen size,
  // including phones; each page here is just one image + a short caption,
  // so there's no equivalent of the landing hero's "too much text to fit"
  // problem driving a separate narrow-width treatment.
  const [staticLayout, setStaticLayout] = useState<boolean | null>(null);
  useEffect(() => {
    const motionMql = window.matchMedia("(prefers-reduced-motion: reduce)");
    const recompute = () => setStaticLayout(motionMql.matches);
    recompute();
    motionMql.addEventListener("change", recompute);
    return () => {
      motionMql.removeEventListener("change", recompute);
    };
  }, []);

  // Every page's <img> is already in the DOM from mount, not added as you
  // scroll to it — but a page that isn't visible yet can still be fetched
  // at a lower priority by the browser's own heuristics, which on a real
  // phone's real connection can mean the last day or two are still
  // downloading by the time their flip reveals them (looking exactly like
  // a stuck animation, even though the flip itself already finished).
  // Explicitly requesting all of them at once, at high priority, closes
  // that gap regardless of scroll position. Only worth doing for the
  // animated path — the static list below loads each image normally, in
  // the order it actually appears on the page.
  if (staticLayout === false) {
    for (const page of pages) preload(page.stripUrl, { as: "image", fetchPriority: "high" });
  }

  const pinRef = useRef<HTMLDivElement>(null);
  const bookRef = useRef<HTMLDivElement>(null);
  const sheetRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    if (staticLayout !== false) return;
    if (sheets.length <= 1) return;
    const pin = pinRef.current;
    const book = bookRef.current;
    if (!pin || !book) return;

    const { dwell, per, endStart } = pageMath(sheets.length);
    const previousOverflowX = document.body.style.overflowX;
    document.body.style.overflowX = "hidden";

    function update() {
      // `visualViewport.height`, not `window.innerHeight`: on mobile Safari,
      // `innerHeight` lags a couple of seconds behind the real visible
      // height while the address-bar chrome animates in/out during scroll,
      // which made the book briefly show the wrong page after scrolling had
      // already stopped. `visualViewport` tracks the actual visible area
      // and fires its own resize event promptly when the chrome settles.
      const vh = window.visualViewport?.height ?? window.innerHeight;
      const sy = window.scrollY;
      const total = pin!.offsetHeight - vh;
      const p = clamp(total > 0 ? sy / total : 0, 0, 1);

      sheets.forEach((_, i) => {
        const sheet = sheetRefs.current[i];
        if (!sheet) return;
        const open = clamp((p - dwell - i * per) / (per * 0.8), 0, 1);
        const e = ease(open);
        const rot = -180 * e;
        sheet.style.transform = `rotateY(${rot}deg) translateZ(${(0.5 - Math.abs(e - 0.5)) * 40}px)`;
        // Unflipped: earlier pages stack on top (ready to flip first).
        // Mid-flip: brought fully to the front so the turning page
        // never disappears behind a still-unflipped one. Flipped:
        // later pages land on top of the left-hand settled pile, the
        // same order a real hand-turned stack would end up in.
        sheet.style.zIndex =
          open <= 0
            ? String(sheets.length - i)
            : open >= 1
              ? String(i)
              : String(sheets.length + 50);
      });

      const tail = clamp((p - endStart) / (1 - endStart), 0, 1);
      const scale = 1 + 0.03 * ease(tail); // small: the book already fills its container
      book!.style.transform = `scale(${scale})`;
    }

    pin.style.height = `${(1 + 1.5 * sheets.length + RUNOFF) * 100}vh`;
    const raf = requestAnimationFrame(update);

    let ticking = false;
    const onScroll = () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(() => {
          update();
          ticking = false;
        });
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", update);
    // `visualViewport` fires its own `resize` repeatedly while a real
    // phone's address-bar chrome animates in/out mid-scroll — reacting to
    // every one of those immediately (each with a momentarily different,
    // not-yet-settled `vh`) is the real cause of "All caught up" and
    // other pages flashing through for a couple of seconds: transient
    // bad `vh` readings transiently mis-flip every sheet before the true
    // height settles. Debouncing to the *last* event in a burst uses only
    // the final, settled height.
    let viewportSettleTimer: number | undefined;
    const onViewportResize = () => {
      if (viewportSettleTimer !== undefined) window.clearTimeout(viewportSettleTimer);
      viewportSettleTimer = window.setTimeout(update, 120);
    };
    window.visualViewport?.addEventListener("resize", onViewportResize);

    return () => {
      cancelAnimationFrame(raf);
      if (viewportSettleTimer !== undefined) window.clearTimeout(viewportSettleTimer);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", update);
      window.visualViewport?.removeEventListener("resize", onViewportResize);
      document.body.style.overflowX = previousOverflowX;
    };
  }, [sheets, staticLayout]);

  if (pages.length === 0) return null;

  if (staticLayout === null) {
    // matchMedia hasn't resolved yet on the client; avoid a flash of the
    // wrong (animated vs. static) variant, same as LandingBook.
    return null;
  }

  if (staticLayout) {
    return (
      <div className="weekly-book wb-static">
        {/* Still a real notebook (spiral binding down the left edge, pages
            rounded only on the right, like a page actually bound there) —
            just not 3D or scroll-jacked. A flat list here read as "no book
            at all"; this keeps the book identity while staying reliable. */}
        <div className="wb-notebook">
          <div className="wb-notebook-spiral" aria-hidden="true" />
          <div className="wb-static-cover">
            <span className="wb-cover-kick">Comic Canvas</span>
            <b>The Week</b>
            {dateRange && <span className="wb-cover-range">{dateRange}</span>}
          </div>
          <div className="wb-static-list">
            {pages.map((page, i) => (
              <div className="wb-static-day" key={page.date}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={page.stripUrl} alt={`${page.date} strip`} loading="lazy" decoding="async" />
                <div className="wb-caption">
                  <span>
                    {dayName(page.date)}
                    {page.mood ? ` · ${page.mood}` : ""}
                  </span>
                  <span>{i + 1}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <p className="wb-hint">That&apos;s the week — all caught up</p>
      </div>
    );
  }

  return (
    <div className="weekly-book">
      <p className="wb-hint">Scroll to flip through the week</p>
      <div className="wb-pin" ref={pinRef}>
        <div className="wb-stage">
          <div className="wb-book" ref={bookRef}>
            <div className="wb-backboard" />
            <div className="wb-board">
              <div className="wb-board-content stack">
                <div className="kick">That&apos;s the week</div>
                <h2>All caught up</h2>
                <p>Scroll back up to flip through again, or download every day below.</p>
              </div>
            </div>
            {sheets.map((sheet, i) => (
              <div
                className="wb-sheet"
                key={sheet.kind === "day" ? sheet.page.date : sheet.kind}
                ref={(el) => {
                  sheetRefs.current[i] = el;
                }}
                style={{ zIndex: sheets.length - i }}
              >
                <div className="wb-face wb-back">
                  <div className="wb-back-pattern" />
                </div>
                {sheet.kind === "cover" ? (
                  <div className="wb-face wb-front wb-cover">
                    <div className="wb-cover-title">
                      <span className="wb-cover-kick">Comic Canvas</span>
                      <b>The Week</b>
                      {dateRange && <span className="wb-cover-range">{dateRange}</span>}
                    </div>
                  </div>
                ) : (
                  <div className="wb-face wb-front wb-pagefront">
                    <div className="wb-figure">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={sheet.page.stripUrl}
                        alt={`${sheet.page.date} strip`}
                        fetchPriority="high"
                        decoding="sync"
                      />
                    </div>
                    <div className="wb-caption">
                      <span>
                        {dayName(sheet.page.date)}
                        {sheet.page.mood ? ` · ${sheet.page.mood}` : ""}
                      </span>
                      <span>{i + 1}</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
            <div className="wb-spiral">
              {Array.from({ length: 14 }).map((_, i) => (
                <i key={i} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
