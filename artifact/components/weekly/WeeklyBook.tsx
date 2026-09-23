"use client";

import { useEffect, useMemo, useRef } from "react";
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
 * back). `prefers-reduced-motion` isn't specially handled here with a
 * static fallback the way the landing page's cover is — for N real
 * content pages (not one decorative cover), the pages themselves are
 * already reachable in the Library without this viewer at all, so a
 * reduced-motion visitor loses a nice-to-have flourish, not access to
 * real content.
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
  const pinRef = useRef<HTMLDivElement>(null);
  const bookRef = useRef<HTMLDivElement>(null);
  const sheetRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    if (sheets.length <= 1) return;
    const pin = pinRef.current;
    const book = bookRef.current;
    if (!pin || !book) return;

    const mql = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mql.matches) return;

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
    window.visualViewport?.addEventListener("resize", onScroll);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", update);
      window.visualViewport?.removeEventListener("resize", onScroll);
      document.body.style.overflowX = previousOverflowX;
    };
  }, [sheets]);

  if (pages.length === 0) return null;

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
                      <img src={sheet.page.stripUrl} alt={`${sheet.page.date} strip`} />
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
