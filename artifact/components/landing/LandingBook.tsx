"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Person } from "../illustration/Person";
import { HalftonePattern } from "../illustration/HalftonePattern";
import { FLOW_SECTIONS, FlowContent, HeroContent, Endpaper } from "./content";
import "./landing.css";

/**
 * Direct TypeScript port of comiccanvas-landing.html's `build()`/
 * `update()`/`layoutInk()` — same dwell/per/ease math, same per-sheet
 * transforms, same ink-line SVG path generation — using refs instead of
 * `getElementById` and a mount effect instead of global listeners.
 *
 * Only variant B (the shipped default) is ported: `VARIANT_SPREADS.B = 1`
 * means exactly one flippable sheet ever exists (the cover) — `FLIPS` is
 * a compile-time constant here, not a generic multi-sheet system, since
 * variants A/C's extra sheets never render live.
 *
 * `prefers-reduced-motion: reduce` skips the whole scroll-jacking path
 * (the source's own reduced-motion rule targets a CSS `transition` the
 * imperative rAF loop never uses, so it does effectively nothing) and
 * instead renders the book already open with the flow content directly
 * below it, statically — a real fix, not a faithful bug-for-bug port.
 */

const FLIPS = 1;
const RUNOFF = 0.7;
const DWELL = 1 / (1 + 1.5 * FLIPS + RUNOFF);
const PER = 1.5 * DWELL;
const END_START = DWELL + FLIPS * PER;

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * The condensed hero copy — not `HeroContent` shrunk down, a genuinely
 * shorter version: no drop-cap paragraph, no tags row, no byline. Used in
 * two different structural contexts, so it's the bare content only, no
 * wrapper: `MobileHero` (below) wraps it for normal page flow (the
 * reduced-motion static path); `NarrowHeroInBoard` wraps it in `.pg` for
 * `.board`'s own absolute-fill, overflow-safe layout (the real animated
 * book, narrow screens). Less text is the actual fix that keeps either
 * container from overflowing — not a cleverer wrapper.
 */
function HeroBody() {
  return (
    <>
      <span className="kick">Daily · Spoken · Drawn</span>
      <h2>Speak your day. Get a comic of it.</h2>
      <p>One minute about your day becomes a four-panel strip, with a character that looks like you.</p>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Link className="pill pink" href="/today">
          Open your diary
        </Link>
        <a className="pill" href="#flow">
          How it works
        </a>
      </div>
    </>
  );
}

function MobileHero() {
  return (
    <div className="mobile-hero">
      <HeroBody />
    </div>
  );
}

/** Same condensed copy, but wrapped for `.board`'s own layout (see `HeroBody`). */
function NarrowHeroInBoard() {
  return (
    <div className="pg">
      <HeroBody />
    </div>
  );
}

function CoverFace() {
  return (
    <div className="face front cover">
      <div className="emboss" />
      <div>
        <div className="year">Diary · 2026</div>
      </div>
      <div>
        <div className="title">
          Comic <u>Canvas</u>
        </div>
        <div className="sub">a diary that draws itself, one strip a day</div>
      </div>
      <Person hair="curly" shirt="#E9D24A" mood="smile" className="mark" />
    </div>
  );
}

/**
 * The mobile/static path's own cover — still genuinely looks like a book
 * (spiral binding down the left edge, the same embossed mustard cover
 * face as the desktop version), just not 3D or animated: no rotateY, no
 * scroll-jacking, no clipped-height container. A flat plain card here
 * read as "no book at all" — this keeps the actual book identity while
 * staying static and reliable.
 */
function MobileCover() {
  return (
    <div className="mobile-book">
      <div className="mobile-book-spiral" aria-hidden="true">
        {Array.from({ length: 9 }).map((_, i) => (
          <i key={i} />
        ))}
      </div>
      <div className="mobile-cover">
        <div className="mobile-cover-emboss" />
        <div className="mobile-cover-kick">Diary · 2026</div>
        <div>
          <div className="mobile-cover-title">
            Comic <u>Canvas</u>
          </div>
          <div className="mobile-cover-sub">a diary that draws itself, one strip a day</div>
        </div>
        <Person hair="curly" shirt="#E9D24A" mood="smile" className="mark" />
      </div>
    </div>
  );
}

export function LandingBook() {
  // Only `prefers-reduced-motion` gets the static layout now — a real
  // accessibility need, not a mobile workaround. The same animated 3D
  // flip now runs at every screen size, including phones; `narrow` below
  // only picks which hero copy renders inside `.board` (shorter on small
  // screens, so there's less text for that box to ever have to fit), it
  // doesn't change which of these two branches renders.
  const [staticLayout, setStaticLayout] = useState<boolean | null>(null);
  const [narrow, setNarrow] = useState(false);

  const pinRef = useRef<HTMLDivElement>(null);
  const bookRef = useRef<HTMLDivElement>(null);
  const sheetRef = useRef<HTMLDivElement>(null);
  const flowRef = useRef<HTMLDivElement>(null);
  const inkPathRef = useRef<SVGPathElement>(null);
  const nibRef = useRef<SVGGElement>(null);
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});
  const inkLenRef = useRef(0);

  useEffect(() => {
    const motionMql = window.matchMedia("(prefers-reduced-motion: reduce)");
    const widthMql = window.matchMedia("(max-width: 760px)");
    const recomputeMotion = () => setStaticLayout(motionMql.matches);
    const recomputeWidth = () => setNarrow(widthMql.matches);
    recomputeMotion();
    recomputeWidth();
    motionMql.addEventListener("change", recomputeMotion);
    widthMql.addEventListener("change", recomputeWidth);
    return () => {
      motionMql.removeEventListener("change", recomputeMotion);
      widthMql.removeEventListener("change", recomputeWidth);
    };
  }, []);

  useEffect(() => {
    if (staticLayout !== false) return;
    const pin = pinRef.current;
    const book = bookRef.current;
    const sheet = sheetRef.current;
    const flow = flowRef.current;
    if (!pin || !book || !sheet || !flow) return;

    const previousOverflowX = document.body.style.overflowX;
    document.body.style.overflowX = "hidden";

    function layoutInk() {
      const inkPath = inkPathRef.current;
      const flowEl = flowRef.current;
      if (!inkPath || !flowEl) return;
      const fr = flowEl.getBoundingClientRect();
      const W = flowEl.offsetWidth;
      const H = flowEl.offsetHeight;
      const svg = inkPath.ownerSVGElement;
      svg?.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const X = 48;
      let d = `M ${X} 0`;
      FLOW_SECTIONS.forEach((key) => {
        const s = sectionRefs.current[key];
        if (!s) return;
        const r = s.getBoundingClientRect();
        const x0 = r.left - fr.left;
        const y0 = r.top - fr.top;
        const w = r.width;
        const h = r.height;
        const f = s.querySelector<SVGRectElement>(".frame rect");
        const svgf = s.querySelector<SVGSVGElement>("svg.frame");
        if (f && svgf) {
          f.setAttribute("width", String(w - 3));
          f.setAttribute("height", String(h - 3));
          svgf.setAttribute("viewBox", `0 0 ${w} ${h}`);
          const per = 2 * (w + h);
          f.style.strokeDasharray = String(per);
          f.dataset.per = String(per);
          f.style.strokeDashoffset = String(per);
        }
        d += ` L ${X} ${y0 - 70} C ${X} ${y0 - 20}, ${x0 - 10} ${y0 - 10}, ${x0 + 18} ${y0 + 1.5}`;
        d += ` C ${x0 - 30} ${y0 + 20}, ${X} ${y0 + h * 0.25}, ${X} ${y0 + h * 0.45}`;
      });
      d += ` L ${X} ${H}`;
      inkPath.setAttribute("d", d);
      const len = inkPath.getTotalLength();
      inkLenRef.current = len;
      inkPath.style.strokeDasharray = String(len);
      inkPath.style.strokeDashoffset = String(len);
    }

    function update() {
      // `visualViewport.height`, not `window.innerHeight` — on mobile
      // Safari, `innerHeight` lags behind the real visible height while
      // the address-bar chrome animates in/out during scroll, the same
      // real bug WeeklyBook's own animation already fixed this way.
      const vh = window.visualViewport?.height ?? window.innerHeight;
      const sy = window.scrollY;
      document.body.classList.toggle("scrolled", sy > 40);
      const total = pin!.offsetHeight - vh;
      const p = clamp(total > 0 ? sy / total : 0, 0, 1);

      const open = clamp((p - DWELL) / (PER * 0.8), 0, 1);
      const bx = -25 * (1 - ease(open));
      const lean = Math.sin(Math.PI * ease(open));
      let scale = 1 + 0.07 * lean;

      const e = ease(open);
      const rot = -180 * e;
      sheet!.style.transform = `rotateY(${rot}deg) translateZ(${(0.5 - Math.abs(e - 0.5)) * 40}px)`;
      sheet!.style.zIndex = open <= 0 ? "3" : open >= 1 ? "1" : "11";

      const tail = clamp((p - END_START) / (1 - END_START), 0, 1);
      const op = 1 - ease(clamp((tail - 0.15) / 0.6, 0, 1));
      scale *= 1 + 0.12 * ease(tail);

      book!.style.transform = `translateX(${bx}%) scale(${scale}) rotateX(${(1 - ease(open)) * 6}deg) rotateY(${(1 - ease(open)) * -2}deg)`;
      book!.style.opacity = String(op);

      document.body.classList.toggle("showspine", tail > 0.35);

      const inkPath = inkPathRef.current;
      const fr = flow!.getBoundingClientRect();
      const prog = clamp(fr.height > 0 ? (vh * 0.72 - fr.top) / fr.height : 0, 0, 1);
      if (inkPath) {
        inkPath.style.strokeDashoffset = String(inkLenRef.current * (1 - prog));
        const pt = inkPath.getPointAtLength(inkLenRef.current * prog);
        nibRef.current?.setAttribute("transform", `translate(${pt.x} ${pt.y}) rotate(-28)`);
        if (nibRef.current) nibRef.current.style.opacity = prog > 0 && prog < 1 ? "1" : "0";
      }
      FLOW_SECTIONS.forEach((key) => {
        const s = sectionRefs.current[key];
        if (!s) return;
        const r = s.getBoundingClientRect();
        const tt = clamp((vh * 0.9 - r.top) / (vh * 0.55), 0, 1);
        const f = s.querySelector<HTMLElement>(".frame rect");
        if (f) {
          const per = Number(f.dataset.per ?? 0);
          f.style.strokeDashoffset = String(per * (1 - ease(tt)));
        }
        s.style.opacity = String(0.35 + 0.65 * ease(clamp(tt * 1.4, 0, 1)));
      });
    }

    pin.style.height = `${(1 + 1.5 * FLIPS + RUNOFF) * 100}vh`;
    const raf = requestAnimationFrame(() => {
      layoutInk();
      update();
    });

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
    const onResize = () => {
      layoutInk();
      update();
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onResize);
    // `visualViewport` fires its own `resize` repeatedly while a real
    // phone's address-bar chrome animates in/out mid-scroll — reacting to
    // every one of those immediately (each with a momentarily different,
    // not-yet-settled `vh`) transiently mis-computes the flip position
    // before the true height settles (the same real bug WeeklyBook's own
    // flip had). Debouncing to the *last* event in a burst uses only the
    // final, settled height.
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
      window.removeEventListener("resize", onResize);
      window.visualViewport?.removeEventListener("resize", onViewportResize);
      document.body.classList.remove("scrolled", "showspine");
      document.body.style.overflowX = previousOverflowX;
    };
  }, [staticLayout]);

  if (staticLayout === null) {
    // matchMedia hasn't resolved yet on the client; avoid a flash of the
    // wrong (animated vs. static) variant by rendering nothing briefly.
    return null;
  }

  if (staticLayout) {
    // `prefers-reduced-motion` only, now — a real accessibility need, not
    // a mobile workaround (the animated book below runs on phones too).
    // Still the same simpler, non-3D layout for that visitor: a small
    // decorative cover, the condensed `MobileHero` in normal page flow
    // below it, then the same flow sections.
    return (
      <div className="landing-scope">
        <HalftonePattern />
        <div className="brand">
          <i />
          Comic Canvas
        </div>
        <MobileCover />
        <MobileHero />
        <main className="flow" style={{ marginTop: 0 }} id="flow">
          {FLOW_SECTIONS.map((key) => (
            <section key={key} className="frame-host" style={{ opacity: 1 }}>
              <svg className="frame" aria-hidden="true">
                <rect x="1.5%" y="1.5%" width="97%" height="97%" />
              </svg>
              <FlowContent sectionKey={key} />
            </section>
          ))}
        </main>
        <footer className="site">
          <span>Comic Canvas · a diary that draws itself</span>
          <span>Made for one reader: you</span>
        </footer>
      </div>
    );
  }

  return (
    <div className="landing-scope">
      <HalftonePattern />
      <div className="brand">
        <i />
        Comic Canvas
      </div>
      <div className="hintscroll">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M12 5v14M5 12l7 7 7-7" />
        </svg>
        Scroll to open
      </div>

      <div className="pin" ref={pinRef}>
        <div className="stage">
          <div className="book" ref={bookRef}>
            <div className="backboard" />
            <div className="board">
              {/* The same animated book at every width — only the copy
                  inside it changes on narrow screens, to less text that's
                  less likely to ever need `.board`'s own overflow scroll
                  (the `@media (max-aspect-ratio: 1/1)` rule below) at all. */}
              {narrow ? <NarrowHeroInBoard /> : <HeroContent />}
            </div>
            <div className="sheet" ref={sheetRef} data-i="0">
              <CoverFace />
              <Endpaper />
              <div className="edge" />
            </div>
            <div className="spiral">
              {Array.from({ length: 14 }).map((_, i) => (
                <i key={i} />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="spine" aria-hidden="true">
        <div className="bar" />
        <div className="rings" />
        <div className="holes" />
      </div>

      <main className="flow" ref={flowRef} id="flow">
        {FLOW_SECTIONS.map((key) => (
          <section
            key={key}
            className="frame-host"
            ref={(el) => {
              sectionRefs.current[key] = el;
            }}
          >
            <svg className="frame" aria-hidden="true">
              <rect x={1.5} y={1.5} width={10} height={10} />
            </svg>
            <FlowContent sectionKey={key} />
          </section>
        ))}
        <svg className="inkline">
          <path ref={inkPathRef} id="inkpath" d="" />
          <g className="nib" ref={nibRef}>
            <path d="M0 0 l-7 -22 h14z" fill="#15161B" />
            <circle cx={0} cy={0} r={2.2} fill="#F4CBDD" stroke="#15161B" strokeWidth={1.5} />
          </g>
        </svg>
      </main>

      <footer className="site">
        <span>Comic Canvas · a diary that draws itself</span>
        <span>Made for one reader: you</span>
      </footer>
    </div>
  );
}
