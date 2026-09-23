"use client";

import { useEffect, useRef, useState } from "react";
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

function BookChrome({
  sheetStyle,
  bookStyle,
}: {
  sheetStyle?: React.CSSProperties;
  bookStyle?: React.CSSProperties;
}) {
  return (
    <div className="book" style={bookStyle}>
      <div className="backboard" />
      <div className="board">
        <HeroContent />
      </div>
      <div className="sheet" data-i="0" style={sheetStyle}>
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
  );
}

export function LandingBook() {
  // Same static rendering path for prefers-reduced-motion AND narrow
  // screens: the pinned/sticky scroll-jack traps the hero's own internal
  // overflow scroll in a way that fights with a real phone's touch-scroll
  // capture (verified only in simulated/mouse-driven testing, not on an
  // actual device) — skipping the scroll-jack there entirely, the same way
  // reduced-motion already does, sidesteps that fight completely rather
  // than trying to patch it further.
  const [staticLayout, setStaticLayout] = useState<boolean | null>(null);

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
    const recompute = () => setStaticLayout(motionMql.matches || widthMql.matches);
    recompute();
    motionMql.addEventListener("change", recompute);
    widthMql.addEventListener("change", recompute);
    return () => {
      motionMql.removeEventListener("change", recompute);
      widthMql.removeEventListener("change", recompute);
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
      const vh = window.innerHeight;
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

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
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
    return (
      <div className="landing-scope">
        <HalftonePattern />
        <div className="brand">
          <i />
          Comic Canvas
        </div>
        <div style={{ position: "relative", height: "min(90vh, 66vw)" }}>
          <BookChrome sheetStyle={{ transform: "rotateY(-180deg)" }} bookStyle={{ opacity: 1 }} />
        </div>
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
              <HeroContent />
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
