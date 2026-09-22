"use client";

import { type CSSProperties, useId, useState } from "react";

const GAP = 8;
const MARGIN = 12;
const WIDTH = 260;

/**
 * A small "i" trigger that reveals extra detail on hover, keyboard focus, or
 * tap — the numbers behind a headline stat, kept out of the way until asked
 * for. Same hover/focus/tap convention as `CandidateCard`'s popover; position
 * is measured the same way too (viewport/fixed coordinates via
 * `getBoundingClientRect`, clamped so it never runs off the right or bottom
 * edge — a fixed `left:0` under the icon overflows whenever the icon itself
 * sits near an edge, which a plain CSS rule can't see coming).
 */
export function Hint({ children }: { children: React.ReactNode }) {
  const id = useId();
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean }>({
    left: 0,
    top: 0,
    above: false,
  });

  function place(el: HTMLElement) {
    const pop = el.querySelector<HTMLElement>(".pl-hint-pop");
    if (!pop) return;
    const trigger = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const w = Math.min(WIDTH, vw - 2 * MARGIN);
    const h = pop.offsetHeight;
    const left = Math.max(MARGIN, Math.min(trigger.left, vw - w - MARGIN));
    const fitsBelow = trigger.bottom + GAP + h + MARGIN <= vh;
    const top = fitsBelow ? trigger.bottom + GAP : Math.max(MARGIN, trigger.top - GAP - h);
    setPos({ left, top, above: !fitsBelow });
  }

  return (
    <span
      className="pl-hint"
      tabIndex={0}
      aria-label="More detail"
      aria-describedby={id}
      style={{ "--pop-left": `${pos.left}px`, "--pop-top": `${pos.top}px` } as CSSProperties}
      onPointerEnter={(e) => place(e.currentTarget)}
      onFocus={(e) => place(e.currentTarget)}
    >
      <span className="pl-hint-i" aria-hidden="true">
        i
      </span>
      <span className="pl-hint-pop" role="tooltip" id={id} data-above={pos.above || undefined}>
        {children}
      </span>
    </span>
  );
}
