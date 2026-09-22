"use client";

import { useEffect, useState } from "react";

/**
 * Public demo mode: every button that would normally call the real backend
 * shows this instead of doing anything. A tiny pub-sub (not React context)
 * because `lib/api.ts`'s plain functions need to trigger it from outside any
 * component tree, the same reason a toast library works this way.
 */
type Listener = () => void;
const listeners = new Set<Listener>();

export function showDemoNotice(): void {
  for (const l of listeners) l();
}

export function DemoNoticeModal() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const listener: Listener = () => setOpen(true);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="demo-notice-title"
      className="demo-notice-backdrop"
      onClick={() => setOpen(false)}
    >
      <div className="demo-notice-card" onClick={(e) => e.stopPropagation()}>
        <div className="demo-notice-kick">This is a live demo</div>
        <h2 id="demo-notice-title">Nothing was changed</h2>
        <p>
          Real generation, taps, and settings all cost real API calls, so this public demo doesn&apos;t
          make any. Everything you&apos;re looking at is real output from actually running the app.
        </p>
        <p>
          Clone the project and run it with your own API keys to generate real comics of your own.
          Thanks for checking it out!
        </p>
        <div className="demo-notice-actions">
          <a
            className="demo-notice-btn primary"
            href="https://github.com/neeraj-pola/Comic-Canavas"
            target="_blank"
            rel="noreferrer"
          >
            View the repo
          </a>
          <button type="button" className="demo-notice-btn" onClick={() => setOpen(false)}>
            Keep looking around
          </button>
        </div>
      </div>
    </div>
  );
}
