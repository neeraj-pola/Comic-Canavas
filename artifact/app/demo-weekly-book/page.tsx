"use client";
import { WeeklyBook } from "@/components/weekly/WeeklyBook";
import "@/components/dashboard/dashboard-globals.css";
import "@/components/dashboard/bento.css";

/** DEMO ONLY — placeholder pages so you can see the book live in your
 * own browser. Not part of the real app; safe to delete once you're
 * done looking. Real usage always shows real composed strip.png
 * images, not these solid-color placeholders. */
const DEMO_PAGES = [
  { date: "Mon · Sep 14", mood: "chaotic but earned", color: "9D7BD8/151515" },
  { date: "Tue · Sep 15", mood: "drained but recovering", color: "E67E5C/151515" },
  { date: "Wed · Sep 16", mood: "grounded", color: "5CAE7E/151515" },
  { date: "Thu · Sep 17", mood: "surprisingly good", color: "E9D24A/151515" },
  { date: "Fri · Sep 18", mood: "accomplished", color: "5C8FE6/151515" },
  { date: "Sat · Sep 19", mood: "warmly productive", color: "D85CA8/151515" },
  { date: "Sun · Sep 20", mood: "easy and full", color: "7BC4D8/151515" },
].map((d) => ({
  date: d.date,
  mood: d.mood,
  stripUrl: `https://placehold.co/1080x1080/${d.color}?text=${encodeURIComponent(d.date + "\\n4-panel strip")}`,
}));

export default function DemoWeeklyBook() {
  return (
    <div className="dashboard-scope">
      <div style={{ padding: 20 }}>
        <div className="head">
          <div>
            <div className="kick">Demo · placeholders</div>
            <h1>The week in 6 panels</h1>
          </div>
          <p>Scroll down to flip through 7 placeholder days.</p>
        </div>
        <WeeklyBook pages={DEMO_PAGES} />
      </div>
    </div>
  );
}
