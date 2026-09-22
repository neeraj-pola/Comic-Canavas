import Link from "next/link";

/** Task 10.7's archive route. No real backend endpoint lists past weeks
 * (the real API only serves one week at a time by iso_week) — kept as
 * an honest placeholder rather than fabricating a list. */
export default function WeeksArchivePage() {
  return (
    <div className="head">
      <div>
        <div className="kick">Archive</div>
        <h1>Past weeks</h1>
      </div>
      <p>
        There&apos;s no endpoint yet that lists every past week. For now,{" "}
        <Link href="/weekly">go to this week</Link>.
      </p>
    </div>
  );
}
