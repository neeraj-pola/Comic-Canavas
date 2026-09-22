/** The Monday..Sunday calendar dates (`YYYY-MM-DD`) an ISO week string
 * covers — the inverse of `currentIsoWeek()`'s own math. Used by the
 * Weekly page's book viewer, which needs each individual day's own real
 * `strip_url` (via `/library`), not just the synthesized 6-panel recap
 * `/weekly/{iso_week}` returns. Standard ISO 8601 week algorithm: week 1
 * is the week containing the year's first Thursday, so the Monday of
 * week 1 is 3 days before that Thursday; every other week's Monday is
 * `(week - 1) * 7` days later. */
export function isoWeekDates(isoWeek: string): string[] {
  const [yearStr, weekStr] = isoWeek.split("-W");
  const year = Number(yearStr);
  const week = Number(weekStr);
  const jan4 = new Date(Date.UTC(year, 0, 4));
  const jan4DayNum = (jan4.getUTCDay() + 6) % 7; // Monday = 0
  const week1Monday = new Date(jan4);
  week1Monday.setUTCDate(jan4.getUTCDate() - jan4DayNum);
  const monday = new Date(week1Monday);
  monday.setUTCDate(week1Monday.getUTCDate() + (week - 1) * 7);

  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date(monday);
    d.setUTCDate(monday.getUTCDate() + i);
    return d.toISOString().slice(0, 10);
  });
}

/** The ISO week (`YYYY-Www`) a calendar date (`YYYY-MM-DD`, or a Date) falls in. */
export function isoWeekOf(date: string | Date): string {
  const now = typeof date === "string" ? new Date(`${date}T12:00:00`) : date;
  const d = new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()));
  const dayNum = (d.getUTCDay() + 6) % 7; // Monday = 0
  d.setUTCDate(d.getUTCDate() - dayNum + 3); // nearest Thursday
  const firstThursday = new Date(Date.UTC(d.getUTCFullYear(), 0, 4));
  const week =
    1 +
    Math.round(
      ((d.getTime() - firstThursday.getTime()) / 86400000 -
        3 +
        ((firstThursday.getUTCDay() + 6) % 7)) /
        7,
    );
  return `${d.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** The ISO week `delta` weeks after (or, negative, before) `isoWeek`. */
export function shiftIsoWeek(isoWeek: string, delta: number): string {
  const monday = isoWeekDates(isoWeek)[0]!;
  const d = new Date(`${monday}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + delta * 7);
  return isoWeekOf(d.toISOString().slice(0, 10));
}

/** Matches `services/api/app/routers/weekly.py`'s real validation regex
 * `^\d{4}-W\d{2}$` (ISO 8601 week-numbering, verified against that
 * source). */
export function currentIsoWeek(): string {
  return isoWeekOf(new Date());
}
