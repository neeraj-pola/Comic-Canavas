import type { PreferenceLearning } from "@/lib/api";
import { shortDate } from "@/lib/preference";

/** The written taste card: a few plain sentences the scriptwriter reads before it
 * writes your day. Empty until there are enough picks (about 30 over 10+ days). */
export function TasteNotes({ notes }: { notes: PreferenceLearning["notes"] }) {
  const items = "items" in notes && notes.items ? notes.items : [];
  const updated = "updated_at" in notes && notes.updated_at ? notes.updated_at.slice(0, 10) : null;
  if (!updated) {
    return (
      <p style={{ color: "var(--mute)" }}>
        Nothing written yet. After about 30 picks spread over 10 days, it writes a few lines about
        your taste (framing, expression, caption voice) that the scriptwriter reads before it
        writes your day.
      </p>
    );
  }
  return (
    <div className="stack" style={{ gap: 8 }}>
      {items.length > 0 ? (
        <ul className="pl-notes">
          {items.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      ) : (
        <p style={{ color: "var(--mute)" }}>
          It looked and found nothing it is sure about yet, so no lines were written.
        </p>
      )}
      <span className="mono">updated {shortDate(updated)} · refreshed weekly</span>
    </div>
  );
}
