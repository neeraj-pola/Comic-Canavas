"use client";

import { type KeyboardEvent, useEffect, useId, useState } from "react";
import type { Candidate, Panel } from "@/lib/api";
import "./strip.css";

const MAX_CAPTION = 60; // `Panel.caption_a` in the contract

/** Today's strip. Each panel has an editable caption (click it, edit, Save) and a
 * per-panel "Regenerate" with an optional note about what should change, both with
 * explicit save/submit actions and visible feedback while they run. */
export function Strip({
  panels,
  candidates,
  regenerating,
  onCaptionSave,
  onRegenerate,
}: {
  panels: Panel[];
  candidates: Candidate[];
  regenerating: Set<number>;
  onCaptionSave: (panelId: number, original: string, edited: string) => Promise<void>;
  onRegenerate: (panelId: number, note: string) => Promise<void>;
}) {
  const chosenUrl = (panelId: number): string | null =>
    candidates.find((c) => c.panel_id === panelId && c.chosen)?.url ?? null;

  return (
    <div className={`strip ${panels.length > 4 ? "six" : ""}`}>
      {panels.map((panel) => (
        <PanelCard
          key={panel.id}
          panel={panel}
          url={chosenUrl(panel.id)}
          busy={regenerating.has(panel.id)}
          onCaptionSave={onCaptionSave}
          onRegenerate={onRegenerate}
        />
      ))}
    </div>
  );
}

function PanelCard({
  panel,
  url,
  busy,
  onCaptionSave,
  onRegenerate,
}: {
  panel: Panel;
  url: string | null;
  busy: boolean;
  onCaptionSave: (panelId: number, original: string, edited: string) => Promise<void>;
  onRegenerate: (panelId: number, note: string) => Promise<void>;
}) {
  const noteId = useId();
  const [caption, setCaption] = useState(panel.caption_a);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(panel.caption_a);
  const [saving, setSaving] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [note, setNote] = useState("");

  // A reload (after a regenerate, or another edit) brings the saved caption back in.
  useEffect(() => {
    if (!editing) setCaption(panel.caption_a);
  }, [panel.caption_a, editing]);

  function startEdit() {
    setDraft(caption);
    setError(null);
    setEditing(true);
  }

  async function save() {
    const edited = draft.trim();
    if (!edited || edited === caption) {
      setEditing(false);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await onCaptionSave(panel.id, caption, edited);
      setCaption(edited); // the saved text is the next "original"
      setEditing(false);
      setSavedFlash(true);
      setTimeout(() => setSavedFlash(false), 3500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the caption.");
    } finally {
      setSaving(false);
    }
  }

  function onCaptionKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void save();
    } else if (event.key === "Escape") {
      setEditing(false);
    }
  }

  async function submitRegenerate() {
    const text = note.trim();
    setAsking(false);
    setNote("");
    await onRegenerate(panel.id, text);
  }

  return (
    <div className={`pnl${busy ? " regenerating" : ""}`}>
      {url ? (
        // real, dynamically-generated media URLs served by
        // LocalFileStorage, not a static asset next/image would help with
        // eslint-disable-next-line @next/next/no-img-element
        <img src={url} alt={panel.action} />
      ) : (
        <div style={{ aspectRatio: "1", background: "var(--field)" }} />
      )}

      {editing ? (
        <div className="cap-edit">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onCaptionKey}
            maxLength={MAX_CAPTION}
            rows={2}
            autoFocus
            disabled={saving}
            aria-label={`Caption for panel ${panel.id}`}
          />
          <div className="row">
            <span className="mono">
              {draft.length}/{MAX_CAPTION}
            </span>
            <button
              type="button"
              className="btn sm pri"
              onClick={() => void save()}
              disabled={saving || !draft.trim() || draft.trim() === caption}
            >
              {saving ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              className="btn sm"
              onClick={() => setEditing(false)}
              disabled={saving}
            >
              Cancel
            </button>
          </div>
          {error && <div className="err">{error}</div>}
        </div>
      ) : (
        <button
          type="button"
          className="c cap-btn"
          onClick={startEdit}
          aria-label={`Edit caption: ${caption}`}
          disabled={busy}
        >
          <span>{caption}</span>
          <span className="cap-pen" aria-hidden="true">
            ✎
          </span>
        </button>
      )}

      {busy ? (
        <div className="regen-status" role="status">
          <span className="spin" aria-hidden="true" /> Regenerating, usually 1 to 2 minutes…
        </div>
      ) : asking ? (
        <div className="regen-box">
          <label htmlFor={noteId}>
            What should change? <span className="mono">(optional)</span>
          </label>
          <textarea
            id={noteId}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            maxLength={300}
            rows={3}
            autoFocus
            placeholder={'e.g. "make me look less tired" or "a sunnier background"'}
          />
          <div className="row">
            <button type="button" className="btn sm pri" onClick={() => void submitRegenerate()}>
              Regenerate
            </button>
            <button type="button" className="btn sm" onClick={() => setAsking(false)}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="pnl-actions">
          {savedFlash && <span className="saved">Saved ✓, the strip is updating</span>}
          <button
            type="button"
            className="btn sm"
            onClick={() => setAsking(true)}
            disabled={editing}
          >
            Regenerate this panel
          </button>
        </div>
      )}
    </div>
  );
}
