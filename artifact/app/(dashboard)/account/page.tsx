"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { apiFetch, downloadExport } from "@/lib/api";
import "@/components/dashboard/bento.css";

/**
 * Ported from `renderAccount()`. Export and delete are real (`GET /account/export` + the same
 * `streamJobEvents` reader Today uses, `DELETE /account`). "Yearbook PDF" has no real endpoint —
 * kept disabled and labeled rather than faked.
 *
 * This app has no sign-in (single-user local mode, `@/lib/auth`), so there's no Profile/password
 * form and "Delete my diary" doesn't sign out or redirect anywhere.
 */
export default function AccountPage() {
  const { getToken } = useAuth();

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  async function handleExport() {
    setExportError(null);
    const token = await getToken();
    if (!token) return;
    setExporting(true);
    try {
      await downloadExport(token);
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  }

  async function handleDelete() {
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    const token = await getToken();
    if (!token) return;
    setDeleting(true);
    try {
      await apiFetch("/account", token, { method: "DELETE" });
      // Public demo: apiFetch only shows the notice modal and never really
      // deletes anything, so the UI shouldn't flip to "deleted" either —
      // that would tell a visitor something happened when it didn't.
      setConfirmingDelete(false);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">Account</div>
          <h1>Your data, your book</h1>
        </div>
      </div>
      <div className="grid">
        <div className="card s6 stack">
          <h2>Export</h2>
          <p>
            Everything, in your hands: strips as PNG, your diary entries and beats as JSON, the
            weekly PDFs, and your identity photos.
          </p>
          <div className="row">
            <button type="button" className="btn sm" onClick={handleExport} disabled={exporting}>
              {exporting ? "Exporting…" : "Export all"}
            </button>
            <button type="button" className="btn sm" disabled title="No real endpoint for this yet">
              Yearbook PDF
            </button>
          </div>
          {exportError && <div className="err">{exportError}</div>}
        </div>
        <div className="card s12 stack" style={{ borderColor: "#C9722E" }}>
          <h2>Delete everything</h2>
          <p style={{ color: "var(--mute)" }}>
            Removes your account, photos, strips, and training pairs from our systems. Recurring
            characters you invited are notified.
          </p>
          <div>
            <button
              type="button"
              className="btn sm"
              style={{ borderColor: "#C9722E", color: "#7A3E10" }}
              onClick={handleDelete}
              disabled={deleting}
            >
              {confirmingDelete ? "Click again to confirm" : "Delete my diary"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
