"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { getSettings, updateSettings, type Settings } from "@/lib/api";
import "@/components/dashboard/bento.css";
import "@/components/settings/settings.css";

/**
 * Ported from `renderStyle()`. Every field (`humor`, `caption_length`,
 * `sensitive_mode`) is a real column on `GET/PUT /settings`.
 * `sensitive_mode` is a real boolean, mapped onto two choices ("off
 * automatically" vs "ask me first").
 */
export default function StylePage() {
  const { getToken } = useAuth();
  const [settings, setSettings] = useState<Settings | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    setSettings(await getSettings(token));
  }, [getToken]);

  useEffect(() => {
    load();
  }, [load]);

  async function save(patch: Partial<Settings>) {
    const token = await getToken();
    if (!token || !settings) return;
    const next = { ...settings, ...patch };
    setSettings(next);
    await updateSettings(token, next);
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  if (!settings) return null;

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">Style &amp; voice</div>
          <h1>How your diary looks and sounds</h1>
        </div>
        <p>
          Changes apply from tonight&apos;s strip.
          {saved && (
            <span className="pill ok" style={{ marginLeft: 10 }}>
              Saved
            </span>
          )}
        </p>
      </div>
      <div className="grid">
        <div className="card s12 stack">
          <h2>Voice</h2>
          <label className="f">
            Humor
            <div className="range">
              <span className="mono">gentle</span>
              <input
                type="range"
                min={0}
                max={10}
                value={settings.humor}
                onChange={(e) => save({ humor: Number(e.target.value) })}
              />
              <span className="mono">roast</span>
            </div>
          </label>
          <label className="f">
            Caption length
            <select
              value={settings.caption_length ?? "short"}
              onChange={(e) => save({ caption_length: e.target.value })}
            >
              <option value="short">Short (≤ 60 chars)</option>
              <option value="medium">Medium</option>
            </select>
          </label>
          <label className="f">
            Sensitive days
            <select
              value={settings.sensitive_mode ? "auto" : "ask"}
              onChange={(e) => save({ sensitive_mode: e.target.value === "auto" })}
            >
              <option value="auto">Turn humor off automatically</option>
              <option value="ask">Ask me first</option>
            </select>
          </label>
        </div>
      </div>
    </>
  );
}
