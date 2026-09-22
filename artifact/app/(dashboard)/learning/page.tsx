"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { getLearning, getSettings, updateSettings, type Learning, type Settings } from "@/lib/api";
import { Hint } from "@/components/learning/Hint";
import { PickRateChart } from "@/components/learning/PickRateChart";
import { PreferenceLearning } from "@/components/learning/PreferenceLearning";
import { RatingsChart } from "@/components/learning/RatingsChart";
import "@/components/dashboard/bento.css";
import "@/components/library/library.css";

/** Task 10.11: ported from `renderLearning()`, wired to real
 * `GET /learning`. See `PickRateChart.tsx` for the chart's own real
 * differences from the mock (real daily data, no fabricated weekly
 * buckets or promotion markers). */
export default function LearningPage() {
  const { getToken } = useAuth();
  const [learning, setLearning] = useState<Learning | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    const [l, s] = await Promise.all([getLearning(token), getSettings(token)]);
    setLearning(l);
    setSettings(s);
  }, [getToken]);

  async function togglePersonalise(on: boolean) {
    if (!settings) return;
    const token = await getToken();
    if (!token) return;
    const next = { ...settings, personalise: on };
    setSettings(next);
    await updateSettings(token, next);
  }

  useEffect(() => {
    load();
  }, [load]);

  if (!learning) return null;

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">Learning</div>
          <h1>What your taps have taught it</h1>
        </div>
        <p>
          Every panel you pick teaches a small model what you like; watch it form below. It only
          takes over from the default ranking once it beats it on picks it predicted before seeing
          them.
        </p>
      </div>
      <div className="card row" style={{ marginBottom: 18, justifyContent: "space-between" }}>
        <div>
          <b>Use what it has learned about my taste</b>
          <p style={{ color: "var(--mute)", margin: 0 }}>
            When on, your defaults shape every new panel&apos;s prompt and the scriptwriter reads
            your taste notes. Off, you get the plain prompts, and it keeps learning from your
            picks, so switching back on is instant.
          </p>
        </div>
        <label className="row" style={{ gap: 8, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={settings?.personalise ?? true}
            onChange={(e) => void togglePersonalise(e.target.checked)}
          />
          <span className="mono">{settings?.personalise === false ? "off" : "on"}</span>
        </label>
      </div>
      <div className="card stack" style={{ marginBottom: 18 }}>
        <h2>How it&apos;s learning your taste</h2>
        <PreferenceLearning data={learning.preference} />
      </div>
      <div className="grid">
        <div className="card s8 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>First-choice agreement by day</h2>
            <span className="mono">how often your pick was the app&apos;s first choice</span>
          </div>
          <PickRateChart
            series={learning.pick_rate_series}
            label="First-choice agreement"
            chance={1 / 3}
          />
        </div>
        <div className="card s12 stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              How good are the images you keep?
              <Hint>
                A rating says whether the image in your strip is good at all. A pick among three
                can&apos;t tell you that, since the best of three weak options is still weak. The
                share rated great is the number that should climb.
              </Hint>
            </h2>
            <span className="mono">
              {learning.preference.accuracy.ratings?.n
                ? `${learning.preference.accuracy.ratings.great} great · ${learning.preference.accuracy.ratings.ok} ok · ${learning.preference.accuracy.ratings.off} off`
                : "your ratings"}
            </span>
          </div>
          <RatingsChart days={learning.ratings_by_day ?? []} />
        </div>
        <div className="card s4 stack soft">
          <div className="kick">Signals</div>
          <div className="row" style={{ gap: 18 }}>
            <div>
              <div className="big">{learning.taps ?? learning.pairs}</div>
              <span className="mono">quick picks</span>
            </div>
            <div>
              <div className="big">{learning.caption_edits}</div>
              <span className="mono">caption edits</span>
            </div>
          </div>
          <p>Next training run: {new Date(learning.next_run).toLocaleString()}.</p>
        </div>
        <div className="card s12">
          <h2 style={{ marginBottom: 10 }}>Checkpoints</h2>
          {learning.checkpoints.length === 0 ? (
            <p style={{ color: "var(--mute)" }}>No checkpoints registered yet.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Kind</th>
                  <th>Metrics</th>
                </tr>
              </thead>
              <tbody>
                {learning.checkpoints.map((c) => (
                  <tr key={c.id}>
                    <td>{new Date(c.created_at).toLocaleDateString()}</td>
                    <td>{c.kind}</td>
                    <td>
                      {/* Real metrics are heterogeneous (nested objects,
                          booleans, numbers) — only scalar entries are
                          worth a one-line summary; anything else stays
                          in the full JSON below rather than rendering
                          "[object Object]". */}
                      {Object.entries(c.metrics)
                        .filter(([, v]) => typeof v !== "object")
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(" · ")}
                      <details>
                        <summary className="mono" style={{ cursor: "pointer" }}>
                          full metrics
                        </summary>
                        <pre
                          style={{
                            whiteSpace: "pre-wrap",
                            fontFamily: "var(--mono)",
                            fontSize: 11,
                          }}
                        >
                          {JSON.stringify(c.metrics, null, 2)}
                        </pre>
                      </details>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  );
}
