"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useAuth, useUser } from "@/lib/auth";
import {
  createPerson,
  getCast,
  getSettings,
  pickMaster,
  presignPersonPhoto,
  processPersonPhotos,
  trainPerson,
  updatePersonDetails,
  updateSettings,
  uploadToPresignedUrl,
  type Person,
  type PersonTraining,
  type Settings,
} from "@/lib/api";
import { STYLE_PACKS } from "@/lib/stylePacks";
import "@/components/dashboard/bento.css";
import "@/components/library/library.css";
import "@/components/settings/settings.css";

const STEP_TITLES = ["Your photos", "Your look", "Your style", "Your evening"];

const TRAIN_STAGES: { key: NonNullable<PersonTraining["stage"]>; label: string }[] = [
  { key: "photos", label: "Checking your photos" },
  { key: "look_card", label: "Writing your look card" },
  { key: "candidates", label: "Drawing 24 character options, the slow part, usually 3 to 6 min" },
  { key: "ranking", label: "Ranking the results and keeping your top 5" },
];

function TrainingProgress({ training, now }: { training: PersonTraining; now: number }) {
  const stageIndex = TRAIN_STAGES.findIndex((s) => s.key === training.stage);
  const elapsed = Math.max(0, Math.floor((now - new Date(training.started_at).getTime()) / 1000));
  const mm = Math.floor(elapsed / 60);
  const ss = String(elapsed % 60).padStart(2, "0");
  return (
    <div className="stack" role="status" aria-live="polite">
      <span className="pill mus">
        Training in progress · {mm}:{ss}
      </span>
      <progress
        value={Math.max(stageIndex, 0)}
        max={TRAIN_STAGES.length}
        style={{ width: "100%" }}
      />
      <ol style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 6 }}>
        {TRAIN_STAGES.map((s, i) => (
          <li
            key={s.key}
            style={{
              fontWeight: i === stageIndex ? 700 : 400,
              color: i < stageIndex ? "var(--mute)" : "inherit",
              opacity: i > stageIndex ? 0.5 : 1,
            }}
          >
            {i < stageIndex ? "✓ " : i === stageIndex ? "● " : ""}
            {s.label}
          </li>
        ))}
      </ol>
      <span className="mono">You can leave this page: it keeps running.</span>
    </div>
  );
}

/**
 * Ported from `renderOnboarding(step)` — the source renders this inside
 * the same shell (`#p-detail`), not as a standalone full-screen flow, so
 * this lives under `(dashboard)` like every other real page.
 *
 * Step 2's look card is real but read-only (no `PUT` endpoint exists for
 * it — it's written once by `POST /cast/{id}/train`, a real,
 * confirm-gated, real-money fal.ai call, not free-text editing), and
 * there's no "test render" preview endpoint. Steps 3/4 are fully real,
 * via `GET/PUT /settings`.
 */
export default function SetupStepPage({ params }: { params: Promise<{ step: string }> }) {
  const { step: stepParam } = use(params);
  const step = Math.min(4, Math.max(1, Number(stepParam) || 1));
  const { getToken } = useAuth();
  const { user } = useUser();

  const [person, setPerson] = useState<Person | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadCount, setUploadCount] = useState(0);
  const [ageInput, setAgeInput] = useState("");
  const [training, setTraining] = useState(false);
  // Photo upload/processing (presign -> PUT -> `/process`) otherwise runs
  // with zero visible feedback, looking identical to "nothing happened"
  // until the usable-photo count jumps. `uploadPhase` tracks the real
  // steps as they happen, not a fake timer.
  const [uploadPhase, setUploadPhase] = useState<"idle" | "uploading" | "processing" | "done">(
    "idle",
  );
  const [uploadProgress, setUploadProgress] = useState({ done: 0, total: 0 });
  const fileInput = useRef<HTMLInputElement | null>(null);
  // React's dev-mode double effect can run `load()` twice concurrently;
  // without guarding, both would see "no people yet" and both would
  // create one (two paid training runs later). A single shared in-flight
  // promise makes person creation once-only.
  const creating = useRef<Promise<string> | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    const [{ people }, real] = await Promise.all([getCast(token), getSettings(token)]);
    let you = people[0] ?? null;
    if (!you) {
      creating.current ??= createPerson(token, user?.firstName ?? "You").then((c) => c.id);
      const createdId = await creating.current;
      const refreshed = await getCast(token);
      you = refreshed.people.find((p) => p.id === createdId) ?? null;
    }
    setPerson(you);
    // Don't clobber what's being typed on a background reload (training polls).
    setAgeInput((current) => current || (you?.age ? String(you.age) : ""));
    setSettings(real);
    setLoading(false);
  }, [getToken, user]);

  useEffect(() => {
    load();
  }, [load]);

  const trainingStatus = person?.training?.status;
  const trainingActive = trainingStatus === "pending" || trainingStatus === "running";
  const characterReady = Boolean(person?.identity?.master_path);
  const masterOptions = person?.identity?.master_candidates ?? [];
  const choosingMaster = !characterReady && !trainingActive && masterOptions.length > 0;
  const [picking, setPicking] = useState<string | null>(null);

  async function handlePick(candidateId: string) {
    if (!person) return;
    setError(null);
    setPicking(candidateId);
    try {
      const token = await getToken();
      if (!token) return;
      await pickMaster(token, person.id, candidateId);
      window.dispatchEvent(new Event("cc:profile-changed")); // top-bar avatar
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your choice.");
    } finally {
      setPicking(null);
    }
  }

  // Poll the real training state every 4s while a run is in flight (and
  // tick a local clock once a second for the elapsed timer).
  useEffect(() => {
    if (!trainingActive) return;
    const poll = setInterval(() => void load(), 4000);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  }, [trainingActive, load]);

  /** Gender is asked here, at the very start of onboarding, and passed straight into the real
   * `people.gender_term` column — `train_character_job` reads it back and threads it into every
   * panel's `character_clause`. */
  async function handleGenderChange(genderTerm: string) {
    if (!person) return;
    const token = await getToken();
    if (!token) return;
    setPerson({ ...person, gender_term: genderTerm });
    await updatePersonDetails(token, person.id, { gender_term: genderTerm });
  }

  /** Age is saved on blur, only when it's a plausible whole number — it becomes
   * "{age}-year-old" on the look card. */
  async function handleAgeCommit() {
    if (!person) return;
    const age = Number(ageInput);
    if (!Number.isInteger(age) || age < 1 || age > 120 || age === person.age) return;
    const token = await getToken();
    if (!token) return;
    setPerson({ ...person, age });
    await updatePersonDetails(token, person.id, { age });
  }

  async function handlePhotos(files: FileList | null) {
    if (!files || files.length === 0 || !person) return;
    setError(null);
    const token = await getToken();
    if (!token) return;
    const fileList = Array.from(files);
    setUploadPhase("uploading");
    setUploadProgress({ done: 0, total: fileList.length });
    try {
      for (const file of fileList) {
        const { upload_url } = await presignPersonPhoto(
          token,
          person.id,
          file.name,
          file.type || "image/jpeg",
        );
        await uploadToPresignedUrl(upload_url, file);
        setUploadProgress((p) => ({ ...p, done: p.done + 1 }));
      }
      setUploadCount((n) => n + fileList.length);
      setUploadPhase("processing");
      await processPersonPhotos(token, person.id);
      await load();
      setUploadPhase("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Photo upload failed.");
      setUploadPhase("idle");
    }
  }

  async function handleTrain() {
    if (!person) return;
    setError(null);
    const token = await getToken();
    if (!token) return;
    setTraining(true);
    try {
      await trainPerson(token, person.id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start training.");
    } finally {
      setTraining(false);
    }
  }

  async function saveSettings(patch: Partial<Settings>) {
    const token = await getToken();
    if (!token || !settings) return;
    const next = { ...settings, ...patch };
    setSettings(next);
    await updateSettings(token, next);
  }

  if (loading || !settings) return null;

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">Set up · step {step} of 4</div>
          <h1>{STEP_TITLES[step - 1]}</h1>
        </div>
        <div className="row">
          {STEP_TITLES.map((t, i) => (
            <span key={t} className={`pill ${i + 1 === step ? "mus" : i + 1 < step ? "ok" : ""}`}>
              {i + 1}. {t}
            </span>
          ))}
        </div>
      </div>
      {error && <div className="err">{error}</div>}

      {step === 1 && (
        <div className="grid">
          <div className="card s12 stack">
            <h2>How should we describe you?</h2>
            <p style={{ color: "var(--mute)" }}>
              This helps your character look like you and stay consistent panel to panel. We never
              guess this from your photos.
            </p>
            <div className="row">
              {[
                { value: "man", label: "Man" },
                { value: "woman", label: "Woman" },
                { value: "", label: "Prefer not to say" },
              ].map((option) => (
                <button
                  key={option.label}
                  type="button"
                  className={`btn sm ${person?.gender_term === option.value ? "pri" : ""}`}
                  onClick={() => handleGenderChange(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <label className="f" style={{ maxWidth: 200 }}>
              Your age
              <input
                type="number"
                inputMode="numeric"
                min={1}
                max={120}
                placeholder="e.g. 34"
                value={ageInput}
                onChange={(e) => setAgeInput(e.target.value)}
                onBlur={handleAgeCommit}
              />
            </label>
          </div>
          <div className="card s7 stack">
            <h2>Add 15 to 25 photos of yourself</h2>
            <p style={{ color: "var(--mute)" }}>
              Different angles, lighting, and expressions. No sunglasses. Two or three in your usual
              outfit help the character wear your clothes.
            </p>
            <label
              className="thumb"
              style={{
                aspectRatio: 1,
                display: "grid",
                placeItems: "center",
                borderStyle: "dashed",
                color: "var(--mute)",
                fontWeight: 700,
                cursor: uploadPhase === "idle" || uploadPhase === "done" ? "pointer" : "default",
                maxWidth: 160,
                opacity: uploadPhase === "idle" || uploadPhase === "done" ? 1 : 0.5,
              }}
            >
              + upload
              <input
                ref={fileInput}
                type="file"
                multiple
                accept="image/*"
                disabled={uploadPhase === "uploading" || uploadPhase === "processing"}
                style={{ display: "none" }}
                onChange={(e) => handlePhotos(e.target.files)}
              />
            </label>
            {uploadPhase === "uploading" && (
              <div className="stack" style={{ gap: 6, maxWidth: 300 }}>
                <span className="mono">
                  Uploading photo {uploadProgress.done + 1} of {uploadProgress.total}…
                </span>
                <progress
                  value={uploadProgress.done}
                  max={uploadProgress.total}
                  style={{ width: "100%" }}
                />
              </div>
            )}
            {uploadPhase === "processing" && (
              <div className="row">
                <span className="pill mus">Detecting faces and embedding photos…</span>
              </div>
            )}
            <div className="row">
              <span className="pill ok">{person?.identity?.source_photo_count ?? 0} usable</span>
              <span className="mono">{uploadCount} uploaded this session</span>
            </div>
          </div>
          <div className="card s5 stack soft">
            <div className="kick">What happens next</div>
            <p>
              Faces are detected and cropped, an identity embedding is stored for the critic, and a
              small model of you is trained. You can delete all of it any time.
            </p>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="grid">
          {choosingMaster && (
            <div className="card s12 stack">
              <h2>Which one feels like you?</h2>
              <p style={{ color: "var(--mute)" }}>
                These are your top {masterOptions.length}. This becomes the face in every panel, so
                take your time: glasses, hair and face shape are what matter most.
              </p>
              <div
                style={{
                  display: "grid",
                  gap: 14,
                  gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))",
                }}
              >
                {masterOptions.map((option, i) => (
                  <button
                    key={option.id}
                    type="button"
                    className="card stack"
                    disabled={picking !== null}
                    onClick={() => handlePick(option.id)}
                    style={{ padding: 10, cursor: "pointer", textAlign: "left" }}
                    aria-label={`Use option ${i + 1} as my character`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={option.url}
                      alt={`Character option ${i + 1}`}
                      style={{ width: "100%", borderRadius: 10, border: "1.5px solid var(--ink)" }}
                    />
                    <span className="row" style={{ justifyContent: "space-between" }}>
                      <b>Option {i + 1}</b>
                      {i === 0 && <span className="pill ok">best match</span>}
                    </span>
                    <span className="btn pri sm">
                      {picking === option.id ? "Saving…" : "Use this one"}
                    </span>
                  </button>
                ))}
              </div>
              <div className="row">
                <button className="btn sm" onClick={handleTrain} disabled={training}>
                  {training ? "Starting…" : "None of these, show new options (~$0.50)"}
                </button>
              </div>
            </div>
          )}
          <div className="card s7 stack">
            <h2>Your look card</h2>
            <p style={{ color: "var(--mute)" }}>
              Written from your photos once training runs. This is what the writer uses to describe
              you. Editing it directly isn&apos;t wired yet (no endpoint for it exists).
            </p>
            {person?.identity?.look_card ? (
              <pre style={{ whiteSpace: "pre-wrap", fontFamily: "var(--mono)", fontSize: 12 }}>
                {JSON.stringify(person.identity.look_card, null, 2)}
              </pre>
            ) : (
              <p style={{ color: "var(--mute)" }}>No look card yet: train your character below.</p>
            )}
          </div>
          <div className="card s5 stack">
            <div className="kick">Train your character</div>
            {characterReady ? (
              <>
                <span className="pill ok">Character ready</span>
                {person?.identity?.master_path && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={person.identity.master_path}
                    alt="Your comic character"
                    style={{ width: "100%", borderRadius: 14, border: "1.5px solid var(--ink)" }}
                  />
                )}
                <Link className="btn pri sm" href="/setup/3">
                  Looks good, continue
                </Link>
              </>
            ) : choosingMaster ? (
              <p>Pick the one that feels most like you: see the options on the left.</p>
            ) : trainingActive ? (
              <TrainingProgress training={person!.training!} now={now} />
            ) : (
              <>
                <p>
                  {trainingStatus === "failed"
                    ? "Training didn't finish. You can try again."
                    : "Uses real photo processing (fal.ai). This spends real money."}
                </p>
                {trainingStatus === "failed" && person?.training?.error && (
                  <div className="err">{person.training.error}</div>
                )}
                <button className="btn pri sm" onClick={handleTrain} disabled={training || !person}>
                  {training
                    ? "Starting…"
                    : trainingStatus === "failed"
                      ? "Try again"
                      : "Train character"}
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="grid">
          <div className="card s12 stack">
            <h2>Pick a style</h2>
            <p style={{ color: "var(--mute)" }}>
              Characters stay simple so they stay consistent; backgrounds get the detail. You can
              switch later without losing anything.
            </p>
            <div className="styles">
              {STYLE_PACKS.map((pack) => (
                <button
                  key={pack.key}
                  type="button"
                  className={`sty ${settings.style === pack.key ? "on" : ""}`}
                  onClick={() => saveSettings({ style: pack.key })}
                >
                  <div className="sw" style={{ background: pack.gradient }} />
                  <b>{pack.name}</b>
                </button>
              ))}
            </div>
            <label className="f" style={{ maxWidth: 420 }}>
              Humor
              <div className="range">
                <span className="mono">gentle</span>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={settings.humor}
                  onChange={(e) => saveSettings({ humor: Number(e.target.value) })}
                />
                <span className="mono">roast</span>
              </div>
            </label>
          </div>
        </div>
      )}

      {step === 4 && (
        <div className="grid">
          <div className="card s6 stack">
            <h2>When should it ask about your day?</h2>
            <label className="f">
              Evening nudge
              <input
                type="time"
                value={settings.reminder_time ?? "20:30"}
                onChange={(e) => saveSettings({ reminder_time: e.target.value })}
              />
            </label>
            <label className="f">
              Channel
              <select
                value={settings.channel ?? "push"}
                onChange={(e) => saveSettings({ channel: e.target.value })}
              >
                <option value="push">Push notification</option>
                <option value="telegram">Telegram</option>
                <option value="email">Email</option>
              </select>
            </label>
          </div>
          <div className="card s6 stack soft">
            <div className="kick">You&apos;re set</div>
            <h2>Draw your first strip tonight</h2>
            <p>Or do it now with a few lines about today.</p>
            <Link className="btn pri" href="/today">
              Go to Today
            </Link>
          </div>
        </div>
      )}

      {/* A sticky, clearly bordered footer bar for step navigation, so it's
          never scrolled out of view and always reads as "the way to
          continue," not a stray pair of buttons. */}
      <div
        className="card row"
        style={{
          marginTop: 18,
          position: "sticky",
          bottom: 16,
          justifyContent: "space-between",
          zIndex: 5,
        }}
      >
        {step > 1 ? (
          <Link className="btn" href={`/setup/${step - 1}`}>
            ← Back
          </Link>
        ) : (
          <span />
        )}
        <span className="mono">Step {step} of 4</span>
        {step < 4 ? (
          <Link className="btn pri" href={`/setup/${step + 1}`}>
            Continue to &quot;{STEP_TITLES[step]}&quot; →
          </Link>
        ) : (
          <span />
        )}
      </div>
    </>
  );
}
