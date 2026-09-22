import { streamJobEvents } from "./sse";
import type { components } from "./api-types.generated";

/**
 * `pnpm gen:types` runs `openapi-typescript` against `docs/openapi.json`
 * into `api-types.generated.ts` (committed, regenerable, but needed for a
 * fresh clone to type-check without a running API). Every router in
 * `services/api` returns a plain `dict`, never a Pydantic
 * `response_model=`, so every GET/response schema in the OpenAPI doc is
 * untyped at that layer — only request *bodies* (the `*In` Pydantic
 * models FastAPI evaluates from typed parameters) carry real generated
 * shapes. So this file's request-body types below are aliases of the
 * generated `components["schemas"]["*In"]` types (real drift protection
 * against the backend contracts), while every response type stays the
 * hand-written, live-data-verified shape used throughout this file.
 */
type CaptionInBody = components["schemas"]["CaptionIn"];
type PairInBody = components["schemas"]["PairIn"];
type PersonInBody = components["schemas"]["PersonIn"];
type PhotoPresignInBody = components["schemas"]["PhotoPresignIn"];
type SettingsInBody = components["schemas"]["SettingsIn"];

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * Attaches a bearer token to a call to the Comic Canvas API. The token
 * comes from `lib/auth.ts`'s `useAuth().getToken()` — this module has no
 * session state of its own.
 */
export async function apiFetch(
  path: string,
  token: string | null,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}

/**
 * Confirms a token is reachable and, via the API's own real
 * upsert-on-first-authenticated-call behavior (`app.deps.current_user_id`),
 * creates the user row. There's no dedicated `/auth` or `/me` endpoint
 * (see `docs/api.md`) — any cheap authenticated GET does this; `/settings`
 * is the smallest real one.
 */
export async function verifyAndBootstrapUser(token: string): Promise<void> {
  const response = await apiFetch("/settings", token);
  if (!response.ok) {
    throw new ApiError(response.status, `Could not reach Comic Canvas (${response.status}).`);
  }
}

/** Shapes below are transcribed verbatim from `services/api/app/routers/
 * days.py`'s real dict responses. */
export type Beat = {
  id: string;
  time: string;
  place: string;
  place_detail: string | null;
  event: string;
  emotion: string;
  people: string[];
  objects: string[];
  importance: number;
  humor: number;
  quote: string | null;
};

export type ImagePrompt = {
  id: string;
  panel_id: number;
  generator: string;
  character_clause: string;
  environment_clause: string;
  positive: string;
  negative: string;
  seed: number | null;
  guidance: number | null;
  steps: number | null;
};

export type Panel = {
  id: number;
  beat_id: string;
  place: string;
  time_of_day: string;
  expression: string;
  action: string;
  framing: string;
  caption_a: string;
  caption_b: string;
  bubble: string;
  cast: string[];
};

export type Candidate = {
  id: string;
  panel_id: number;
  url: string;
  seed: number;
  scores: Record<string, number>;
  face_box: [number, number, number, number] | null;
  chosen: boolean;
  rejected_reason: string | null;
  prompt_id: string | null;
};

export type Day = {
  /** The pick you made per panel (`panel_id` -> the image url you chose). */
  picks?: Record<string, string>;
  /** Your rating per image url: 0 off, 1 ok, 2 great. */
  ratings?: Record<string, number>;
  date: string;
  source: "text" | "audio";
  text: string | null;
  transcript: string | null;
  mood: string | null;
  quiet_day: boolean | null;
  strip_url: string | null;
  story_url: string | null;
  layout: Record<string, unknown> | null;
  cost_usd: number;
  versions: Record<string, string>;
  beats: Beat[];
  panels: Panel[];
  candidates: Candidate[];
  prompts: ImagePrompt[];
};

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

/** `POST /days` always uses the server's own "today" — there is no way
 * to create a day for an arbitrary date (verified against the real
 * route: `today = date_cls.today().isoformat()`, not a request field). */
export async function createDay(
  token: string,
  body: { text?: string; audio_url?: string; date?: string },
): Promise<{ job_id: string; status: string }> {
  const response = await apiFetch("/days", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

export async function getDay(token: string, date: string): Promise<Day | null> {
  const response = await apiFetch(`/days/${date}`, token);
  if (response.status === 404) return null;
  return json(response);
}

export async function submitCaptionFeedback(token: string, body: CaptionInBody): Promise<void> {
  const response = await apiFetch("/feedback/caption", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await json(response);
}

/** Rate one image: 0 off, 1 ok, 2 great (usually the one in your strip). */
export async function submitRating(
  token: string,
  body: components["schemas"]["RatingIn"],
): Promise<void> {
  const response = await apiFetch("/feedback/rating", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await json(response);
}

/** Best-of-3 quick tap: the favourite of a panel's options (`others` = the ones passed on). */
export async function submitPick(
  token: string,
  body: components["schemas"]["PickIn"],
): Promise<void> {
  const response = await apiFetch("/feedback/pick", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await json(response);
}

export async function submitPairFeedback(token: string, body: PairInBody): Promise<void> {
  const response = await apiFetch("/feedback/pair", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await json(response);
}

export async function regeneratePanel(
  token: string,
  date: string,
  panelId: number,
  note: string,
): Promise<{ job_id: string; status: string }> {
  const body: components["schemas"]["RegenerateIn"] = { note };
  const response = await apiFetch(`/days/${date}/panels/${panelId}/regenerate`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

/** `services/api/app/routers/library.py`'s real response shapes. */
export type LibraryDay = {
  date: string;
  mood: string | null;
  quiet_day: boolean | null;
  strip_url: string | null;
  thumbs: number[];
};

export async function getLibrary(
  token: string,
  month: string,
  mood?: string,
): Promise<{ month: string; days: LibraryDay[] }> {
  const params = new URLSearchParams({ month });
  if (mood) params.set("mood", mood);
  const response = await apiFetch(`/library?${params}`, token);
  return json(response);
}

export type SearchHit = {
  date: string;
  beat_id: number;
  event: string | null;
  place: string | null;
  quote: string | null;
  panel_urls: string[];
};

export async function search(token: string, q: string): Promise<{ q: string; hits: SearchHit[] }> {
  const response = await apiFetch(`/search?q=${encodeURIComponent(q)}`, token);
  return json(response);
}

/** `services/api/app/routers/weekly.py`'s real response shapes. */
export type WeeklyPanel = { id: number; caption: string; url: string };
export type WeeklyContext = {
  week_label: string;
  date_range: string;
  user_label: string;
  mood: string | null;
  quiet_day: boolean | null;
  strip_url: string | null;
  story_url: string | null;
  panels: WeeklyPanel[];
  mood_series: string[];
  stats: { days: number; panels: number; pairs: number; first_pick_pct: number };
  cast: { name: string }[];
  training_run: { id: string; kind: string; decision: string; created_at: string } | null;
};

export async function getWeekly(token: string, isoWeek: string): Promise<WeeklyContext | null> {
  const response = await apiFetch(`/weekly/${isoWeek}`, token);
  if (response.status === 404) return null;
  return json(response);
}

export async function generateWeekly(
  token: string,
  isoWeek: string,
): Promise<{ job_id: string; status: string }> {
  const response = await apiFetch(`/weekly/${isoWeek}/generate`, token, { method: "POST" });
  return json(response);
}

/** `GET /weekly/{iso_week}.pdf` needs the bearer header too — a plain
 * `<a href>` download wouldn't send it (same real problem `lib/sse.ts`
 * solves for the job-events stream). Fetches the real PDF bytes with
 * auth, then triggers a normal browser download via a blob URL. */
export async function downloadWeeklyPdf(token: string, isoWeek: string): Promise<void> {
  const response = await apiFetch(`/weekly/${isoWeek}.pdf`, token);
  if (!response.ok) {
    throw new ApiError(response.status, `Could not download the PDF (${response.status}).`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `comiccanvas-${isoWeek}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

/** `services/api/app/routers/cast.py`'s real response shapes. */
/** One of the top-ranked master options `train_character_job` saved for the
 * person to choose from (`identity_models.master_candidates`). */
export type MasterCandidate = {
  id: string;
  url: string;
  seed: number;
  rank_score: number;
  style_score: number;
  checklist_score: number;
  style_card_version: string;
};

export type PersonIdentity = {
  person_id: string;
  source_photo_count: number;
  mean_self_cosine: number | null;
  look_card: Record<string, unknown> | null;
  master_path: string | null;
  generation_path: string | null;
  master_candidates: MasterCandidate[] | null;
};

/** Real training progress (`jobs` row `train:{person_id}`, surfaced by
 * `GET /cast`). `null` = never started. `stage` is the latest of
 * photos → look_card → candidates → master. */
export type PersonTraining = {
  status: "pending" | "running" | "done" | "failed";
  stage: "photos" | "look_card" | "candidates" | "ranking" | "review" | null;
  error: string | null;
  started_at: string;
  updated_at: string;
};

export type Person = {
  id: string;
  name: string;
  // `LookCard.gender_term` is a real, user-provided value, never
  // vision-inferred (guessing gender from photos risks misgendering) —
  // collected up front in the setup wizard, not inferred.
  gender_term: string;
  /** Asked up front alongside gender; becomes "{age}-year-old" on the
   * look card. `null` = not provided. */
  age: number | null;
  consent_at: string | null;
  revoked_at: string | null;
  created_at: string;
  identity: PersonIdentity | null;
  training: PersonTraining | null;
};

export async function getCast(token: string): Promise<{ people: Person[] }> {
  const response = await apiFetch("/cast", token);
  return json(response);
}

export async function createPerson(
  token: string,
  name: string,
  genderTerm = "",
): Promise<{ id: string; name: string; gender_term: string; created_at: string }> {
  const body: PersonInBody = { name, gender_term: genderTerm };
  const response = await apiFetch("/cast", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

/** For the common real case: a person is auto-created (just a name) the
 * moment the setup wizard's first screen loads, before there's been any
 * chance to ask for a gender — this fills it in right after. */
export async function updatePersonDetails(
  token: string,
  personId: string,
  details: { gender_term?: string; age?: number },
): Promise<{ id: string; name: string; gender_term: string; age: number | null }> {
  const body: components["schemas"]["PersonPatchIn"] = details;
  const response = await apiFetch(`/cast/${personId}`, token, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

export async function presignPersonPhoto(
  token: string,
  personId: string,
  filename: string,
  contentType: string,
): Promise<{ id: number; upload_url: string; key: string }> {
  const body: PhotoPresignInBody = { filename, content_type: contentType };
  const response = await apiFetch(`/cast/${personId}/photos/presign`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

/** The presigned `upload_url` is itself the full, already-signed target
 * — a plain unauthenticated PUT of the raw bytes, matching
 * `LocalFileStorage.put_presigned`'s contract. */
export async function uploadToPresignedUrl(uploadUrl: string, file: File): Promise<void> {
  const response = await fetch(uploadUrl, { method: "PUT", body: file });
  if (!response.ok) {
    throw new ApiError(response.status, `Upload failed (${response.status}).`);
  }
}

export async function processPersonPhotos(
  token: string,
  personId: string,
): Promise<{ photos: { status: string }[]; identity: PersonIdentity | null }> {
  const response = await apiFetch(`/cast/${personId}/process`, token, { method: "POST" });
  return json(response);
}

export async function trainPerson(
  token: string,
  personId: string,
): Promise<{ person_id: string; status: string }> {
  const body: components["schemas"]["TrainIn"] = { confirm: true };
  const response = await apiFetch(`/cast/${personId}/train`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

/** Free: promotes the chosen candidate to the person's master (no generation). */
export async function pickMaster(
  token: string,
  personId: string,
  candidateId: string,
): Promise<{ person_id: string; master_path: string }> {
  const body: components["schemas"]["MasterPickIn"] = { candidate_id: candidateId };
  const response = await apiFetch(`/cast/${personId}/master`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

export async function revokePerson(token: string, personId: string): Promise<void> {
  const response = await apiFetch(`/cast/${personId}`, token, { method: "DELETE" });
  await json(response);
}

/** `InviteIn.expires_in` defaults to 604800s (7 days) server-side when
 * omitted; the generated type treats it as always-present (openapi-
 * typescript's `defaultNonNullable`, real for any Pydantic field with a
 * default), so the default is spelled out explicitly here rather than
 * sending `{}`. */
export async function createInvite(
  token: string,
  expiresIn = 604_800,
): Promise<{ token: string; expires_at: number }> {
  const body: components["schemas"]["InviteIn"] = { expires_in: expiresIn };
  const response = await apiFetch("/cast/invite", token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

/** `services/api/app/routers/settings.py`'s real shape — every field is
 * optional with a real server-side default (`GET /settings` on a user
 * with no row yet returns these same defaults, not a 404). */
export type Settings = {
  style: string | null;
  humor: number;
  caption_length: string | null;
  sensitive_mode: boolean;
  reminder_time: string | null;
  channel: string | null;
  generator: string;
  detail_level: string | null;
  /** Off: what it has learned about your taste no longer steers prompts or scripts. */
  personalise: boolean;
};

export async function getSettings(token: string): Promise<Settings> {
  const response = await apiFetch("/settings", token);
  return json(response);
}

export async function updateSettings(token: string, settings: Settings): Promise<Settings> {
  const body: SettingsInBody = settings;
  const response = await apiFetch("/settings", token, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json(response);
}

/** `services/api/app/routers/learning.py`'s real response shape. */
export type Checkpoint = {
  id: string;
  kind: string;
  /** Real, live-checked shape: heterogeneous, not flat numbers — e.g.
   * the reward-model checkpoint's real metrics include nested objects
   * (`learned_weights`) and a boolean (`beats_champion`), not just
   * scalars. */
  metrics: Record<string, unknown>;
  created_at: string;
};

/** One A/B tap as the preference model saw it (`preference_snapshots`):
 * `prob` is what it predicted for your pick BEFORE learning from the tap;
 * `mu`/`sd` are the weights (and their uncertainty) AFTER; `delta` is the
 * scaled feature difference chosen-minus-rejected. Arrays follow
 * `PreferenceLearning.features`. */
export type PreferenceSnapshot = {
  tap: number;
  date: string | null;
  panel_id: number;
  chosen_url: string;
  /** The first option you passed on (kept for older rows). */
  rejected_url: string;
  /** Every option you passed on: two for a best-of-3 tap, one for an older pair. */
  rejected_urls?: string[];
  /** Whether you picked the default option (no step either side); null when unknown. */
  default_picked?: boolean | null;
  prob: number;
  correct: number;
  hand_correct: number;
  mu: number[];
  sd: number[];
  delta: number[];
  axis: number | null;
};

export type KnobPoint = {
  level: number;
  mean: number;
  sd: number;
  /** 1 when this level was ever on screen; 0 = an extrapolation, not something it has seen. */
  seen?: number;
};

export type KnobModel = {
  n_taps: number;
  half_life: number;
  enter_p: number;
  keep_p: number;
  min_taps: number;
  uncertainty: Record<string, number>;
  /** The levels every panel's three options are drawn around (confident default + softer guess). */
  guess?: Record<string, number>;
  /** Highest-mean level per knob (before any caution): the model's best guess. */
  best: Record<string, number>;
  /** Utility of each level relative to level 0, with its standard deviation. */
  curves: Record<string, KnobPoint[]>;
};

export type PreferenceLearning = {
  features: string[];
  n_taps: number;
  active: boolean;
  lean: Record<string, number>;
  z: Record<string, number>;
  accuracy: {
    learned?: number | null;
    hand?: number | null;
    n_eval?: number;
    min_taps?: number;
    warmup?: number;
    /** Net calls the learned model is ahead of the default (needs `min_edge` to take over). */
    edge?: number;
    min_edge?: number;
    learned_wins?: number;
    hand_wins?: number;
    /** How often the tap matched the default ranking over ALL taps (warm-up included). */
    overall_hand?: number | null;
    overall_n?: number;
    /** KPI: how often you picked the default option; `chance` is 1 / options shown. */
    default_pick?: number | null;
    default_pick_n?: number;
    default_pick_chance?: number | null;
    /** What you have said about single images, and where the model puts the cut points. */
    ratings?: {
      n: number;
      off: number;
      ok: number;
      great: number;
      thresholds: number[] | null;
    };
  };
  /** The knob model: what it thinks about warmth / framing / expression as levels -2..2. */
  knobs: KnobModel | Record<string, never>;
  /** Written taste notes (appear after a few weeks of taps). */
  notes: { items?: string[]; updated_at?: string; n_taps?: number } | Record<string, never>;
  /** Per day: how often you picked the default option. */
  default_pick_by_day: { day_id: string; pick_rate: number; taps: number }[];
  snapshots: PreferenceSnapshot[];
  explore_by_day: { date: string; panels: number; explored: number }[];
};

export type Learning = {
  preference: PreferenceLearning;
  /** Per day: how many images you rated off / ok / great (the latest rating per image). */
  ratings_by_day?: { day_id: string; off: number; ok: number; great: number }[];
  /** Per day: how often your tap matched the app's own first choice. */
  pick_rate_series: { day_id: string; pick_rate: number; taps?: number }[];
  /** Distinct A/B taps the model learns from (one per panel). Was every `image_pairs` row. */
  pairs: number;
  taps?: number;
  regenerations?: number;
  caption_edits: number;
  checkpoints: Checkpoint[];
  next_run: string;
};

export async function getLearning(token: string): Promise<Learning> {
  const response = await apiFetch("/learning", token);
  return json(response);
}

/**
 * `GET /account/export` is an async job (`export_account_job`,
 * `services/worker/app/worker.py`) whose last event carries
 * `{download_url}`, not a distinct event name — the SSE wire format
 * reuses `event: node` for every entry regardless of job kind. The
 * download URL itself is a real, unauthenticated `/media/*` path
 * (`LocalFileStorage`'s own contract), so a plain link open is enough —
 * no bearer header needed for the download itself.
 */
export async function downloadExport(token: string): Promise<void> {
  const response = await apiFetch("/account/export", token, { method: "GET" });
  const { job_id } = await json<{ job_id: string; status: string }>(response);
  await streamJobEvents(job_id, token, (evt) => {
    const url = evt.data.download_url;
    if (typeof url === "string") {
      // already a full, real URL (same convention as strip_url/candidate.url
      // elsewhere) — the media endpoint itself needs no auth header.
      window.open(url, "_blank");
    }
  });
}
