import demoData from "./demo-data.json";
import { showDemoNotice } from "./demo-notice";

/**
 * Public demo build of `lib/api.ts`. Every exported name and type below is
 * kept identical to the real app's `lib/api.ts` (see the real project's
 * `apps/web/lib/api.ts`) so every page/component works completely
 * unchanged — only what's inside each function differs: reads come from
 * `demo-data.json` (real generated output from the real app, frozen as a
 * static fixture), and every write shows `showDemoNotice()` instead of
 * calling a real backend.
 *
 * Which writes throw vs. resolve harmlessly is deliberate, not arbitrary:
 * it matches whether that call site already has a try/catch (throwing is
 * then a clean, silent no-op) or is fire-and-forget (throwing there would
 * be an unhandled promise rejection in the browser console).
 */

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** The one direct `apiFetch` caller left in the app (Account page's
 * `DELETE /account`) doesn't check `.ok`, so a fake-but-successful Response
 * keeps that flow's own "deleted" confirmation working harmlessly instead
 * of an unhandled rejection. */
export async function apiFetch(
  _path: string,
  _token: string | null,
  init: RequestInit = {},
): Promise<Response> {
  if ((init.method ?? "GET") !== "GET") showDemoNotice();
  return new Response("{}", { status: 200 });
}

export async function verifyAndBootstrapUser(_token: string): Promise<void> {
  return;
}

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
  picks?: Record<string, string>;
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

type DemoData = {
  days: Record<string, Day>;
  library_days: {
    date: string;
    mood: string | null;
    quiet_day: boolean | null;
    strip_url: string | null;
    thumbs: number[];
  }[];
  weekly: Record<string, WeeklyContext>;
  person: Person | null;
  settings: Settings;
  learning: Learning;
};

const DEMO = demoData as unknown as DemoData;

/** Every real generated day this demo has, oldest first — used to fall
 * back to a real day when a route asks for a date that was never
 * generated (e.g. "today" in whatever timezone visits this demo). */
const DEMO_DATES = Object.keys(DEMO.days).sort();
const LATEST_DEMO_DATE = DEMO_DATES[DEMO_DATES.length - 1]!;

export async function createDay(
  _token: string,
  _body: { text?: string; audio_url?: string; date?: string },
): Promise<{ job_id: string; status: string }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: this doesn't generate a real comic. See the message above.");
}

export async function getDay(_token: string, date: string): Promise<Day | null> {
  return DEMO.days[date] ?? null;
}

export async function submitCaptionFeedback(
  _token: string,
  _body: { date: string; panel_id: number; original: string; edited: string },
): Promise<void> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: caption edits aren't saved here.");
}

export async function submitRating(
  _token: string,
  _body: { date: string; panel_id: number; url: string; rating: number },
): Promise<void> {
  showDemoNotice();
}

export async function submitPick(
  _token: string,
  _body: { date: string; panel_id: number; picked: string; others: string[] },
): Promise<void> {
  showDemoNotice();
}

export async function submitPairFeedback(
  _token: string,
  _body: { date: string; panel_id: number; picked: string; others: string[] },
): Promise<void> {
  showDemoNotice();
}

export async function regeneratePanel(
  _token: string,
  _date: string,
  _panelId: number,
  _note: string,
): Promise<{ job_id: string; status: string }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: regenerating isn't wired up here.");
}

export type LibraryDay = {
  date: string;
  mood: string | null;
  quiet_day: boolean | null;
  strip_url: string | null;
  thumbs: number[];
};

export async function getLibrary(
  _token: string,
  month: string,
  _mood?: string,
): Promise<{ month: string; days: LibraryDay[] }> {
  const days = DEMO.library_days.filter((d) => d.date.startsWith(month));
  return { month, days };
}

export type SearchHit = {
  date: string;
  beat_id: number;
  event: string | null;
  place: string | null;
  quote: string | null;
  panel_urls: string[];
};

export async function search(_token: string, q: string): Promise<{ q: string; hits: SearchHit[] }> {
  const needle = q.trim().toLowerCase();
  const hits: SearchHit[] = [];
  if (needle) {
    for (const day of Object.values(DEMO.days)) {
      for (const beat of day.beats) {
        const haystack = `${beat.event} ${beat.place} ${beat.quote ?? ""}`.toLowerCase();
        if (haystack.includes(needle)) {
          hits.push({
            date: day.date,
            beat_id: 0,
            event: beat.event,
            place: beat.place,
            quote: beat.quote,
            panel_urls: day.panels
              .filter((p) => p.beat_id === beat.id)
              .map((p) => day.candidates.find((c) => c.panel_id === p.id && c.chosen)?.url ?? "")
              .filter(Boolean),
          });
        }
      }
    }
  }
  return { q, hits };
}

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

export async function getWeekly(_token: string, isoWeek: string): Promise<WeeklyContext | null> {
  return DEMO.weekly[isoWeek] ?? null;
}

export async function generateWeekly(
  _token: string,
  _isoWeek: string,
): Promise<{ job_id: string; status: string }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: this doesn't generate a real recap.");
}

export async function downloadWeeklyPdf(_token: string, _isoWeek: string): Promise<void> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: the weekly PDF needs a real backend to render.");
}

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
  gender_term: string;
  age: number | null;
  consent_at: string | null;
  revoked_at: string | null;
  created_at: string;
  identity: PersonIdentity | null;
  training: PersonTraining | null;
};

export async function getCast(_token: string): Promise<{ people: Person[] }> {
  return { people: DEMO.person ? [DEMO.person] : [] };
}

export async function createPerson(
  _token: string,
  name: string,
  genderTerm = "",
): Promise<{ id: string; name: string; gender_term: string; created_at: string }> {
  showDemoNotice();
  return { id: "demo", name, gender_term: genderTerm, created_at: new Date().toISOString() };
}

export async function updatePersonDetails(
  _token: string,
  personId: string,
  details: { gender_term?: string; age?: number },
): Promise<{ id: string; name: string; gender_term: string; age: number | null }> {
  showDemoNotice();
  return {
    id: personId,
    name: DEMO.person?.name ?? "You",
    gender_term: details.gender_term ?? DEMO.person?.gender_term ?? "",
    age: details.age ?? DEMO.person?.age ?? null,
  };
}

export async function presignPersonPhoto(
  _token: string,
  _personId: string,
  _filename: string,
  _contentType: string,
): Promise<{ id: number; upload_url: string; key: string }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: photo upload needs a real backend.");
}

export async function uploadToPresignedUrl(_uploadUrl: string, _file: File): Promise<void> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: photo upload needs a real backend.");
}

export async function processPersonPhotos(
  _token: string,
  _personId: string,
): Promise<{ photos: { status: string }[]; identity: PersonIdentity | null }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: photo processing needs a real backend.");
}

export async function trainPerson(
  _token: string,
  _personId: string,
): Promise<{ person_id: string; status: string }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: training a character needs a real backend and real cost.");
}

export async function pickMaster(
  _token: string,
  _personId: string,
  _candidateId: string,
): Promise<{ person_id: string; master_path: string }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: picking a master needs a real backend.");
}

export async function revokePerson(_token: string, _personId: string): Promise<void> {
  showDemoNotice();
}

export async function createInvite(
  _token: string,
  _expiresIn = 604_800,
): Promise<{ token: string; expires_at: number }> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: invites need a real backend.");
}

export type Settings = {
  style: string | null;
  humor: number;
  caption_length: string | null;
  sensitive_mode: boolean;
  reminder_time: string | null;
  channel: string | null;
  generator: string;
  detail_level: string | null;
  personalise: boolean;
};

export async function getSettings(_token: string): Promise<Settings> {
  return DEMO.settings;
}

export async function updateSettings(_token: string, settings: Settings): Promise<Settings> {
  showDemoNotice();
  // Resolves with what was asked for, so the form still feels responsive
  // (the change just never survives a reload) rather than reverting itself.
  return settings;
}

export type Checkpoint = {
  id: string;
  kind: string;
  metrics: Record<string, unknown>;
  created_at: string;
};

export type PreferenceSnapshot = {
  tap: number;
  date: string | null;
  panel_id: number;
  chosen_url: string;
  rejected_url: string;
  rejected_urls?: string[];
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
  seen?: number;
};

export type KnobModel = {
  n_taps: number;
  half_life: number;
  enter_p: number;
  keep_p: number;
  min_taps: number;
  uncertainty: Record<string, number>;
  guess?: Record<string, number>;
  best: Record<string, number>;
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
    edge?: number;
    min_edge?: number;
    learned_wins?: number;
    hand_wins?: number;
    overall_hand?: number | null;
    overall_n?: number;
    default_pick?: number | null;
    default_pick_n?: number;
    default_pick_chance?: number | null;
    ratings?: {
      n: number;
      off: number;
      ok: number;
      great: number;
      thresholds: number[] | null;
    };
  };
  knobs: KnobModel | Record<string, never>;
  notes: { items?: string[]; updated_at?: string; n_taps?: number } | Record<string, never>;
  default_pick_by_day: { day_id: string; pick_rate: number; taps: number }[];
  snapshots: PreferenceSnapshot[];
  explore_by_day: { date: string; panels: number; explored: number }[];
};

export type Learning = {
  preference: PreferenceLearning;
  ratings_by_day?: { day_id: string; off: number; ok: number; great: number }[];
  pick_rate_series: { day_id: string; pick_rate: number; taps?: number }[];
  pairs: number;
  taps?: number;
  regenerations?: number;
  caption_edits: number;
  checkpoints: Checkpoint[];
  next_run: string;
};

export async function getLearning(_token: string): Promise<Learning> {
  return DEMO.learning;
}

export async function downloadExport(_token: string): Promise<void> {
  showDemoNotice();
  throw new ApiError(0, "Demo mode: exporting your data needs a real backend.");
}

/** Not part of the real app's exports, but handy for pages that want "the
 * most recent real day" instead of a literal `new Date()` that this demo
 * has no generated content for. */
export function latestDemoDate(): string {
  return LATEST_DEMO_DATE;
}

export function demoDates(): string[] {
  return DEMO_DATES;
}

const DEMO_WEEKS = Object.keys(DEMO.weekly).sort();

/** The one real weekly recap this demo has (there's no reason to expect the
 * visitor's own real current ISO week to be a week this fixed dataset ever
 * covers). */
export function latestDemoWeek(): string {
  return DEMO_WEEKS[DEMO_WEEKS.length - 1] ?? "2026-W37";
}
