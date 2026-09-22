/** What each learned feature means to a person. `pos`/`neg` are the words for
 * a positive / negative weight ("warmer" / "cooler"), which is how the charts
 * describe taste without showing raw numbers. Order = display order. Keep the
 * feature NAMES in sync with `services/worker/app/preference/features.py`. */
export type FeatureMeta = {
  key: string;
  label: string;
  neg: string;
  pos: string;
  group: "look" | "critic";
  /** What this number actually is, for the hover hint next to its row. */
  desc: string;
};

export const FEATURES: FeatureMeta[] = [
  {
    key: "warmth",
    label: "Color warmth",
    neg: "cooler",
    pos: "warmer",
    group: "look",
    desc: "How warm or cool the image's colors are, measured directly from the pixels.",
  },
  {
    key: "brightness",
    label: "Brightness",
    neg: "darker",
    pos: "brighter",
    group: "look",
    desc: "How light or dark the image is overall, measured directly from the pixels.",
  },
  {
    key: "saturation",
    label: "Color intensity",
    neg: "more muted",
    pos: "more vivid",
    group: "look",
    desc: "How vivid or washed-out the colors are, measured directly from the pixels.",
  },
  {
    key: "contrast",
    label: "Contrast",
    neg: "softer",
    pos: "punchier",
    group: "look",
    desc: "How much the light and dark areas of the image differ, measured directly from the pixels.",
  },
  {
    key: "closeness",
    label: "Framing",
    neg: "wider",
    pos: "closer",
    group: "look",
    desc: "How close up the shot is: how much of the frame the character fills.",
  },
  {
    key: "expression",
    label: "Expression",
    neg: "subtler",
    pos: "bolder",
    group: "look",
    desc: "How subtle or bold the character's expression was asked to be. This is what the prompt requested, not something measured from the image, since there's no reliable way to measure expression strength from pixels alone.",
  },
  {
    key: "identity",
    label: "Looks like you",
    neg: "less like you",
    pos: "more like you",
    group: "critic",
    desc: "The critic's identity check: does this candidate show the approved character's real features (hair, glasses, face shape, etc.)?",
  },
  {
    key: "style",
    label: "Comic style match",
    neg: "off-style",
    pos: "on-style",
    group: "critic",
    desc: "How closely the image matches the comic's flat-ink art style, by comparing it to a bank of reference renders in that style.",
  },
  {
    key: "judge_scene",
    label: "Shows the scene",
    neg: "off-scene",
    pos: "on-scene",
    group: "critic",
    desc: "A vision model's rating of whether the image actually shows the scene the panel's caption describes.",
  },
  {
    key: "judge_face",
    label: "Face looks right",
    neg: "off-face",
    pos: "good face",
    group: "critic",
    desc: "A vision model's rating of whether the face is well formed: not distorted, warped, or wrong.",
  },
  {
    key: "judge_body",
    label: "Hands & body look right",
    neg: "glitchy",
    pos: "clean",
    group: "critic",
    desc: "A vision model's rating of whether hands and body are anatomically clean, not glitchy or malformed, a common AI-image failure.",
  },
  {
    key: "judge_outfit",
    label: "Same outfit",
    neg: "outfit changed",
    pos: "outfit kept",
    group: "critic",
    desc: "A vision model's rating of whether the character is wearing the same outfit as the master reference image.",
  },
  {
    key: "judge_overall",
    label: "Looks polished",
    neg: "rough",
    pos: "polished",
    group: "critic",
    desc: "A vision model's overall impression of how clean and finished the image looks, beyond the other specific checks.",
  },
  {
    key: "alignment",
    label: "Matches the caption",
    neg: "off-story",
    pos: "on-story",
    group: "critic",
    desc: "How well the image matches its intended action/caption. Catches a candidate that's on-identity and on-style but shows the wrong scene.",
  },
  {
    key: "detail",
    label: "Background detail",
    neg: "simpler",
    pos: "busier",
    group: "critic",
    desc: "How much texture and detail the background/environment has (edge density outside the face), separate from the character's own linework.",
  },
];

export const FEATURE_BY_KEY: Record<string, FeatureMeta> = Object.fromEntries(
  FEATURES.map((f) => [f.key, f]),
);

/** The three axes a panel's candidates vary along (index = `axis` id). */
export const AXIS_KEYS = ["warmth", "closeness", "expression"] as const;

/** 95% credible range half-width in standard deviations. */
export const CI_Z = 1.96;

export function shortDate(date: string | null): string {
  if (!date) return "";
  return new Date(`${date}T12:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}
