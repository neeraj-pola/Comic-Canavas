/**
 * Direct port of comiccanvas-landing.html / comiccanvas-dashboard.html's
 * `person(hair, shirt, mood)` procedural SVG generator — both prototypes
 * redefine near-identical copies of this; unified here as one shared
 * component (dead-code dedup, not a visual change).
 */
export type PersonHair = "bun" | "fringe" | "curly";
export type PersonMood = "smile" | "oh" | "flat";

const MOUTHS: Record<PersonMood, React.ReactNode> = {
  smile: <path d="M27 40 q5 5 10 0" fill="none" stroke="#15161B" strokeWidth={2} />,
  oh: <ellipse cx={32} cy={41} rx={3} ry={4.5} fill="#15161B" />,
  flat: <path d="M28 41 h8" stroke="#15161B" strokeWidth={2} />,
};

const HAIR: Record<PersonHair, React.ReactNode> = {
  bun: (
    <>
      <circle cx={18} cy={14} r={7} fill="#15161B" />
      <path d="M14 26 q18 -18 36 0" fill="#15161B" />
    </>
  ),
  fringe: <path d="M13 30 q19 -20 38 0 l-4 -2 q-15 -10 -30 0z" fill="#15161B" />,
  curly: (
    <path
      d="M12 28 q4 -18 20 -14 q16 -4 20 14 q-6 -6 -12 -2 q-8 -8 -16 0 q-6 -4 -12 2z"
      fill="#15161B"
    />
  ),
};

export function Person({
  hair,
  shirt,
  mood,
  className,
}: {
  hair: PersonHair;
  shirt: string;
  mood: PersonMood;
  className?: string;
}) {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true" className={className}>
      <circle cx={32} cy={32} r={18} fill="#fff" stroke="#15161B" strokeWidth={2} />
      {HAIR[hair]}
      <circle cx={26} cy={33} r={1.8} fill="#15161B" />
      <circle cx={38} cy={33} r={1.8} fill="#15161B" />
      {MOUTHS[mood]}
      <path d="M14 64 q18 -20 36 0z" fill={shirt} stroke="#15161B" strokeWidth={2} />
    </svg>
  );
}
