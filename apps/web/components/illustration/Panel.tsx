/**
 * Direct port of the prototypes' `panel(bg, scene, exp)` mini comic-panel
 * SVG generator. Relies on the shared `<pattern id="ht">` def rendered
 * once by `<HalftonePattern />` (SVG pattern references resolve by id
 * anywhere in the document, so this component never renders its own
 * copy of the pattern).
 */
export type PanelScene = "bed" | "desk" | "kitchen" | "gym";
export type PanelExpression = "panic" | "content" | "tired" | "angry";

const FACES: Record<PanelExpression, React.ReactNode> = {
  panic: (
    <>
      <circle cx={-5} cy={-1} r={3} fill="#fff" stroke="#15161B" strokeWidth={1.5} />
      <circle cx={5} cy={-1} r={3} fill="#fff" stroke="#15161B" strokeWidth={1.5} />
      <ellipse cx={0} cy={7} rx={3} ry={4.5} fill="#15161B" />
    </>
  ),
  content: (
    <path
      d="M-8 -1 q3 2 6 0 M2 -1 q3 2 6 0 M-5 5 q5 6 10 0"
      fill="none"
      stroke="#15161B"
      strokeWidth={1.6}
    />
  ),
  tired: (
    <path d="M-8 -1 h6 M2 -1 h6 M-4 7 q4 -2 8 0" fill="none" stroke="#15161B" strokeWidth={1.6} />
  ),
  angry: (
    <path d="M-8 -5 l6 2 M8 -5 l-6 2 M-4 7 h8" fill="none" stroke="#15161B" strokeWidth={1.6} />
  ),
};

const SCENES: Record<PanelScene, React.ReactNode> = {
  bed: (
    <>
      <rect
        x={90}
        y={70}
        width={80}
        height={26}
        rx={5}
        fill="#C9A27A"
        stroke="#15161B"
        strokeWidth={1.5}
      />
      <rect
        x={20}
        y={16}
        width={44}
        height={36}
        fill="#A9CFF7"
        stroke="#15161B"
        strokeWidth={1.5}
      />
    </>
  ),
  desk: (
    <>
      <rect
        x={10}
        y={84}
        width={160}
        height={7}
        fill="#A68A6A"
        stroke="#15161B"
        strokeWidth={1.5}
      />
      <rect x={105} y={50} width={56} height={36} rx={3} fill="#2B2B2B" />
      <rect x={110} y={55} width={46} height={26} fill="#1E2A38" />
      <rect x={114} y={60} width={30} height={3} fill="#4CAF7D" />
      <rect x={114} y={67} width={36} height={4} fill="#E5484D" />
    </>
  ),
  kitchen: (
    <>
      <rect
        x={10}
        y={88}
        width={160}
        height={7}
        fill="#D9C9B4"
        stroke="#15161B"
        strokeWidth={1.5}
      />
      <ellipse cx={48} cy={86} rx={24} ry={6} fill="#fff" stroke="#15161B" strokeWidth={1.5} />
      <ellipse cx={48} cy={82} rx={19} ry={6} fill="#E0A244" stroke="#15161B" strokeWidth={1.5} />
    </>
  ),
  gym: (
    <>
      <rect
        x={10}
        y={90}
        width={160}
        height={6}
        fill="#8a8a8a"
        stroke="#15161B"
        strokeWidth={1.5}
      />
      <rect
        x={118}
        y={44}
        width={42}
        height={46}
        fill="#6e6e6e"
        stroke="#15161B"
        strokeWidth={1.5}
      />
      <rect
        x={24}
        y={76}
        width={66}
        height={8}
        rx={3}
        fill="#3a3a3a"
        stroke="#15161B"
        strokeWidth={1.5}
      />
    </>
  ),
};

const CENTER_X: Record<PanelScene, number> = { bed: 60, desk: 70, kitchen: 100, gym: 70 };

export function Panel({ bg, scene, exp }: { bg: string; scene: PanelScene; exp: PanelExpression }) {
  const cx = CENTER_X[scene];
  return (
    <svg viewBox="0 0 180 100" aria-hidden="true">
      <rect width={180} height={100} fill={bg} />
      <rect width={180} height={100} fill="url(#ht)" />
      {SCENES[scene]}
      <g transform={`translate(${cx} 44)`}>
        <rect
          x={-12}
          y={12}
          width={24}
          height={34}
          rx={4}
          fill="#5B7FB5"
          stroke="#15161B"
          strokeWidth={1.5}
        />
        <circle cx={0} cy={0} r={12} fill="#C68A5A" stroke="#15161B" strokeWidth={1.5} />
        <path
          d="M-12 -4 q4 -14 14 -8 q6 -5 10 8 l-4 -3 l-3 4 l-4 -4 l-4 4 l-4 -4 l-3 4z"
          fill="#15161B"
        />
        {FACES[exp]}
        <rect
          x={-8}
          y={-4}
          width={6}
          height={4}
          rx={1}
          fill="none"
          stroke="#15161B"
          strokeWidth={1.2}
        />
        <rect
          x={2}
          y={-4}
          width={6}
          height={4}
          rx={1}
          fill="none"
          stroke="#15161B"
          strokeWidth={1.2}
        />
      </g>
    </svg>
  );
}
