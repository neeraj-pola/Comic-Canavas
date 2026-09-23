/**
 * Real pipeline stages (`services/worker/app/graph.py`'s `NODE_ORDER`,
 * verified against the source rather than reusing the mock prototype's
 * own fabricated 7-item list `['Listen','Beats','Script','Prompts',
 * 'Draw ×3','Critic','Compose']` — the real graph has 8 nodes, and each
 * `event: node` SSE frame's `data` is exactly `{"node": "<name>"}`
 * (`services/worker/app/worker.py`'s `append_job_event` call site).
 */
export const REAL_STAGES = [
  "input",
  "beats",
  "script",
  "prompts",
  "generate",
  "critic",
  "compose",
  "memory",
] as const;

const LABELS: Record<(typeof REAL_STAGES)[number], string> = {
  input: "Read",
  beats: "Beats",
  script: "Script",
  prompts: "Prompts",
  generate: "Draw",
  critic: "Critic",
  compose: "Compose",
  memory: "Remember",
};

export type StageStatus = "pending" | "run" | "done";

export function Stages({ statuses }: { statuses: Record<string, StageStatus> }) {
  return (
    <ul className="stages">
      {REAL_STAGES.map((stage) => (
        <li key={stage} className={statuses[stage] ?? "pending"}>
          {LABELS[stage]}
        </li>
      ))}
    </ul>
  );
}
