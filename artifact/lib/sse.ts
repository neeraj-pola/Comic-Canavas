const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type JobEvent = { event: "node" | "done"; data: Record<string, unknown> };

/**
 * `GET /jobs/{id}/events` requires a bearer token, but the browser's
 * native `EventSource` can't set custom headers — so this reads the same
 * `event: <name>\ndata: <json>\n\n` wire format by hand over a plain
 * authenticated `fetch` + `ReadableStream`, matching the format
 * `services/api/app/routers/days.py`'s `_sse()` helper emits.
 */
export async function streamJobEvents(
  jobId: string,
  token: string,
  onEvent: (event: JobEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/events`, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`could not open the job event stream (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });

    let frameEnd: number;
    while ((frameEnd = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, frameEnd);
      buffer = buffer.slice(frameEnd + 2);

      let eventName: JobEvent["event"] | null = null;
      let dataLine: string | null = null;
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) eventName = line.slice(7) as JobEvent["event"];
        else if (line.startsWith("data: ")) dataLine = line.slice(6);
      }
      if (eventName && dataLine) {
        onEvent({ event: eventName, data: JSON.parse(dataLine) });
        if (eventName === "done") return;
      }
    }
  }
}
