const COLORS: { name: string; varName: string }[] = [
  { name: "field", varName: "--field" },
  { name: "card", varName: "--card" },
  { name: "ink", varName: "--ink" },
  { name: "mute", varName: "--mute" },
  { name: "rule", varName: "--rule" },
  { name: "rule-soft", varName: "--rule-soft" },
  { name: "mustard", varName: "--mustard" },
  { name: "mustard-soft", varName: "--mustard-soft" },
  { name: "mustard-deep", varName: "--mustard-deep" },
  { name: "charcoal", varName: "--charcoal" },
  { name: "charcoal-deep", varName: "--charcoal-deep" },
  { name: "paper", varName: "--paper" },
  { name: "paper-line", varName: "--paper-line" },
  { name: "spiral", varName: "--spiral" },
  { name: "ok", varName: "--ok" },
  { name: "warn", varName: "--warn" },
];

export default function TokensPage() {
  return (
    <main style={{ padding: 40, maxWidth: 960, margin: "0 auto" }}>
      <h1 style={{ fontFamily: "var(--sans)", fontWeight: 800 }}>Design tokens</h1>
      <p style={{ fontFamily: "var(--sans)", color: "var(--mute)" }}>
        Every color and font from the two frozen prototypes, rendered without a Storybook.
      </p>

      <h2 style={{ fontFamily: "var(--sans)" }}>Colors</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
          gap: 16,
        }}
      >
        {COLORS.map((c) => (
          <div key={c.name}>
            <div
              style={{
                background: `var(${c.varName})`,
                height: 72,
                borderRadius: "var(--r)",
                border: "1.5px solid var(--ink)",
                boxShadow: "var(--shadow)",
              }}
            />
            <div style={{ fontFamily: "var(--mono)", fontSize: 12, marginTop: 6 }}>{c.varName}</div>
          </div>
        ))}
      </div>

      <h2 style={{ fontFamily: "var(--sans)", marginTop: 40 }}>Fonts</h2>
      <div style={{ display: "grid", gap: 20 }}>
        <div>
          <div style={{ fontFamily: "var(--sans)", fontWeight: 400, fontSize: 22 }}>
            Inter 400: Comic Canvas, a diary that draws itself.
          </div>
          <div style={{ fontFamily: "var(--sans)", fontWeight: 500, fontSize: 22 }}>
            Inter 500: Comic Canvas, a diary that draws itself.
          </div>
          <div style={{ fontFamily: "var(--sans)", fontWeight: 600, fontSize: 22 }}>
            Inter 600: Comic Canvas, a diary that draws itself.
          </div>
          <div style={{ fontFamily: "var(--sans)", fontWeight: 700, fontSize: 22 }}>
            Inter 700: Comic Canvas, a diary that draws itself.
          </div>
          <div style={{ fontFamily: "var(--sans)", fontWeight: 800, fontSize: 22 }}>
            Inter 800: Comic Canvas, a diary that draws itself.
          </div>
        </div>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontWeight: 400, fontSize: 18 }}>
            JetBrains Mono 400: 2026-09-16 · 12 panels · $0.31
          </div>
          <div style={{ fontFamily: "var(--mono)", fontWeight: 500, fontSize: 18 }}>
            JetBrains Mono 500: 2026-09-16 · 12 panels · $0.31
          </div>
        </div>
      </div>

      <h2 style={{ fontFamily: "var(--sans)", marginTop: 40 }}>Radius &amp; shadow</h2>
      <div style={{ display: "flex", gap: 24 }}>
        <div>
          <div
            style={{
              width: 160,
              height: 100,
              background: "var(--card)",
              border: "1.5px solid var(--ink)",
              borderRadius: "var(--r)",
              boxShadow: "var(--shadow)",
            }}
          />
          <div style={{ fontFamily: "var(--mono)", fontSize: 12, marginTop: 6 }}>
            landing scope (--r: 14px)
          </div>
        </div>
        <div className="dashboard-scope">
          <div
            style={{
              width: 160,
              height: 100,
              background: "var(--card)",
              border: "1.5px solid var(--ink)",
              borderRadius: "var(--r)",
              boxShadow: "var(--shadow)",
            }}
          />
          <div style={{ fontFamily: "var(--mono)", fontSize: 12, marginTop: 6 }}>
            dashboard scope (--r: 16px)
          </div>
        </div>
      </div>
    </main>
  );
}
