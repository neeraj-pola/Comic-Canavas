/** Shared halftone-dot pattern def `<Panel>` SVGs reference via `url(#ht)`.
 * Render exactly once per page — SVG pattern refs resolve by id anywhere
 * in the document, regardless of which SVG the referencing element is in. */
export function HalftonePattern() {
  return (
    <svg width={0} height={0} style={{ position: "absolute" }} aria-hidden="true">
      <defs>
        <pattern id="ht" width={5} height={5} patternUnits="userSpaceOnUse">
          <circle cx={2.5} cy={2.5} r={0.7} fill="#15161B" opacity={0.14} />
        </pattern>
      </defs>
    </svg>
  );
}
