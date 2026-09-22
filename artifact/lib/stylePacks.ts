/** Ported verbatim from comiccanvas-dashboard.html's inline style-pack
 * array — shared by the setup wizard and the Style & voice page. */
export const STYLE_PACKS = [
  { key: "ink", name: "Flat ink", gradient: "linear-gradient(135deg,#F6EBD5,#E4E9EE)" },
  {
    key: "paper",
    name: "Newspaper",
    gradient: "repeating-linear-gradient(0deg,#f3f0e6 0 3px,#e6e2d4 3px 4px)",
  },
  {
    key: "water",
    name: "Watercolor",
    gradient: "radial-gradient(circle at 30% 30%,#F3E9A8,#E1EFD9 60%,#F6E1D0)",
  },
  { key: "night", name: "Night mode", gradient: "linear-gradient(135deg,#1E1E1C,#3a3a36)" },
] as const;
