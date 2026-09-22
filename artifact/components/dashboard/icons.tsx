/** Direct port of comiccanvas-dashboard.html's `I` icon object — hand-
 * authored stroke SVGs, no icon library, copied verbatim. */

function Svg({ children }: { children: React.ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      {children}
    </svg>
  );
}

export const TodayIcon = () => (
  <Svg>
    <rect x={3} y={4} width={18} height={17} rx={3} />
    <path d="M8 2v4M16 2v4M3 10h18" />
  </Svg>
);
export const LibraryIcon = () => (
  <Svg>
    <path d="M4 4h6v16H4zM14 4h6v16h-6z" />
  </Svg>
);
export const WeekIcon = () => (
  <Svg>
    <path d="M3 17l5-6 4 3 5-7 4 4" />
  </Svg>
);
export const CastIcon = () => (
  <Svg>
    <circle cx={9} cy={9} r={3.5} />
    <path d="M2.5 20a6.5 6.5 0 0113 0M16 4a3.5 3.5 0 010 7M21.5 20a6.5 6.5 0 00-5-6.3" />
  </Svg>
);
export const StyleIcon = () => (
  <Svg>
    <path d="M12 3a9 9 0 100 18c1.5 0 2-1 2-2s-1-2 0-3 3 0 4-1a9 9 0 00-6-12z" />
    <circle cx={8} cy={10} r={1} />
    <circle cx={12} cy={7} r={1} />
    <circle cx={16} cy={10} r={1} />
  </Svg>
);
export const LearnIcon = () => (
  <Svg>
    <path d="M4 20V10M10 20V4M16 20v-8M22 20H2" />
  </Svg>
);
export const AccountIcon = () => (
  <Svg>
    <circle cx={12} cy={8} r={4} />
    <path d="M4 21a8 8 0 0116 0" />
  </Svg>
);
export const SearchIcon = () => (
  <svg viewBox="0 0 24 24" width={18} height={18} fill="none" stroke="currentColor" strokeWidth={2}>
    <circle cx={11} cy={11} r={7} />
    <path d="M20 20l-3.5-3.5" />
  </svg>
);
export const SignOutIcon = () => (
  <Svg>
    <path d="M10 4H5v16h5M14 8l4 4-4 4M18 12H9" />
  </Svg>
);
export const DownloadIcon = () => (
  <Svg>
    <path d="M12 4v11M7 10l5 5 5-5M4 20h16" />
  </Svg>
);
