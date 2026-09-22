# design/ — frozen visual reference

CLAUDE.md §0.1 and §10 are explicit: the visual design is **fixed**, and
Phase 10 ports it to Next.js verbatim rather than redesigning it.

- `comiccanvas-landing.html` — mustard cover, scroll-driven 3D book, dissolve-to-page landing.
- `comiccanvas-dashboard.html` — the app shell, Today/Library/Weekly/Cast/etc. screens.
- `comiccanvas-explorer.html` — the design exploration file (named `daycanvas.html` in CLAUDE.md §2's repo-layout comment; not referenced by path in any task's Accept line, so the filename actually supplied is used as-is rather than renamed to match).

Do not invent replacements for these or redesign them — Phase 10 (task
10.1+) reproduces them, it doesn't reinterpret them.
