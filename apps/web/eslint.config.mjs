import { FlatCompat } from "@eslint/eslintrc";
import rootConfig from "../../eslint.config.mjs";

// `eslint-config-next` (as of 15.5.25) still ships a legacy .eslintrc-shape
// config, not a flat one (verified by reading its real index.js — no
// flat/ export exists) — bridged via FlatCompat, the pattern Next.js's own
// docs recommend for flat-config consumers. Layered on top of the root
// config per that file's own comment about how Phase 10 should extend it.
//
// Real, live-verified bug: eslint-config-next's peer range tops out at
// eslint ^9.0.0 and its `@rushstack/eslint-patch` module-resolution hack
// hard-crashes ("Failed to patch ESLint because the calling module was
// not recognized") under the root's eslint@^10 — not just a peer-warning,
// a real failure. Rather than downgrade the shared root eslint (used
// repo-wide by `pnpm lint`), apps/web pins its own eslint@^9.39.5 as a
// local devDependency; pnpm's per-package node_modules resolves this
// package's own eslint binary independently of root's ^10.
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  ...rootConfig,
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [".next/**", "next-env.d.ts", "playwright-report/**", "test-results/**"],
  },
];

export default config;
