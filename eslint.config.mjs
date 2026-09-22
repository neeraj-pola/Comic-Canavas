// Root ESLint flat config. Phase 10 adds `apps/web`'s Next.js/TypeScript
// rules on top of this; kept minimal for now so `make lint` (task 0.1's
// accept line: "make lint runs on empty packages") has something to run.
//
// `apps/web/**` is excluded here, not merely left unconfigured: ESLint's
// flat-config system resolves the CLOSEST config file per linted file,
// so running `eslint .` from the repo root would otherwise still reach
// into apps/web and load its `eslint.config.mjs` — which extends
// `eslint-config-next` and real-crashes under this root's `eslint@^10`
// (`eslint-config-next`'s `@rushstack/eslint-patch` hard-fails past
// ESLint 9; verified live, not a guess). apps/web pins its own
// `eslint@^9.39.5` and lints itself via `pnpm --filter web lint`, wired
// as a second step in the root `lint` script (package.json) — see that
// script and apps/web/eslint.config.mjs's own comment for the full
// account.
export default [
  {
    ignores: [
      "**/node_modules/**",
      "**/.next/**",
      ".venv/**",
      "ml/goldens/**",
      "design/**",
      "apps/web/**",
    ],
  },
];
