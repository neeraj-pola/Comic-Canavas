import { FlatCompat } from "@eslint/eslintrc";

// Self-contained: this project is deployed standalone (its own Vercel
// project / its own repo root), so unlike the real app's copy of this
// file it doesn't extend a parent monorepo's root config.
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [".next/**", "next-env.d.ts", "node_modules/**"],
  },
  {
    // lib/api.ts's demo functions keep the real app's exact signatures for
    // drop-in parity, so most params are intentionally unused; underscore
    // marks that on purpose rather than by accident.
    rules: {
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
    },
  },
];

export default config;
