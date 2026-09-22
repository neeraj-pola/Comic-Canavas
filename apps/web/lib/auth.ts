/**
 * Single-user local mode: this app has no sign-in flow. There is exactly
 * one person using it, and every request is made as them. The API's own
 * `AUTH_MODE=mock` treats whatever bearer token it receives as the user
 * id, with no cryptographic check, so `DEV_USER_ID` below is not a secret
 * and not a real credential, just which existing rows this app reads.
 *
 * This file exists so every page that used to call `useAuth()` /
 * `useUser()` from `@clerk/nextjs` keeps the same shape
 * (`{ getToken, isLoaded, isSignedIn }`, `{ user }`) and needs only its
 * import changed, not its logic.
 *
 * Configurable via `NEXT_PUBLIC_*` env vars (inlined at build time, so
 * `apps/web/.env` needs its own copy — see the README), with generic
 * defaults for a brand-new user, so a fresh clone works with no source
 * changes.
 */

export const DEV_USER_ID = process.env.NEXT_PUBLIC_DEV_USER_ID?.trim() || "local-user";

/** Used where the UI used to read `user?.firstName` from a real Clerk session. */
const DISPLAY_NAME = process.env.NEXT_PUBLIC_DEV_USER_NAME?.trim() || "You";

async function getDevToken(): Promise<string> {
  return DEV_USER_ID;
}

/** A module-level constant, not a freshly-built object per call: every page puts `getToken` in a
 * `useEffect`/`useCallback` dependency array, so a new reference on every render would re-run
 * those effects endlessly. Nothing here depends on props or state, so there's nothing to
 * recompute per render in the first place. */
const AUTH_RESULT = {
  getToken: getDevToken,
  isLoaded: true as const,
  isSignedIn: true as const,
};

const USER_RESULT = { user: { firstName: DISPLAY_NAME } };

export function useAuth(): typeof AUTH_RESULT {
  return AUTH_RESULT;
}

export function useUser(): typeof USER_RESULT {
  return USER_RESULT;
}
