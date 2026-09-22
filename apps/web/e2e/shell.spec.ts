import { expect, test } from "@playwright/test";

/**
 * The app shell (`(dashboard)/layout.tsx`). This app has no sign-in (single-user local mode,
 * `@/lib/auth`), so every route renders the shell directly. Run without a real backend, so each
 * page's own data fetches fail; what this checks is that the shell itself (the sidebar nav,
 * unaffected by that) always renders rather than crashing or redirecting anywhere.
 */
for (const route of [
  "/today",
  "/library",
  "/weekly",
  "/cast",
  "/style",
  "/learning",
  "/account",
  "/day/2026-09-01",
  "/weeks",
  "/person/me",
  "/setup/1",
]) {
  test(`${route} renders the app shell directly (no sign-in)`, async ({ page }) => {
    await page.goto(route);
    await expect(page.getByRole("link", { name: "Today" })).toBeVisible();
    expect(new URL(page.url()).pathname).toBe(route);
  });
}
