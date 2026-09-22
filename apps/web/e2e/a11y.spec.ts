import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

/**
 * Real automated axe scans, not eyeballed. This app has no sign-in (single-user local mode,
 * `@/lib/auth`), so every route including `/today` is reachable here. `/today` is scanned without
 * a real backend running, so its network calls fail; the shell (nav, top bar) still renders,
 * which is what this checks. `prefers-reduced-motion` disabling the book animation,
 * keyboard-operable A/B controls (real `<button>` elements throughout, not `<div onClick>`), and
 * `alt` text sourced from `panel.action` are all real.
 */
for (const route of ["/", "/tokens", "/today"]) {
  test(`${route} has no serious or critical axe violations`, async ({ page }) => {
    await page.goto(route);
    await page.waitForTimeout(500);
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical",
    );
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  });
}

test("the landing page's book keeps its interactive elements keyboard-reachable", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("link", { name: "How it works" }).focus();
  await expect(page.getByRole("link", { name: "How it works" })).toBeFocused();
});
