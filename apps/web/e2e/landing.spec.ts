import { expect, test } from "@playwright/test";

/**
 * Task 10.2's literal accept line: "Playwright scroll test: cover
 * rotation reaches −180°, spine appears, ink path length > 0."
 *
 * Scroll position is computed from the same dwell/per/ease constants
 * `LandingBook.tsx` uses (FLIPS=1, RUNOFF=0.7) rather than a magic
 * pixel offset, so this test keeps working if the viewport height (and
 * therefore `.pin`'s pixel height) changes.
 */
test("scrolling the landing page opens the book and reveals the spine + ink line", async ({
  page,
}) => {
  await page.goto("/");
  await page.waitForSelector(".sheet");

  const pinHeight = await page.evaluate(
    () => (document.querySelector(".pin") as HTMLElement).scrollHeight,
  );
  const viewportHeight = page.viewportSize()!.height;
  const total = pinHeight - viewportHeight;

  // p = 0.95 lands comfortably inside both "sheet fully open" (open>=1,
  // needs p >= dwell + per*0.8 ≈ 0.685) and "spine shown" (tail>0.35,
  // needs p >= end_start + 0.35*(1-end_start) ≈ 0.858) for FLIPS=1.
  await page.evaluate((y) => window.scrollTo(0, y), total * 0.95);
  // 200ms for the rAF-driven `update()` to run, plus the spine's own
  // 0.5s CSS opacity transition (`.spine{transition:opacity .5s}`) to
  // finish before asserting its resolved, not mid-fade, opacity.
  await page.waitForTimeout(700);

  const sheetTransform = await page.evaluate(
    () => (document.querySelector(".sheet") as HTMLElement).style.transform,
  );
  const rotateMatch = sheetTransform.match(/rotateY\((-?[\d.]+)deg\)/);
  expect(rotateMatch).not.toBeNull();
  expect(Number(rotateMatch![1])).toBeLessThanOrEqual(-179);

  const hasShowSpine = await page.evaluate(() => document.body.classList.contains("showspine"));
  expect(hasShowSpine).toBe(true);

  const spineOpacity = await page.evaluate(
    () => getComputedStyle(document.querySelector(".spine") as HTMLElement).opacity,
  );
  expect(Number(spineOpacity)).toBeCloseTo(1, 1);

  const inkPathLength = await page.evaluate(() => {
    const path = document.querySelector("#inkpath") as SVGPathElement | null;
    return path ? path.getTotalLength() : 0;
  });
  expect(inkPathLength).toBeGreaterThan(0);
});

test("prefers-reduced-motion shows the book already open, with no scroll-jacking spacer", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.waitForSelector(".landing-scope");

  const hasPin = await page.evaluate(() => !!document.querySelector(".pin"));
  expect(hasPin).toBe(false);

  const sectionCount = await page.evaluate(
    () => document.querySelectorAll(".flow .frame-host").length,
  );
  expect(sectionCount).toBe(5);
});
