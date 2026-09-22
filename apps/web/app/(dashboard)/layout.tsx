"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { getCast } from "@/lib/api";
import { Sidebar } from "@/components/dashboard/Sidebar";
import { TopBar } from "@/components/dashboard/TopBar";
import "../../components/dashboard/dashboard-globals.css";
import "../../components/dashboard/shell.css";

/**
 * The app shell — ported from comiccanvas-dashboard.html's `renderShell()` (`.shell` grid:
 * sidebar + main). This app has no sign-in flow (single-user local mode, `@/lib/auth`), so every
 * page always renders. The onboarding redirect (no approved character yet -> the setup wizard)
 * is unrelated to sign-in and stays.
 */
export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { getToken } = useAuth();

  // Until an approved character exists, every page except setup itself
  // (and Account, so delete-my-data stays reachable) sends you into the
  // wizard — straight to step 2 if training is already running.
  useEffect(() => {
    if (pathname.startsWith("/setup") || pathname.startsWith("/account")) return;
    let cancelled = false;
    (async () => {
      try {
        const token = await getToken();
        if (!token) return;
        const { people } = await getCast(token);
        if (cancelled || people.some((p) => p.identity?.master_path)) return;
        const training = people.some(
          (p) => p.training?.status === "pending" || p.training?.status === "running",
        );
        // Training finished but no master picked yet -> the picker (step 2).
        const awaitingPick = people.some((p) => p.identity?.master_candidates?.length);
        router.replace(training || awaitingPick ? "/setup/2" : "/setup/1");
      } catch {
        // API unreachable: leave the page as-is rather than bouncing.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pathname, getToken, router]);

  return (
    <div className="dashboard-scope">
      <div className="shell">
        <Sidebar />
        <div className="main">
          <TopBar />
          {children}
        </div>
      </div>
    </div>
  );
}
