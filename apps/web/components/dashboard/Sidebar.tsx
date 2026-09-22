"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  AccountIcon,
  CastIcon,
  LearnIcon,
  LibraryIcon,
  StyleIcon,
  TodayIcon,
  WeekIcon,
} from "./icons";

/** Direct port of comiccanvas-dashboard.html's `NAV` array + sidebar
 * markup. Sub-routes (`/day/[date]`, `/person/[id]`, `/weeks`,
 * `/setup/[n]`) highlight their parent per the prototype's own `SUB`
 * mapping (`go()`).
 *
 * No sign-out button (this app has no sign-in, `@/lib/auth`, so there is
 * nothing to sign out of) and no streak container (it was already an
 * honest placeholder with no real endpoint behind it). */
const NAV: { href: string; label: string; icon: React.ComponentType }[] = [
  { href: "/today", label: "Today", icon: TodayIcon },
  { href: "/library", label: "Library", icon: LibraryIcon },
  { href: "/weekly", label: "Weekly", icon: WeekIcon },
  { href: "/cast", label: "Cast", icon: CastIcon },
  { href: "/style", label: "Style & voice", icon: StyleIcon },
  { href: "/learning", label: "Learning", icon: LearnIcon },
  { href: "/account", label: "Account", icon: AccountIcon },
];

const PARENT_OF: Record<string, string> = {
  "/day": "/library",
  "/person": "/cast",
  "/weeks": "/weekly",
  "/setup": "/cast",
};

function activeNavHref(pathname: string): string {
  for (const [prefix, parent] of Object.entries(PARENT_OF)) {
    if (pathname.startsWith(prefix)) return parent;
  }
  return pathname;
}

export function Sidebar() {
  const pathname = usePathname();
  const active = activeNavHref(pathname);

  return (
    <aside className="side">
      <div className="brand">
        <i />
        Comic Canvas
      </div>
      <nav className="navcard">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} className={active === href ? "on" : undefined}>
            <Icon />
            {label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
