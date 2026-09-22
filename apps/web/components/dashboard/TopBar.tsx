"use client";

import { type FormEvent, useCallback, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth, useUser } from "@/lib/auth";
import { getCast } from "@/lib/api";
import { Person } from "../illustration/Person";
import { SearchIcon } from "./icons";

/** Direct port of comiccanvas-dashboard.html's `.top` bar (search + tags
 * + user chip). The tag row is mock content with no real endpoint behind
 * it — kept as a static placeholder rather than fabricating a "trending
 * tags" feature. Search submits to `/library?q=`, which wires to the
 * real `/search` endpoint. No mic/"ask by voice" button: ASR isn't
 * built, so voice-input affordances shouldn't appear anywhere in the UI. */
export function TopBar() {
  const router = useRouter();
  const { user } = useUser();
  const { getToken } = useAuth();
  const pathname = usePathname();

  // The profile picture is the person's own chosen master. Refetched on
  // navigation and whenever setup picks a new one (`cc:profile-changed`);
  // `v` busts the browser's copy of the same URL.
  const [avatar, setAvatar] = useState<{ url: string; v: number } | null>(null);
  const loadAvatar = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;
      const { people } = await getCast(token);
      const url = people[0]?.identity?.master_path ?? null;
      setAvatar((prev) => (url ? { url, v: prev?.url === url ? prev.v : Date.now() } : null));
    } catch {
      // Keep whatever is showing; the illustration is the fallback.
    }
  }, [getToken]);

  useEffect(() => {
    void loadAvatar();
  }, [loadAvatar, pathname]);

  useEffect(() => {
    const refresh = () => {
      setAvatar((prev) => (prev ? { ...prev, v: Date.now() } : prev));
      void loadAvatar();
    };
    window.addEventListener("cc:profile-changed", refresh);
    return () => window.removeEventListener("cc:profile-changed", refresh);
  }, [loadAvatar]);

  function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const q = new FormData(event.currentTarget).get("q");
    if (typeof q === "string" && q.trim()) {
      router.push(`/library?q=${encodeURIComponent(q.trim())}`);
    }
  }

  const displayName =
    user?.firstName ?? "you";

  return (
    <div className="top">
      <div>
        <form className="search" onSubmit={handleSearch}>
          <SearchIcon />
          <input name="q" placeholder="Ask your diary: when did I last go to the gym?" />
        </form>
        <div className="tags">#gym #cooking #shipped #weekend</div>
      </div>
      <div className="user">
        {avatar ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img className="avatar" src={`${avatar.url}?v=${avatar.v}`} alt="" />
        ) : (
          <Person hair="curly" shirt="#E9D24A" mood="smile" />
        )}
        {displayName}
      </div>
    </div>
  );
}
