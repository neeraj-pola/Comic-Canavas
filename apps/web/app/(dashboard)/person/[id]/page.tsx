"use client";

import { use, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { getCast, type Person } from "@/lib/api";
import "@/components/dashboard/bento.css";
import "@/components/weekly/weekly.css";

/** Person-detail sub-route. There is no dedicated `GET /cast/{id}`
 * endpoint (only the aggregate `GET /cast` list), nor a real "photos
 * grid" or "appearances" endpoint. This finds the person in the real
 * `/cast` list and shows every real field that exists (identity stats,
 * look card, consent/revoked timestamps) rather than fabricating a photo
 * grid or appearance history the API has no data for. */
export default function PersonDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { getToken } = useAuth();
  const [person, setPerson] = useState<Person | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const token = await getToken();
      if (!token || cancelled) return;
      const { people } = await getCast(token);
      if (cancelled) return;
      setPerson(people.find((p) => p.id === id) ?? null);
      setLoading(false);
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [getToken, id]);

  if (loading) return null;
  if (!person) {
    return (
      <div className="head">
        <h1>No one found for {id}</h1>
      </div>
    );
  }

  return (
    <>
      <div className="head">
        <h1>{person.name}</h1>
      </div>
      <div className="grid">
        <div className="card s6 stack">
          <h2>Identity model</h2>
          {person.identity ? (
            <>
              <p>{person.identity.source_photo_count} source photos</p>
              <p className="mono">
                generation path: {person.identity.generation_path ?? "not set"}
              </p>
              {person.identity.mean_self_cosine != null && (
                <p className="mono">
                  mean self-cosine: {person.identity.mean_self_cosine.toFixed(3)}
                </p>
              )}
              {person.identity.master_path ? (
                <p className="pill ok">approved master exists</p>
              ) : (
                <p className="pill warn">no approved master yet</p>
              )}
            </>
          ) : (
            <p style={{ color: "var(--mute)" }}>No photos processed yet.</p>
          )}
        </div>
        <div className="card s6 stack">
          <h2>Consent</h2>
          <p className="mono">consented: {person.consent_at ?? "not recorded"}</p>
          <p className="mono">revoked: {person.revoked_at ?? "no"}</p>
          <p className="mono">created: {person.created_at}</p>
        </div>
        {person.identity?.look_card && (
          <div className="card s12 stack">
            <h2>Look card</h2>
            <pre style={{ whiteSpace: "pre-wrap", fontFamily: "var(--mono)", fontSize: 12 }}>
              {JSON.stringify(person.identity.look_card, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </>
  );
}
