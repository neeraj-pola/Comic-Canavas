"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useAuth, useUser } from "@/lib/auth";
import {
  createInvite,
  getCast,
  presignPersonPhoto,
  processPersonPhotos,
  revokePerson,
  uploadToPresignedUrl,
  type Person,
} from "@/lib/api";
import { Person as PersonIcon } from "@/components/illustration/Person";
import "@/components/dashboard/bento.css";
import "@/components/weekly/weekly.css";

/**
 * Task 10.8: ported from `renderCast()`, wired to real `GET/POST /cast`,
 * `/cast/{id}/photos/presign`, `/cast/{id}/process`, `DELETE /cast/{id}`,
 * `POST /cast/invite`. Real deviations: the mock's per-person `hair`/
 * `shirt` illustration attributes don't exist on real `Person` records
 * (only a name + real identity stats) — every card uses the same
 * generic illustration rather than fabricating look details; status
 * pills reflect the real `identity.master_path` (approved) vs. none yet
 * (training/pending), not the mock's arbitrary "ready"/"training" flags.
 */
export default function CastPage() {
  const { getToken } = useAuth();
  const { user } = useUser();
  const [people, setPeople] = useState<Person[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [inviteLink, setInviteLink] = useState<string | null>(null);
  const fileInputs = useRef<Record<string, HTMLInputElement | null>>({});

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    const { people: real } = await getCast(token);
    setPeople(real);
    setLoading(false);
  }, [getToken]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleAddPhotos(personId: string, files: FileList | null) {
    if (!files || files.length === 0) return;
    setError(null);
    const token = await getToken();
    if (!token) return;
    try {
      for (const file of Array.from(files)) {
        const { upload_url } = await presignPersonPhoto(
          token,
          personId,
          file.name,
          file.type || "image/jpeg",
        );
        await uploadToPresignedUrl(upload_url, file);
      }
      await processPersonPhotos(token, personId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Photo upload failed.");
    }
  }

  async function handleRemove(personId: string) {
    const token = await getToken();
    if (!token) return;
    await revokePerson(token, personId);
    await load();
  }

  async function handleInvite() {
    setError(null);
    const token = await getToken();
    if (!token) return;
    try {
      const { token: inviteToken } = await createInvite(token);
      const link = `${window.location.origin}/invite/${inviteToken}`;
      setInviteLink(link);
      await navigator.clipboard.writeText(link).catch(() => {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create an invite link.");
    }
  }

  if (loading) return null;

  const you = user?.firstName ?? "You";

  return (
    <>
      <div className="head">
        <div>
          <div className="kick">Cast</div>
          <h1>You and your recurring characters</h1>
        </div>
        <p>
          Each person uploads their own photos and can remove them any time. Fifteen to twenty-five
          photos make a consistent character.
        </p>
      </div>
      {error && <div className="err">{error}</div>}
      <div className="people">
        {people.map((p) => {
          const ready = Boolean(p.identity?.master_path);
          const isYou = p.name === you || people.indexOf(p) === 0;
          return (
            <div className="card stack" key={p.id}>
              <div className="person">
                {p.identity?.master_path ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img className="avatar" src={p.identity.master_path} alt="" />
                ) : (
                  <PersonIcon hair="curly" shirt="#fff" mood="smile" />
                )}
                <div>
                  <b>{p.name}</b>
                  <span>
                    {p.identity ? `${p.identity.source_photo_count} photos` : "no photos yet"}
                  </span>
                </div>
              </div>
              <div className="row">
                <span className={`pill ${ready ? "ok" : "warn"}`}>
                  {ready ? "ready" : "training"}
                </span>
              </div>
              <div className="row">
                <Link className="btn sm" href={`/person/${p.id}`}>
                  Open
                </Link>
                <button
                  type="button"
                  className="btn sm"
                  onClick={() => fileInputs.current[p.id]?.click()}
                >
                  Add photos
                </button>
                <input
                  ref={(el) => {
                    fileInputs.current[p.id] = el;
                  }}
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(e) => handleAddPhotos(p.id, e.target.files)}
                />
                {isYou ? (
                  <Link className="btn sm" href="/setup/1">
                    Redo setup
                  </Link>
                ) : (
                  <button type="button" className="btn sm" onClick={() => handleRemove(p.id)}>
                    Remove
                  </button>
                )}
              </div>
            </div>
          );
        })}
        <div className="card stack soft" style={{ borderStyle: "dashed" }}>
          <h3>Add someone</h3>
          <p>
            Send them a link; they upload photos and set a name. You never upload photos of other
            people yourself.
          </p>
          <button type="button" className="btn pri sm" onClick={handleInvite}>
            Create invite link
          </button>
          {inviteLink && <p className="mono">{inviteLink} (copied)</p>}
        </div>
      </div>
    </>
  );
}
