import Link from "next/link";
import { Person } from "../illustration/Person";

/**
 * Direct port of comiccanvas-landing.html's `B` content object. Variant B
 * (the shipped default: `data-variant="B"`, the only `build()` call site)
 * uses `hero` on the book's static right page and
 * `['how','sample','learn','cast','join']` as flow sections — `endpaper`
 * is the cover sheet's back face and `colophon` is genuinely unreachable
 * in variant B (SPREADS.slice(1).flat() never includes it, and it isn't
 * selected as board content either) — both ported as components below,
 * but colophon is not rendered anywhere, matching the live prototype's
 * own real behavior rather than inventing a place to show it.
 */

export function Endpaper() {
  return (
    <div className="face back endpaper">
      <div className="pat" />
      <div className="note">
        <b>This diary writes itself. Well, draws.</b>
        Every evening you talk for a minute. It listens, finds the four moments that matter, and
        draws them as a strip with you in it. Turn the page.
      </div>
    </div>
  );
}

export function HeroContent() {
  return (
    <div className="pg">
      <span className="kick">Daily · Spoken · Drawn</span>
      <h2>
        Speak your day.
        <br />
        Get a comic of it.
      </h2>
      <p>
        <span className="dc">Y</span>ou talk for sixty seconds about what actually happened. Comic
        Canvas transcribes it, picks the setup, the complication, the relief and the punchline, and
        draws a four-panel strip in a style that stays yours, with a character that looks like you.
      </p>
      <div className="tags">#daily #comic #text #yours</div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Link className="pill pink" href="/today">
          Open your diary
        </Link>
        <a className="pill" href="#flow">
          How it works
        </a>
      </div>
      <div className="byline">
        <Person hair="bun" shirt="#E9D24A" mood="smile" />
        <div>
          <b>Made for one reader: you</b>
          <span>No feed. No likes. A book you keep.</span>
        </div>
      </div>
    </div>
  );
}

function HowContent() {
  return (
    <div className="pg">
      <span className="kick">How a day becomes a strip</span>
      <h2>Three steps, sixty seconds</h2>
      <div className="cards c3">
        <div className="card">
          <div className="num">1</div>
          <h3>Say it</h3>
          <p>A few lines about your day, in your own words.</p>
        </div>
        <div className="card pink">
          <div className="num">2</div>
          <h3>Find the beats</h3>
          <p>An agent extracts what happened, when, where, and how you felt. Nothing invented.</p>
        </div>
        <div className="card blue">
          <div className="num">3</div>
          <h3>Draw it</h3>
          <p>
            A script agent writes four panels in your voice; the illustrator draws them with you in
            frame.
          </p>
        </div>
      </div>
      <div className="byline">
        <Person hair="fringe" shirt="#F7F6F1" mood="oh" />
        <div>
          <b>Quiet day?</b>
          <span>Two panels, gentler jokes. It reads the room.</span>
        </div>
      </div>
    </div>
  );
}

function SampleContent() {
  return (
    <div className="pg">
      <span className="kick">Monday, Sep 21</span>
      <h2>Up before the world</h2>
      <div className="strip">
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/landing/sample-1-run.png"
            alt="A comic panel of the character running down a garden path at dawn, captioned 'Up before the world, feet already moving'"
          />
        </div>
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/landing/sample-2-laptop.png"
            alt="A comic panel of the character working at a laptop with coffee, captioned 'Grinding through problems, one tab at a time'"
          />
        </div>
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/landing/sample-3-cricket.png"
            alt="A comic panel of the character batting at sunset, captioned 'Evening reset: bat in hand, worries benched'"
          />
        </div>
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/landing/sample-4-cook.png"
            alt="A comic panel of the character cooking in the kitchen, captioned 'Cooked enough for a small army. No regrets.'"
          />
        </div>
      </div>
      <p>
        A real strip, drawn from a real day. Captions reuse your own phrasing, so it sounds like you,
        not like a model.
      </p>
    </div>
  );
}

function LearnContent() {
  return (
    <div className="pg">
      <span className="kick">It learns your taste</span>
      <h2>Four taps a day teach it</h2>
      <p>
        Every panel is drawn three ways. Tap your favourite, since every pick teaches it your
        taste. Those picks train a small reward model, and once a week the illustrator is nudged
        toward what you kept.
      </p>
      <div className="ab three">
        <div className="opt win">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/landing/pick-cricket-chosen.png" alt="Option 1: the character mid-swing, batting at sunset" />
          You picked this
        </div>
        <div className="opt">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/landing/pick-cricket-b.png" alt="Option 2: a different angle on the same batting swing" />
          Option 2
        </div>
        <div className="opt">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/landing/pick-cricket-c.png" alt="Option 3: the character batting with gloves, a closer framing" />
          Option 3
        </div>
      </div>
      <div className="stat">
        <div>
          <b>82%</b>first pick rate
        </div>
        <div>
          <b>15 s</b>a day
        </div>
        <div>
          <b>0</b>surveys
        </div>
      </div>
    </div>
  );
}

function CastContent() {
  // Just names and icons — the intro paragraph and the per-card subtitle
  // line were real, measured content, but on narrower widths the `c3`
  // grid only had room to format about half of it before the text ran
  // out of space. Less text here, not a cleverer grid.
  return (
    <div className="pg">
      <span className="kick">Your cast</span>
      <h2>Friends become characters</h2>
      <div className="cards c3">
        <div className="card">
          <div className="person">
            <Person hair="curly" shirt="#fff" mood="smile" />
            <b>You</b>
          </div>
        </div>
        <div className="card pink">
          <div className="person">
            <Person hair="bun" shirt="#fff" mood="oh" />
            <b>Friend</b>
          </div>
        </div>
        <div className="card blue">
          <div className="person">
            <Person hair="fringe" shirt="#fff" mood="flat" />
            <b>Coach</b>
          </div>
        </div>
      </div>
    </div>
  );
}

function JoinContent() {
  return (
    <div className="pg">
      <span className="kick">Open your diary</span>
      <h2>One strip a day, yours to keep</h2>
      <p>
        Your entries, your face, and your strips stay in your own book; nothing is shared unless you
        share it.
      </p>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Link className="pill pink" href="/today">
          Open your diary
        </Link>
      </div>
      <div className="cards c2">
        <div className="card pink">
          <span className="tag">Weekly recap</span>
          <p style={{ marginTop: 8 }}>Sundays, a six-panel week.</p>
        </div>
        <div className="card blue">
          <span className="tag">Yearbook</span>
          <p style={{ marginTop: 8 }}>Your favourite strips, printed.</p>
        </div>
      </div>
    </div>
  );
}

export const FLOW_SECTIONS = ["how", "sample", "learn", "cast", "join"] as const;
export type FlowSectionKey = (typeof FLOW_SECTIONS)[number];

export function FlowContent({ sectionKey }: { sectionKey: FlowSectionKey }) {
  switch (sectionKey) {
    case "how":
      return <HowContent />;
    case "sample":
      return <SampleContent />;
    case "learn":
      return <LearnContent />;
    case "cast":
      return <CastContent />;
    case "join":
      return <JoinContent />;
  }
}
