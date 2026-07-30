"use client";
/**
 * SetupModal — shown on first load or when no active session exists.
 * Collects league size, draft position, rounds, and an optional Sleeper draft ID.
 */

import { useEffect, useRef, useState } from "react";
import { api, SleeperPrefill } from "@/lib/api";

/** Elements that can hold focus inside the dialog, for the Tab trap below. */
const FOCUSABLE =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

interface Props {
  onStart: (config: {
    league_size: number;
    my_draft_position: number;
    total_rounds: number;
    qb_slots: number;
    rb_slots: number;
    wr_slots: number;
    te_slots: number;
    flex_slots: number;
    dst_slots: number;
    sleeper_draft_id?: string;
  }) => void;
}

// Today's implicit standard 1-QB PPR lineup — matches DraftConfig's own
// field defaults on the backend. Selecting these in the UI changes nothing
// from current behavior; only deviating from them does.
const DEFAULT_LINEUP = { qb: 1, rb: 2, wr: 2, te: 1, flex: 1, dst: 1 };

// Mirrors the ge/le bounds on DraftConfigRequest in backend/app/schemas.py —
// kept in sync manually since there's no shared schema between front and
// back end yet.
const SLOT_BOUNDS = {
  qb: { min: 0, max: 4 },
  rb: { min: 0, max: 6 },
  wr: { min: 0, max: 6 },
  te: { min: 0, max: 4 },
  flex: { min: 0, max: 4 },
  dst: { min: 0, max: 2 },
};

function Stepper({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min: number;
  max: number;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-slate-300">{label}</span>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => onChange(Math.max(min, value - 1))}
          disabled={value <= min}
          className="w-7 h-7 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-slate-800 font-bold"
        >
          −
        </button>
        <span className="w-4 text-center text-sm font-semibold text-slate-100">
          {value}
        </span>
        <button
          type="button"
          onClick={() => onChange(Math.min(max, value + 1))}
          disabled={value >= max}
          className="w-7 h-7 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-slate-800 font-bold"
        >
          +
        </button>
      </div>
    </div>
  );
}

export default function SetupModal({ onStart }: Props) {
  const [leagueSize, setLeagueSize] = useState(12);
  const [draftPos, setDraftPos] = useState(1);
  const [rounds, setRounds] = useState(15);
  const [sleeperDraftId, setSleeperDraftId] = useState("");
  const [sleeperUsername, setSleeperUsername] = useState("");
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectWarnings, setDetectWarnings] = useState<string[]>([]);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [detectedScoring, setDetectedScoring] = useState<string | null>(null);

  const [showRosterSettings, setShowRosterSettings] = useState(false);
  const [qbSlots, setQbSlots] = useState(DEFAULT_LINEUP.qb);
  const [rbSlots, setRbSlots] = useState(DEFAULT_LINEUP.rb);
  const [wrSlots, setWrSlots] = useState(DEFAULT_LINEUP.wr);
  const [teSlots, setTeSlots] = useState(DEFAULT_LINEUP.te);
  const [flexSlots, setFlexSlots] = useState(DEFAULT_LINEUP.flex);
  const [dstSlots, setDstSlots] = useState(DEFAULT_LINEUP.dst);

  const isDefaultLineup =
    qbSlots === DEFAULT_LINEUP.qb &&
    rbSlots === DEFAULT_LINEUP.rb &&
    wrSlots === DEFAULT_LINEUP.wr &&
    teSlots === DEFAULT_LINEUP.te &&
    flexSlots === DEFAULT_LINEUP.flex &&
    dstSlots === DEFAULT_LINEUP.dst;

  // Best-effort — populates whatever Sleeper reports as suggestions, never
  // blocks manual entry. See SleeperPrefillResponse's docstring in
  // backend/app/schemas.py for why detected_scoring_format isn't applied
  // to any field here (this app is PPR-only regardless of league scoring).
  async function handleDetect() {
    if (!sleeperDraftId) return;
    setIsDetecting(true);
    setDetectError(null);
    setDetectWarnings([]);
    setDetectedScoring(null);
    try {
      const r: SleeperPrefill = await api.getSleeperPrefill(
        sleeperDraftId,
        sleeperUsername || undefined
      );
      if (r.league_size != null) setLeagueSize(r.league_size);
      if (r.total_rounds != null) setRounds(r.total_rounds);
      if (r.my_draft_position != null) setDraftPos(r.my_draft_position);

      let rosterChanged = false;
      if (r.qb_slots != null) { setQbSlots(r.qb_slots); rosterChanged = true; }
      if (r.rb_slots != null) { setRbSlots(r.rb_slots); rosterChanged = true; }
      if (r.wr_slots != null) { setWrSlots(r.wr_slots); rosterChanged = true; }
      if (r.te_slots != null) { setTeSlots(r.te_slots); rosterChanged = true; }
      if (r.flex_slots != null) { setFlexSlots(r.flex_slots); rosterChanged = true; }
      if (r.dst_slots != null) { setDstSlots(r.dst_slots); rosterChanged = true; }
      if (rosterChanged) setShowRosterSettings(true);

      setDetectedScoring(r.detected_scoring_format);
      setDetectWarnings(r.warnings);
    } catch (e) {
      setDetectError(
        e instanceof Error ? e.message : "Couldn't detect settings from Sleeper."
      );
    } finally {
      setIsDetecting(false);
    }
  }

  // ------------------------------------------------------------------
  // Focus management
  // ------------------------------------------------------------------

  const dialogRef = useRef<HTMLDivElement>(null);

  // Move focus into the dialog on mount, so keyboard and screen-reader users
  // start inside it rather than at the top of an inert page.
  useEffect(() => {
    const first = dialogRef.current?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();
  }, []);

  /**
   * Keeps Tab inside the dialog. Without this, tabbing past the last control
   * walks into the page behind the overlay — which is visually hidden and
   * completely unusable, so focus simply appears to vanish.
   */
  function handleTrapTab(e: React.KeyboardEvent) {
    if (e.key !== "Tab") return;

    const focusable = Array.from(
      dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []
    ).filter((el) => !el.hasAttribute("disabled") && el.offsetParent !== null);

    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;

    if (e.shiftKey && active === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      {/*
        A real dialog. This is a modal in every sense except the semantics —
        it covers the app and nothing behind it is usable — but it announced
        as a plain div, and Tab walked straight out of it into the page
        underneath. There's deliberately no Escape-to-close: there is nothing
        to go back to until a session exists.
      */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="setup-title"
        aria-describedby="setup-description"
        onKeyDown={handleTrapTab}
        className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-700 p-8 shadow-2xl"
      >
        <h1 id="setup-title" className="text-2xl font-bold text-slate-100 mb-1">
          Fantasy Draft Assistant
        </h1>
        <p id="setup-description" className="text-slate-300 text-sm mb-8">
          Configure your draft settings to get started.
        </p>

        <div className="space-y-5">
          {/* League size */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              League size
            </label>
            <div className="flex gap-2">
              {[8, 10, 12, 14].map((n) => (
                <button
                  key={n}
                  onClick={() => setLeagueSize(n)}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    leagueSize === n
                      ? "bg-emerald-500 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Draft position */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              My draft position
            </label>
            <div className="grid grid-cols-6 gap-2">
              {Array.from({ length: leagueSize }, (_, i) => i + 1).map((pos) => (
                <button
                  key={pos}
                  onClick={() => setDraftPos(pos)}
                  className={`py-2 rounded-lg text-sm font-semibold transition-colors ${
                    draftPos === pos
                      ? "bg-blue-500 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {pos}
                </button>
              ))}
            </div>
          </div>

          {/* Rounds */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Rounds
            </label>
            <div className="flex gap-2">
              {[14, 15, 16].map((n) => (
                <button
                  key={n}
                  onClick={() => setRounds(n)}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    rounds === n
                      ? "bg-emerald-500 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Roster settings — advanced, collapsed by default */}
          <div>
            <button
              type="button"
              onClick={() => setShowRosterSettings((v) => !v)}
              className="flex items-center gap-1.5 text-sm font-medium text-slate-400 hover:text-slate-300 transition-colors"
            >
              <span className={`transition-transform ${showRosterSettings ? "rotate-90" : ""}`}>
                ›
              </span>
              Advanced: Roster settings
              {!isDefaultLineup && (
                <span className="text-xs font-normal text-emerald-400">(customized)</span>
              )}
            </button>

            {showRosterSettings && (
              <div className="mt-3 p-4 rounded-xl bg-slate-800/60 border border-slate-700 space-y-2.5">
                <p className="text-xs text-slate-400 mb-3">
                  How many starters at each position. Defaults match a standard
                  1-QB PPR lineup — only change these if your league's roster
                  settings are different.
                </p>
                <Stepper label="QB" value={qbSlots} onChange={setQbSlots} {...SLOT_BOUNDS.qb} />
                <Stepper label="RB" value={rbSlots} onChange={setRbSlots} {...SLOT_BOUNDS.rb} />
                <Stepper label="WR" value={wrSlots} onChange={setWrSlots} {...SLOT_BOUNDS.wr} />
                <Stepper label="TE" value={teSlots} onChange={setTeSlots} {...SLOT_BOUNDS.te} />
                <Stepper label="FLEX" value={flexSlots} onChange={setFlexSlots} {...SLOT_BOUNDS.flex} />
                <Stepper label="DST" value={dstSlots} onChange={setDstSlots} {...SLOT_BOUNDS.dst} />
                {!isDefaultLineup && (
                  <button
                    type="button"
                    onClick={() => {
                      setQbSlots(DEFAULT_LINEUP.qb);
                      setRbSlots(DEFAULT_LINEUP.rb);
                      setWrSlots(DEFAULT_LINEUP.wr);
                      setTeSlots(DEFAULT_LINEUP.te);
                      setFlexSlots(DEFAULT_LINEUP.flex);
                      setDstSlots(DEFAULT_LINEUP.dst);
                    }}
                    className="text-xs text-slate-400 hover:text-slate-200 underline mt-1"
                  >
                    Reset to standard
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Sleeper draft ID — optional */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Sleeper Draft ID{" "}
              <span className="text-slate-400 font-normal">(optional)</span>
            </label>
            <p className="text-xs text-slate-400 mb-2">
              Paste your Sleeper draft ID to sync picks automatically.
              Found in Sleeper under league settings → Drafts, or in the draft URL.
            </p>
            <input
              type="text"
              placeholder="e.g. 1234567890"
              value={sleeperDraftId}
              onChange={(e) => setSleeperDraftId(e.target.value.trim())}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:border-slate-400"
            />

            {sleeperDraftId && (
              <div className="mt-3 space-y-2">
                <input
                  type="text"
                  placeholder="Sleeper username (optional — to auto-detect your draft slot)"
                  value={sleeperUsername}
                  onChange={(e) => setSleeperUsername(e.target.value.trim())}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:border-slate-400"
                />
                <button
                  type="button"
                  onClick={handleDetect}
                  disabled={isDetecting}
                  className="w-full py-2 rounded-lg bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-200 text-sm font-semibold transition-colors"
                >
                  {isDetecting ? "Detecting…" : "Autofill from Sleeper"}
                </button>

                {detectError && (
                  <p className="text-xs text-red-400">{detectError}</p>
                )}
                {detectWarnings.length > 0 && (
                  <ul className="text-xs text-amber-400 space-y-1 list-disc list-inside">
                    {detectWarnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                )}
                {!detectError && detectWarnings.length === 0 && detectedScoring && (
                  <p className="text-xs text-emerald-400">
                    ✓ Settings detected ({detectedScoring.replace("_", " ")} scoring)
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 rounded-xl bg-slate-800 text-slate-300 text-sm mb-6">
          <div>
            {leagueSize}-team PPR · Slot {draftPos} of {leagueSize} · {rounds} rounds
          </div>
          {!isDefaultLineup && (
            <div className="mt-1 text-xs text-slate-400">
              Lineup: {qbSlots} QB, {rbSlots} RB, {wrSlots} WR, {teSlots} TE,{" "}
              {flexSlots} FLEX, {dstSlots} DST
            </div>
          )}
          {sleeperDraftId && (
            <div className="text-emerald-400 mt-1 text-xs">
              ✓ Sleeper sync enabled — picks will update automatically
            </div>
          )}
        </div>

        <button
          onClick={() =>
            onStart({
              league_size: leagueSize,
              my_draft_position: draftPos,
              total_rounds: rounds,
              qb_slots: qbSlots,
              rb_slots: rbSlots,
              wr_slots: wrSlots,
              te_slots: teSlots,
              flex_slots: flexSlots,
              dst_slots: dstSlots,
              ...(sleeperDraftId ? { sleeper_draft_id: sleeperDraftId } : {}),
            })
          }
          className="w-full py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-bold text-base transition-colors"
        >
          Start Draft
        </button>
      </div>
    </div>
  );
}
