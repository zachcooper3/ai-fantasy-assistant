"use client";
/**
 * Main draft room page.
 *
 * Layout (desktop):  BigBoard | DraftRoom | AIPanel
 * Layout (mobile):   Tab-based: Board / Room / AI
 */

import { useEffect, useMemo, useState } from "react";
import { LayoutGrid, Users, Lightbulb, RotateCcw } from "lucide-react";

import { useDraft } from "@/hooks/useDraft";
import { hasModifier, isTypingTarget } from "@/lib/keyboard";
import SetupModal from "@/components/SetupModal";
import StatusBar from "@/components/StatusBar";
import BigBoard from "@/components/BigBoard";
import DraftRoom from "@/components/DraftRoom";
import AIPanel from "@/components/AIPanel";

type MobileTab = "board" | "room" | "ai";

/**
 * Human-readable start time for the resume notice — "today at 11:14" reads
 * faster mid-draft than a raw timestamp. Falls back to the raw string if the
 * backend ever sends something unparseable, rather than rendering "Invalid Date".
 */
function formatStartedAt(iso: string): string {
  const started = new Date(iso);
  if (Number.isNaN(started.getTime())) return iso;

  const time = started.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const isToday = started.toDateString() === new Date().toDateString();
  return isToday ? `today at ${time}` : `${started.toLocaleDateString()} at ${time}`;
}

export default function DraftPage() {
  const {
    session,
    board,
    recommendation,
    recHistory,
    syncStatus,
    isSyncing,
    isConnected,
    isLoadingRec,
    autoRecommend,
    setAutoRecommend,
    error,
    startSession,
    endSession,
    recordPick,
    undoPick,
    fetchRecommendation,
    clearError,
  } = useDraft();

  const [mobileTab, setMobileTab] = useState<MobileTab>("board");

  // "Draft Complete!" overlay dismissal — lets you get back to the board
  // to review picks after the draft. Re-arms whenever the session is no
  // longer complete (i.e. a new draft started), so the next completion
  // shows the overlay again.
  const [completeDismissed, setCompleteDismissed] = useState(false);
  const draftComplete = session?.draft_complete ?? false;
  useEffect(() => {
    if (!draftComplete) setCompleteDismissed(false);
  }, [draftComplete]);

  // Resume notice. Sessions persist across backend restarts by design, so
  // relaunching mid-draft silently drops you back into the old draft — which
  // looks like the app being stuck if you meant to start a new one. Dismissal
  // is keyed to the session's start time so a *different* resumed draft
  // announces itself again.
  const [resumeDismissed, setResumeDismissed] = useState<string | null>(null);
  const showResumeNotice =
    (session?.was_restored ?? false) && resumeDismissed !== (session?.started_at ?? "unknown");

  // Global shortcuts that aren't about the player list (BigBoard owns those).
  // Declared before the early return below: hooks can't run conditionally.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (hasModifier(e) || isTypingTarget(e)) return;

      if (e.key === "g") {
        e.preventDefault();
        fetchRecommendation();
      } else if (e.key === "u") {
        e.preventDefault();
        undoPick();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [fetchRecommendation, undoPick]);

  // Lookup for team/bye, which the recommendation payload doesn't carry — the
  // AI response has only id/name/position/adp. Built once per board refresh
  // rather than scanning the array per suggestion.
  // Declared before the early return below: hooks can't run conditionally.
  const playersById = useMemo(
    () => new Map((board?.players ?? []).map((p) => [p.id, p])),
    [board]
  );

  // Show setup modal until a session is active
  if (!session?.is_active) {
    return <SetupModal onStart={startSession} />;
  }

  const recommendedId = recommendation?.recommendation?.player_id;
  const scarcity = board?.scarcity ?? null;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-950">

      {/* Status bar */}
      <StatusBar
        session={session}
        isConnected={isConnected}
        syncStatus={syncStatus}
        onUndo={undoPick}
        onNewDraft={endSession}
        onReset={() => startSession({
          league_size: session.league_size,
          my_draft_position: session.my_draft_position,
          total_rounds: session.total_rounds,
          qb_slots: session.qb_slots,
          rb_slots: session.rb_slots,
          wr_slots: session.wr_slots,
          te_slots: session.te_slots,
          flex_slots: session.flex_slots,
          dst_slots: session.dst_slots,
        })}
      />

      {/* Resumed-draft notice */}
      {showResumeNotice && session && (
        <div
          role="status"
          className="mx-3 mt-2 px-4 py-2.5 bg-sky-950/80 border border-sky-800 rounded-lg text-sky-100 text-sm flex items-center gap-3 shrink-0"
        >
          <RotateCcw size={14} className="shrink-0 text-sky-300" aria-hidden="true" />
          <span className="flex-1">
            Resumed your draft in progress
            {session.started_at && ` from ${formatStartedAt(session.started_at)}`} —{" "}
            {session.picks.length} pick{session.picks.length !== 1 ? "s" : ""} recorded.
            Starting a different draft? Use{" "}
            <span className="font-semibold">New draft</span> above.
          </span>
          <button
            type="button"
            onClick={() => setResumeDismissed(session.started_at ?? "unknown")}
            className="shrink-0 text-xs text-sky-300 hover:text-sky-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded px-1"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Error toast.
          role="alert" so failures are announced rather than silently appearing,
          and a real <button> to dismiss — this was a clickable div, so keyboard
          users had no way to clear it. */}
      {error && (
        <div
          role="alert"
          className="mx-3 mt-2 px-4 py-2 bg-red-900/80 border border-red-700 rounded-lg text-red-100 text-sm flex items-center justify-between gap-3 shrink-0"
        >
          <span>{error}</span>
          <button
            type="button"
            onClick={clearError}
            aria-label="Dismiss error"
            className="shrink-0 text-red-200 hover:text-white text-xs px-1 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
          >
            ✕ dismiss
          </button>
        </div>
      )}

      {/* Desktop 3-column layout.
          The draft-room column was a fixed 280px, which forced Opponent
          Tracking's per-slot chips to wrap into an unreadable block in a
          12-team league. minmax() lets it breathe on wide screens while
          minmax(0,1fr) keeps the board from being pushed out of the grid. */}
      <div className="hidden md:grid md:grid-cols-[minmax(0,1fr)_320px_300px] xl:grid-cols-[minmax(0,1fr)_380px_340px] gap-3 p-3 flex-1 min-h-0">
        <BigBoard
          players={board?.players ?? []}
          isMyTurn={session.is_my_turn}
          recommendedId={recommendedId}
          onPick={recordPick}
          isSyncing={isSyncing}
          sessionKey={session.started_at ?? ""}
        />
        <DraftRoom session={session} />
        <AIPanel
          recommendation={recommendation}
          recHistory={recHistory}
          scarcity={scarcity}
          isLoading={isLoadingRec}
          isMyTurn={session.is_my_turn}
          onFetch={fetchRecommendation}
          onDraftRecommended={recordPick}
          playersById={playersById}
          currentPickNumber={session.current_pick_number}
          autoRecommend={autoRecommend}
          onAutoRecommendChange={setAutoRecommend}
          isSyncing={isSyncing}
        />
      </div>

      {/* Mobile tab layout */}
      <div className="md:hidden flex-1 min-h-0 overflow-hidden p-2">
        {mobileTab === "board" && (
          <BigBoard
            players={board?.players ?? []}
            isMyTurn={session.is_my_turn}
            recommendedId={recommendedId}
            onPick={recordPick}
            isSyncing={isSyncing}
            sessionKey={session.started_at ?? ""}
          />
        )}
        {mobileTab === "room" && <DraftRoom session={session} />}
        {mobileTab === "ai" && (
          <AIPanel
            recommendation={recommendation}
            recHistory={recHistory}
            scarcity={scarcity}
            isLoading={isLoadingRec}
            isMyTurn={session.is_my_turn}
            onFetch={fetchRecommendation}
            onDraftRecommended={recordPick}
            playersById={playersById}
          currentPickNumber={session.current_pick_number}
          autoRecommend={autoRecommend}
          onAutoRecommendChange={setAutoRecommend}
            isSyncing={isSyncing}
          />
        )}
      </div>

      {/* Mobile bottom tab bar.
          A real tablist: these were plain buttons, so nothing conveyed that
          they're a set or which one is current. */}
      <nav
        role="tablist"
        aria-label="Draft room sections"
        className="md:hidden flex border-t border-slate-800 bg-slate-900 shrink-0"
      >
        {(
          [
            { tab: "board", Icon: LayoutGrid, label: "Board" },
            { tab: "room",  Icon: Users,      label: "Draft" },
            { tab: "ai",    Icon: Lightbulb,  label: "AI" },
          ] as const
        ).map(({ tab, Icon, label }) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={mobileTab === tab}
            onClick={() => setMobileTab(tab)}
            className={`flex-1 flex flex-col items-center gap-1 py-3 min-h-[56px] text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-emerald-500 ${
              mobileTab === tab
                ? "text-emerald-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Icon size={20} aria-hidden="true" />
            {label}
          </button>
        ))}
      </nav>

      {/* Draft complete banner */}
      {session.draft_complete && !completeDismissed && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
          onKeyDown={(e) => e.key === "Escape" && setCompleteDismissed(true)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="draft-complete-title"
            className="bg-slate-900 border border-slate-700 rounded-2xl p-10 text-center max-w-sm max-h-[85vh] overflow-y-auto"
          >
            <div className="text-4xl mb-4" aria-hidden="true">🏈</div>
            <h2 id="draft-complete-title" className="text-2xl font-bold text-white mb-2">
              Draft Complete!
            </h2>
            <p className="text-slate-300 mb-6">
              You drafted {session.my_roster.length} players across {session.total_rounds} rounds.
            </p>
            <div className="flex flex-col gap-2">
              {session.my_roster.map((p) => (
                <div key={p.pick_number} className="text-sm text-slate-300 flex justify-between">
                  <span>Rd {p.round_number}</span>
                  <span className="font-medium">{p.player_name}</span>
                  <span className="text-slate-400">{p.position}</span>
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-2 mt-8">
              <button
                type="button"
                onClick={endSession}
                autoFocus
                className="w-full py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-bold text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300"
              >
                Start New Draft
              </button>
              <button
                type="button"
                onClick={() => setCompleteDismissed(true)}
                className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400"
              >
                Review Board
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
