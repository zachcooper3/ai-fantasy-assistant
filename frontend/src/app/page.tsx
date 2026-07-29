"use client";
/**
 * Main draft room page.
 *
 * Layout (desktop):  BigBoard | DraftRoom | AIPanel
 * Layout (mobile):   Tab-based: Board / Room / AI
 */

import { useEffect, useState } from "react";
import { LayoutGrid, Users, Lightbulb } from "lucide-react";

import { useDraft } from "@/hooks/useDraft";
import SetupModal from "@/components/SetupModal";
import StatusBar from "@/components/StatusBar";
import BigBoard from "@/components/BigBoard";
import DraftRoom from "@/components/DraftRoom";
import AIPanel from "@/components/AIPanel";

type MobileTab = "board" | "room" | "ai";

export default function DraftPage() {
  const {
    session,
    board,
    recommendation,
    syncStatus,
    isConnected,
    isLoadingRec,
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

      {/* Error toast */}
      {error && (
        <div
          className="mx-4 mt-2 px-4 py-2 bg-red-900/80 border border-red-700 rounded-lg text-red-200 text-sm flex items-center justify-between cursor-pointer"
          onClick={clearError}
        >
          <span>{error}</span>
          <span className="text-red-400 text-xs ml-4">✕ dismiss</span>
        </div>
      )}

      {/* Desktop 3-column layout */}
      <div className="hidden md:grid md:grid-cols-[1fr_280px_300px] gap-3 p-3 flex-1 min-h-0">
        <BigBoard
          players={board?.players ?? []}
          isMyTurn={session.is_my_turn}
          recommendedId={recommendedId}
          onPick={recordPick}
        />
        <DraftRoom session={session} />
        <AIPanel
          recommendation={recommendation}
          scarcity={scarcity}
          isLoading={isLoadingRec}
          isMyTurn={session.is_my_turn}
          onFetch={fetchRecommendation}
          onDraftRecommended={recordPick}
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
          />
        )}
        {mobileTab === "room" && <DraftRoom session={session} />}
        {mobileTab === "ai" && (
          <AIPanel
            recommendation={recommendation}
            scarcity={scarcity}
            isLoading={isLoadingRec}
            isMyTurn={session.is_my_turn}
            onFetch={fetchRecommendation}
            onDraftRecommended={recordPick}
          />
        )}
      </div>

      {/* Mobile bottom tab bar */}
      <nav className="md:hidden flex border-t border-slate-800 bg-slate-900 shrink-0">
        {(
          [
            { tab: "board", Icon: LayoutGrid, label: "Board" },
            { tab: "room",  Icon: Users,      label: "Draft" },
            { tab: "ai",    Icon: Lightbulb,  label: "AI" },
          ] as const
        ).map(({ tab, Icon, label }) => (
          <button
            key={tab}
            onClick={() => setMobileTab(tab)}
            className={`flex-1 flex flex-col items-center gap-1 py-3 text-xs font-medium transition-colors ${
              mobileTab === tab
                ? "text-emerald-400"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <Icon size={20} />
            {label}
          </button>
        ))}
      </nav>

      {/* Draft complete banner */}
      {session.draft_complete && !completeDismissed && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl p-10 text-center max-w-sm max-h-[85vh] overflow-y-auto">
            <div className="text-4xl mb-4">🏈</div>
            <h2 className="text-2xl font-bold text-white mb-2">Draft Complete!</h2>
            <p className="text-slate-400 mb-6">
              You drafted {session.my_roster.length} players across {session.total_rounds} rounds.
            </p>
            <div className="flex flex-col gap-2">
              {session.my_roster.map((p) => (
                <div key={p.pick_number} className="text-sm text-slate-300 flex justify-between">
                  <span>Rd {p.round_number}</span>
                  <span className="font-medium">{p.player_name}</span>
                  <span className="text-slate-500">{p.position}</span>
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-2 mt-8">
              <button
                onClick={endSession}
                className="w-full py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-bold text-sm transition-colors"
              >
                Start New Draft
              </button>
              <button
                onClick={() => setCompleteDismissed(true)}
                className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-sm transition-colors"
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
