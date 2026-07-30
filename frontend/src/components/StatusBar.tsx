"use client";
/**
 * StatusBar — top bar showing current pick number, round, and whose turn it is.
 * Highlights in green when it's the user's turn.
 */

import { Undo2, Wifi, WifiOff, RefreshCw, RotateCcw, LogOut } from "lucide-react";
import { DraftState, SyncStatus } from "@/lib/api";
import ConfirmButton from "@/components/ConfirmButton";

interface Props {
  session: DraftState;
  isConnected: boolean;
  syncStatus: SyncStatus | null;
  onUndo: () => void;
  /** Clears the picks but keeps this league's settings. */
  onReset: () => void;
  /**
   * Ends the session entirely and returns to the setup screen.
   *
   * Without this there was no way out of an active draft: the session is
   * persisted and rehydrated on boot, so restarting the backend resumed the
   * old draft, and "Reset" restarts with the same config rather than letting
   * you reconfigure. The only route back to setup was the draft-complete
   * overlay — unreachable unless you played the draft out.
   */
  onNewDraft: () => void;
}

const POS_COLORS: Record<string, string> = {
  QB: "text-red-400",
  RB: "text-emerald-400",
  WR: "text-blue-400",
  TE: "text-amber-400",
  DST: "text-purple-400",
  K: "text-slate-400",
};

export default function StatusBar({
  session,
  isConnected,
  syncStatus,
  onUndo,
  onReset,
  onNewDraft,
}: Props) {
  const myTurn = session.is_my_turn;

  // Undoing a sync-recorded pick restores the player locally while Sleeper
  // still has the pick, and the backend's synced-pick counter isn't rewound —
  // so sync never re-records it and the two boards diverge permanently
  // (audit W12). Sleeper is the source of truth while it's live; undo there.
  const isSyncing = syncStatus?.status === "syncing";
  const undoDisabled = session.picks.length === 0 || isSyncing;

  const lastPick = session.picks[session.picks.length - 1];

  return (
    <div
      className={`flex items-center gap-4 px-4 py-3 border-b text-sm font-medium transition-colors ${
        myTurn
          ? "bg-emerald-950 border-emerald-700"
          : "bg-slate-900 border-slate-700"
      }`}
    >
      {/*
        Screen-reader announcements. The visual UI conveys "it's your turn"
        with colour, a pulsing dot and a layout shift — none of which reach a
        screen reader. This is the one event in the app that's genuinely
        time-critical, so it's assertive; the pick feed below it is polite.
      */}
      <span aria-live="assertive" aria-atomic="true" className="sr-only">
        {myTurn
          ? "You are on the clock."
          : `${session.picks_until_my_turn} picks until your turn.`}
      </span>
      <span aria-live="polite" aria-atomic="true" className="sr-only">
        {lastPick
          ? `Pick ${lastPick.pick_number}: ${lastPick.player_name}, ${lastPick.position}, ` +
            `${lastPick.is_mine ? "your pick" : `slot ${lastPick.team_slot}`}.`
          : ""}
      </span>

      {/* On the clock indicator */}
      {myTurn ? (
        <span className="flex items-center gap-2 text-emerald-400 font-bold text-base">
          <span className="relative flex h-2.5 w-2.5" aria-hidden="true">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
          </span>
          ON THE CLOCK
        </span>
      ) : (
        <span className="text-slate-300">
          {session.picks_until_my_turn} pick{session.picks_until_my_turn !== 1 ? "s" : ""} away
        </span>
      )}

      <span className="text-slate-500" aria-hidden="true">|</span>

      {/* Pick / round info */}
      <span className="text-slate-300">
        Pick <span className="text-white font-bold">#{session.current_pick_number}</span>
      </span>
      <span className="text-slate-300">
        Round <span className="text-white font-bold">{session.current_round}</span>
        <span className="text-slate-400"> / {session.total_rounds}</span>
      </span>

      {/* Current team on clock */}
      {!myTurn && (
        <span className="text-slate-300">
          Slot <span className="text-slate-100">{session.current_team_slot}</span> picking
        </span>
      )}

      {/* My next pick */}
      {session.my_next_pick_number && !myTurn && (
        <span className="text-slate-400 text-xs">
          My next: #{session.my_next_pick_number} (Rd {Math.ceil(session.my_next_pick_number / session.league_size)})
        </span>
      )}

      {/* Roster summary */}
      {session.my_roster.length > 0 && (
        <span className="hidden lg:flex items-center gap-1 text-xs text-slate-400 ml-2">
          My roster:
          {session.my_roster.map((p) => (
            <span key={p.pick_number} className={`${POS_COLORS[p.position] ?? "text-slate-400"}`}>
              {p.position}
            </span>
          ))}
        </span>
      )}

      {/* Spacer */}
      <span className="flex-1" />

      {/* Controls */}
      <button
        onClick={onUndo}
        disabled={undoDisabled}
        title={
          isSyncing
            ? "Undo is disabled while Sleeper sync is live — undo the pick in Sleeper instead"
            : "Undo last pick"
        }
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        <Undo2 size={14} />
        Undo
      </button>

      {/* Clears picks, keeps this league's settings */}
      <ConfirmButton
        label={
          <span className="flex items-center gap-1.5">
            <RotateCcw size={12} aria-hidden="true" />
            Reset picks
          </span>
        }
        confirmLabel="Clear all picks?"
        onConfirm={onReset}
        ariaLabel="Reset picks, keeping the current league settings"
        title="Clear every pick and start this same draft over"
        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors text-xs focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
      />

      {/* Ends the session and returns to the setup screen */}
      <ConfirmButton
        label={
          <span className="flex items-center gap-1.5">
            <LogOut size={12} aria-hidden="true" />
            New draft
          </span>
        }
        confirmLabel="End this draft?"
        onConfirm={onNewDraft}
        ariaLabel="End this draft and return to setup"
        title="End this session and go back to the configuration screen"
        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-red-900 text-slate-400 hover:text-red-300 transition-colors text-xs focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
        confirmClassName="bg-red-600 hover:bg-red-500 text-white"
      />

      {/* Sleeper sync indicator */}
      {syncStatus && syncStatus.status !== "idle" && (
        <span
          role="status"
          title={
            syncStatus.status === "syncing"
              ? `Sleeper sync active (${syncStatus.synced_pick_count} picks synced)`
              : syncStatus.status === "complete"
              ? "Sleeper draft complete"
              : `Sync error: ${syncStatus.error}`
          }
          className="flex items-center gap-1 text-xs"
        >
          <RefreshCw
            size={12}
            aria-hidden="true"
            className={
              syncStatus.status === "syncing"
                ? "text-emerald-400 animate-spin"
                : syncStatus.status === "complete"
                ? "text-slate-400"
                : "text-red-400"
            }
          />
          <span
            className={
              syncStatus.status === "syncing"
                ? "text-emerald-400"
                : syncStatus.status === "complete"
                ? "text-slate-400"
                : "text-red-400"
            }
          >
            {syncStatus.status === "syncing"
              ? "Sleeper live"
              : syncStatus.status === "complete"
              ? "Draft complete"
              : "Sync error"}
          </span>
        </span>
      )}

      {/* WebSocket connection indicator */}
      <span
        role="status"
        title={isConnected ? "Live" : "Reconnecting…"}
        aria-label={isConnected ? "Connected to the draft server" : "Disconnected — reconnecting"}
      >
        {isConnected
          ? <Wifi size={14} className="text-emerald-400" aria-hidden="true" />
          : <WifiOff size={14} className="text-red-400 animate-pulse" aria-hidden="true" />
        }
      </span>
    </div>
  );
}
