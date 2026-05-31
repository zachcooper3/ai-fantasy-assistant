"use client";
/**
 * StatusBar — top bar showing current pick number, round, and whose turn it is.
 * Highlights in green when it's the user's turn.
 */

import { Undo2, Wifi, WifiOff } from "lucide-react";
import { DraftState } from "@/lib/api";

interface Props {
  session: DraftState;
  isConnected: boolean;
  onUndo: () => void;
  onReset: () => void;
}

const POS_COLORS: Record<string, string> = {
  QB: "text-red-400",
  RB: "text-emerald-400",
  WR: "text-blue-400",
  TE: "text-amber-400",
  DST: "text-purple-400",
  K: "text-slate-400",
};

export default function StatusBar({ session, isConnected, onUndo, onReset }: Props) {
  const myTurn = session.is_my_turn;

  return (
    <div
      className={`flex items-center gap-4 px-4 py-3 border-b text-sm font-medium transition-colors ${
        myTurn
          ? "bg-emerald-950 border-emerald-700"
          : "bg-slate-900 border-slate-700"
      }`}
    >
      {/* On the clock indicator */}
      {myTurn ? (
        <span className="flex items-center gap-2 text-emerald-400 font-bold text-base">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
          </span>
          ON THE CLOCK
        </span>
      ) : (
        <span className="text-slate-400">
          {session.picks_until_my_turn} pick{session.picks_until_my_turn !== 1 ? "s" : ""} away
        </span>
      )}

      <span className="text-slate-600">|</span>

      {/* Pick / round info */}
      <span className="text-slate-300">
        Pick <span className="text-white font-bold">#{session.current_pick_number}</span>
      </span>
      <span className="text-slate-300">
        Round <span className="text-white font-bold">{session.current_round}</span>
        <span className="text-slate-500"> / {session.total_rounds}</span>
      </span>

      {/* Current team on clock */}
      {!myTurn && (
        <span className="text-slate-400">
          Slot <span className="text-slate-200">{session.current_team_slot}</span> picking
        </span>
      )}

      {/* My next pick */}
      {session.my_next_pick_number && !myTurn && (
        <span className="text-slate-500 text-xs">
          My next: #{session.my_next_pick_number} (Rd {Math.ceil(session.my_next_pick_number / session.league_size)})
        </span>
      )}

      {/* Roster summary */}
      {session.my_roster.length > 0 && (
        <span className="hidden lg:flex items-center gap-1 text-xs text-slate-500 ml-2">
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
        disabled={session.picks.length === 0}
        title="Undo last pick"
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        <Undo2 size={14} />
        Undo
      </button>

      <button
        onClick={onReset}
        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-red-900 text-slate-400 hover:text-red-300 transition-colors text-xs"
      >
        Reset
      </button>

      {/* Connection indicator */}
      <span title={isConnected ? "Live" : "Reconnecting…"}>
        {isConnected
          ? <Wifi size={14} className="text-emerald-500" />
          : <WifiOff size={14} className="text-red-400 animate-pulse" />
        }
      </span>
    </div>
  );
}
