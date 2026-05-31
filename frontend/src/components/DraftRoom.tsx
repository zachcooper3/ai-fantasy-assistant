"use client";
/**
 * DraftRoom — recent pick feed + my roster.
 * Shows the last N picks across all teams and the user's drafted players.
 */

import { DraftState } from "@/lib/api";

const POS_COLORS: Record<string, string> = {
  QB:  "text-red-400 bg-red-900/30",
  RB:  "text-emerald-400 bg-emerald-900/30",
  WR:  "text-blue-400 bg-blue-900/30",
  TE:  "text-amber-400 bg-amber-900/30",
  DST: "text-purple-400 bg-purple-900/30",
  K:   "text-slate-400 bg-slate-700/30",
};

// Standard roster slot layout for display
const ROSTER_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DST"];

interface Props {
  session: DraftState;
}

export default function DraftRoom({ session }: Props) {
  const recentPicks = [...session.picks].reverse().slice(0, 12);

  // Assign my roster picks to display slots (simplified — just shows in draft order)
  const myPicks = session.my_roster;

  return (
    <div className="flex flex-col gap-4 h-full overflow-y-auto">

      {/* My Roster */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 shrink-0">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          My Roster
        </h2>

        {myPicks.length === 0 ? (
          <p className="text-slate-600 text-sm">No picks yet</p>
        ) : (
          <div className="space-y-1.5">
            {myPicks.map((pick) => (
              <div
                key={pick.pick_number}
                className="flex items-center gap-2 text-sm"
              >
                <span
                  className={`text-xs font-bold px-1.5 py-0.5 rounded shrink-0 ${
                    POS_COLORS[pick.position] ?? "text-slate-400 bg-slate-700/30"
                  }`}
                >
                  {pick.position}
                </span>
                <span className="text-slate-200">{pick.player_name}</span>
                <span className="text-xs text-slate-500 ml-auto">{pick.nfl_team}</span>
                <span className="text-xs text-slate-600">Rd {pick.round_number}</span>
              </div>
            ))}
          </div>
        )}

        {/* Empty roster slots */}
        {myPicks.length < ROSTER_SLOTS.length && (
          <div className="mt-2 space-y-1">
            {ROSTER_SLOTS.slice(myPicks.length).map((slot, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs text-slate-700 border border-dashed border-slate-800 rounded px-2 py-1"
              >
                <span className="font-bold w-8">{slot}</span>
                <span>—</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent picks */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 flex-1 min-h-0 flex flex-col">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 shrink-0">
          Recent Picks
        </h2>

        {recentPicks.length === 0 ? (
          <p className="text-slate-600 text-sm">Draft not started</p>
        ) : (
          <div className="space-y-1.5 overflow-y-auto flex-1 min-h-0">
            {recentPicks.map((pick) => (
              <div
                key={pick.pick_number}
                className={`flex items-center gap-2 text-sm rounded-lg px-2 py-1.5 ${
                  pick.is_mine
                    ? "bg-emerald-950/50 border border-emerald-800/50"
                    : "hover:bg-slate-800/50"
                }`}
              >
                {/* Pick number */}
                <span className="text-xs text-slate-600 tabular-nums w-6 shrink-0">
                  {pick.pick_number}
                </span>

                {/* Position badge */}
                <span
                  className={`text-xs font-bold px-1 py-0.5 rounded shrink-0 ${
                    POS_COLORS[pick.position] ?? "text-slate-400 bg-slate-700/30"
                  }`}
                >
                  {pick.position}
                </span>

                {/* Player name */}
                <span className={pick.is_mine ? "text-emerald-300 font-medium" : "text-slate-300"}>
                  {pick.player_name}
                </span>

                {/* Slot */}
                <span className="ml-auto text-xs text-slate-600 shrink-0">
                  {pick.is_mine ? "Me" : `Slot ${pick.team_slot}`}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Opponent position counts */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 shrink-0">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          Opponent Tracking
        </h2>
        <div className="grid grid-cols-2 gap-1 text-xs">
          {Array.from({ length: session.league_size }, (_, i) => i + 1)
            .filter((slot) => slot !== session.my_draft_position)
            .map((slot) => {
              const slotPicks = session.picks.filter((p) => p.team_slot === slot);
              const posCounts: Record<string, number> = {};
              slotPicks.forEach((p) => {
                posCounts[p.position] = (posCounts[p.position] ?? 0) + 1;
              });
              return (
                <div key={slot} className="flex items-center gap-1 text-slate-500">
                  <span className="text-slate-600 w-10">Slot {slot}</span>
                  {Object.entries(posCounts).map(([pos, n]) => (
                    <span
                      key={pos}
                      className={`px-1 rounded text-xs font-bold ${
                        POS_COLORS[pos] ?? "text-slate-400 bg-slate-700/30"
                      }`}
                    >
                      {pos}{n > 1 ? `×${n}` : ""}
                    </span>
                  ))}
                  {slotPicks.length === 0 && <span>—</span>}
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}
