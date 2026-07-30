"use client";
/**
 * DraftRoom — my roster, recent pick feed, and opponent position tracking.
 *
 * Layout notes: each of the three cards owns its own scroll area with a
 * min-height floor. Previously the column scrolled as a whole and only
 * Recent Picks was `flex-1`, so a full roster plus eleven opponent rows
 * squeezed the pick feed down to two or three visible rows.
 */

import { useState } from "react";
import { ChevronDown } from "lucide-react";

import { DraftState } from "@/lib/api";
import { rosterSlots, slotsBeforeMyNextPick } from "@/lib/draft";

const POS_COLORS: Record<string, string> = {
  QB:  "text-red-400 bg-red-900/30",
  RB:  "text-emerald-400 bg-emerald-900/30",
  WR:  "text-blue-400 bg-blue-900/30",
  TE:  "text-amber-400 bg-amber-900/30",
  DST: "text-purple-400 bg-purple-900/30",
  K:   "text-slate-300 bg-slate-700/30",
};

// Fixed display order for the opponent chips. Without this the chips appeared
// in whatever order that team happened to draft, so the same position sat in a
// different spot on every row and the block couldn't be scanned vertically.
const POS_ORDER = ["QB", "RB", "WR", "TE", "DST", "K"] as const;

const RECENT_PICK_LIMIT = 12;

interface Props {
  session: DraftState;
}

/** Card shell with a collapsible header, so you can fold away what you're not using. */
function Card({
  title,
  children,
  className = "",
  defaultOpen = true,
  badge,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div
      className={`bg-slate-900 rounded-xl border border-slate-700 flex flex-col min-h-0 ${
        open ? className : "shrink-0"
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-2 w-full px-4 py-3 shrink-0 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded-xl"
      >
        <ChevronDown
          size={14}
          aria-hidden="true"
          className={`text-slate-500 transition-transform ${open ? "" : "-rotate-90"}`}
        />
        <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">{title}</h2>
        {badge}
      </button>
      {open && <div className="px-4 pb-4 min-h-0 overflow-y-auto">{children}</div>}
    </div>
  );
}

export default function DraftRoom({ session }: Props) {
  const recentPicks = [...session.picks].reverse().slice(0, RECENT_PICK_LIMIT);
  const myPicks = session.my_roster;
  const slots = rosterSlots(session);

  // Opponents who pick before you're back on the clock — the only rows in the
  // tracker that change what you'd do right now.
  const upcoming = slotsBeforeMyNextPick(session);
  const upcomingSet = new Set(upcoming);

  const opponentSlots = Array.from({ length: session.league_size }, (_, i) => i + 1)
    .filter((slot) => slot !== session.my_draft_position)
    // Teams picking before your next turn float to the top.
    .sort((a, b) => {
      const ai = upcoming.indexOf(a);
      const bi = upcoming.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a - b;
    });

  return (
    <div className="flex flex-col gap-3 h-full min-h-0 overflow-hidden">

      {/* My Roster */}
      <Card
        title="My Roster"
        className="max-h-[38%]"
        badge={
          <span className="ml-auto text-xs text-slate-400 tabular-nums">
            {myPicks.length} / {session.total_rounds}
          </span>
        }
      >
        {myPicks.length === 0 ? (
          <p className="text-slate-400 text-sm">No picks yet</p>
        ) : (
          <div className="space-y-1.5">
            {myPicks.map((pick) => (
              <div key={pick.pick_number} className="flex items-center gap-2 text-sm">
                <span
                  className={`text-xs font-bold px-1.5 py-0.5 rounded shrink-0 ${
                    POS_COLORS[pick.position] ?? "text-slate-300 bg-slate-700/30"
                  }`}
                >
                  {pick.position}
                </span>
                <span className="text-slate-200 truncate">{pick.player_name}</span>
                <span className="text-xs text-slate-400 ml-auto shrink-0">{pick.nfl_team}</span>
                <span className="text-xs text-slate-500 shrink-0">Rd {pick.round_number}</span>
              </div>
            ))}
          </div>
        )}

        {/* Remaining slots, derived from this session's lineup config */}
        {myPicks.length < slots.length && (
          <div className="mt-2 space-y-1">
            {slots.slice(myPicks.length).map((slot, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs text-slate-500 border border-dashed border-slate-700 rounded px-2 py-1"
              >
                <span className="font-bold w-10">{slot}</span>
                <span aria-hidden="true">—</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Recent picks */}
      <Card title="Recent Picks" className="flex-1 min-h-[7rem]">
        {recentPicks.length === 0 ? (
          <p className="text-slate-400 text-sm">Draft not started</p>
        ) : (
          <div className="space-y-1.5">
            {recentPicks.map((pick) => (
              <div
                key={pick.pick_number}
                className={`flex items-center gap-2 text-sm rounded-lg px-2 py-1.5 ${
                  pick.is_mine
                    ? "bg-emerald-950/50 border border-emerald-800/50"
                    : "hover:bg-slate-800/50"
                }`}
              >
                <span className="text-xs text-slate-500 tabular-nums w-6 shrink-0">
                  {pick.pick_number}
                </span>

                <span
                  className={`text-xs font-bold px-1 py-0.5 rounded shrink-0 ${
                    POS_COLORS[pick.position] ?? "text-slate-300 bg-slate-700/30"
                  }`}
                >
                  {pick.position}
                </span>

                <span
                  className={`truncate ${
                    pick.is_mine ? "text-emerald-300 font-medium" : "text-slate-300"
                  }`}
                >
                  {pick.player_name}
                </span>

                <span className="ml-auto text-xs text-slate-500 shrink-0">
                  {pick.is_mine ? "Me" : `Slot ${pick.team_slot}`}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Opponent position counts.
          Was a 2-column grid at text-xs inside a 280px column — eleven rows of
          wrapped chips that couldn't be read at a glance. Now one row per team,
          fixed chip order, with the teams picking before your next turn pinned
          to the top and outlined. */}
      <Card
        title="Opponent Tracking"
        className="max-h-[34%]"
        badge={
          upcoming.length > 0 ? (
            <span className="ml-auto text-xs text-emerald-400">
              {upcoming.length} pick{upcoming.length !== 1 ? "s" : ""} before you
            </span>
          ) : undefined
        }
      >
        <div className="flex flex-col gap-1">
          {opponentSlots.map((slot) => {
            const slotPicks = session.picks.filter((p) => p.team_slot === slot);
            const posCounts: Record<string, number> = {};
            slotPicks.forEach((p) => {
              posCounts[p.position] = (posCounts[p.position] ?? 0) + 1;
            });
            const isNext = upcomingSet.has(slot);

            return (
              <div
                key={slot}
                className={`flex items-center gap-1.5 rounded-lg px-2 py-1.5 ${
                  isNext ? "bg-slate-800/80 border border-slate-600" : "border border-transparent"
                }`}
              >
                <span
                  className={`text-xs font-semibold shrink-0 w-12 ${
                    isNext ? "text-slate-200" : "text-slate-400"
                  }`}
                >
                  Slot {slot}
                </span>

                <div className="flex flex-wrap items-center gap-1 min-w-0">
                  {POS_ORDER.filter((pos) => posCounts[pos]).map((pos) => (
                    <span
                      key={pos}
                      className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                        POS_COLORS[pos] ?? "text-slate-300 bg-slate-700/30"
                      }`}
                    >
                      {pos}
                      {posCounts[pos] > 1 ? `×${posCounts[pos]}` : ""}
                    </span>
                  ))}
                  {slotPicks.length === 0 && (
                    <span className="text-xs text-slate-500">No picks</span>
                  )}
                </div>

                {isNext && (
                  <span className="ml-auto text-xs font-medium text-emerald-400 shrink-0">
                    on deck
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
