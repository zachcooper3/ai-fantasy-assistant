"use client";
import React, { useState, useEffect, useRef } from "react";
/**
 * BigBoard — the main player list, ordered by ADP.
 *
 * Counter display: "available / total" where:
 *   - numerator   = players still available matching the current filter
 *   - denominator = total players that exist for that filter (fixed on first load,
 *                   before any picks are made — captured in initialTotals ref)
 *
 * This means "56 / 56" for QB at the start, dropping to "55 / 56" after a QB is drafted.
 */

import { Player } from "@/lib/api";

const POSITIONS = ["All", "QB", "RB", "WR", "TE", "DST"] as const;
type PosFilter = (typeof POSITIONS)[number];

const INITIAL_LIMIT = 50;
const LOAD_MORE_STEP = 50;

const POS_BADGE: Record<string, string> = {
  QB:  "bg-red-900   text-red-300",
  RB:  "bg-emerald-900 text-emerald-300",
  WR:  "bg-blue-900  text-blue-300",
  TE:  "bg-amber-900 text-amber-300",
  DST: "bg-purple-900 text-purple-300",
  K:   "bg-slate-700 text-slate-300",
};

interface Props {
  players: Player[];
  isMyTurn: boolean;
  recommendedId?: number;
  onPick: (playerId: number) => void;
}

export default function BigBoard({ players, isMyTurn, recommendedId, onPick }: Props) {
  const [posFilter, setPosFilter] = useState<PosFilter>("All");
  const [search, setSearch] = useState("");
  const [displayLimit, setDisplayLimit] = useState(INITIAL_LIMIT);

  // Capture total counts per position on first non-empty load.
  // These become the fixed denominators — they don't change as players are drafted.
  const initialTotals = useRef<Record<string, number> | null>(null);
  useEffect(() => {
    if (players.length > 0 && initialTotals.current === null) {
      const totals: Record<string, number> = { All: players.length };
      players.forEach((p) => {
        totals[p.position] = (totals[p.position] ?? 0) + 1;
      });
      initialTotals.current = totals;
    }
  }, [players]);

  // Reset display limit when filter changes
  useEffect(() => {
    setDisplayLimit(INITIAL_LIMIT);
  }, [posFilter]);

  const filtered = players.filter((p) => {
    if (posFilter !== "All" && p.position !== posFilter) return false;
    if (search && !p.name.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  // Numerator: available players matching the current filter
  const availableCount = filtered.length;

  // Denominator: total that exist for this filter (fixed at first load)
  const totalCount =
    initialTotals.current?.[posFilter === "All" ? "All" : posFilter] ?? availableCount;

  // Paginate "All" with no search; show everything for position filters and searches
  const shouldPaginate = posFilter === "All" && !search;
  const visible = shouldPaginate ? filtered.slice(0, displayLimit) : filtered;
  const hasMore = shouldPaginate && filtered.length > displayLimit;

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-2 border-b border-slate-700 shrink-0">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
            Big Board
          </h2>
          <span className="text-xs text-slate-500">
            {availableCount} / {totalCount} available
          </span>
        </div>

        {/* Position filter */}
        <div className="flex gap-1 mb-2">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosFilter(pos)}
              className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-colors ${
                posFilter === pos
                  ? "bg-slate-200 text-slate-900"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              {pos}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="Search player…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:border-slate-400"
        />
      </div>

      {/* Player list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            No players match
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-slate-900 border-b border-slate-700 z-10">
              <tr className="text-xs text-slate-500 uppercase">
                <th className="text-left pl-4 py-2 w-10">#</th>
                <th className="text-left py-2">Player</th>
                <th className="text-center py-2 w-12">Pos</th>
                <th className="text-center py-2 w-12">ADP</th>
                <th className="py-2 w-16" />
              </tr>
            </thead>
            <tbody>
              {visible.map((player, idx) => {
                const prev = visible[idx - 1];
                const tierBreak = prev && player.adp - prev.adp > 5;
                const isRec = player.id === recommendedId;

                return (
                  <React.Fragment key={player.id}>
                    {tierBreak && (
                      <tr>
                        <td colSpan={5} className="py-0.5">
                          <div className="mx-4 border-t border-dashed border-slate-700" />
                        </td>
                      </tr>
                    )}
                    <tr
                      className={`border-b border-slate-800 hover:bg-slate-800/50 transition-colors ${
                        isRec ? "bg-emerald-950/40" : ""
                      }`}
                    >
                      <td className="pl-4 py-2.5 text-slate-500 text-xs">{player.rank}</td>

                      <td className="py-2.5 pr-2">
                        <div className="flex items-center gap-2">
                          {isRec && (
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                          )}
                          <div>
                            <span className="text-slate-100 font-medium">{player.name}</span>
                            <span className="ml-1.5 text-xs text-slate-500">{player.team}</span>
                          </div>
                        </div>
                      </td>

                      <td className="py-2.5 text-center">
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded text-xs font-bold ${
                            POS_BADGE[player.position] ?? "bg-slate-700 text-slate-300"
                          }`}
                        >
                          {player.pos_rank}
                        </span>
                      </td>

                      <td className="py-2.5 text-center text-slate-400 text-xs tabular-nums">
                        {player.adp}
                      </td>

                      <td className="py-2.5 pr-3 text-right">
                        <button
                          onClick={() => onPick(player.id)}
                          className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-colors ${
                            isMyTurn
                              ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                              : "bg-slate-700 hover:bg-slate-600 text-slate-300"
                          }`}
                        >
                          Draft
                        </button>
                      </td>
                    </tr>
                  </React.Fragment>
                );
              })}

              {/* Show more row */}
              {hasMore && (
                <tr>
                  <td colSpan={5} className="py-3 text-center">
                    <button
                      onClick={() => setDisplayLimit((n) => n + LOAD_MORE_STEP)}
                      className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 text-xs font-medium transition-colors"
                    >
                      Show {Math.min(LOAD_MORE_STEP, filtered.length - displayLimit)} more
                      <span className="text-slate-600 ml-1">
                        ({filtered.length - displayLimit} remaining)
                      </span>
                    </button>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
