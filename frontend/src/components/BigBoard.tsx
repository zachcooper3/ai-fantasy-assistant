"use client";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
/**
 * BigBoard — the main player list, ordered by ADP.
 *
 * Counter display: "available / total" where:
 *   - numerator   = players still available matching the current filter
 *   - denominator = total players that exist for that filter (fixed on first load,
 *                   before any picks are made — captured in initialTotals ref)
 *
 * This means "56 / 56" for QB at the start, dropping to "55 / 56" after a QB is drafted.
 *
 * Keyboard (see the shortcut list in the header tooltip):
 *   /        focus search        ↑ ↓   move selection
 *   Enter    draft selected      Esc   clear search / selection
 *   1–6      position filter
 */

import { ChevronLeft, ChevronRight } from "lucide-react";

import { Player } from "@/lib/api";
import { matchesQuery } from "@/lib/search";
import { hasModifier, isTypingTarget } from "@/lib/keyboard";

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
  /**
   * True while picks are streaming in from Sleeper. The per-row Draft button
   * is removed entirely rather than disabled: while sync is live the board is
   * a read-only mirror of Sleeper, a manual pick actively corrupts team-slot
   * attribution (audit W2), and a greyed-out button reads as "broken" rather
   * than "not applicable". Dropping the column also gives the player name
   * ~64px more room.
   */
  isSyncing?: boolean;
  /**
   * Changes when a new draft session starts. Resets the fixed "total players"
   * denominators, which are captured once on first load — without this they
   * survived a reset-to-new-session and showed stale counts until a full page
   * reload.
   */
  sessionKey?: string;
  /**
   * Collapsed to a thin rail so the AI panel can take the freed grid column
   * (the "focus mode" toggle). The component stays mounted rather than being
   * conditionally rendered by the parent, so search/filter/scroll state
   * survives a collapse-expand round trip.
   */
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export default function BigBoard({
  players,
  isMyTurn,
  recommendedId,
  onPick,
  isSyncing = false,
  sessionKey = "",
  collapsed = false,
  onToggleCollapse,
}: Props) {
  const [posFilter, setPosFilter] = useState<PosFilter>("All");
  const [search, setSearch] = useState("");
  const [displayLimit, setDisplayLimit] = useState(INITIAL_LIMIT);
  // Index into `visible`. -1 means nothing is selected — the board opens in
  // that state so an accidental Enter can't draft the ADP leader.
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const searchRef = useRef<HTMLInputElement>(null);
  const selectedRowRef = useRef<HTMLTableRowElement>(null);

  // Capture total counts per position on first non-empty load.
  // These become the fixed denominators — they don't change as players are drafted.
  const initialTotals = useRef<Record<string, number> | null>(null);

  // Clear the captured totals when the session changes so the next load
  // recaptures them for the new draft.
  useEffect(() => {
    initialTotals.current = null;
  }, [sessionKey]);

  useEffect(() => {
    if (players.length > 0 && initialTotals.current === null) {
      const totals: Record<string, number> = { All: players.length };
      players.forEach((p) => {
        totals[p.position] = (totals[p.position] ?? 0) + 1;
      });
      initialTotals.current = totals;
    }
  }, [players]);

  // Reset paging and selection when the view changes underneath them.
  useEffect(() => {
    setDisplayLimit(INITIAL_LIMIT);
    setSelectedIndex(-1);
  }, [posFilter, search]);

  const filtered = useMemo(
    () =>
      players.filter((p) => {
        if (posFilter !== "All" && p.position !== posFilter) return false;
        if (search && !matchesQuery(p, search)) return false;
        return true;
      }),
    [players, posFilter, search]
  );

  // Numerator: available players matching the current filter
  const availableCount = filtered.length;

  // Denominator: total that exist for this filter (fixed at first load)
  const totalCount =
    initialTotals.current?.[posFilter === "All" ? "All" : posFilter] ?? availableCount;

  // Paginate every view, not just unfiltered "All". Filtering to WR used to
  // render the entire filtered set — 100+ rows — in one go.
  //
  // Memoised so its identity is stable across renders: the keyboard effect
  // below depends on it, and a fresh array every render would tear down and
  // re-attach the document listener on every state change.
  const visible = useMemo(
    () => filtered.slice(0, displayLimit),
    [filtered, displayLimit]
  );
  const hasMore = filtered.length > displayLimit;

  const columnCount = isSyncing ? 4 : 5;

  const draftSelected = useCallback(() => {
    if (isSyncing) return;
    const player = visible[selectedIndex];
    if (player) onPick(player.id);
  }, [isSyncing, visible, selectedIndex, onPick]);

  // ------------------------------------------------------------------
  // Keyboard shortcuts
  //
  // Document-level rather than on a focused container: mid-draft your hands
  // shouldn't have to find the list first. isTypingTarget keeps the
  // single-letter shortcuts from firing while you're typing in the search box.
  // ------------------------------------------------------------------
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (hasModifier(e)) return;
      // Collapsed rail has no search box or rows to select — these shortcuts
      // would otherwise silently focus/act on hidden controls.
      if (collapsed) return;

      // Escape works even while typing — it's the way out of the search box.
      if (e.key === "Escape") {
        if (search) setSearch("");
        setSelectedIndex(-1);
        searchRef.current?.blur();
        return;
      }

      if (isTypingTarget(e)) {
        // Arrows still move the selection while the search box has focus, so
        // you can type a name and then arrow into the results without
        // reaching for the mouse.
        if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      }

      if (e.key === "/") {
        e.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
        return;
      }

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, visible.length - 1));
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
        return;
      }

      if (e.key === "Enter") {
        // Don't hijack Enter when a button or link has focus — that keypress
        // is already activating it, and drafting the selected player at the
        // same time would fire two actions from one keystroke.
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "BUTTON" || tag === "A") return;

        if (selectedIndex >= 0) {
          e.preventDefault();
          draftSelected();
        }
        return;
      }

      // 1-6 pick a position filter, in the order they're shown.
      const digit = Number(e.key);
      if (Number.isInteger(digit) && digit >= 1 && digit <= POSITIONS.length) {
        e.preventDefault();
        setPosFilter(POSITIONS[digit - 1]);
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [visible.length, selectedIndex, search, draftSelected, collapsed]);

  // Keep the selected row on screen when arrowing past the fold.
  useEffect(() => {
    selectedRowRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  // Arrowing to the end of the page pulls in the next batch, so keyboard
  // navigation isn't silently capped by pagination.
  useEffect(() => {
    if (hasMore && selectedIndex >= visible.length - 1 && selectedIndex >= 0) {
      setDisplayLimit((n) => n + LOAD_MORE_STEP);
    }
  }, [selectedIndex, visible.length, hasMore]);

  // Collapsed rail — the "focus mode" state. Kept as an early return rather
  // than a wrapping conditional so the rest of the component (search,
  // pagination, keyboard effects) reads the same as before; only the header
  // button and this branch know about collapsing at all.
  if (collapsed) {
    return (
      <div className="flex flex-col items-center h-full bg-slate-900 rounded-xl border border-slate-700 overflow-hidden py-3 gap-3">
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Expand Big Board"
          title="Expand Big Board (b)"
          className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          <ChevronRight size={16} aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Expand Big Board"
          className="flex-1 flex items-center justify-center focus:outline-none"
        >
          <span
            className="text-xs font-bold text-slate-400 uppercase tracking-wider"
            style={{ writingMode: "vertical-rl" }}
          >
            Big Board
          </span>
        </button>
        <span
          className="text-[10px] text-slate-500 tabular-nums pb-1"
          style={{ writingMode: "vertical-rl" }}
        >
          {availableCount}/{totalCount}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-2 border-b border-slate-700 shrink-0">
        <div className="flex items-center justify-between mb-3 gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            {onToggleCollapse && (
              <button
                type="button"
                onClick={onToggleCollapse}
                aria-label="Collapse Big Board"
                title="Collapse Big Board (b) — gives the AI panel more room"
                className="shrink-0 p-1 -ml-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
              >
                <ChevronLeft size={14} aria-hidden="true" />
              </button>
            )}
            <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider truncate">
              Big Board
            </h2>
          </div>
          <div className="flex items-center gap-2">
            {/* Explains the missing Draft column — without this the controls
                just silently vanish, which reads as a bug. */}
            {isSyncing && (
              <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-950 border border-emerald-800/60 text-xs font-medium text-emerald-300">
                <span className="relative flex h-1.5 w-1.5" aria-hidden="true">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
                </span>
                Picks syncing from Sleeper
              </span>
            )}
            <span className="text-xs text-slate-400 tabular-nums">
              {availableCount} / {totalCount} available
            </span>
          </div>
        </div>

        {/* Position filter */}
        <div className="flex gap-1 mb-2">
          {POSITIONS.map((pos, i) => (
            <button
              key={pos}
              type="button"
              onClick={() => setPosFilter(pos)}
              aria-pressed={posFilter === pos}
              title={`${pos} (${i + 1})`}
              className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
                posFilter === pos
                  ? "bg-slate-200 text-slate-900"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {pos}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          ref={searchRef}
          type="text"
          placeholder="Search player or team…  ( / )"
          aria-label="Search players by name or team"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-slate-200 text-sm placeholder-slate-400 focus:outline-none focus:border-slate-400"
        />
      </div>

      {/* Player list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400 text-sm">
            No players match
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-slate-900 border-b border-slate-700 z-10">
              <tr className="text-xs text-slate-400 uppercase">
                <th scope="col" className="text-left pl-4 py-2 w-10">#</th>
                <th scope="col" className="text-left py-2">Player</th>
                <th scope="col" className="text-center py-2 w-12">Pos</th>
                <th scope="col" className="text-center py-2 w-12">ADP</th>
                {!isSyncing && <th scope="col" className="py-2 w-16"><span className="sr-only">Draft</span></th>}
              </tr>
            </thead>
            <tbody>
              {visible.map((player, idx) => {
                const prev = visible[idx - 1];
                const tierBreak = prev && player.adp - prev.adp > 5;
                const isRec = player.id === recommendedId;
                const isSelected = idx === selectedIndex;

                return (
                  <React.Fragment key={player.id}>
                    {tierBreak && (
                      <tr>
                        <td colSpan={columnCount} className="py-0.5">
                          <div className="mx-4 border-t border-dashed border-slate-700" />
                        </td>
                      </tr>
                    )}
                    <tr
                      ref={isSelected ? selectedRowRef : undefined}
                      aria-selected={isSelected}
                      onClick={() => setSelectedIndex(idx)}
                      className={`border-b border-slate-800 transition-colors ${
                        isSelected
                          ? "bg-slate-700/70 outline outline-1 outline-emerald-500"
                          : isRec
                          ? "bg-emerald-950/40 hover:bg-slate-800/50"
                          : "hover:bg-slate-800/50"
                      }`}
                    >
                      <td className="pl-4 py-2.5 text-slate-400 text-xs">{player.rank}</td>

                      <td className="py-2.5 pr-2">
                        <div className="flex items-center gap-2">
                          {isRec && (
                            <span
                              className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0"
                              title="AI recommendation"
                            />
                          )}
                          <div>
                            <span className="text-slate-100 font-medium">{player.name}</span>
                            <span className="ml-1.5 text-xs text-slate-400">{player.team}</span>
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

                      <td className="py-2.5 text-center text-slate-300 text-xs tabular-nums">
                        {player.adp}
                      </td>

                      {!isSyncing && (
                        <td className="py-2.5 pr-3 text-right">
                          <button
                            type="button"
                            onClick={() => onPick(player.id)}
                            aria-label={`Draft ${player.name}`}
                            className={`px-2.5 py-1.5 min-h-[32px] rounded-md text-xs font-semibold transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                              isMyTurn
                                ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                                : "bg-slate-700 hover:bg-slate-600 text-slate-200"
                            }`}
                          >
                            Draft
                          </button>
                        </td>
                      )}
                    </tr>
                  </React.Fragment>
                );
              })}

              {/* Show more row */}
              {hasMore && (
                <tr>
                  <td colSpan={columnCount} className="py-3 text-center">
                    <button
                      type="button"
                      onClick={() => setDisplayLimit((n) => n + LOAD_MORE_STEP)}
                      className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-slate-100 text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                    >
                      Show {Math.min(LOAD_MORE_STEP, filtered.length - displayLimit)} more
                      <span className="text-slate-400 ml-1">
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
