"use client";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
/**
 * BigBoard — the main player list, ordered by ADP.
 *
 * Counter display: "available / total" for the current position filter,
 * where BOTH numbers come from the board's scarcity counts, not from the
 * rendered player list.
 *
 * That distinction is the whole fix. The list is capped by the board's fetch
 * limit (400), and the pool is larger than that, so counting rows made the
 * numerator max out at the cap — and the denominator, captured on first
 * load, maxed out at the same 400. The counter therefore read "400 / 400
 * available" at the START of a draft and still read "400 / 400" at the END
 * of one, having been incapable of showing anything else. It also
 * contradicted the Position Depth panel directly beneath it, which counts
 * the real pool.
 *
 * Scarcity counts come from an uncapped server-side query, so "412 / 592"
 * now means what it says. They exclude IR/PUP/Suspended/Out — those players
 * ARE listed on the board (see the board route's include_undraftable) but
 * aren't startable supply, which is why a row can exist without being
 * counted.
 *
 * Keyboard (the canonical list users see is the ? overlay — see
 * ShortcutsOverlay; keep the two in step):
 *   /        focus search        ↑ ↓   move selection
 *   Enter    draft selected      Esc   clear search / selection
 *   1–7      position filter (indexes into POSITIONS below)
 */

import { ChevronLeft, ChevronRight } from "lucide-react";

import { Player, Scarcity } from "@/lib/api";
import { matchesQuery } from "@/lib/search";
import { hasModifier, isTypingTarget } from "@/lib/keyboard";
import { isUndraftable } from "@/lib/injury";
import InjuryBadge from "@/components/InjuryBadge";

// K is here even though there's no kicker roster slot (DraftConfigRequest has
// no k_slots — see rosterSlots in lib/draft.ts). The board is the pool, not
// the lineup: there are 51 kickers in the ADP data and the recommendation
// engine now surfaces one in the final rounds, so a filter that can't reach
// them left the last pick of every draft to name-search only.
const POSITIONS = ["All", "QB", "RB", "WR", "TE", "DST", "K"] as const;

/** The positions Scarcity actually carries — POSITIONS minus the "All" pseudo-filter. */
const SCARCITY_POSITIONS = ["QB", "RB", "WR", "TE", "DST", "K"] as const;
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
  /**
   * Uncapped available-player counts per position, from the board response.
   * Drives the header counter — see the module docstring on why the
   * rendered list can't. Null before the first board load.
   */
  scarcity: Scarcity | null;
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
   * Every pick is in. Suppresses the Draft column for the same reason
   * isSyncing does — the action isn't available, and the backend rejects it
   * with a 400 ("Draft is already complete") — but says so with its own
   * chip, since "syncing from Sleeper" would be a lie.
   */
  draftComplete?: boolean;
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
  /**
   * Opens the player detail drawer. Optional so the board still renders
   * standalone (and in tests) without one — when omitted the name is plain
   * text rather than a button that does nothing.
   */
  onShowDetail?: (playerId: number) => void;
}

export default function BigBoard({
  players,
  scarcity,
  isMyTurn,
  recommendedId,
  onPick,
  isSyncing = false,
  draftComplete = false,
  sessionKey = "",
  collapsed = false,
  onToggleCollapse,
  onShowDetail,
}: Props) {
  const [posFilter, setPosFilter] = useState<PosFilter>("All");
  const [search, setSearch] = useState("");
  const [displayLimit, setDisplayLimit] = useState(INITIAL_LIMIT);
  // Index into `visible`. -1 means nothing is selected — the board opens in
  // that state so an accidental Enter can't draft the ADP leader.
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const searchRef = useRef<HTMLInputElement>(null);
  const selectedRowRef = useRef<HTMLTableRowElement>(null);

  // Available counts for the current filter, straight from scarcity.
  // `All` is the sum rather than a separate figure so the six position
  // tallies and the total can never disagree.
  const availableByFilter = useMemo(() => {
    if (!scarcity) return null;
    const counts: Record<string, number> = { All: 0 };
    for (const pos of SCARCITY_POSITIONS) {
      counts[pos] = scarcity[pos];
      counts.All += scarcity[pos];
    }
    return counts;
  }, [scarcity]);

  // Denominators, captured from the first non-empty scarcity payload —
  // i.e. the size of the pool before anything was drafted. Held in a ref
  // rather than state: this is write-once-per-session bookkeeping, and
  // making it state would re-render the whole table when it settles.
  const initialTotals = useRef<Record<string, number> | null>(null);

  // Clear the captured totals when the session changes so the next load
  // recaptures them for the new draft.
  useEffect(() => {
    initialTotals.current = null;
  }, [sessionKey]);

  useEffect(() => {
    if (availableByFilter && availableByFilter.All > 0 && initialTotals.current === null) {
      initialTotals.current = availableByFilter;
    }
  }, [availableByFilter]);

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

  // Numerator and denominator for the header counter. Both null until the
  // first board response lands — the counter is hidden rather than showing
  // a placeholder that looks like a real number.
  const availableCount = availableByFilter?.[posFilter] ?? null;
  const totalCount = initialTotals.current?.[posFilter] ?? availableCount;

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

  // Both reasons the pick controls go away collapse to one question here.
  const canPick = !isSyncing && !draftComplete;
  const columnCount = canPick ? 5 : 4;

  const draftSelected = useCallback(() => {
    if (!canPick) return;
    const player = visible[selectedIndex];
    if (player) onPick(player.id);
  }, [canPick, visible, selectedIndex, onPick]);

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

      // "i" opens the detail drawer for the selected player, so the whole
      // arrow-to-a-name / inspect / draft loop stays on the keyboard.
      if (e.key === "i") {
        const player = visible[selectedIndex];
        if (player && onShowDetail) {
          e.preventDefault();
          onShowDetail(player.id);
        }
        return;
      }

      // 1-N pick a position filter, in the order they're shown.
      const digit = Number(e.key);
      if (Number.isInteger(digit) && digit >= 1 && digit <= POSITIONS.length) {
        e.preventDefault();
        setPosFilter(POSITIONS[digit - 1]);
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [visible, selectedIndex, search, draftSelected, collapsed, onShowDetail]);

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
        {availableCount !== null && (
          <span
            className="text-[10px] text-slate-500 tabular-nums pb-1"
            style={{ writingMode: "vertical-rl" }}
          >
            {availableCount}/{totalCount}
          </span>
        )}
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
            {draftComplete && (
              <span className="px-2 py-0.5 rounded-full bg-slate-800 border border-slate-600 text-xs font-medium text-slate-300">
                Draft complete — review only
              </span>
            )}
            {isSyncing && !draftComplete && (
              <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-950 border border-emerald-800/60 text-xs font-medium text-emerald-300">
                <span className="relative flex h-1.5 w-1.5" aria-hidden="true">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
                </span>
                Picks syncing from Sleeper
              </span>
            )}
            {availableCount !== null && (
              <span
                className="text-xs text-slate-400 tabular-nums"
                title="Undrafted players who can currently play. IR/PUP/Suspended players are listed below but not counted."
              >
                {availableCount} / {totalCount} available
              </span>
            )}
          </div>
        </div>

        {/* Position filter */}
        <div className="flex flex-wrap gap-1 mb-2">
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
                {canPick && <th scope="col" className="py-2 w-16"><span className="sr-only">Draft</span></th>}
              </tr>
            </thead>
            <tbody>
              {visible.map((player, idx) => {
                const prev = visible[idx - 1];
                const tierBreak = prev && player.adp - prev.adp > 5;
                const isRec = player.id === recommendedId;
                const isSelected = idx === selectedIndex;
                // Listed, but not part of the available count and never
                // suggested by the AI. Dimmed so the discrepancy between
                // "rows on screen" and "N available" reads as deliberate
                // rather than as an off-by-something.
                const cannotPlay = isUndraftable(player.injury_status);

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
                      } ${cannotPlay && !isSelected ? "opacity-55" : ""}`}
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
                          <div className="min-w-0">
                            {/* The name is the affordance for the detail
                                drawer — a separate icon button would need
                                its own column in a table that's already
                                fighting for width at the rail end of the
                                resize range. */}
                            {onShowDetail ? (
                              <button
                                type="button"
                                onClick={() => onShowDetail(player.id)}
                                aria-label={`Details for ${player.name}`}
                                className="text-slate-100 font-medium text-left hover:text-emerald-300 hover:underline decoration-dotted underline-offset-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
                              >
                                {player.name}
                              </button>
                            ) : (
                              <span className="text-slate-100 font-medium">{player.name}</span>
                            )}
                            <span className="ml-1.5 text-xs text-slate-400">{player.team}</span>
                            {/* whitespace-nowrap: this is two words to the
                                layout engine, and in a narrow board column it
                                broke between them — a row reading "DET bye"
                                with a bare "6" on the next line. */}
                            {player.bye != null && (
                              <span className="ml-1.5 text-xs text-slate-500 whitespace-nowrap">
                                bye {player.bye}
                              </span>
                            )}
                            <InjuryBadge status={player.injury_status} className="ml-1.5" />
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

                      {canPick && (
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
