"use client";
/**
 * PlayerDetailDrawer — everything the app knows about one player.
 *
 * This app ingests five categories of nflverse metrics, draft capital, college
 * production and the real NFL schedule, and until now fed all of it straight
 * into Claude's prompt and nowhere else. The consequence wasn't cosmetic: the
 * only way to see WHY a recommendation was made was to hope the model spelled
 * it out in prose, which makes a bad pick indistinguishable from a bad input.
 * This is the audit surface for that — the same numbers the prompt gets.
 *
 * Presentation rules, all of which exist because the underlying data is
 * deliberately sparse (see PlayerMetrics' docstring — every field is Optional
 * and a missing one means "unknown", not zero):
 *
 *   - A metric with no value is omitted, never rendered as 0 or "—" in a
 *     grid that implies it was measured.
 *   - A section with nothing in it doesn't render at all.
 *   - games_played is shown next to the season, and a small sample is called
 *     out explicitly — a 3-game row and a 17-game row look identical once
 *     they're both averages.
 *   - Trends state their direction in words. depth_chart_trend is NEGATIVE
 *     when a player is moving UP the chart, which is exactly the kind of sign
 *     convention that gets read backwards at speed.
 */

import { useEffect, useRef, useState } from "react";
import { X, TrendingUp, TrendingDown, AlertTriangle, GraduationCap } from "lucide-react";

import { api, PlayerDetail } from "@/lib/api";
import InjuryBadge from "@/components/InjuryBadge";

const POS_COLORS: Record<string, string> = {
  QB:  "text-red-400 bg-red-900/30",
  RB:  "text-emerald-400 bg-emerald-900/30",
  WR:  "text-blue-400 bg-blue-900/30",
  TE:  "text-amber-400 bg-amber-900/30",
  DST: "text-purple-400 bg-purple-900/30",
  K:   "text-slate-300 bg-slate-700/30",
};

/**
 * Below this many games, per-game averages are noise dressed as a rate.
 * Matches the spirit of the backend's own small-sample guard on QB ppg.
 */
const SMALL_SAMPLE_GAMES = 6;

/** Weeks of schedule to show. Fantasy regular seasons are decided early and a
 *  full 17-week strip is unreadable in a 28rem drawer. */
const SCHEDULE_WEEKS = 8;

// ---------------------------------------------------------------------------
// Formatters. Each returns null for a null input so callers can drop the row.
// ---------------------------------------------------------------------------

function pct(v: number | null | undefined): string | null {
  return v == null ? null : `${(v * 100).toFixed(0)}%`;
}

function num(v: number | null | undefined, digits = 1): string | null {
  return v == null ? null : v.toFixed(digits);
}

function whole(v: number | null | undefined): string | null {
  return v == null ? null : String(v);
}

/** Percentage-point delta with an explicit sign, e.g. "+3.4pp". */
function pctPoints(v: number | null | undefined): string | null {
  if (v == null) return null;
  const points = v * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)}pp`;
}

// ---------------------------------------------------------------------------

interface StatItem {
  label: string;
  /** null means "not measured" — the tile is dropped, never zero-filled. */
  value: string | null;
  hint?: string;
}

/**
 * A grid of stats, driven by data rather than by children.
 *
 * Deliberately not a component that filters its own JSX children: doing it
 * that way means the "is this section empty?" decision depends on reaching
 * into child element props, which type-checks poorly and breaks silently the
 * first time a section wraps a stat in anything. A plain array can just be
 * filtered.
 *
 * Renders nothing at all when every item is null — an empty "Efficiency"
 * heading over blank space claims the numbers are zero when they're unknown.
 */
function StatGrid({ title, items }: { title: string; items: StatItem[] }) {
  const present = items.filter((i) => i.value !== null);
  if (present.length === 0) return null;

  return (
    <div>
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        {title}
      </h3>
      <div className="grid grid-cols-3 gap-1.5">
        {present.map((i) => (
          <div key={i.label} className="rounded-lg bg-slate-800/60 px-2.5 py-2" title={i.hint}>
            <div className="text-[10px] uppercase tracking-wide text-slate-400">
              {i.label}
            </div>
            <div className="text-sm font-semibold text-slate-100 tabular-nums">
              {i.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** A trend row — the arrow and the sentence, since the sign alone is ambiguous. */
function Trend({
  label,
  value,
  goodDirection,
  format,
}: {
  label: string;
  value: number | null;
  /** Which sign means the player's role is growing. */
  goodDirection: "positive" | "negative";
  format: (v: number) => string;
}) {
  if (value == null || value === 0) return null;
  const rising = goodDirection === "positive" ? value > 0 : value < 0;
  const Icon = rising ? TrendingUp : TrendingDown;
  return (
    <div className="flex items-center gap-2 text-xs">
      <Icon
        size={13}
        aria-hidden="true"
        className={rising ? "text-emerald-400 shrink-0" : "text-amber-400 shrink-0"}
      />
      <span className="text-slate-400">{label}</span>
      <span
        className={`ml-auto font-semibold tabular-nums ${
          rising ? "text-emerald-400" : "text-amber-400"
        }`}
      >
        {format(value)}
      </span>
      <span className={`text-[10px] ${rising ? "text-emerald-400" : "text-amber-400"}`}>
        {rising ? "rising" : "falling"}
      </span>
    </div>
  );
}

export default function PlayerDetailDrawer({
  playerId,
  onClose,
}: {
  playerId: number;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<PlayerDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    api
      .playerDetail(playerId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Couldn't load player details.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [playerId]);

  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  // Escape closes. Capture phase, because BigBoard's own document-level
  // Escape handler would otherwise clear the search box behind the drawer
  // instead of closing the thing the user is actually looking at.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [onClose]);

  const p = detail?.player;
  const m = detail?.metrics ?? null;
  const dp = detail?.draft_profile ?? null;
  const smallSample = m != null && m.games_played < SMALL_SAMPLE_GAMES;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/50"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={p ? `${p.name} details` : "Player details"}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md h-full overflow-y-auto bg-slate-900 border-l border-slate-700 shadow-2xl"
      >
        {/* Header */}
        <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-5 py-4 z-10">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-lg font-bold text-white break-words">
                  {p?.name ?? "Loading…"}
                </h2>
                {p && (
                  <span
                    className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                      POS_COLORS[p.position] ?? "text-slate-300 bg-slate-700/30"
                    }`}
                  >
                    {p.pos_rank}
                  </span>
                )}
                <InjuryBadge status={p?.injury_status} />
              </div>
              {p && (
                <div className="flex items-center gap-2 mt-1 text-xs text-slate-400 tabular-nums">
                  <span>{p.team}</span>
                  {p.bye != null && <span>Bye {p.bye}</span>}
                  <span>ADP {p.adp}</span>
                  <span>Rank {p.rank}</span>
                </div>
              )}
            </div>
            <button
              ref={closeRef}
              type="button"
              onClick={onClose}
              aria-label="Close player details"
              className="shrink-0 p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </div>

        <div className="px-5 py-4 space-y-5">
          {error && (
            <div
              role="alert"
              className="flex items-start gap-2 text-xs text-red-200 bg-red-900/25 border border-red-800/50 rounded-lg px-3 py-2"
            >
              <AlertTriangle size={12} className="shrink-0 mt-0.5" aria-hidden="true" />
              {error}
            </div>
          )}

          {!detail && !error && (
            <p className="text-sm text-slate-400">Loading player details…</p>
          )}

          {detail && (
            <>
              {/* Draft capital — the primary signal for anyone without an NFL
                  season behind them, and still worth seeing for anyone who
                  has one (see the backend's _experience_context). */}
              {dp && (
                <div>
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                    Draft capital
                  </h3>
                  <div className="rounded-lg bg-slate-800/60 px-3 py-2.5 text-sm text-slate-200">
                    {dp.draft_round != null ? (
                      <span className="font-semibold">
                        Round {dp.draft_round}
                        {dp.draft_pick != null && ` · #${dp.draft_pick} overall`}
                      </span>
                    ) : (
                      <span className="font-semibold">{dp.draft_year} draft class</span>
                    )}
                    <span className="text-slate-400">
                      {dp.draft_round != null && ` · ${dp.draft_year}`}
                      {dp.draft_team && ` · ${dp.draft_team}`}
                    </span>
                    {dp.college && (
                      <div className="flex items-center gap-1.5 mt-1 text-xs text-slate-400">
                        <GraduationCap size={12} aria-hidden="true" />
                        {dp.college}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* College production — the closest thing to a track record for
                  a player with no NFL snaps. Counting stats only; the fields
                  that don't apply to a position come back null and drop out. */}
              {dp?.college_season != null && (
                <StatGrid
                  title={`College — ${dp.college_season}`}
                  items={[
                    { label: "Pass yds", value: whole(dp.passing_yards) },
                    { label: "Pass TD", value: whole(dp.passing_td) },
                    { label: "INT", value: whole(dp.interceptions_thrown) },
                    { label: "Rush yds", value: whole(dp.rushing_yards) },
                    { label: "Rush TD", value: whole(dp.rushing_td) },
                    { label: "Carries", value: whole(dp.carries) },
                    { label: "Rec", value: whole(dp.receptions) },
                    { label: "Rec yds", value: whole(dp.receiving_yards) },
                    { label: "Rec TD", value: whole(dp.receiving_td) },
                  ]}
                />
              )}

              {/* NFL metrics */}
              {m ? (
                <>
                  <div className="flex items-center gap-2 flex-wrap text-xs">
                    <span className="text-slate-300 font-semibold">
                      {m.season} season
                    </span>
                    <span className="text-slate-400">
                      {m.games_played} game{m.games_played !== 1 ? "s" : ""} · through
                      week {m.through_week}
                    </span>
                    {m.team && m.team !== p?.team && (
                      <span
                        title="These numbers were earned with a different team than his current one."
                        className="px-1.5 py-0.5 rounded border border-slate-600 bg-slate-800 text-[10px] font-medium text-slate-300"
                      >
                        with {m.team}
                      </span>
                    )}
                  </div>

                  {smallSample && (
                    <div className="flex items-start gap-2 text-xs text-amber-200 bg-amber-900/25 border border-amber-800/50 rounded-lg px-3 py-2">
                      <AlertTriangle size={12} className="shrink-0 mt-0.5" aria-hidden="true" />
                      Only {m.games_played} game{m.games_played !== 1 ? "s" : ""} of data —
                      treat every per-game figure below as a small sample, not a rate.
                    </div>
                  )}

                  <StatGrid
                    title="Opportunity"
                    items={[
                      { label: "Snap %", value: pct(m.snap_pct) },
                      { label: "Tgt share", value: pct(m.target_share) },
                      { label: "Carry share", value: pct(m.carry_share) },
                      { label: "Tgt / gm", value: num(m.targets_per_game) },
                      { label: "Car / gm", value: num(m.carries_per_game) },
                      {
                        label: "RZ / gm",
                        value: num(m.red_zone_touches_per_game),
                        hint: "Red-zone touches per game",
                      },
                    ]}
                  />

                  <StatGrid
                    title="Efficiency"
                    items={[
                      { label: "Y / tgt", value: num(m.yards_per_target) },
                      { label: "Y / carry", value: num(m.yards_per_carry) },
                      { label: "Catch %", value: pct(m.catch_rate) },
                      {
                        label: "YAC / rec",
                        value: num(m.yac_per_reception),
                        hint: "Yards after catch per reception",
                      },
                      {
                        label: "RACR",
                        value: num(m.racr, 2),
                        hint: "Receiving yards divided by air yards",
                      },
                    ]}
                  />

                  <StatGrid
                    title="Role & team context"
                    items={[
                      {
                        label: "Depth rank",
                        value: whole(m.depth_chart_rank),
                        hint: "1 = starter at the position",
                      },
                      {
                        label: "Team pass %",
                        value: pct(m.team_pass_rate),
                        hint: "How pass-heavy his offense was",
                      },
                    ]}
                  />

                  <StatGrid
                    title="Consistency & risk"
                    items={[
                      { label: "PPR / gm", value: num(m.fantasy_points_avg) },
                      {
                        label: "Std dev",
                        value: num(m.fantasy_points_stdev),
                        hint: "Week-to-week PPR swing — higher is more boom/bust",
                      },
                      { label: "Games missed", value: whole(m.games_missed) },
                      {
                        label: "Injury rpts",
                        value: whole(m.injury_report_appearances),
                        hint: "Weeks appearing on any injury report",
                      },
                    ]}
                  />

                  {/* Trends. Rendered as sentences because the sign convention
                      is not self-evident — depth_chart_trend in particular is
                      negative when a player is moving UP the chart. */}
                  {(m.target_share_trend || m.snap_pct_trend || m.depth_chart_trend) && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                        Late-season trend
                      </h3>
                      <div className="space-y-1.5 rounded-lg bg-slate-800/60 px-3 py-2.5">
                        <Trend
                          label="Target share"
                          value={m.target_share_trend}
                          goodDirection="positive"
                          format={(v) => pctPoints(v)!}
                        />
                        <Trend
                          label="Snap share"
                          value={m.snap_pct_trend}
                          goodDirection="positive"
                          format={(v) => pctPoints(v)!}
                        />
                        <Trend
                          label="Depth chart"
                          value={m.depth_chart_trend}
                          goodDirection="negative"
                          format={(v) => `${v > 0 ? "+" : ""}${v}`}
                        />
                        <p className="text-[10px] text-slate-500 pt-1">
                          Last 3 weeks against the season average.
                        </p>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                !dp && (
                  <p className="text-sm text-slate-400">
                    No NFL production data for this player yet.
                  </p>
                )
              )}

              {m == null && dp != null && (
                <p className="text-xs text-slate-400">
                  No NFL snaps yet — draft capital and college production above are
                  the whole record.
                </p>
              )}

              {/* Schedule */}
              {detail.schedule.length > 0 && (
                <div>
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                    Opening schedule
                    {detail.season != null && (
                      <span className="ml-1.5 font-medium normal-case tracking-normal text-slate-500">
                        {detail.season}
                      </span>
                    )}
                  </h3>
                  <div className="flex flex-wrap gap-1.5">
                    {detail.schedule.slice(0, SCHEDULE_WEEKS).map((g) => (
                      <div
                        key={g.week}
                        className="rounded-lg bg-slate-800/60 px-2 py-1.5 text-center"
                        title={`Week ${g.week} ${g.is_home ? "vs" : "at"} ${g.opponent}`}
                      >
                        <div className="text-[10px] text-slate-500 tabular-nums">
                          W{g.week}
                        </div>
                        <div className="text-xs font-semibold text-slate-200">
                          <span className="text-slate-500">{g.is_home ? "vs" : "@"}</span>{" "}
                          {g.opponent}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
