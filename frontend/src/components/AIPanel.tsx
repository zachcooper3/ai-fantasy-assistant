"use client";
/**
 * AIPanel — AI recommendation + positional scarcity alerts.
 *
 * Renders sections instead of one pick plus a footnote list of
 * alternatives: Main (the model's synthesized pick, with real reasoning),
 * Best Available (up to 2, cheapest by ADP regardless of need), Needs (up
 * to 2, fills your highest-priority open starting slot), Depth (0-2, one QB
 * stash and one TE stash — see the backend's _depth_pick — which can appear
 * alongside Needs, not just once it empties), and Opportunity (0-1, an
 * RB/WR/TE whose role looks bigger than his ADP/tier reflects — see the
 * backend's _opportunity_pick_from). Main and Opportunity are the only
 * model-generated sections — the rest are computed server-side from the
 * same board data, so they're never asked of Claude at all. See the
 * backend's RecommendationResult docstring.
 *
 * Sections are allowed to overlap: the main pick is routinely also the best
 * value on the board, or the neediest-position fill. Rather than hiding
 * that, a card that qualifies for more than one section shows a badge for
 * each — see PickSuggestion.tags and TAG_LABELS below.
 *
 * Falls back gracefully when no API key is configured (the backend's ADP
 * fallback reports confidence "low" and carries no strategy).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Lightbulb,
  RefreshCw,
  AlertTriangle,
  Info,
  History,
  Compass,
  Zap,
  PanelLeftClose,
  PanelLeftOpen,
  TrendingUp,
  Target,
  Layers,
  Rocket,
} from "lucide-react";

import {
  api,
  Confidence,
  Player,
  PickSuggestion,
  Recommendation,
  Scarcity,
  ScarcityAlert,
  ScarcityAnalysis,
  SectionTag,
  Survival,
} from "@/lib/api";
import { PastRecommendation } from "@/hooks/useDraft";
import { adpValue } from "@/lib/draft";
import ConfirmButton from "@/components/ConfirmButton";
import InjuryBadge from "@/components/InjuryBadge";

const POS_COLORS: Record<string, string> = {
  QB:  "text-red-400",
  RB:  "text-emerald-400",
  WR:  "text-blue-400",
  TE:  "text-amber-400",
  DST: "text-purple-400",
  K:   "text-slate-300",
};

const CONFIDENCE_STYLES: Record<Confidence, { label: string; className: string }> = {
  high:   { label: "High confidence",   className: "bg-emerald-950 text-emerald-300 border-emerald-800/60" },
  medium: { label: "Medium confidence", className: "bg-slate-800 text-slate-300 border-slate-600" },
  low:    { label: "Low confidence",    className: "bg-amber-950/60 text-amber-300 border-amber-800/60" },
};

const VALUE_STYLES = {
  value: "text-emerald-400",
  reach: "text-amber-400",
  even:  "text-slate-400",
} as const;

/**
 * Position Depth tiles, keyed by the backend's scarcity tier.
 *
 * The tier is NOT computed here. It used to be — two absolute thresholds
 * (`count < 20` low, `count < 8` critical) applied to every position in every
 * league — which is wrong wherever demand isn't uniform across positions. A
 * 12-team league starting 2 RB + 1 FLEX has demand for ~36 running backs, so
 * 25 left is genuinely critical, while 25 quarterbacks against demand for 12
 * is a healthy supply. The old rule called both of those "ok" and only ever
 * fired late, after the scarcity had already cost you something.
 *
 * GET /api/recommend/scarcity does the real math from THIS session's league
 * size and starting lineup, sharing compute_position_scarcity with the
 * recommendation prompt's own Positional Availability section — so what the
 * panel says and what the model was told can't drift apart.
 *
 * "unknown" is a real state, not a default: it covers both the pre-first-fetch
 * moment and a position this league doesn't start at all (the backend drops
 * zero-slot positions rather than emitting "critical: 0 needed"). Either way
 * the count still renders, without a tier claim attached to it.
 */
const TIER_STYLES: Record<
  ScarcityAlert["tier"] | "unknown",
  { tile: string; count: string; label: string | null }
> = {
  critical: {
    tile: "bg-red-900/30 border-red-800/50",
    count: "text-red-400",
    label: "Critical",
  },
  low: {
    tile: "bg-amber-900/20 border-amber-800/30",
    count: "text-amber-400",
    label: "Low",
  },
  ok: {
    tile: "bg-slate-800 border-slate-700",
    count: "text-slate-200",
    label: null,
  },
  unknown: {
    tile: "bg-slate-800 border-slate-700",
    count: "text-slate-200",
    label: null,
  },
};

/**
 * Panel width, in px, above which a 2-entry section (Best Available, Needs)
 * switches from a stacked list to a 2-column grid.
 *
 * This can't be a Tailwind breakpoint (md:, xl:) because those respond to
 * *viewport* width, and this panel's actual width changes independently of
 * the viewport when the Big Board is collapsed into its rail — the panel can
 * go from ~340px to over 700px on the same screen size. Measured directly
 * with a ResizeObserver instead.
 */
const SECTION_GRID_MIN_WIDTH = 560;

/** Label + icon for each PickSuggestion.tags entry, used as a small badge on
 * a card that qualifies for a section other than the one it's rendered
 * under — see the module docstring on why overlap is shown, not hidden. */
const TAG_META: Record<SectionTag, { label: string; icon: typeof TrendingUp }> = {
  main: { label: "Main Pick", icon: Lightbulb },
  best_available: { label: "Best Value", icon: TrendingUp },
  needs: { label: "Fills Need", icon: Target },
  depth: { label: "Depth Stash", icon: Layers },
  opportunity: { label: "Opportunity", icon: Rocket },
};

/** Tags to badge on a card, excluding the section it's already shown under —
 * that one is implied by which section the card appears in. */
function otherTags(tags: SectionTag[], currentSection: SectionTag): SectionTag[] {
  return tags.filter((t) => t !== currentSection);
}

function TagBadges({ tags, currentSection }: { tags: SectionTag[]; currentSection: SectionTag }) {
  const shown = otherTags(tags, currentSection);
  if (shown.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-1.5">
      {shown.map((tag) => {
        const meta = TAG_META[tag];
        const Icon = meta.icon;
        return (
          <span
            key={tag}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-600 bg-slate-800 text-[10px] font-medium text-slate-300"
          >
            <Icon size={9} aria-hidden="true" />
            {meta.label}
          </span>
        );
      })}
    </div>
  );
}

/**
 * Survival badge. This is the single most decision-relevant fact on the row —
 * a player who will still be there next turn costs you nothing to skip, and
 * one who won't is your only chance — so it reads as a badge rather than
 * another number in the grey meta line.
 *
 * Colour runs red-to-slate on urgency, NOT on quality: "will last" is not a
 * worse player, he is a later one. The label says so in words because colour
 * alone would imply a ranking.
 */
const SURVIVAL_STYLES: Record<
  Exclude<Survival, "">,
  { label: string; title: string; className: string }
> = {
  take_now: {
    label: "TAKE NOW",
    title: "Very unlikely to still be available at your next turn — this is your only chance at him.",
    className: "bg-red-950 text-red-300 border-red-800/70",
  },
  might_last: {
    label: "MIGHT LAST",
    title: "Could go either way before your next turn.",
    className: "bg-amber-950/60 text-amber-300 border-amber-800/60",
  },
  will_last: {
    label: "WILL LAST",
    title: "Very likely still available at your next turn — not a worse player, just one you can take later.",
    className: "bg-slate-800 text-slate-400 border-slate-600",
  },
};

function SurvivalBadge({ survival }: { survival: Survival }) {
  if (!survival) return null;              // last pick of the draft
  const s = SURVIVAL_STYLES[survival];
  if (!s) return null;                     // unknown code from a newer backend
  return (
    <span
      title={s.title}
      className={`px-1.5 py-0.5 rounded border text-[10px] font-bold tracking-wide ${s.className}`}
    >
      {s.label}
    </span>
  );
}

interface Props {
  recommendation: Recommendation | null;
  recHistory: PastRecommendation[];
  /** Available counts per position, from the board — updated locally on every
   *  pick, so this is the responsive half of the Position Depth panel. */
  scarcity: Scarcity | null;
  /** Tiered critical/low/ok analysis for those counts, computed server-side
   *  against this league's real roster shape. Null before the first fetch or
   *  if it failed — the panel then shows counts with no tier claim. */
  scarcityAnalysis: ScarcityAnalysis | null;
  isLoading: boolean;
  isMyTurn: boolean;
  onFetch: () => void;
  onDraftRecommended: (playerId: number) => void;
  /** Board players by id — supplies team/bye, which the AI response omits. */
  playersById: Map<number, Player>;
  /** The pick currently on the clock, for the staleness check below. */
  currentPickNumber: number;
  /** Whether the recommendation is fetched automatically as your turn nears. */
  autoRecommend: boolean;
  onAutoRecommendChange: (on: boolean) => void;
  /** See the isSyncing docs in useDraft: sync-active means no manual picks. */
  isSyncing?: boolean;
  /** Every pick is in. There is no next pick to advise on, so Get pick and
   *  Auto are both disabled rather than spending a Claude call on a board
   *  nobody can act against. */
  draftComplete?: boolean;
  /** Whether the Big Board is currently collapsed to its rail ("focus mode"). */
  boardCollapsed?: boolean;
  /** Toggles the Big Board's collapsed state. Omitted (e.g. on mobile, where
   *  each section is already full-width) hides the header toggle button. */
  onToggleBoardCollapse?: () => void;
  /**
   * Opens the player detail drawer. This matters most here: the metrics and
   * draft capital behind a recommendation are exactly the numbers the model
   * reasoned over, and being able to open them is the difference between
   * taking the pick on faith and checking it.
   */
  onShowDetail?: (playerId: number) => void;
}

/** ADP + value/reach delta + team/bye, the numbers you compare picks on. */
function PlayerMeta({
  suggestion,
  pickNumber,
  player,
}: {
  suggestion: PickSuggestion;
  pickNumber: number;
  player?: Player;
}) {
  const value = adpValue(suggestion.adp, pickNumber);

  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-slate-400">
      {/* First, ahead of ADP: whether you can still have him later decides
          more picks than any other figure on this row. */}
      <SurvivalBadge survival={suggestion.survival} />
      {/* An injury designation outranks every number here: a player who
          cannot play is not a pick to weigh. The AI service excludes them,
          but this panel is also how you check its work. */}
      <InjuryBadge status={player?.injury_status} />
      <span className="tabular-nums">ADP {suggestion.adp}</span>
      <span className={`font-medium ${VALUE_STYLES[value.label]}`}>
        <span className="tabular-nums">{value.text}</span>
        {/* The number's sign is ambiguous on its own — negative is the *good*
            direction here — so the verdict is spelled out next to it. */}
        <span className="ml-1">· {value.word}</span>
      </span>
      {player?.team && <span>{player.team}</span>}
      {player?.bye != null && <span>Bye {player.bye}</span>}
    </div>
  );
}

/**
 * Draft button with an inline confirm step. A single click used to be enough
 * to record a pick — and on the alternatives it was a click anywhere on the
 * row — which is an unforgiving interaction for an action that is effectively
 * irreversible once Sleeper sync is involved.
 */
function DraftButton({
  playerName,
  onConfirm,
  variant = "primary",
}: {
  playerName: string;
  onConfirm: () => void;
  variant?: "primary" | "subtle";
}) {
  return (
    <ConfirmButton
      label="Draft"
      onConfirm={onConfirm}
      ariaLabel={`Draft ${playerName}`}
      confirmAriaLabel={`Confirm drafting ${playerName}`}
      className={`shrink-0 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
        variant === "primary"
          ? "bg-emerald-600 hover:bg-emerald-500 text-white"
          : "bg-slate-700 hover:bg-slate-600 text-slate-200"
      }`}
    />
  );
}

/**
 * Haiku/Sonnet switch. Self-contained: fetches the current choice on mount
 * and posts a switch directly through `api`, rather than routing through
 * useDraft — unlike Auto (a client-only localStorage preference), this is
 * server state (AIService.set_model, persisted on the active DraftSession),
 * so the panel talking to the endpoint directly is the more honest shape.
 *
 * Switches take effect on the NEXT "Get pick" — there is deliberately no
 * attempt to re-fetch the current recommendation on toggle, since that
 * would spend a second paid Claude call nobody asked for.
 */
function ModelToggle() {
  const [model, setModel] = useState<string | null>(null);
  const [isSwitching, setIsSwitching] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.getModel()
      .then((choice) => { if (!cancelled) setModel(choice.model); })
      .catch(() => {
        // Best-effort — an unreachable backend already surfaces loudly
        // elsewhere (Get pick will fail too). No point duplicating an
        // error state for a settings toggle nobody's clicked yet.
      });
    return () => { cancelled = true; };
  }, []);

  // "custom" covers a CLAUDE_MODEL env override that matches neither
  // option (see the backend's AIService.model_alias) — nothing to
  // highlight as active, but the switch still works from here.
  if (model === null) return null;

  const isSonnet = model === "sonnet";

  async function toggle() {
    const next = isSonnet ? "haiku" : "sonnet";
    setIsSwitching(true);
    try {
      const choice = await api.setModel(next);
      setModel(choice.model);
    } catch {
      // Leave the displayed value as whatever it last confirmed to be —
      // safer than optimistically showing a switch that didn't take.
    } finally {
      setIsSwitching(false);
    }
  }

  return (
    <button
      type="button"
      role="switch"
      aria-checked={isSonnet}
      onClick={toggle}
      disabled={isSwitching}
      title={
        isSonnet
          ? "Sonnet — richer analysis, slower and costs more. Click for Haiku."
          : "Haiku — fast and cheap. Click for Sonnet (richer analysis, higher cost)."
      }
      className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium border transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
        isSonnet
          ? "bg-violet-950 border-violet-800/60 text-violet-300"
          : "bg-slate-800 border-slate-600 text-slate-300"
      }`}
    >
      {model === "custom" ? "Model: custom" : isSonnet ? "Sonnet" : "Haiku"}
    </button>
  );
}

/**
 * One player row, shared by the Main card and every Best Available/Needs/
 * Depth card — the only real differences between them are size (`variant`)
 * and which section's own tag gets excluded from the overlap badges (see
 * TagBadges above).
 */
function SuggestionCard({
  suggestion,
  section,
  pickNumber,
  player,
  canDraft,
  onDraft,
  onShowDetail,
  variant = "subtle",
}: {
  suggestion: PickSuggestion;
  section: SectionTag;
  pickNumber: number;
  player?: Player;
  canDraft: boolean;
  onDraft: () => void;
  onShowDetail?: (playerId: number) => void;
  variant?: "primary" | "subtle";
}) {
  const isPrimary = variant === "primary";
  const nameClass = isPrimary
    ? "text-lg font-bold text-white break-words"
    : "text-slate-100 text-sm font-semibold break-words";

  return (
    <div className={isPrimary ? "bg-slate-800 rounded-xl p-4" : "bg-slate-800/60 rounded-xl p-3"}>
      <div className="flex items-start justify-between gap-2 mb-1">
        <div className="min-w-0">
          {onShowDetail ? (
            <button
              type="button"
              onClick={() => onShowDetail(suggestion.player_id)}
              aria-label={`Details for ${suggestion.player_name}`}
              className={`${nameClass} text-left hover:text-emerald-300 hover:underline decoration-dotted underline-offset-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded`}
            >
              {suggestion.player_name}
            </button>
          ) : (
            <span className={nameClass}>{suggestion.player_name}</span>
          )}
          <span
            className={`${isPrimary ? "ml-2 text-sm" : "ml-1.5 text-xs"} font-semibold ${
              POS_COLORS[suggestion.position] ?? "text-slate-300"
            }`}
          >
            {suggestion.position}
          </span>
        </div>
        {canDraft && (
          <DraftButton
            playerName={suggestion.player_name}
            onConfirm={onDraft}
            variant={variant}
          />
        )}
      </div>

      <PlayerMeta suggestion={suggestion} pickNumber={pickNumber} player={player} />

      {suggestion.reasoning && (
        <p className={isPrimary ? "text-slate-300 text-sm leading-relaxed mt-2" : "text-xs text-slate-300 leading-relaxed mt-1.5"}>
          {suggestion.reasoning}
        </p>
      )}

      <TagBadges tags={suggestion.tags} currentSection={section} />
    </div>
  );
}

export default function AIPanel({
  recommendation,
  recHistory,
  scarcity,
  scarcityAnalysis,
  isLoading,
  isMyTurn,
  onFetch,
  onDraftRecommended,
  playersById,
  currentPickNumber,
  autoRecommend,
  onAutoRecommendChange,
  isSyncing = false,
  draftComplete = false,
  boardCollapsed = false,
  onToggleBoardCollapse,
  onShowDetail,
}: Props) {
  const [showHistory, setShowHistory] = useState(false);

  // Alerts arrive as a list sorted critical-first (the backend orders them for
  // callers that render a feed). This panel is a fixed-order grid instead —
  // the same argument as DraftRoom's POS_ORDER: a tile that moves around
  // between refreshes can't be found at a glance mid-draft — so it indexes by
  // position rather than iterating the list.
  const alertsByPosition = useMemo(
    () => new Map((scarcityAnalysis?.alerts ?? []).map((a) => [a.position, a])),
    [scarcityAnalysis]
  );

  // Tracks this panel's actual rendered width so Best Available/Needs can
  // switch to a 2-column grid once there's genuinely room for it — see
  // SECTION_GRID_MIN_WIDTH above for why this is measured rather than
  // derived from a viewport breakpoint.
  const panelRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);

  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      setIsWide(width >= SECTION_GRID_MIN_WIDTH);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Defence in depth. useDraft already discards responses that arrive after
  // the draft has moved on, and clears the recommendation on every pick — but
  // this app has twice shipped a bug where drafted players were presented as
  // available, so the panel refuses to render advice as current when it
  // demonstrably isn't, rather than trusting the layer above to be perfect.
  const isStale = recommendation != null && recommendation.pick_number !== currentPickNumber;

  // Drafting off stale advice is exactly the mistake the staleness check
  // exists to prevent.
  const canDraft = !isSyncing && !isStale && !draftComplete;

  // The fallback alert (whatever the specific reason — see FALLBACK_REASON_RE
  // below) is already communicated by the badge on the card itself; repeating
  // it as a full alert banner is noise. Matched by suffix rather than the old
  // "AI service unavailable" substring, now that the backend names the
  // SPECIFIC cause (missing key, API error, unparseable response, ...)
  // instead of one generic sentence for all of them.
  const alerts = recommendation
    ? recommendation.alerts.filter((a) => !a.endsWith("showing best available by ADP only."))
    : [];

  const isFallback = recommendation?.model.includes("fallback") ?? false;

  // The fallback alert's reason prefix, e.g. "No ANTHROPIC_API_KEY configured"
  // or "Claude API error" — see _fallback's `reason` param in ai_service.py.
  // Falls back to a generic label on the off chance alerts is empty (should
  // not happen alongside isFallback, but the badge shouldn't blank out if it
  // ever does).
  const fallbackReason =
    recommendation?.alerts
      .find((a) => a.endsWith("showing best available by ADP only."))
      ?.replace(/ — showing best available by ADP only\.$/, "") ?? "AI service unavailable";
  const confidence = CONFIDENCE_STYLES[recommendation?.confidence ?? "medium"];

  return (
    <div ref={panelRef} className="flex flex-col gap-3 h-full overflow-y-auto">

      {/* Alerts.
          These sit above the recommendation now. They were rendered below the
          alternatives, which in a 300px column put scarcity warnings and tier
          drop-off flags off the bottom of the panel — exactly the information
          that should change what you do before you read the pick. */}
      {alerts.length > 0 && (
        <div className="space-y-1.5 shrink-0" role="status">
          {alerts.map((alert, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs text-amber-200 bg-amber-900/25 border border-amber-800/50 rounded-lg px-3 py-2"
            >
              <AlertTriangle size={12} className="shrink-0 mt-0.5" aria-hidden="true" />
              {alert}
            </div>
          ))}
        </div>
      )}

      {/* Recommendation card */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 shrink-0">
        <div className="flex items-center justify-between mb-3 gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Lightbulb size={14} className="text-amber-400 shrink-0" aria-hidden="true" />
            <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider truncate">
              AI Recommendation
            </h2>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {/* Focus mode. Collapses the Big Board rail so this panel gets its
                grid column — the same toggle lives on the board itself, but
                it's not obviously discoverable from here without a mirror. */}
            {onToggleBoardCollapse && (
              <button
                type="button"
                onClick={onToggleBoardCollapse}
                title={
                  boardCollapsed
                    ? "Show Big Board (b)"
                    : "Focus mode — collapse Big Board for more room (b)"
                }
                aria-label={boardCollapsed ? "Show Big Board" : "Collapse Big Board"}
                aria-pressed={boardCollapsed}
                className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
                  boardCollapsed
                    ? "bg-emerald-950 border-emerald-800/60 text-emerald-300"
                    : "bg-slate-800 border-slate-600 text-slate-300"
                }`}
              >
                {boardCollapsed ? (
                  <PanelLeftOpen size={11} aria-hidden="true" />
                ) : (
                  <PanelLeftClose size={11} aria-hidden="true" />
                )}
                Focus
              </button>
            )}

            <ModelToggle />

            {/* Auto-recommend toggle. Each automatic fetch is a paid Claude
                call, so this is a real preference rather than something to
                decide on the user's behalf. */}
            <button
              type="button"
              role="switch"
              aria-checked={autoRecommend}
              onClick={() => onAutoRecommendChange(!autoRecommend)}
              disabled={draftComplete}
              title={
                autoRecommend
                  ? "Auto-recommend on — fetches as your turn comes up"
                  : "Auto-recommend off — fetch manually with Get pick or g"
              }
              className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium border transition-colors disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
                autoRecommend
                  ? "bg-emerald-950 border-emerald-800/60 text-emerald-300"
                  : "bg-slate-800 border-slate-600 text-slate-300"
              }`}
            >
              <Zap size={11} aria-hidden="true" />
              Auto
            </button>

            <button
              type="button"
              onClick={onFetch}
              disabled={isLoading || draftComplete}
              title={draftComplete ? "The draft is over — there's no pick to advise on." : undefined}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            >
              <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} aria-hidden="true" />
              {isLoading ? "Thinking…" : "Get pick"}
            </button>
          </div>
        </div>

        {draftComplete && !recommendation ? (
          <div className="text-center py-6 text-slate-400 text-sm">
            <p>Your draft is finished — nothing left to recommend.</p>
            <p className="mt-1 text-xs">
              Earlier advice is still below, if you kept any.
            </p>
          </div>
        ) : !recommendation ? (
          <div className="text-center py-6 text-slate-400 text-sm">
            <p>
              Click <span className="text-slate-200 font-medium">&quot;Get pick&quot;</span> for a
              recommendation.
            </p>
            {isMyTurn && (
              <p className="text-emerald-400 mt-2 text-xs font-medium">You&apos;re on the clock!</p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {/* Stale advice — the board moved on after this was generated, so
                the players named may already be gone. Drafting is suppressed
                until it's refreshed. */}
            {isStale && (
              <div
                role="status"
                className="flex items-start gap-2 text-xs text-amber-200 bg-amber-900/25 border border-amber-800/50 rounded-lg px-3 py-2"
              >
                <AlertTriangle size={12} className="shrink-0 mt-0.5" aria-hidden="true" />
                <span>
                  This was for pick #{recommendation.pick_number}; the board is now on #
                  {currentPickNumber}. Some of these players may be gone — hit{" "}
                  <span className="font-semibold">Get pick</span> to refresh.
                </span>
              </div>
            )}

            {/* Strategy — the plan, above the individual name it argues for */}
            {recommendation.strategy && (
              <div className="flex items-start gap-2 text-xs text-slate-300 bg-slate-800/60 rounded-lg px-3 py-2">
                <Compass size={12} className="shrink-0 mt-0.5 text-slate-400" aria-hidden="true" />
                <p className="leading-relaxed">{recommendation.strategy}</p>
              </div>
            )}

            {/* Main — the model's synthesized pick, the only one backed by
                real reasoning (tiers, opportunity cost, VOR, news). */}
            <div>
              <SuggestionCard
                suggestion={recommendation.main}
                section="main"
                pickNumber={recommendation.pick_number}
                player={playersById.get(recommendation.main.player_id)}
                canDraft={canDraft}
                onDraft={() => onDraftRecommended(recommendation.main.player_id)}
                onShowDetail={onShowDetail}
                variant="primary"
              />
              <div className="flex items-center justify-between gap-2 mt-2 px-1">
                <span
                  className={`px-2 py-0.5 rounded-full border text-xs font-medium ${confidence.className}`}
                >
                  {confidence.label}
                </span>
                <span className="text-xs text-slate-400 truncate" title={isFallback ? fallbackReason : undefined}>
                  {isFallback ? `⚠ Fallback — ${fallbackReason}` : recommendation.model}
                </span>
              </div>
            </div>

            {/* While streaming, the pick arrives well before the rest. Say so
                explicitly: an empty Best Available/Needs section would
                otherwise read as "there's nothing else", which is a
                different and much stronger claim than "they haven't
                finished generating". */}
            {recommendation.isPartial && (
              <p className="flex items-center gap-2 text-xs text-slate-400">
                <RefreshCw size={12} className="animate-spin" aria-hidden="true" />
                Loading best available and needs…
              </p>
            )}

            {/* Best Available — pure ADP value, roster needs ignored. Never
                model output; see the Recommendation type docstring. */}
            {recommendation.best_available.length > 0 && (
              <div>
                <p className="flex items-center gap-1.5 text-xs text-slate-400 uppercase font-semibold mb-2">
                  <TrendingUp size={11} aria-hidden="true" />
                  Best Available
                </p>
                <div className={isWide ? "grid grid-cols-2 gap-2 items-start" : "space-y-2"}>
                  {recommendation.best_available.map((p) => (
                    <SuggestionCard
                      key={p.player_id}
                      suggestion={p}
                      section="best_available"
                      pickNumber={recommendation.pick_number}
                      player={playersById.get(p.player_id)}
                      canDraft={canDraft}
                      onDraft={() => onDraftRecommended(p.player_id)}
                      onShowDetail={onShowDetail}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Needs — fills your highest-priority open starting slot.
                Empty once the skill lineup is filled, EXCEPT it can pick
                back up recommending DST/K once those become the last open
                slots — see the backend's _needs docstring. Can render
                alongside Depth below during that phase. */}
            {recommendation.needs.length > 0 && (
              <div>
                <p className="flex items-center gap-1.5 text-xs text-slate-400 uppercase font-semibold mb-2">
                  <Target size={11} aria-hidden="true" />
                  Fills a Need
                </p>
                <div className={isWide ? "grid grid-cols-2 gap-2 items-start" : "space-y-2"}>
                  {recommendation.needs.map((p) => (
                    <SuggestionCard
                      key={p.player_id}
                      suggestion={p}
                      section="needs"
                      pickNumber={recommendation.pick_number}
                      player={playersById.get(p.player_id)}
                      canDraft={canDraft}
                      onDraft={() => onDraftRecommended(p.player_id)}
                      onShowDetail={onShowDetail}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Depth — a QB stash and a TE stash (up to 2), once the skill
                starting lineup is filled. Can render alongside Needs above
                during the DST/K phase — see the backend's _depth_pick. */}
            {recommendation.depth.length > 0 && (
              <div>
                <p className="flex items-center gap-1.5 text-xs text-slate-400 uppercase font-semibold mb-2">
                  <Layers size={11} aria-hidden="true" />
                  Depth Stash
                </p>
                <div className={isWide ? "grid grid-cols-2 gap-2 items-start" : "space-y-2"}>
                  {recommendation.depth.map((p) => (
                    <SuggestionCard
                      key={p.player_id}
                      suggestion={p}
                      section="depth"
                      pickNumber={recommendation.pick_number}
                      player={playersById.get(p.player_id)}
                      canDraft={canDraft}
                      onDraft={() => onDraftRecommended(p.player_id)}
                      onShowDetail={onShowDetail}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Opportunity — an RB/WR/TE whose role looks bigger than his
                ADP/tier reflects (0-1). Model-generated, unlike every other
                section here except Main — see the backend's
                _opportunity_pick_from. Empty whenever the AI is unavailable
                or names someone invalid, not gated by any other section. */}
            {recommendation.opportunity.length > 0 && (
              <div>
                <p className="flex items-center gap-1.5 text-xs text-slate-400 uppercase font-semibold mb-2">
                  <Rocket size={11} aria-hidden="true" />
                  Opportunity
                </p>
                <div className={isWide ? "grid grid-cols-2 gap-2 items-start" : "space-y-2"}>
                  {recommendation.opportunity.map((p) => (
                    <SuggestionCard
                      key={p.player_id}
                      suggestion={p}
                      section="opportunity"
                      pickNumber={recommendation.pick_number}
                      player={playersById.get(p.player_id)}
                      canDraft={canDraft}
                      onDraft={() => onDraftRecommended(p.player_id)}
                      onShowDetail={onShowDetail}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Past recommendations */}
      {recHistory.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-700 shrink-0">
          <button
            type="button"
            onClick={() => setShowHistory((s) => !s)}
            aria-expanded={showHistory}
            className="flex items-center gap-2 w-full px-4 py-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded-xl"
          >
            <History size={14} className="text-slate-500" aria-hidden="true" />
            <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Earlier advice
            </h2>
            <span className="ml-auto text-xs text-slate-400">{recHistory.length}</span>
          </button>

          {showHistory && (
            <div className="px-4 pb-4 space-y-2">
              {recHistory.map((past) => (
                <div key={past.pickNumber} className="text-xs bg-slate-800/60 rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 tabular-nums">#{past.pickNumber}</span>
                    <span className="text-slate-100 font-medium">
                      {past.recommendation.main.player_name}
                    </span>
                    <span
                      className={
                        POS_COLORS[past.recommendation.main.position] ?? "text-slate-300"
                      }
                    >
                      {past.recommendation.main.position}
                    </span>
                  </div>
                  <p className="text-slate-400 leading-relaxed mt-1">
                    {past.recommendation.main.reasoning}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Positional scarcity */}
      {scarcity && (
        <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 shrink-0">
          <div className="flex items-center gap-2 mb-3">
            <Info size={14} className="text-slate-500" aria-hidden="true" />
            <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Position Depth
            </h2>
            {scarcityAnalysis === null && (
              <span
                className="ml-auto text-[10px] text-slate-500"
                title="Scarcity tiers haven't loaded — showing raw counts only."
              >
                counts only
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2">
            {(["QB", "RB", "WR", "TE", "DST", "K"] as const).map((pos) => {
              // Count comes from the board (decremented locally on every pick,
              // so it's instant); the tier comes from the analysis, which is
              // fetched in the same round as the board — see refreshBoard.
              const count = scarcity[pos];
              const alert = alertsByPosition.get(pos);
              const style = TIER_STYLES[alert?.tier ?? "unknown"];

              return (
                <div
                  key={pos}
                  title={alert?.message}
                  className={`rounded-lg p-2.5 text-center border ${style.tile}`}
                >
                  <div className={`text-xs font-bold ${POS_COLORS[pos]}`}>{pos}</div>
                  <div className={`text-lg font-bold tabular-nums ${style.count}`}>
                    {count}
                  </div>
                  {style.label && (
                    <div className={`text-xs ${style.count}`}>{style.label}</div>
                  )}
                </div>
              );
            })}
          </div>
          <p className="mt-2.5 text-[10px] text-slate-500 leading-relaxed">
            Measured against what your league actually starts — critical means
            fewer left than half the league needs.
          </p>
        </div>
      )}
    </div>
  );
}
