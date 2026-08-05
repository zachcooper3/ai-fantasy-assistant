"use client";
/**
 * AIPanel — AI recommendation + positional scarcity alerts.
 *
 * Shows the Claude-generated pick recommendation with its reasoning, the
 * alternatives it weighed (each with the trade-off against the main pick),
 * scarcity alerts, and recent past recommendations.
 *
 * Alternatives are rendered as peers of the main recommendation rather than a
 * truncated footnote list: on the clock you are choosing between them, not
 * reading a ranked answer, and the previous one-line clamp cut the model's
 * justification off after roughly six words.
 *
 * Falls back gracefully when no API key is configured (the backend's ADP
 * fallback reports confidence "low" and carries no strategy or trade-offs).
 */

import { useState } from "react";
import {
  Lightbulb,
  RefreshCw,
  AlertTriangle,
  Info,
  History,
  Compass,
  Zap,
} from "lucide-react";

import { Confidence, Player, PickSuggestion, Recommendation, Scarcity, Survival } from "@/lib/api";
import { PastRecommendation } from "@/hooks/useDraft";
import { adpValue } from "@/lib/draft";
import ConfirmButton from "@/components/ConfirmButton";

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
  scarcity: Scarcity | null;
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

export default function AIPanel({
  recommendation,
  recHistory,
  scarcity,
  isLoading,
  isMyTurn,
  onFetch,
  onDraftRecommended,
  playersById,
  currentPickNumber,
  autoRecommend,
  onAutoRecommendChange,
  isSyncing = false,
}: Props) {
  const [showHistory, setShowHistory] = useState(false);

  // Defence in depth. useDraft already discards responses that arrive after
  // the draft has moved on, and clears the recommendation on every pick — but
  // this app has twice shipped a bug where drafted players were presented as
  // available, so the panel refuses to render advice as current when it
  // demonstrably isn't, rather than trusting the layer above to be perfect.
  const isStale = recommendation != null && recommendation.pick_number !== currentPickNumber;

  // Drafting off stale advice is exactly the mistake the staleness check
  // exists to prevent.
  const canDraft = !isSyncing && !isStale;

  // "AI service unavailable" is already communicated by the fallback notice on
  // the card itself; repeating it as an alert is noise.
  const alerts = recommendation
    ? recommendation.alerts.filter((a) => !a.includes("AI service unavailable"))
    : [];

  const isFallback = recommendation?.model.includes("fallback") ?? false;
  const confidence = CONFIDENCE_STYLES[recommendation?.confidence ?? "medium"];

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">

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
            {/* Auto-recommend toggle. Each automatic fetch is a paid Claude
                call, so this is a real preference rather than something to
                decide on the user's behalf. */}
            <button
              type="button"
              role="switch"
              aria-checked={autoRecommend}
              onClick={() => onAutoRecommendChange(!autoRecommend)}
              title={
                autoRecommend
                  ? "Auto-recommend on — fetches as your turn comes up"
                  : "Auto-recommend off — fetch manually with Get pick or g"
              }
              className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
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
              disabled={isLoading}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            >
              <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} aria-hidden="true" />
              {isLoading ? "Thinking…" : "Get pick"}
            </button>
          </div>
        </div>

        {!recommendation ? (
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

            {/* Main recommendation */}
            <div className="bg-slate-800 rounded-xl p-4">
              <div className="flex items-start justify-between gap-2 mb-1">
                <div className="min-w-0">
                  <span className="text-lg font-bold text-white break-words">
                    {recommendation.recommendation.player_name}
                  </span>
                  <span
                    className={`ml-2 text-sm font-semibold ${
                      POS_COLORS[recommendation.recommendation.position] ?? "text-slate-300"
                    }`}
                  >
                    {recommendation.recommendation.position}
                  </span>
                </div>
                {canDraft && (
                  <DraftButton
                    playerName={recommendation.recommendation.player_name}
                    onConfirm={() =>
                      onDraftRecommended(recommendation.recommendation.player_id)
                    }
                  />
                )}
              </div>

              <PlayerMeta
                suggestion={recommendation.recommendation}
                pickNumber={recommendation.pick_number}
                player={playersById.get(recommendation.recommendation.player_id)}
              />

              <p className="text-slate-300 text-sm leading-relaxed mt-2">
                {recommendation.recommendation.reasoning}
              </p>

              <div className="flex items-center justify-between gap-2 mt-3">
                <span
                  className={`px-2 py-0.5 rounded-full border text-xs font-medium ${confidence.className}`}
                >
                  {confidence.label}
                </span>
                <span className="text-xs text-slate-400 truncate">
                  {isFallback ? "⚠ Fallback — no ANTHROPIC_API_KEY" : recommendation.model}
                </span>
              </div>
            </div>

            {/* Alternatives — peers of the recommendation, with the trade-off
                against it spelled out. */}
            {recommendation.alternatives.length > 0 && (
              <div>
                <p className="text-xs text-slate-400 uppercase font-semibold mb-2">
                  Also considered
                </p>
                <div className="space-y-2">
                  {recommendation.alternatives.map((alt) => (
                    <div key={alt.player_id} className="bg-slate-800/60 rounded-xl p-3">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <div className="min-w-0">
                          <span className="text-slate-100 text-sm font-semibold break-words">
                            {alt.player_name}
                          </span>
                          <span
                            className={`ml-1.5 text-xs font-semibold ${
                              POS_COLORS[alt.position] ?? "text-slate-300"
                            }`}
                          >
                            {alt.position}
                          </span>
                        </div>
                        {canDraft && (
                          <DraftButton
                            playerName={alt.player_name}
                            onConfirm={() => onDraftRecommended(alt.player_id)}
                            variant="subtle"
                          />
                        )}
                      </div>

                      <PlayerMeta
                        suggestion={alt}
                        pickNumber={recommendation.pick_number}
                        player={playersById.get(alt.player_id)}
                      />

                      {alt.reasoning && (
                        <p className="text-xs text-slate-300 leading-relaxed mt-1.5">
                          {alt.reasoning}
                        </p>
                      )}
                      {alt.tradeoff && (
                        <p className="text-xs text-slate-400 leading-relaxed mt-1.5 pl-2 border-l-2 border-slate-600">
                          <span className="text-slate-400 font-medium">vs. pick: </span>
                          {alt.tradeoff}
                        </p>
                      )}
                    </div>
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
                      {past.recommendation.recommendation.player_name}
                    </span>
                    <span
                      className={
                        POS_COLORS[past.recommendation.recommendation.position] ?? "text-slate-300"
                      }
                    >
                      {past.recommendation.recommendation.position}
                    </span>
                  </div>
                  <p className="text-slate-400 leading-relaxed mt-1">
                    {past.recommendation.recommendation.reasoning}
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
          </div>
          <div className="grid grid-cols-3 gap-2">
            {(["QB", "RB", "WR", "TE", "DST", "K"] as const).map((pos) => {
              const count = scarcity[pos];
              const low = count < 20;
              const critical = count < 8;
              return (
                <div
                  key={pos}
                  className={`rounded-lg p-2.5 text-center border ${
                    critical
                      ? "bg-red-900/30 border-red-800/50"
                      : low
                      ? "bg-amber-900/20 border-amber-800/30"
                      : "bg-slate-800 border-slate-700"
                  }`}
                >
                  <div className={`text-xs font-bold ${POS_COLORS[pos]}`}>{pos}</div>
                  <div
                    className={`text-lg font-bold tabular-nums ${
                      critical ? "text-red-400" : low ? "text-amber-400" : "text-slate-200"
                    }`}
                  >
                    {count}
                  </div>
                  {critical && <div className="text-xs text-red-400">Critical</div>}
                  {low && !critical && <div className="text-xs text-amber-400">Low</div>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
