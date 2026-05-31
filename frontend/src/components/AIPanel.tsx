"use client";
/**
 * AIPanel — AI recommendation + positional scarcity alerts.
 *
 * Shows the Claude-generated pick recommendation with reasoning,
 * alternative picks, and any scarcity/tier alerts.
 * Falls back gracefully when no API key is configured.
 */

import { Lightbulb, RefreshCw, ChevronRight, AlertTriangle, Info } from "lucide-react";
import { Recommendation, Scarcity } from "@/lib/api";

const POS_COLORS: Record<string, string> = {
  QB:  "text-red-400",
  RB:  "text-emerald-400",
  WR:  "text-blue-400",
  TE:  "text-amber-400",
  DST: "text-purple-400",
  K:   "text-slate-400",
};

interface Props {
  recommendation: Recommendation | null;
  scarcity: Scarcity | null;
  isLoading: boolean;
  isMyTurn: boolean;
  onFetch: () => void;
  onDraftRecommended: (playerId: number) => void;
}

export default function AIPanel({
  recommendation,
  scarcity,
  isLoading,
  isMyTurn,
  onFetch,
  onDraftRecommended,
}: Props) {

  return (
    <div className="flex flex-col gap-4 h-full overflow-y-auto">

      {/* Recommendation card */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Lightbulb size={14} className="text-amber-400" />
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              AI Recommendation
            </h2>
          </div>
          <button
            onClick={onFetch}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition-colors disabled:opacity-50"
          >
            <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} />
            {isLoading ? "Thinking…" : "Get pick"}
          </button>
        </div>

        {!recommendation ? (
          <div className="text-center py-6 text-slate-600 text-sm">
            <p>Click <span className="text-slate-400 font-medium">"Get pick"</span> to get a recommendation.</p>
            {isMyTurn && (
              <p className="text-emerald-500 mt-2 text-xs font-medium">You&apos;re on the clock!</p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Main recommendation */}
            <div className="bg-slate-800 rounded-xl p-4">
              <div className="flex items-start justify-between gap-2 mb-2">
                <div>
                  <span className="text-lg font-bold text-white">
                    {recommendation.recommendation.player_name}
                  </span>
                  <span className={`ml-2 text-sm font-semibold ${POS_COLORS[recommendation.recommendation.position] ?? "text-slate-400"}`}>
                    {recommendation.recommendation.position}
                  </span>
                  <span className="ml-2 text-xs text-slate-500">
                    ADP {recommendation.recommendation.adp}
                  </span>
                </div>
                <button
                  onClick={() => onDraftRecommended(recommendation.recommendation.player_id)}
                  className="shrink-0 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition-colors"
                >
                  Draft
                </button>
              </div>
              <p className="text-slate-400 text-sm leading-relaxed">
                {recommendation.recommendation.reasoning}
              </p>
              <p className="text-xs text-slate-600 mt-2">
                {recommendation.model.includes("fallback")
                  ? "⚠ Fallback mode — add ANTHROPIC_API_KEY for AI reasoning"
                  : `Model: ${recommendation.model}`}
              </p>
            </div>

            {/* Alternatives */}
            {recommendation.alternatives.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase font-semibold mb-2">Alternatives</p>
                <div className="space-y-2">
                  {recommendation.alternatives.map((alt) => (
                    <div
                      key={alt.player_id}
                      className="flex items-center gap-3 bg-slate-800/60 rounded-lg px-3 py-2 hover:bg-slate-800 transition-colors cursor-pointer group"
                      onClick={() => onDraftRecommended(alt.player_id)}
                    >
                      <ChevronRight size={12} className="text-slate-600 group-hover:text-slate-400 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <span className="text-slate-200 text-sm font-medium">{alt.player_name}</span>
                        <span className={`ml-1.5 text-xs ${POS_COLORS[alt.position] ?? "text-slate-400"}`}>
                          {alt.position}
                        </span>
                        {alt.reasoning && (
                          <p className="text-xs text-slate-500 truncate mt-0.5">{alt.reasoning}</p>
                        )}
                      </div>
                      <span className="text-xs text-slate-500 shrink-0">{alt.adp}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Alerts */}
            {recommendation.alerts.filter(a => !a.includes("AI service unavailable")).length > 0 && (
              <div className="space-y-1.5">
                {recommendation.alerts
                  .filter(a => !a.includes("AI service unavailable"))
                  .map((alert, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs text-amber-300 bg-amber-900/20 border border-amber-900/40 rounded-lg px-3 py-2">
                      <AlertTriangle size={12} className="shrink-0 mt-0.5" />
                      {alert}
                    </div>
                  ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Positional scarcity */}
      {scarcity && (
        <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 shrink-0">
          <div className="flex items-center gap-2 mb-3">
            <Info size={14} className="text-slate-500" />
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
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
                  <div className={`text-lg font-bold tabular-nums ${critical ? "text-red-400" : low ? "text-amber-400" : "text-slate-200"}`}>
                    {count}
                  </div>
                  {critical && <div className="text-xs text-red-500">Critical</div>}
                  {low && !critical && <div className="text-xs text-amber-500">Low</div>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
