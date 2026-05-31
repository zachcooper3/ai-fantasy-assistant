"use client";
/**
 * SetupModal — shown on first load or when no active session exists.
 * Collects league_size and my_draft_position before starting the draft.
 */

import { useState } from "react";

interface Props {
  onStart: (config: {
    league_size: number;
    my_draft_position: number;
    total_rounds: number;
  }) => void;
}

export default function SetupModal({ onStart }: Props) {
  const [leagueSize, setLeagueSize] = useState(12);
  const [draftPos, setDraftPos] = useState(1);
  const [rounds, setRounds] = useState(15);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-700 p-8 shadow-2xl">
        <h1 className="text-2xl font-bold text-slate-100 mb-1">
          Fantasy Draft Assistant
        </h1>
        <p className="text-slate-400 text-sm mb-8">
          Configure your draft settings to get started.
        </p>

        <div className="space-y-5">
          {/* League size */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              League size
            </label>
            <div className="flex gap-2">
              {[8, 10, 12, 14].map((n) => (
                <button
                  key={n}
                  onClick={() => setLeagueSize(n)}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    leagueSize === n
                      ? "bg-emerald-500 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Draft position */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              My draft position
            </label>
            <div className="grid grid-cols-6 gap-2">
              {Array.from({ length: leagueSize }, (_, i) => i + 1).map((pos) => (
                <button
                  key={pos}
                  onClick={() => setDraftPos(pos)}
                  className={`py-2 rounded-lg text-sm font-semibold transition-colors ${
                    draftPos === pos
                      ? "bg-blue-500 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {pos}
                </button>
              ))}
            </div>
          </div>

          {/* Rounds */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Rounds
            </label>
            <div className="flex gap-2">
              {[14, 15, 16].map((n) => (
                <button
                  key={n}
                  onClick={() => setRounds(n)}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    rounds === n
                      ? "bg-emerald-500 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Summary + start */}
        <div className="mt-8 p-4 rounded-xl bg-slate-800 text-slate-300 text-sm mb-6">
          {leagueSize}-team PPR · Slot {draftPos} of {leagueSize} · {rounds} rounds
        </div>

        <button
          onClick={() =>
            onStart({ league_size: leagueSize, my_draft_position: draftPos, total_rounds: rounds })
          }
          className="w-full py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-bold text-base transition-colors"
        >
          Start Draft
        </button>
      </div>
    </div>
  );
}
