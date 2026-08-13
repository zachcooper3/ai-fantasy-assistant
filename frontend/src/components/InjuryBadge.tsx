"use client";
/**
 * InjuryBadge — Sleeper's injury designation for a player.
 *
 * This data had been on the wire since Player.injury_status was added but
 * was never rendered: the frontend's Player interface didn't even declare
 * the field.
 *
 * Two severities, because the distinction that changes a draft-day decision
 * is "cannot play at all" vs. "dinged". The split is NOT a local judgement
 * call — it's `isUndraftable`, which mirrors the backend set that decides
 * who the AI is allowed to recommend and who counts toward positional
 * scarcity. So a red badge means something precise: the server has excluded
 * this player from its own supply math, and he's on your board only because
 * the board deliberately opts into showing him.
 *
 * Codes outside both sets (a future Sleeper value, "DNR", "NFI") render
 * amber with their raw text rather than being swallowed — unknown is not
 * the same as fine, but it also isn't a claim this component gets to make.
 */

import { injuryCode, isUndraftable } from "@/lib/injury";

const LABELS: Record<string, string> = {
  IR: "IR",
  PUP: "PUP",
  DNR: "DNR",
  NFI: "NFI",
  SUSPENDED: "SUSP",
  OUT: "OUT",
  DOUBTFUL: "DOUBT",
  QUESTIONABLE: "QUEST",
};

const TITLES: Record<string, string> = {
  IR: "On injured reserve — cannot play. Excluded from AI suggestions and from positional depth counts.",
  PUP: "Physically unable to perform — cannot play to start the season. Excluded from AI suggestions and from positional depth counts.",
  SUSPENDED: "Suspended — cannot play. Excluded from AI suggestions and from positional depth counts.",
  OUT: "Ruled out. Excluded from AI suggestions and from positional depth counts.",
  DNR: "Did not report.",
  NFI: "Non-football injury list — unlikely to play to start the season.",
  DOUBTFUL: "Doubtful — unlikely to play.",
  QUESTIONABLE: "Questionable — game-time decision.",
};

export default function InjuryBadge({
  status,
  className = "",
}: {
  status?: string | null;
  className?: string;
}) {
  const code = injuryCode(status);
  if (!code) return null;

  const cannotPlay = isUndraftable(status);

  return (
    <span
      title={TITLES[code] ?? `Injury designation: ${status}`}
      className={`shrink-0 px-1 py-px rounded border text-[10px] font-bold leading-tight tracking-wide whitespace-nowrap ${
        cannotPlay
          ? "bg-red-950 text-red-300 border-red-800/70"
          : "bg-amber-950/60 text-amber-300 border-amber-800/60"
      } ${className}`}
    >
      {LABELS[code] ?? code}
    </span>
  );
}
