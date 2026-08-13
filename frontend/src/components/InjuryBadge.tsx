"use client";
/**
 * InjuryBadge — Sleeper's injury designation for a player.
 *
 * This data has been on the wire since Player.injury_status was added
 * (PlayerResponse carries it on every board refresh) but was never rendered
 * anywhere: the frontend's Player interface didn't even declare the field.
 * The AI service treats "cannot play" as a hard exclusion rather than a risk
 * to weigh — a designation the human is expected to see too, since the Big
 * Board is also how you check the model's work.
 *
 * Two severities, not five. The distinction that changes a draft-day decision
 * is "this player is unavailable" vs. "this player is dinged", so PUP/IR/
 * Suspended/DNR read as a hard red badge and Questionable/Doubtful as a soft
 * amber one. Unrecognised codes from a future Sleeper change fall through to
 * the soft style with their raw text, rather than being swallowed.
 *
 * Observed live values in this database: Questionable, PUP, IR, NA, DNR.
 */

/** Designations meaning the player is not expected to play at all. */
const OUT_CODES = new Set(["IR", "PUP", "OUT", "SUSPENDED", "NFI", "DNR"]);

/**
 * Sleeper reports "NA" for a player it has no status for, which is
 * indistinguishable from healthy and would otherwise render a badge on
 * someone who has nothing wrong with them.
 */
const NON_STATUSES = new Set(["", "NA", "ACTIVE", "HEALTHY"]);

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
  IR: "On injured reserve — will not play.",
  PUP: "Physically unable to perform — will not play to start the season.",
  DNR: "Did not report.",
  NFI: "Non-football injury list — will not play to start the season.",
  SUSPENDED: "Suspended — will not play.",
  OUT: "Ruled out.",
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
  if (!status) return null;

  const code = status.trim().toUpperCase();
  if (NON_STATUSES.has(code)) return null;

  const isOut = OUT_CODES.has(code);

  return (
    <span
      title={TITLES[code] ?? `Injury designation: ${status}`}
      className={`shrink-0 px-1 py-px rounded border text-[10px] font-bold leading-tight tracking-wide ${
        isOut
          ? "bg-red-950 text-red-300 border-red-800/70"
          : "bg-amber-950/60 text-amber-300 border-amber-800/60"
      } ${className}`}
    >
      {LABELS[code] ?? code}
    </span>
  );
}
