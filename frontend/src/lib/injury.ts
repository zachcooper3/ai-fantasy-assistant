/**
 * One client-side definition of what an injury designation means.
 *
 * Three places need to agree on this and previously two of them didn't
 * exist: the badge's severity styling, the Big Board's row dimming, and
 * useDraft's decision about whether a drafted player should decrement a
 * scarcity count. That last one is the reason this is a shared module
 * rather than a detail of the badge — an IR player is deliberately absent
 * from the backend's scarcity counts, so decrementing one when he's drafted
 * would walk the count down past the truth, permanently, with no way to
 * notice.
 */

/**
 * Mirrors `UNDRAFTABLE_STATUSES` in backend/db/player_repo.py — the set the
 * server uses to exclude a player from the AI's candidate pool and from
 * every scarcity count.
 *
 * KEEP IN SYNC. There is no shared schema between front and back end (see
 * SetupModal's SLOT_BOUNDS for the same caveat). If the backend set changes
 * and this doesn't, the scarcity guard in useDraft silently starts counting
 * the wrong players. `injury.test.ts` pins the exact membership so a change
 * here is at least deliberate.
 *
 * Compared case-insensitively: the DB stores Sleeper's own casing ("IR",
 * "PUP", "Questionable") and there's nothing enforcing it.
 */
const UNDRAFTABLE_STATUSES = new Set(["IR", "PUP", "SUSPENDED", "OUT"]);

/**
 * Designations that carry no information. Sleeper reports "NA" for a player
 * it simply has no status for, which is indistinguishable from healthy —
 * badging it would put a warning on someone with nothing wrong.
 */
const NON_STATUSES = new Set(["", "NA", "ACTIVE", "HEALTHY"]);

/** Normalised code, or "" for anything that isn't a real designation. */
export function injuryCode(status?: string | null): string {
  if (!status) return "";
  const code = status.trim().toUpperCase();
  return NON_STATUSES.has(code) ? "" : code;
}

/**
 * True when the backend considers this player unable to play at all — i.e.
 * he is absent from scarcity counts and from the AI's slate, and appears on
 * the big board only because the board opts into showing him.
 */
export function isUndraftable(status?: string | null): boolean {
  return UNDRAFTABLE_STATUSES.has(injuryCode(status));
}

/** True when there's any designation worth showing the user. */
export function hasInjuryDesignation(status?: string | null): boolean {
  return injuryCode(status) !== "";
}
