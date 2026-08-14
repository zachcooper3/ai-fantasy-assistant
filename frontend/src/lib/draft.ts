/**
 * Presentation-only draft helpers.
 *
 * These derive display data from the session the backend already sends. They
 * deliberately live outside the components so the roster-slot layout and the
 * snake math each exist in exactly one place on the client.
 */

import { DraftState } from "@/lib/api";

/**
 * Returns the 1-indexed team slot that owns a given overall pick number.
 *
 * Mirrors `DraftStateService.slot_for_pick` in
 * backend/app/services/draft_state.py. Duplicated on purpose: the backend
 * only ever reports the *current* slot, and highlighting "who picks between
 * now and your next turn" requires looking ahead. Same caveat as the backend
 * (audit W3) — this assumes a pure snake, so 3rd-round reversal and traded
 * picks will attribute wrongly. It's used for emphasis only; nothing
 * load-bearing depends on it.
 */
export function slotForPick(pickNumber: number, leagueSize: number): number {
  if (leagueSize <= 0) return 1;
  const round = Math.ceil(pickNumber / leagueSize);
  const posInRound = pickNumber - (round - 1) * leagueSize;
  return round % 2 === 1 ? posInRound : leagueSize + 1 - posInRound;
}

/**
 * The team slots that pick between right now and the user's next turn, in
 * order, excluding the user's own slot. This is the only part of opponent
 * tracking that's actionable while you're on the clock — these are the teams
 * that can take the player you're eyeing.
 *
 * Returns an empty array when the user is on the clock or has no picks left.
 */
export function slotsBeforeMyNextPick(session: DraftState): number[] {
  const { current_pick_number, my_next_pick_number, league_size, my_draft_position } = session;
  if (session.is_my_turn || my_next_pick_number == null) return [];

  const seen = new Set<number>();
  const out: number[] = [];
  for (let pick = current_pick_number; pick < my_next_pick_number; pick++) {
    const slot = slotForPick(pick, league_size);
    if (slot === my_draft_position || seen.has(slot)) continue;
    seen.add(slot);
    out.push(slot);
  }
  return out;
}

/**
 * The roster slot layout for the *current* session config.
 *
 * This used to be a module-level constant hardcoded to 2 RB / 2 WR / 1 K,
 * which meant the empty-slot placeholders were simply wrong for any league
 * that wasn't that exact shape — and the app lets you configure all of them.
 * Note there is no kicker slot: the backend's DraftConfigRequest has no
 * `k_slots` field, so the old "K" placeholder was showing a slot the app
 * doesn't actually model.
 *
 * Everything past the starters is bench ("BN"), sized so the placeholders
 * account for every remaining round rather than stopping at the starters.
 */
export interface AdpValue {
  /**
   * `adp - pickNumber`, i.e. the player's market slot relative to this pick.
   *
   * NEGATIVE is the good direction: the field takes this player at `adp` on
   * average and they're somehow still on the board at a later pick, so they've
   * fallen to you. Positive means you'd be buying ahead of the market.
   */
  delta: number;
  label: "value" | "reach" | "even";
  /** e.g. "-7 vs pick 31" */
  text: string;
  /** "value" / "reach" spelled out, so the sign convention needn't be known. */
  word: string;
}

/**
 * How a player's ADP compares to the pick you'd spend on them.
 *
 * ADP is where the field drafts a player on average, so the comparison is
 * against `pickNumber`:
 *
 *   ADP 16, you're at pick 26  → they fell 10 past their slot  → VALUE
 *   ADP 32, you're at pick 26  → you'd pay 6 picks early       → REACH
 *
 * The direction is easy to get backwards — this shipped inverted, painting
 * genuine value amber and reaches green — so the mapping is spelled out above
 * and the label is rendered as a word alongside the number.
 *
 * The neutral band exists because ADP is an average over noisy inputs: a point
 * or two either way isn't a signal, and treating it as one would paint almost
 * every player as a value or a reach.
 */
export function adpValue(adp: number, pickNumber: number, neutralBand = 3): AdpValue {
  const delta = Math.round(adp - pickNumber);

  const label: AdpValue["label"] =
    delta < -neutralBand ? "value" : delta > neutralBand ? "reach" : "even";

  const sign = delta > 0 ? "+" : "";
  const word = label === "value" ? "value" : label === "reach" ? "reach" : "at ADP";

  return { delta, label, text: `${sign}${delta} vs pick ${pickNumber}`, word };
}

/**
 * How many players have been drafted at each position so far, plus an
 * `All` total.
 *
 * Exists to give the Big Board's counter a denominator that is a property
 * of the DRAFT rather than of the browser tab. The denominator used to be
 * captured from the first board response — "the pool as it looked when this
 * page loaded" — which is only the starting pool if you never reload. Open
 * the app mid-draft, or after it finishes, and it captured the *current*
 * count, so the counter collapsed to "414 / 414" and claimed nothing had
 * been drafted. Adding this to the live available count reconstructs the
 * original pool from data that's on every session payload, so a reload
 * can't change it.
 *
 * KNOWN OVERCOUNT: available counts exclude IR/PUP/Suspended/Out players
 * (see the board route), but the pick journal doesn't record whether a
 * drafted player was one of them — PickResponse carries no injury_status.
 * So drafting an IR stash inflates that position's denominator by one, for
 * the rest of the draft. It's bounded by the handful of such players who
 * ever get taken, and it only ever moves the total, never the "available"
 * figure people actually read.
 */
export function draftedCountsByPosition(session: DraftState): Record<string, number> {
  const counts: Record<string, number> = { All: 0 };
  for (const pick of session.picks) {
    counts[pick.position] = (counts[pick.position] ?? 0) + 1;
    counts.All += 1;
  }
  return counts;
}

/**
 * How many picks out the automatic recommendation fires.
 *
 * Zero: on the clock only. Each fire is a paid Claude call, and every pick
 * invalidates the current recommendation, so a threshold of N fires roughly
 * N+1 times per turn. At 1 this fired twice — once while the team ahead of you
 * was picking, then again once you were actually on the clock — and only the
 * second described the board you were drafting from. The first was discarded
 * by the same pick that made it worth having.
 *
 * The cost of 0 is that the Claude call happens inside your pick window rather
 * than ahead of it. That's the deliberate trade: one call per turn, always
 * against the real board.
 *
 * If the wait ever needs hiding, keep the previous recommendation on screen
 * while the new one loads rather than fetching earlier — that costs nothing
 * and doesn't present stale advice as current.
 */
export const REC_AUTO_WITHIN_PICKS = 0;

export interface AutoRecommendInputs {
  /** Has the stored auto-recommend preference been read from localStorage? */
  prefsLoaded: boolean;
  /** The user's auto-recommend preference. */
  autoRecommend: boolean;
  session: DraftState | null;
  /** Pick number an automatic request has already been made for, if any. */
  requestedForPick: number | null;
  /** Is a recommendation already on screen? */
  hasRecommendation: boolean;
  /** Is a request already in flight? */
  isLoading: boolean;
}

/**
 * Whether the automatic recommendation should fire right now.
 *
 * Extracted from the effect that used to hold it inline so it can be tested
 * directly — the frontend's vitest setup runs in a node environment with no
 * jsdom or React testing library, so a predicate is testable where a hook is
 * not. Every guard here is load-bearing and each one has cost money or shown
 * wrong advice at some point:
 *
 *  - `prefsLoaded` — without it a saved "off" still let one automatic call
 *    through on every page load, before the preference had been read.
 *  - `requestedForPick` — the effect re-runs whenever the session or board
 *    object changes identity, which is often; this pins it to once per pick.
 *  - `hasRecommendation` / `isLoading` — stops a second request stacking on
 *    top of advice that's already there or already coming.
 */
export function shouldAutoRecommend({
  prefsLoaded,
  autoRecommend,
  session,
  requestedForPick,
  hasRecommendation,
  isLoading,
}: AutoRecommendInputs): boolean {
  if (!prefsLoaded || !autoRecommend) return false;
  if (!session?.is_active || session.draft_complete) return false;
  if (session.picks_until_my_turn > REC_AUTO_WITHIN_PICKS) return false;
  if (requestedForPick === session.current_pick_number) return false;
  if (hasRecommendation || isLoading) return false;
  return true;
}

export function rosterSlots(session: DraftState): string[] {
  const starters: string[] = [
    ...Array<string>(session.qb_slots).fill("QB"),
    ...Array<string>(session.rb_slots).fill("RB"),
    ...Array<string>(session.wr_slots).fill("WR"),
    ...Array<string>(session.te_slots).fill("TE"),
    ...Array<string>(session.flex_slots).fill("FLEX"),
    ...Array<string>(session.dst_slots).fill("DST"),
  ];
  const bench = Math.max(0, session.total_rounds - starters.length);
  return [...starters, ...Array<string>(bench).fill("BN")];
}
