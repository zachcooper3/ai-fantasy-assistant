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
