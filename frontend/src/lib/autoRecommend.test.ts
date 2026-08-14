/**
 * Tests for the automatic-recommendation trigger.
 *
 * Every fire is a paid Claude call, so the interesting assertions here are
 * mostly about NOT firing. The trigger previously ran at one pick out, which
 * meant two calls per turn: the first was discarded by the pick that landed on
 * top of it, and only the second described the board being drafted from.
 */

import { describe, expect, it } from "vitest";

import { DraftState } from "@/lib/api";
import { REC_AUTO_WITHIN_PICKS, shouldAutoRecommend } from "@/lib/draft";

function session(over: Partial<DraftState> = {}): DraftState {
  return {
    is_active: true,
    league_size: 12,
    my_draft_position: 5,
    total_rounds: 15,
    scoring_format: "ppr",
    qb_slots: 1, rb_slots: 2, wr_slots: 2, te_slots: 1, flex_slots: 1, dst_slots: 1,
    current_pick_number: 17,
    current_round: 2,
    current_team_slot: 5,
    is_my_turn: true,
    picks_until_my_turn: 0,
    my_next_pick_number: 17,
    draft_complete: false,
    was_restored: false,
    started_at: null,
    picks: [],
    my_roster: [],
    ...over,
  };
}

const ready = {
  prefsLoaded: true,
  autoRecommend: true,
  session: session(),
  requestedForPick: null,
  hasRecommendation: false,
  isLoading: false,
};

describe("shouldAutoRecommend", () => {
  it("fires when you are on the clock", () => {
    expect(shouldAutoRecommend(ready)).toBe(true);
  });

  it("does not fire one pick out", () => {
    // The regression this change exists to prevent: firing here cost a second
    // Claude call whose advice the next pick immediately invalidated.
    expect(
      shouldAutoRecommend({ ...ready, session: session({ picks_until_my_turn: 1, is_my_turn: false }) })
    ).toBe(false);
  });

  it.each([2, 3, 5, 11])("does not fire %i picks out", (n) => {
    expect(
      shouldAutoRecommend({ ...ready, session: session({ picks_until_my_turn: n, is_my_turn: false }) })
    ).toBe(false);
  });

  it("is pinned to on-the-clock only", () => {
    // Guards the intent rather than the number — if someone raises this, the
    // double-call behaviour comes back and this test should make them say so.
    expect(REC_AUTO_WITHIN_PICKS).toBe(0);
  });

  it("waits for the stored preference to load", () => {
    // Otherwise a saved "off" still let one call through on every page load.
    expect(shouldAutoRecommend({ ...ready, prefsLoaded: false })).toBe(false);
  });

  it("respects the preference being off", () => {
    expect(shouldAutoRecommend({ ...ready, autoRecommend: false })).toBe(false);
  });

  it("does not fire without a session", () => {
    expect(shouldAutoRecommend({ ...ready, session: null })).toBe(false);
  });

  it("does not fire on an inactive session", () => {
    expect(shouldAutoRecommend({ ...ready, session: session({ is_active: false }) })).toBe(false);
  });

  it("does not fire once the draft is complete", () => {
    expect(shouldAutoRecommend({ ...ready, session: session({ draft_complete: true }) })).toBe(false);
  });

  it("fires only once per pick", () => {
    expect(shouldAutoRecommend({ ...ready, requestedForPick: 17 })).toBe(false);
  });

  it("fires again on the next turn", () => {
    // requestedForPick still holds the previous turn's number.
    expect(
      shouldAutoRecommend({
        ...ready,
        requestedForPick: 17,
        session: session({ current_pick_number: 32 }),
      })
    ).toBe(true);
  });

  it("does not stack on advice already showing", () => {
    expect(shouldAutoRecommend({ ...ready, hasRecommendation: true })).toBe(false);
  });

  it("does not stack on a request already in flight", () => {
    expect(shouldAutoRecommend({ ...ready, isLoading: true })).toBe(false);
  });

  it("fires after a pick clears the previous recommendation", () => {
    // The real sequence: your turn arrives, the websocket has just blanked the
    // panel, nothing is in flight, and this pick has not been asked about.
    expect(
      shouldAutoRecommend({
        ...ready,
        requestedForPick: 8,
        hasRecommendation: false,
        session: session({ current_pick_number: 17, picks_until_my_turn: 0 }),
      })
    ).toBe(true);
  });
});
