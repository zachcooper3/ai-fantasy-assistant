/**
 * Tests for the pure draft helpers.
 *
 * adpValue is the reason this file exists: it shipped with value and reach
 * swapped, so the panel painted genuine fallers amber and reaches green on the
 * exact number you use to judge a pick.
 */

import { describe, expect, it } from "vitest";

import { DraftState } from "@/lib/api";
import { adpValue, rosterSlots, slotForPick, slotsBeforeMyNextPick } from "@/lib/draft";

// ---------------------------------------------------------------------------
// adpValue
// ---------------------------------------------------------------------------

describe("adpValue", () => {
  it("calls a player who fell past their ADP a value", () => {
    // The field takes them around 16; they're still here at 26.
    const v = adpValue(16, 26);
    expect(v.label).toBe("value");
    expect(v.delta).toBe(-10);
    expect(v.text).toBe("-10 vs pick 26");
    expect(v.word).toBe("value");
  });

  it("calls a player taken ahead of their ADP a reach", () => {
    // Market says pick 32; spending pick 26 is paying 6 picks early.
    const v = adpValue(32, 26);
    expect(v.label).toBe("reach");
    expect(v.delta).toBe(6);
    expect(v.text).toBe("+6 vs pick 26");
    expect(v.word).toBe("reach");
  });

  it("treats small gaps as neutral, since ADP is an average", () => {
    expect(adpValue(26, 26).label).toBe("even");
    expect(adpValue(29, 26).label).toBe("even"); // +3, on the band edge
    expect(adpValue(23, 26).label).toBe("even"); // -3, on the band edge
  });

  it("tips out of the neutral band just past the threshold", () => {
    expect(adpValue(30, 26).label).toBe("reach"); // +4
    expect(adpValue(22, 26).label).toBe("value"); // -4
  });

  it("rounds fractional ADP before comparing", () => {
    // -3.6 rounds to -4, which is outside the +/-3 band.
    expect(adpValue(22.4, 26).delta).toBe(-4);
    expect(adpValue(22.4, 26).label).toBe("value");
  });

  it("honours a custom neutral band", () => {
    expect(adpValue(30, 26, 10).label).toBe("even");
    expect(adpValue(16, 26, 10).label).toBe("even");
  });

  it("never renders a bare sign-ambiguous number without a word", () => {
    for (const [adp, pick] of [[16, 26], [32, 26], [26, 26]]) {
      expect(adpValue(adp, pick).word).toBeTruthy();
    }
  });
});

// ---------------------------------------------------------------------------
// slotForPick
// ---------------------------------------------------------------------------

describe("slotForPick", () => {
  it("matches the backend's documented 12-team examples", () => {
    // From DraftStateService.slot_for_pick's docstring.
    expect(slotForPick(1, 12)).toBe(1);
    expect(slotForPick(12, 12)).toBe(12);
    expect(slotForPick(13, 12)).toBe(12); // snake reverses
    expect(slotForPick(24, 12)).toBe(1);
    expect(slotForPick(25, 12)).toBe(1);
  });

  it("ascends on odd rounds and descends on even ones", () => {
    const round1 = [1, 2, 3, 4].map((p) => slotForPick(p, 4));
    const round2 = [5, 6, 7, 8].map((p) => slotForPick(p, 4));
    expect(round1).toEqual([1, 2, 3, 4]);
    expect(round2).toEqual([4, 3, 2, 1]);
  });

  it("gives every slot exactly one pick per round", () => {
    const size = 10;
    for (let round = 1; round <= 4; round++) {
      const slots = new Set<number>();
      for (let i = 1; i <= size; i++) {
        slots.add(slotForPick((round - 1) * size + i, size));
      }
      expect(slots.size).toBe(size);
    }
  });

  it("doesn't divide by zero on a degenerate league size", () => {
    expect(slotForPick(5, 0)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// slotsBeforeMyNextPick
// ---------------------------------------------------------------------------

function state(overrides: Partial<DraftState> = {}): DraftState {
  return {
    is_active: true,
    league_size: 12,
    my_draft_position: 3,
    total_rounds: 15,
    scoring_format: "ppr",
    qb_slots: 1, rb_slots: 2, wr_slots: 2, te_slots: 1, flex_slots: 1, dst_slots: 1,
    current_pick_number: 1,
    current_round: 1,
    current_team_slot: 1,
    is_my_turn: false,
    picks_until_my_turn: 2,
    my_next_pick_number: 3,
    draft_complete: false,
    was_restored: false,
    started_at: null,
    picks: [],
    my_roster: [],
    ...overrides,
  };
}

describe("slotsBeforeMyNextPick", () => {
  it("lists the teams picking between now and your turn", () => {
    expect(
      slotsBeforeMyNextPick(state({ current_pick_number: 1, my_next_pick_number: 3 }))
    ).toEqual([1, 2]);
  });

  it("is empty while you're on the clock", () => {
    expect(slotsBeforeMyNextPick(state({ is_my_turn: true }))).toEqual([]);
  });

  it("is empty when you have no picks left", () => {
    expect(slotsBeforeMyNextPick(state({ my_next_pick_number: null }))).toEqual([]);
  });

  it("excludes your own slot and never repeats a team", () => {
    // Pick 3 is mine (slot 3); across the turn, slot 12 picks twice in a row.
    const slots = slotsBeforeMyNextPick(
      state({ current_pick_number: 4, my_next_pick_number: 22 })
    );
    expect(slots).not.toContain(3);
    expect(new Set(slots).size).toBe(slots.length);
  });
});

// ---------------------------------------------------------------------------
// rosterSlots
// ---------------------------------------------------------------------------

describe("rosterSlots", () => {
  it("builds starters from the session config, then fills with bench", () => {
    const slots = rosterSlots(state({ total_rounds: 12 }));
    expect(slots.slice(0, 8)).toEqual(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DST"]);
    expect(slots.filter((s) => s === "BN")).toHaveLength(4);
  });

  it("reflects a non-default lineup rather than a hardcoded one", () => {
    // The bug this replaced: 2RB/2WR/K was baked in regardless of config.
    const slots = rosterSlots(
      state({ qb_slots: 2, rb_slots: 1, wr_slots: 3, te_slots: 0, flex_slots: 2, dst_slots: 0 })
    );
    expect(slots.filter((s) => s === "QB")).toHaveLength(2);
    expect(slots.filter((s) => s === "WR")).toHaveLength(3);
    expect(slots).not.toContain("TE");
    expect(slots).not.toContain("DST");
  });

  it("never emits a K slot — the backend has no k_slots field", () => {
    expect(rosterSlots(state())).not.toContain("K");
  });

  it("doesn't go negative when starters outnumber rounds", () => {
    const slots = rosterSlots(state({ total_rounds: 2 }));
    expect(slots.filter((s) => s === "BN")).toHaveLength(0);
    expect(slots.length).toBeGreaterThan(0);
  });
});
