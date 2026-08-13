import { describe, expect, it } from "vitest";

import { hasInjuryDesignation, injuryCode, isUndraftable } from "@/lib/injury";

describe("injuryCode", () => {
  it("normalises casing and whitespace", () => {
    expect(injuryCode("questionable")).toBe("QUESTIONABLE");
    expect(injuryCode("  IR  ")).toBe("IR");
  });

  it("treats Sleeper's non-statuses as no designation", () => {
    // "NA" is what Sleeper sends for a player it has no status for. Badging
    // it would warn about someone with nothing wrong.
    expect(injuryCode("NA")).toBe("");
    expect(injuryCode("Active")).toBe("");
    expect(injuryCode(null)).toBe("");
    expect(injuryCode(undefined)).toBe("");
    expect(injuryCode("")).toBe("");
  });
});

describe("isUndraftable", () => {
  /**
   * This set must match UNDRAFTABLE_STATUSES in backend/db/player_repo.py.
   * It's pinned exactly — not with a "contains IR" spot check — because the
   * failure mode of drift is silent: useDraft would decrement a scarcity
   * count for a player the backend never counted, walking the number down
   * past the truth with nothing to surface it.
   */
  it.each(["IR", "PUP", "Suspended", "Out"])("treats %s as unable to play", (status) => {
    expect(isUndraftable(status)).toBe(true);
  });

  it.each(["Questionable", "Doubtful", "DNR", "NFI"])(
    "treats %s as playable — the backend still counts and offers him",
    (status) => {
      expect(isUndraftable(status)).toBe(false);
    }
  );

  it("is false for a healthy player", () => {
    expect(isUndraftable(null)).toBe(false);
    expect(isUndraftable("NA")).toBe(false);
  });
});

describe("hasInjuryDesignation", () => {
  it("is true for anything worth showing, playable or not", () => {
    expect(hasInjuryDesignation("Questionable")).toBe(true);
    expect(hasInjuryDesignation("IR")).toBe(true);
  });

  it("is false for healthy and for Sleeper's placeholder", () => {
    expect(hasInjuryDesignation(null)).toBe(false);
    expect(hasInjuryDesignation("NA")).toBe(false);
  });
});
