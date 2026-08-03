/**
 * Tests for board search matching.
 *
 * The cases here are the ones that failed under the old
 * `name.toLowerCase().includes(query)` implementation — all of them names you'd
 * plausibly type in a hurry mid-draft.
 */

import { describe, expect, it } from "vitest";

import { matchesQuery, normalize } from "@/lib/search";

const player = (name: string, team = "CIN") => ({ name, team });

describe("normalize", () => {
  it("strips case, punctuation and accents", () => {
    expect(normalize("Ja'Marr Chase")).toBe("jamarrchase");
    expect(normalize("A.J. Brown")).toBe("ajbrown");
    expect(normalize("D'Andre Swift")).toBe("dandreswift");
  });
});

describe("matchesQuery", () => {
  it("matches regardless of capitalisation", () => {
    expect(matchesQuery(player("CeeDee Lamb", "DAL"), "ceedee")).toBe(true);
  });

  it("matches across apostrophes and periods you didn't type", () => {
    expect(matchesQuery(player("Ja'Marr Chase"), "jamarr")).toBe(true);
    expect(matchesQuery(player("A.J. Brown", "PHI"), "aj")).toBe(true);
  });

  it("requires every token to match, so extra words narrow the list", () => {
    const chase = player("Ja'Marr Chase", "CIN");
    expect(matchesQuery(chase, "ja chase")).toBe(true);
    expect(matchesQuery(chase, "chase cin")).toBe(true);
    expect(matchesQuery(chase, "chase phi")).toBe(false);
  });

  it("finds players by team", () => {
    expect(matchesQuery(player("Joe Burrow", "CIN"), "cin")).toBe(true);
    expect(matchesQuery(player("Joe Burrow", "CIN"), "c")).toBe(true);
  });

  it("matches mid-name, not just from the start", () => {
    expect(matchesQuery(player("Amon-Ra St. Brown", "DET"), "stbrown")).toBe(true);
    expect(matchesQuery(player("Amon-Ra St. Brown", "DET"), "ra")).toBe(true);
  });

  it("treats an empty or whitespace query as matching everything", () => {
    expect(matchesQuery(player("Anyone"), "")).toBe(true);
    expect(matchesQuery(player("Anyone"), "   ")).toBe(true);
  });

  it("rejects a genuine non-match", () => {
    expect(matchesQuery(player("Joe Burrow", "CIN"), "mahomes")).toBe(false);
  });

  it("ignores punctuation typed in the query itself", () => {
    expect(matchesQuery(player("Ja'Marr Chase"), "ja'marr")).toBe(true);
  });
});
