/**
 * Player search matching.
 *
 * The board search was a raw `name.toLowerCase().includes(query)`, which fails
 * exactly when you're in a hurry: "ceedee" misses "CeeDee Lamb" the moment you
 * type it without the capital, punctuation in names ("Ja'Marr", "A.J.") has to
 * be typed exactly, and you can't find a player by their NFL team at all.
 */

/** Lowercase, strip accents, drop everything that isn't alphanumeric. */
export function normalize(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]/g, "");
}

/**
 * True if a player matches the query.
 *
 * Every whitespace-separated token must match something — so "ja chase" finds
 * Ja'Marr Chase, and "chase cin" narrows to the Bengals one. A token matches if
 * it's a substring of the punctuation-stripped full name, or a prefix of the
 * team code. Punctuation-stripping is what makes "jamarr" and "aj" work.
 */
export function matchesQuery(
  player: { name: string; team: string },
  query: string
): boolean {
  const tokens = query.trim().split(/\s+/).map(normalize).filter(Boolean);
  if (tokens.length === 0) return true;

  const name = normalize(player.name);
  const team = normalize(player.team);

  return tokens.every((token) => name.includes(token) || team.startsWith(token));
}
