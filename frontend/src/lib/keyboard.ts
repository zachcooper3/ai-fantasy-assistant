/**
 * Shared guard for global keyboard shortcuts.
 *
 * Every document-level shortcut has to answer the same question first: is the
 * user typing? Without this, searching for "Gibbs" fires the "g = get pick"
 * shortcut on the first keystroke.
 */

export function isTypingTarget(event: KeyboardEvent): boolean {
  const el = event.target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    el.isContentEditable
  );
}

/**
 * True if a modifier is held. Browser and OS shortcuts (Cmd+R, Ctrl+F) must
 * keep working, so single-key shortcuts should ignore these.
 */
export function hasModifier(event: KeyboardEvent): boolean {
  return event.metaKey || event.ctrlKey || event.altKey;
}
