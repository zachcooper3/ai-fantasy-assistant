"use client";
/**
 * ShortcutsOverlay — the keyboard shortcut reference, opened with "?".
 *
 * Every shortcut in this app is document-level and invisible: BigBoard's
 * search/selection keys, the page's g/u/b, and the resize handle's arrows are
 * spread across three files with no user-facing list anywhere. BigBoard's own
 * docstring claimed the list lived "in the header tooltip" — there was no such
 * tooltip, and there never had been.
 *
 * This is the canonical list. When a shortcut is added or changed, change it
 * here too — the handlers themselves are in BigBoard.tsx (search/selection/
 * position filters), app/page.tsx (g/u/b/?) and hooks/useBoardResize.ts (the
 * handle's arrow keys).
 */

import { useEffect, useRef } from "react";
import { X } from "lucide-react";

interface Group {
  title: string;
  items: { keys: string[]; description: string }[];
}

const GROUPS: Group[] = [
  {
    title: "Big Board",
    items: [
      { keys: ["/"], description: "Focus the search box" },
      { keys: ["↑", "↓"], description: "Move the selection (works while searching)" },
      { keys: ["Enter"], description: "Draft the selected player" },
      { keys: ["i"], description: "Open details for the selected player" },
      { keys: ["Esc"], description: "Clear the search and selection" },
      { keys: ["1", "–", "7"], description: "Filter to All / QB / RB / WR / TE / DST / K" },
    ],
  },
  {
    title: "Draft",
    items: [
      { keys: ["g"], description: "Get an AI recommendation for this pick" },
      { keys: ["u"], description: "Undo the last pick" },
    ],
  },
  {
    title: "Layout",
    items: [
      { keys: ["b"], description: "Toggle focus mode — collapse the Big Board" },
      {
        keys: ["Tab", "→", "←"],
        description: "Focus the drag handle, then nudge the board's width",
      },
      { keys: ["Home", "End"], description: "With the handle focused: rail / maximum width" },
      { keys: ["?"], description: "Open this list" },
    ],
  },
];

function Key({ children }: { children: React.ReactNode }) {
  // "–" is a separator in a range like 1–7, not a key to press.
  if (children === "–") return <span className="text-slate-500 text-xs">–</span>;
  return (
    <kbd className="inline-flex items-center justify-center min-w-[1.5rem] px-1.5 py-0.5 rounded border border-slate-600 bg-slate-800 text-[11px] font-semibold text-slate-200">
      {children}
    </kbd>
  );
}

export default function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  // Focus the close button on open, so Enter/Space dismisses without a
  // reach for the mouse and screen readers land inside the dialog.
  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  // Escape closes. Captured at the document level rather than on the dialog
  // because BigBoard's own Escape handler is document-level too and would
  // otherwise clear the search box underneath instead.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcuts-title"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg max-h-[85vh] overflow-y-auto rounded-2xl bg-slate-900 border border-slate-700 p-6 shadow-2xl"
      >
        <div className="flex items-center justify-between mb-5">
          <h2 id="shortcuts-title" className="text-sm font-bold text-slate-200 uppercase tracking-wider">
            Keyboard shortcuts
          </h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close keyboard shortcuts"
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        <div className="space-y-5">
          {GROUPS.map((group) => (
            <div key={group.title}>
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                {group.title}
              </h3>
              <dl className="space-y-1.5">
                {group.items.map((item) => (
                  <div key={item.description} className="flex items-center gap-3 text-sm">
                    <dt className="flex items-center gap-1 shrink-0 w-32">
                      {item.keys.map((k, i) => (
                        <Key key={i}>{k}</Key>
                      ))}
                    </dt>
                    <dd className="text-slate-300 min-w-0">{item.description}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>

        <p className="mt-5 pt-4 border-t border-slate-800 text-xs text-slate-400">
          Shortcuts are ignored while you&apos;re typing in a text field, except Escape
          and the arrow keys.
        </p>
      </div>
    </div>
  );
}
