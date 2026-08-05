"use client";
/**
 * Big Board column width — drag-to-resize with a collapse threshold.
 *
 * Originally "focus mode" was binary: BigBoard was either its normal width
 * or a 56px rail, and AIPanel took whatever was freed. That's still exactly
 * what happens at the two ends of this hook's range — toggleCollapse jumps
 * between RAIL_WIDTH and the last remembered expanded width — but in between,
 * a drag handle lets you park the board (and therefore the AI panel) at
 * whatever width actually suits the screen, instead of only choosing between
 * "full board" and "thin strip."
 */

import { useCallback, useEffect, useRef, useState } from "react";

/** Absolute minimum drag width — matches the collapsed rail's rendered width. */
export const RAIL_WIDTH = 56;

/** At or below this, BigBoard renders its rail — a real table doesn't fit. */
const COLLAPSE_THRESHOLD = 90;

/**
 * First-ever-visit width, before any drag or persisted preference exists.
 * Deliberately not "fill all available space" (the old 1fr default): now
 * that width is a real user-adjustable control, a fixed, reasonable start
 * point is easier to reason about than trying to replicate exact auto-fill
 * sizing across arbitrary monitor widths.
 */
const DEFAULT_WIDTH = 640;

/**
 * Floor for the AI panel, mirrored from the fixed widths it used to render
 * at pre-resize. Expressed as a ceiling on BigBoard's drag range rather than
 * enforced on AIPanel directly — AIPanel's grid column is `minmax(0,1fr)`
 * and just gets whatever BigBoard + DraftRoom don't take.
 */
const AI_MIN_WIDTH_MD = 300;
const AI_MIN_WIDTH_XL = 340;

/** DraftRoom's fixed width — mirrors the md:/xl: breakpoint it used to use. */
const DRAFT_ROOM_WIDTH_MD = 320;
const DRAFT_ROOM_WIDTH_XL = 380;
const XL_BREAKPOINT = "(min-width: 1280px)";

/** Visual + hit-target width of the drag handle. */
export const HANDLE_WIDTH = 10;

/** Matches the grid's gap-3; there are two gaps to account for (3 columns). */
const GAP = 12;

/** px per Arrow key press when the handle has keyboard focus. */
const KEYBOARD_STEP = 24;

/** Used as the max width before the container has been measured at least once. */
const UNMEASURED_MAX_WIDTH = 900;

const WIDTH_KEY = "fda:board-width";
const EXPANDED_WIDTH_KEY = "fda:board-expanded-width";

function clamp(width: number, min: number, max: number): number {
  return Math.min(Math.max(width, min), Math.max(min, max));
}

export interface BoardResize {
  /** Attach to the grid container — used to measure available width. */
  gridRef: React.RefObject<HTMLDivElement | null>;
  /** Current BigBoard column width, in px. */
  boardWidth: number;
  /** True at/below the rail threshold — pass through to BigBoard/AIPanel. */
  collapsed: boolean;
  /** Current DraftRoom column width, in px (tracks the xl breakpoint). */
  draftRoomWidth: number;
  /** Jump between the rail and the last remembered expanded width. */
  toggleCollapse: () => void;
  /** Props to spread onto the drag handle element. */
  handle: {
    role: "separator";
    "aria-orientation": "vertical";
    "aria-label": string;
    "aria-valuenow": number;
    "aria-valuemin": number;
    "aria-valuemax": number;
    tabIndex: 0;
    onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
    onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => void;
    onPointerUp: (e: React.PointerEvent<HTMLDivElement>) => void;
    onDoubleClick: () => void;
    onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void;
  };
}

export function useBoardResize(): BoardResize {
  const gridRef = useRef<HTMLDivElement>(null);

  // DraftRoom's width steps at the xl breakpoint. Tracked via matchMedia
  // rather than Tailwind classes because grid-template-columns is now set
  // inline (it needs boardWidth, a runtime value Tailwind can't express).
  const [isXl, setIsXl] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(XL_BREAKPOINT);
    setIsXl(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setIsXl(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  const draftRoomWidth = isXl ? DRAFT_ROOM_WIDTH_XL : DRAFT_ROOM_WIDTH_MD;
  const aiMinWidth = isXl ? AI_MIN_WIDTH_XL : AI_MIN_WIDTH_MD;

  const [containerWidth, setContainerWidth] = useState(0);
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      setContainerWidth(entries[0]?.contentRect.width ?? 0);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const maxWidth = useCallback(
    (container: number, draftRoom: number) =>
      container > 0
        ? Math.max(RAIL_WIDTH, container - draftRoom - aiMinWidth - GAP * 2 - HANDLE_WIDTH)
        : UNMEASURED_MAX_WIDTH,
    [aiMinWidth]
  );

  const [boardWidth, setBoardWidthState] = useState(DEFAULT_WIDTH);
  // Mirrors boardWidth for reads inside callbacks that must stay dep-free
  // (the drag handlers) or that fire from a stale closure (pointerup after
  // several pointermove-driven state updates).
  const boardWidthRef = useRef(boardWidth);
  const expandedWidthRef = useRef(DEFAULT_WIDTH);

  // Load persisted width once on mount. Read in an effect, not during
  // render, so the server/first-client render stay identical (localStorage
  // isn't available server-side) and there's no hydration mismatch.
  useEffect(() => {
    try {
      const storedExpanded = window.localStorage.getItem(EXPANDED_WIDTH_KEY);
      if (storedExpanded !== null) {
        const n = Number(storedExpanded);
        if (Number.isFinite(n)) expandedWidthRef.current = n;
      }
      const stored = window.localStorage.getItem(WIDTH_KEY);
      if (stored !== null) {
        const n = Number(stored);
        if (Number.isFinite(n)) {
          boardWidthRef.current = n;
          setBoardWidthState(n);
        }
      }
    } catch {
      // Storage can be unavailable (private mode, blocked cookies) —
      // resizing still works this session, it just won't persist.
    }
  }, []);

  const persist = useCallback((width: number) => {
    try {
      window.localStorage.setItem(WIDTH_KEY, String(width));
      if (width > COLLAPSE_THRESHOLD) {
        window.localStorage.setItem(EXPANDED_WIDTH_KEY, String(width));
      }
    } catch {
      // Non-fatal, see above.
    }
  }, []);

  const setWidth = useCallback(
    (width: number, opts: { persist?: boolean } = {}) => {
      const clamped = clamp(width, RAIL_WIDTH, maxWidth(containerWidth, draftRoomWidth));
      boardWidthRef.current = clamped;
      setBoardWidthState(clamped);
      if (clamped > COLLAPSE_THRESHOLD) expandedWidthRef.current = clamped;
      if (opts.persist !== false) persist(clamped);
    },
    [containerWidth, draftRoomWidth, maxWidth, persist]
  );

  // Re-clamp whenever the available space shrinks (window resize, DraftRoom
  // swapping between its md/xl widths) so a width saved on a wide monitor
  // can't push the AI panel below its floor — or off-screen — on a
  // narrower one.
  useEffect(() => {
    if (containerWidth === 0) return;
    setBoardWidthState((w) => {
      const clamped = clamp(w, RAIL_WIDTH, maxWidth(containerWidth, draftRoomWidth));
      boardWidthRef.current = clamped;
      return clamped === w ? w : clamped;
    });
  }, [containerWidth, draftRoomWidth, maxWidth]);

  const collapsed = boardWidth <= COLLAPSE_THRESHOLD;

  const toggleCollapse = useCallback(() => {
    if (boardWidthRef.current <= COLLAPSE_THRESHOLD) {
      setWidth(expandedWidthRef.current || DEFAULT_WIDTH);
    } else {
      setWidth(RAIL_WIDTH);
    }
  }, [setWidth]);

  const resetWidth = useCallback(() => setWidth(DEFAULT_WIDTH), [setWidth]);

  // ------------------------------------------------------------------
  // Drag handle. Pointer capture keeps move/up events targeting the handle
  // even once the cursor leaves its 10px hit area mid-drag — without it, a
  // fast drag outside the strip would silently stop tracking.
  // ------------------------------------------------------------------
  const dragStartXRef = useRef(0);
  const dragStartWidthRef = useRef(0);
  const draggingRef = useRef(false);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    dragStartXRef.current = e.clientX;
    dragStartWidthRef.current = boardWidthRef.current;
    e.currentTarget.setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      const delta = e.clientX - dragStartXRef.current;
      // Skip the localStorage write on every pixel of movement — only the
      // width at pointer-up needs to persist.
      setWidth(dragStartWidthRef.current + delta, { persist: false });
    },
    [setWidth]
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      e.currentTarget.releasePointerCapture(e.pointerId);
      persist(boardWidthRef.current);
    },
    [persist]
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      // Standard WAI-ARIA "window splitter" keyboard behaviour.
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setWidth(boardWidthRef.current - KEYBOARD_STEP);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setWidth(boardWidthRef.current + KEYBOARD_STEP);
      } else if (e.key === "Home") {
        e.preventDefault();
        setWidth(RAIL_WIDTH);
      } else if (e.key === "End") {
        e.preventDefault();
        setWidth(maxWidth(containerWidth, draftRoomWidth));
      }
    },
    [setWidth, maxWidth, containerWidth, draftRoomWidth]
  );

  return {
    gridRef,
    boardWidth,
    collapsed,
    draftRoomWidth,
    toggleCollapse,
    handle: {
      role: "separator",
      "aria-orientation": "vertical",
      "aria-label": "Resize Big Board",
      "aria-valuenow": Math.round(boardWidth),
      "aria-valuemin": RAIL_WIDTH,
      "aria-valuemax": Math.round(maxWidth(containerWidth, draftRoomWidth)),
      tabIndex: 0,
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onDoubleClick: resetWidth,
      onKeyDown,
    },
  };
}
