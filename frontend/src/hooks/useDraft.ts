"use client";
/**
 * Central state hook for the draft room.
 *
 * Manages:
 * - Draft session state (from the API)
 * - The big board (available players + scarcity)
 * - AI recommendation
 * - WebSocket connection for real-time pick updates
 * - Optional Sleeper live draft sync
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, API_TOKEN, Board, DraftState, Recommendation, SyncStatus, WsEvent } from "@/lib/api";

// Browsers can't set an Authorization header on a WebSocket handshake, so
// the shared token (if configured) rides a query param instead — checked
// server-side in backend/app/api/websocket.py.
const WS_URL =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000")
    .replace(/^http/, "ws") + "/ws/draft" +
  (API_TOKEN ? `?token=${encodeURIComponent(API_TOKEN)}` : "");

// How often to re-poll GET /api/sync/status while a Sleeper sync is live.
// The endpoint is a pure in-memory read on the backend (no Sleeper call), so
// this is cheap; it exists to keep synced_pick_count fresh and to notice the
// poller dying (status → "error") without waiting for a user action.
const SYNC_POLL_MS = 5000;

/** Field-wise equality, so an unchanged poll response doesn't trigger a render. */
function sameSyncStatus(a: SyncStatus | null, b: SyncStatus | null): boolean {
  if (a === b) return true;
  if (a === null || b === null) return false;
  return (
    a.status === b.status &&
    a.draft_id === b.draft_id &&
    a.synced_pick_count === b.synced_pick_count &&
    a.error === b.error
  );
}

export interface DraftHook {
  session: DraftState | null;
  board: Board | null;
  recommendation: Recommendation | null;
  syncStatus: SyncStatus | null;
  /**
   * True while picks are flowing in from Sleeper. When this is set the UI must
   * not offer manual pick controls: a manual pick races the 2-second poll loop
   * and local pick numbers are inferred from the pick count, so an interleaved
   * manual entry misattributes every subsequent pick to the wrong team slot
   * (audit W2). Undo is worse still — it restores availability locally while
   * Sleeper still has the pick, and the synced-pick counter isn't rewound, so
   * sync never re-records it (audit W12).
   */
  isSyncing: boolean;
  isConnected: boolean;
  isLoadingRec: boolean;
  error: string | null;
  startSession: (config: {
    league_size: number;
    my_draft_position: number;
    total_rounds: number;
    qb_slots?: number;
    rb_slots?: number;
    wr_slots?: number;
    te_slots?: number;
    flex_slots?: number;
    dst_slots?: number;
    sleeper_draft_id?: string;
  }) => Promise<void>;
  endSession: () => Promise<void>;
  recordPick: (playerId: number) => Promise<void>;
  undoPick: () => Promise<void>;
  fetchRecommendation: () => Promise<void>;
  clearError: () => void;
}

export function useDraft(): DraftHook {
  const [session, setSession] = useState<DraftState | null>(null);
  const [board, setBoard] = useState<Board | null>(null);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoadingRec, setIsLoadingRec] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Bumping this restarts the sync-status poll loop immediately. Anything that
  // changes sync state locally (starting a session with a Sleeper draft id,
  // ending the session, a reset arriving over the WebSocket) bumps it, which
  // also cancels any in-flight poll whose stale response would otherwise
  // clobber the fresh value.
  const [syncNonce, setSyncNonce] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);

  // ------------------------------------------------------------------
  // Board refresh — fetch ALL available players so position filters
  // have the full pool, not just the top 40.
  // ------------------------------------------------------------------

  const refreshBoard = useCallback(async () => {
    try {
      const b = await api.getBoard(400);
      setBoard(b);
    } catch {
      // Board fetch failures are non-fatal; keep showing the old board
    }
  }, []);

  // ------------------------------------------------------------------
  // WebSocket — connects once on mount, reconnects on drop
  // ------------------------------------------------------------------

  useEffect(() => {
    let retryTimeout: ReturnType<typeof setTimeout> | undefined;
    // Reconnect-leak guard (audit W9): closing the socket in the effect
    // cleanup fires onclose, which used to schedule a fresh connect()
    // whose timer id overwrote the one the cleanup had already cleared —
    // leaving a ghost socket reconnecting forever after unmount (visible
    // under React StrictMode's dev double-mount). `disposed` makes the
    // cleanup's close terminal.
    let disposed = false;

    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setIsConnected(true);

      ws.onmessage = (event) => {
        let msg: WsEvent;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return; // one malformed frame shouldn't kill the handler
        }

        if (msg.type === "connected") {
          if (msg.state) {
            setSession(msg.state);
            refreshBoard();
          }
        } else if (msg.type === "pick" || msg.type === "undo") {
          setSession(msg.state);
          refreshBoard();
          setRecommendation(null);
        } else if (msg.type === "reset") {
          setSession(null);
          setBoard(null);
          setRecommendation(null);
          setSyncStatus(null);
          setSyncNonce((n) => n + 1);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (!disposed) {
          retryTimeout = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      disposed = true;
      clearTimeout(retryTimeout);
      wsRef.current?.close();
    };
  }, [refreshBoard]);

  // ------------------------------------------------------------------
  // Sleeper sync status
  //
  // This used to be set exactly once, by startSession(). That meant a page
  // reload mid-draft dropped syncStatus to null while the backend poller was
  // still happily pulling picks from Sleeper — and a null sync status is
  // indistinguishable from "no sync", so the UI re-offered the manual pick
  // controls that corrupt a synced draft. Polling the backend makes the
  // client's view of sync derive from server truth instead of from whatever
  // happened to occur in this browser tab's lifetime.
  // ------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const status = await api.getSyncStatus();
        if (cancelled) return;
        // Normalise "idle" to null so consumers can treat null as "no sync"
        // without also having to special-case the idle string.
        const next = status.status === "idle" ? null : status;
        // Only commit when something actually changed. The response is a fresh
        // object every poll, so setting it unconditionally would re-render the
        // whole draft room (including a several-hundred-row board) every 5
        // seconds for no reason.
        setSyncStatus((prev) => (sameSyncStatus(prev, next) ? prev : next));
      } catch {
        // Non-fatal: keep the last known status rather than flapping the UI
        // (and, more importantly, rather than re-enabling manual pick
        // controls) because one status request failed.
      }
      if (!cancelled) timer = setTimeout(poll, SYNC_POLL_MS);
    }

    poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [syncNonce]);

  // ------------------------------------------------------------------
  // Session management
  // ------------------------------------------------------------------

  const startSession = useCallback(
    async (config: {
      league_size: number;
      my_draft_position: number;
      total_rounds: number;
      qb_slots?: number;
      rb_slots?: number;
      wr_slots?: number;
      te_slots?: number;
      flex_slots?: number;
      dst_slots?: number;
      sleeper_draft_id?: string;
    }) => {
      try {
        const { sleeper_draft_id, ...sessionConfig } = config;
        const state = await api.startSession(sessionConfig);
        setSession(state);
        setRecommendation(null);
        setSyncStatus(null);
        await refreshBoard();

        // If a Sleeper draft ID was provided, start live sync immediately
        if (sleeper_draft_id) {
          try {
            const status = await api.startSync(sleeper_draft_id);
            setSyncStatus(status.status === "idle" ? null : status);
            // Restart the poll loop so it tracks the new sync from here on
            // (and so any in-flight poll from before the sync started can't
            // land afterwards and clobber this with a stale "idle").
            setSyncNonce((n) => n + 1);
          } catch (e) {
            setError(
              `Session started, but Sleeper sync failed: ${
                e instanceof Error ? e.message : "unknown error"
              }`
            );
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to start session");
      }
    },
    [refreshBoard]
  );

  const endSession = useCallback(async () => {
    try {
      await api.endSession();
      // The backend broadcasts a "reset" event that clears this state via
      // the WebSocket handler too — clearing locally as well makes the
      // setup modal appear immediately even if the socket is mid-reconnect.
      setSession(null);
      setBoard(null);
      setRecommendation(null);
      setSyncStatus(null);
      setSyncNonce((n) => n + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to end session");
    }
  }, []);

  // ------------------------------------------------------------------
  // Picks
  // ------------------------------------------------------------------

  // Hiding the pick controls (see the isSyncing docs above) is the primary
  // defence, but these guards close the gap for anything that can reach the
  // callbacks without going through a rendered button — keyboard shortcuts,
  // an in-flight click that lands just as sync starts, a stale component.
  // A ref rather than the state value so the callbacks can stay dep-free and
  // never read a stale closure.
  const isSyncingRef = useRef(false);
  useEffect(() => {
    isSyncingRef.current = syncStatus?.status === "syncing";
  }, [syncStatus]);

  const recordPick = useCallback(async (playerId: number) => {
    if (isSyncingRef.current) {
      setError("Picks are syncing from Sleeper — manual picks are disabled.");
      return;
    }
    try {
      await api.recordPick(playerId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to record pick");
    }
  }, []);

  const undoPick = useCallback(async () => {
    if (isSyncingRef.current) {
      setError("Undo is disabled while syncing — Sleeper is the source of truth.");
      return;
    }
    try {
      await api.undoPick();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to undo pick");
    }
  }, []);

  // ------------------------------------------------------------------
  // AI recommendation
  // ------------------------------------------------------------------

  const fetchRecommendation = useCallback(async () => {
    if (!session?.is_active) return;
    setIsLoadingRec(true);
    try {
      const rec = await api.getRecommendation();
      setRecommendation(rec);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch recommendation");
    } finally {
      setIsLoadingRec(false);
    }
  }, [session?.is_active]);

  // ------------------------------------------------------------------

  return {
    session,
    board,
    recommendation,
    syncStatus,
    isSyncing: syncStatus?.status === "syncing",
    isConnected,
    isLoadingRec,
    error,
    startSession,
    endSession,
    recordPick,
    undoPick,
    fetchRecommendation,
    clearError: () => setError(null),
  };
}
