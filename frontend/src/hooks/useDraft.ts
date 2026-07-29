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

export interface DraftHook {
  session: DraftState | null;
  board: Board | null;
  recommendation: Recommendation | null;
  syncStatus: SyncStatus | null;
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
    let retryTimeout: ReturnType<typeof setTimeout>;

    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setIsConnected(true);

      ws.onmessage = (event) => {
        const msg: WsEvent = JSON.parse(event.data);

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
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        retryTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      clearTimeout(retryTimeout);
      wsRef.current?.close();
    };
  }, [refreshBoard]);

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
            setSyncStatus(status);
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

  // ------------------------------------------------------------------
  // Picks
  // ------------------------------------------------------------------

  const recordPick = useCallback(async (playerId: number) => {
    try {
      await api.recordPick(playerId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to record pick");
    }
  }, []);

  const undoPick = useCallback(async () => {
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
    isConnected,
    isLoadingRec,
    error,
    startSession,
    recordPick,
    undoPick,
    fetchRecommendation,
    clearError: () => setError(null),
  };
}
