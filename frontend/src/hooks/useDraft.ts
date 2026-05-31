"use client";
/**
 * Central state hook for the draft room.
 *
 * Manages:
 * - Draft session state (from the API)
 * - The big board (available players + scarcity)
 * - AI recommendation
 * - WebSocket connection for real-time pick updates
 *
 * All components read from this hook — there's no global store needed.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, Board, DraftState, Recommendation, WsEvent } from "@/lib/api";

const WS_URL =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000")
    .replace(/^http/, "ws") + "/ws/draft";

export interface DraftHook {
  session: DraftState | null;
  board: Board | null;
  recommendation: Recommendation | null;
  isConnected: boolean;
  isLoadingRec: boolean;
  error: string | null;
  startSession: (config: {
    league_size: number;
    my_draft_position: number;
    total_rounds: number;
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
          // Clear stale recommendation after a pick
          setRecommendation(null);
        } else if (msg.type === "reset") {
          setSession(null);
          setBoard(null);
          setRecommendation(null);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        // Reconnect after 3 s
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
    async (config: { league_size: number; my_draft_position: number; total_rounds: number }) => {
      try {
        const state = await api.startSession(config);
        setSession(state);
        setRecommendation(null);
        await refreshBoard();
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
      // Board + session refresh via the WebSocket "pick" event
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to record pick");
    }
  }, []);

  const undoPick = useCallback(async () => {
    try {
      await api.undoPick();
      // Session + board refresh via the WebSocket "undo" event
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
