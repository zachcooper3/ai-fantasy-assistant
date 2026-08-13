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

/** How many past recommendations to keep for review. */
const REC_HISTORY_LIMIT = 3;

/**
 * Auto-fetch the recommendation once you're this many picks away.
 *
 * One, deliberately. Each fire is a paid Claude call, and because the
 * recommendation is invalidated by every pick, a threshold of N fires roughly
 * N+1 times per turn. At 1 that means: once while the team ahead of you is
 * picking (so there's something on screen immediately), then once more when
 * you're actually on the clock — which is the one that reflects the real board
 * and is safe to draft from. At 2 it was three calls a turn, most of them
 * describing a board that no longer existed by the time you looked.
 */
const REC_PREFETCH_WITHIN_PICKS = 1;

/** localStorage key for the auto-recommend preference. */
const AUTO_RECOMMEND_KEY = "fda:auto-recommend";

/** A recommendation plus the pick it was given for, so history can be labelled. */
export interface PastRecommendation {
  recommendation: Recommendation;
  /** Overall pick number the advice was requested at. */
  pickNumber: number;
}

export interface DraftHook {
  session: DraftState | null;
  board: Board | null;
  recommendation: Recommendation | null;
  /**
   * Previous recommendations, newest first. The live recommendation used to be
   * discarded outright on every pick event, so the moment anyone drafted you
   * lost both the advice and the reasoning behind it — including your own pick,
   * where you'd most want to re-read what it said.
   */
  recHistory: PastRecommendation[];
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
  /**
   * True until the first GET /api/draft/session has settled.
   *
   * Nothing about the session is known before that, and "unknown" must not
   * be rendered as "no draft": the setup screen's primary button POSTs a new
   * session, which clears the picks of a draft already in progress. Callers
   * must show a neutral loading state while this is true rather than falling
   * through to setup. See the hydration effect below.
   */
  isHydrating: boolean;
  isLoadingRec: boolean;
  /**
   * When on, the recommendation is fetched automatically as your turn comes
   * up instead of waiting for a click. Persisted across sessions.
   */
  autoRecommend: boolean;
  setAutoRecommend: (on: boolean) => void;
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
  const [recHistory, setRecHistory] = useState<PastRecommendation[]>([]);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isHydrating, setIsHydrating] = useState(true);
  const [isLoadingRec, setIsLoadingRec] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Bumping this restarts the sync-status poll loop immediately. Anything that
  // changes sync state locally (starting a session with a Sleeper draft id,
  // ending the session, a reset arriving over the WebSocket) bumps it, which
  // also cancels any in-flight poll whose stale response would otherwise
  // clobber the fresh value.
  const [syncNonce, setSyncNonce] = useState(0);

  // Auto-recommend preference. Defaults on, but the stored value wins once
  // it's read — `prefsLoaded` gates the prefetch effect until then so a saved
  // "off" isn't briefly ignored on mount. Reading localStorage in an effect
  // (not during render) also keeps the server and first client render
  // identical, avoiding a hydration mismatch.
  const [autoRecommend, setAutoRecommendState] = useState(true);
  const [prefsLoaded, setPrefsLoaded] = useState(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(AUTO_RECOMMEND_KEY);
      if (stored !== null) setAutoRecommendState(stored === "true");
    } catch {
      // Storage can be unavailable (private mode, blocked cookies) — fall
      // back to the default rather than breaking the draft room.
    }
    setPrefsLoaded(true);
  }, []);

  const setAutoRecommend = useCallback((on: boolean) => {
    setAutoRecommendState(on);
    try {
      window.localStorage.setItem(AUTO_RECOMMEND_KEY, String(on));
    } catch {
      // Preference just won't persist; the toggle still works this session.
    }
  }, []);

  const wsRef = useRef<WebSocket | null>(null);

  // Mirrors `recommendation` so the WebSocket handler — which is created once
  // per connection and would otherwise close over a stale value — can read the
  // live one when retiring it into history.
  const recommendationRef = useRef<Recommendation | null>(null);
  useEffect(() => {
    recommendationRef.current = recommendation;
  }, [recommendation]);

  // The pick currently on the clock, readable from inside async callbacks that
  // started before it changed. Used to throw away recommendation responses the
  // draft has already moved past — see fetchRecommendation.
  const currentPickRef = useRef<number | null>(null);
  useEffect(() => {
    currentPickRef.current = session?.current_pick_number ?? null;
  }, [session?.current_pick_number]);

  // ------------------------------------------------------------------
  // Board refresh — fetch ALL available players so position filters
  // have the full pool, not just the top 40.
  // ------------------------------------------------------------------

  // Debounce handle for the reconcile refetch below.
  const reconcileTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const refreshBoard = useCallback(async () => {
    try {
      const b = await api.getBoard(400);
      setBoard(b);
    } catch {
      // Board fetch failures are non-fatal; keep showing the old board
    }
  }, []);

  /**
   * Refetch the board shortly, collapsing bursts into one request.
   *
   * Every pick used to trigger a full 400-player fetch — roughly 180 of them
   * over a 12-team draft, each re-rendering the entire table, and Sleeper sync
   * can deliver several picks in a single poll. Picks are applied locally for
   * instant feedback (see applyPickLocally); this reconciles against the
   * server afterwards so any drift is short-lived.
   */
  const scheduleReconcile = useCallback(() => {
    clearTimeout(reconcileTimer.current);
    reconcileTimer.current = setTimeout(refreshBoard, 1500);
  }, [refreshBoard]);

  useEffect(() => () => clearTimeout(reconcileTimer.current), []);

  /**
   * Drop a drafted player from the board without a round trip.
   *
   * Returns the board unchanged if the player isn't on it, so a duplicate
   * WebSocket frame can't decrement a scarcity count twice.
   */
  const applyPickLocally = useCallback((playerId: number) => {
    setBoard((prev) => {
      if (!prev) return prev;
      const drafted = prev.players.find((p) => p.id === playerId);
      if (!drafted) return prev;

      const position = drafted.position as keyof typeof prev.scarcity;
      return {
        ...prev,
        players: prev.players.filter((p) => p.id !== playerId),
        scarcity: {
          ...prev.scarcity,
          [position]: Math.max(0, (prev.scarcity[position] ?? 0) - 1),
        },
      };
    });
  }, []);

  // ------------------------------------------------------------------
  // Initial hydration
  //
  // Session state used to arrive ONLY on the WebSocket's "connected" frame,
  // which means `session` was null for as long as the handshake took. The
  // page renders the setup modal whenever there's no active session, so
  // every load of a draft in progress flashed the setup screen first — and
  // its Start Draft button POSTs a new session, which resets availability
  // and clears the pick journal. A slow socket therefore put a
  // draft-destroying button under the cursor for a few hundred milliseconds.
  //
  // The socket still hydrates too (it's the path that survives a reconnect);
  // this just means the answer no longer depends on the handshake. Whichever
  // arrives first wins, and they carry the same payload.
  // ------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const state = await api.getSession();
        if (cancelled) return;
        if (state.is_active) {
          setSession(state);
          await refreshBoard();
        }
      } catch {
        // A 404 is the ordinary "no draft yet" answer, and an unreachable
        // backend lands here too. Both end at the setup screen, which warns
        // about the latter via isConnected rather than by throwing here.
      } finally {
        if (!cancelled) setIsHydrating(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [refreshBoard]);

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

          if (msg.type === "pick") {
            // Instant: drop the drafted player and decrement their position
            // count, then reconcile in the background.
            applyPickLocally(msg.pick.player_id);
            scheduleReconcile();
          } else {
            // An undo puts a player *back* on the board, and the pick payload
            // doesn't carry their rank/adp/bye — only a refetch can restore
            // the full row, so undo stays a straight refresh.
            refreshBoard();
          }

          // Retire the current recommendation into history rather than
          // dropping it — it was computed for a board state that no longer
          // exists, so it must not stay presented as live advice, but it's
          // still worth being able to read back.
          //
          // Read via ref, not inside a setRecommendation updater: updaters
          // must stay pure, and StrictMode double-invokes them, which would
          // push the same entry into history twice.
          const retiring = recommendationRef.current;
          if (retiring) {
            setRecHistory((prev) =>
              [
                { recommendation: retiring, pickNumber: retiring.pick_number },
                ...prev.filter((h) => h.pickNumber !== retiring.pick_number),
              ].slice(0, REC_HISTORY_LIMIT)
            );
          }
          setRecommendation(null);
        } else if (msg.type === "reset") {
          setSession(null);
          setBoard(null);
          setRecommendation(null);
          // History belongs to the draft that just ended — carrying it into a
          // new session would show advice about a different board.
          setRecHistory([]);
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
  }, [refreshBoard, applyPickLocally, scheduleReconcile]);

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
        setRecHistory([]);
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
      setRecHistory([]);
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
      // Streamed so the pick appears well before the rest. The partial
      // result is rendered with empty best_available/needs/depth/alerts and
      // replaced by the full payload when it lands — see
      // api.streamRecommendation.
      const rec = await api.streamRecommendation((pick, pickNumber) => {
        // Same staleness rule as the full response below: a pick computed
        // against a board that has since moved on must not be shown, and
        // prefetching makes that routine rather than rare.
        if (pickNumber !== currentPickRef.current) return;
        setRecommendation({
          main: pick,
          best_available: [],
          needs: [],
          depth: [],
          opportunity: [],
          alerts: [],
          model: "",
          strategy: "",
          confidence: "medium",
          considered: [],
          pick_number: pickNumber,
          is_my_turn: true,
          picks_until_my_turn: 0,
          isPartial: true,
        });
      });

      // Discard advice the draft has already moved past.
      //
      // A recommendation takes several seconds. If picks land while the
      // request is in flight — which prefetching makes routine, since it fires
      // while other teams are still picking — the response describes a board
      // that no longer exists, and the players it names may since have been
      // drafted. The WebSocket handler clears `recommendation` on every pick,
      // but that happens *before* this response arrives, so without this check
      // the stale result simply repopulates the panel.
      //
      // rec.pick_number is the pick the server actually computed against,
      // which is more trustworthy than the pick we think we asked at.
      if (rec.pick_number !== currentPickRef.current) return;

      setRecommendation(rec);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch recommendation");
    } finally {
      setIsLoadingRec(false);
    }
  }, [session?.is_active]);

  /**
   * Fetch the recommendation ahead of your turn instead of on a click.
   *
   * A recommendation takes several seconds (Claude call plus retrieval), and
   * asking for one only when you're already on the clock spends that time out
   * of your pick window. Firing it a couple of picks early means the advice is
   * usually on screen by the time it's your turn.
   *
   * Fires at most once per pick number: `prefetchedForPick` is what stops the
   * effect from re-requesting every time the board or session object changes
   * identity.
   */
  const prefetchedForPick = useRef<number | null>(null);

  useEffect(() => {
    // Parked until the stored preference has been read — otherwise a saved
    // "off" would still let one automatic call through on every page load.
    if (!prefsLoaded || !autoRecommend) return;
    if (!session?.is_active || session.draft_complete) return;
    if (session.picks_until_my_turn > REC_PREFETCH_WITHIN_PICKS) return;
    if (prefetchedForPick.current === session.current_pick_number) return;
    if (recommendation || isLoadingRec) return;

    prefetchedForPick.current = session.current_pick_number;
    fetchRecommendation();
  }, [
    prefsLoaded,
    autoRecommend,
    session?.is_active,
    session?.draft_complete,
    session?.picks_until_my_turn,
    session?.current_pick_number,
    recommendation,
    isLoadingRec,
    fetchRecommendation,
    session,
  ]);

  // ------------------------------------------------------------------

  return {
    session,
    board,
    recommendation,
    recHistory,
    syncStatus,
    isSyncing: syncStatus?.status === "syncing",
    isConnected,
    isHydrating,
    isLoadingRec,
    autoRecommend,
    setAutoRecommend,
    error,
    startSession,
    endSession,
    recordPick,
    undoPick,
    fetchRecommendation,
    clearError: () => setError(null),
  };
}
