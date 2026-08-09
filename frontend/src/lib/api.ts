/**
 * Typed API client for the FastAPI backend.
 * All types mirror the Pydantic schemas in backend/app/schemas.py.
 *
 * Base URL comes from NEXT_PUBLIC_API_URL (defaults to http://localhost:8000).
 * In dev, Next.js rewrites /api/* to the backend, so we can call /api/* directly.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// Shared API token — must match the backend's APP_AUTH_TOKEN. Empty (the
// local-dev default) sends no header, matching a backend with auth
// disabled. NOTE: NEXT_PUBLIC_* values are baked into the public JS
// bundle, so anyone who can load the deployed frontend can extract this —
// it protects the backend from scanners and strangers, not from someone
// you've shared the frontend URL with. See backend/app/auth.py.
export const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

function authHeaders(): Record<string, string> {
  return API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {};
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface Player {
  id: number;
  rank: number;
  name: string;
  team: string;
  bye: number | null;
  position: string;
  pos_rank: string;
  adp: number;
  is_available: boolean;
}

export interface Scarcity {
  QB: number;
  RB: number;
  WR: number;
  TE: number;
  DST: number;
  K: number;
}

// ---------------------------------------------------------------------------
// Draft types
// ---------------------------------------------------------------------------

export interface Pick {
  pick_number: number;
  round_number: number;
  team_slot: number;
  player_id: number;
  player_name: string;
  position: string;
  nfl_team: string;
  is_mine: boolean;
}

export interface DraftState {
  is_active: boolean;
  league_size: number;
  my_draft_position: number;
  total_rounds: number;
  scoring_format: string;
  qb_slots: number;
  rb_slots: number;
  wr_slots: number;
  te_slots: number;
  flex_slots: number;
  dst_slots: number;
  current_pick_number: number;
  current_round: number;
  current_team_slot: number;
  is_my_turn: boolean;
  picks_until_my_turn: number;
  my_next_pick_number: number | null;
  draft_complete: boolean;
  /** True when the backend rehydrated this session from disk at boot. */
  was_restored: boolean;
  /** ISO timestamp of when the session originally began, if known. */
  started_at: string | null;
  picks: Pick[];
  my_roster: Pick[];
}

export interface Board {
  players: Player[];
  scarcity: Scarcity;
  picks_until_my_turn: number;
  is_my_turn: boolean;
}

// ---------------------------------------------------------------------------
// Recommendation types
// ---------------------------------------------------------------------------

/** Which Recommendation section(s) a player appears in — see Recommendation's
 * docstring. A player can carry more than one tag; overlap is meaningful
 * (e.g. "this is both the main pick and your best value on the board") and
 * is never deduplicated away. */
export type SectionTag = "main" | "best_available" | "needs" | "depth";

export interface PickSuggestion {
  player_id: number;
  player_name: string;
  position: string;
  adp: number;
  reasoning: string;
  /** Which Recommendation section(s) this entry appears in. */
  tags: SectionTag[];
  /**
   * Whether this player is expected to survive to your next turn, computed
   * server-side from ADP vs. the horizon pick. Empty on the last pick of a
   * draft, where there is no next turn to survive to.
   *
   * Deliberately a code rather than the prompt's own wording: that wording
   * has been rewritten once already (it used to read "GONE", which the model
   * mistook for "unavailable") and the UI must not move with it.
   */
  survival: Survival;
}

export type Survival = "take_now" | "might_last" | "will_last" | "";

export type Confidence = "high" | "medium" | "low";

export interface Recommendation {
  /**
   * The model's single synthesized pick — the only entry backed by real
   * reasoning (tiers, opportunity cost, VOR, news). best_available/needs/
   * depth below are NOT model output: they're computed server-side straight
   * from the same board data ("cheapest by ADP" / "fills your open slot" is
   * a lookup, not a judgement call), specifically to keep generation cost
   * down. See the backend's RecommendationResult docstring.
   */
  main: PickSuggestion;
  /** Up to 2, cheapest by ADP regardless of roster need. */
  best_available: PickSuggestion[];
  /**
   * Up to 2, the realistic slate at your highest-priority open starting
   * slot. Empty once every starting slot is filled — see `depth`.
   */
  needs: PickSuggestion[];
  /**
   * 0 or 1. A QB/TE stash pick, only ever present when `needs` is empty —
   * nowhere else does this app ever suggest a second QB/TE.
   */
  depth: PickSuggestion[];
  alerts: string[];
  model: string;
  /** One sentence on the roster's shape and what this pick does about it. */
  strategy: string;
  /** How clear-cut the call is. The no-AI ADP fallback always reports "low". */
  confidence: Confidence;
  /**
   * One line per must-evaluate player: taken, or passed and why. Exists to
   * make omission visible — every live mis-recommendation so far has been a
   * top-of-board player never mentioned at all, rather than one rejected on
   * the merits. Empty on the ADP fallback, which evaluates nothing. Not
   * rendered anywhere in the UI — a server-side reliability signal only.
   */
  considered: string[];
  pick_number: number;
  is_my_turn: boolean;
  picks_until_my_turn: number;
  /**
   * True while only `main` has arrived and best_available/needs/depth,
   * verdicts and alerts are still generating. Set client-side by the
   * streaming path, never sent by the server — it describes how much of the
   * response we have, not anything about the recommendation itself.
   */
  isPartial?: boolean;
}

export interface ScarcityAlert {
  position: string;
  available: number;
  tier: "critical" | "low" | "ok";
  message: string;
}

export interface ScarcityAnalysis {
  alerts: ScarcityAlert[];
  available_counts: Record<string, number>;
}

export interface SyncStatus {
  status: "idle" | "syncing" | "complete" | "error";
  draft_id: string | null;
  synced_pick_count: number;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Sleeper prefill types
// ---------------------------------------------------------------------------

export interface SleeperPrefill {
  league_size: number | null;
  total_rounds: number | null;
  my_draft_position: number | null;
  qb_slots: number | null;
  rb_slots: number | null;
  wr_slots: number | null;
  te_slots: number | null;
  flex_slots: number | null;
  dst_slots: number | null;
  // Informational only — this app's ADP/rankings/AI prompt are PPR-only
  // regardless of a league's actual scoring, so this is never fed back
  // into a startSession() call. See SleeperPrefillResponse's docstring
  // in backend/app/schemas.py.
  detected_scoring_format: string | null;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// WebSocket event types
// ---------------------------------------------------------------------------

export type WsEvent =
  | { type: "connected"; state: DraftState | null }
  | { type: "pick";  pick: Pick; state: DraftState }
  | { type: "undo";  pick: Pick; state: DraftState }
  | { type: "reset" };

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `POST ${path} → ${res.status}`);
  }
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `DELETE ${path} → ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

export const api = {
  // Players
  players: (position?: string, availableOnly = false, limit = 50) =>
    get<Player[]>(
      `/api/players?available_only=${availableOnly}&limit=${limit}` +
      (position ? `&position=${position}` : "")
    ),

  // Draft session
  startSession: (config: {
    league_size: number;
    my_draft_position: number;
    total_rounds: number;
    scoring_format?: string;
    qb_slots?: number;
    rb_slots?: number;
    wr_slots?: number;
    te_slots?: number;
    flex_slots?: number;
    dst_slots?: number;
  }) => post<DraftState>("/api/draft/session", config),

  getSession: () => get<DraftState>("/api/draft/session"),

  endSession: () => del<void>("/api/draft/session"),

  // Picks
  recordPick: (playerId: number) =>
    post<Pick>("/api/draft/pick", { player_id: playerId }),

  undoPick: () => del<Pick>("/api/draft/pick"),

  // Board
  getBoard: (limit = 40) => get<Board>(`/api/draft/board?limit=${limit}`),

  // Recommendations
  getRecommendation: () => get<Recommendation>("/api/recommend/pick"),

  /**
   * Streams the recommendation, calling `onPick` as soon as the pick itself
   * has been generated and resolving with the full response.
   *
   * Generation is sequential and output-bound, so the plain endpoint shows
   * nothing until the whole response lands even though `main` — near the
   * front of the schema — finished much earlier. This does not make the
   * model faster; it stops hiding the answer until best_available/needs/
   * depth, verdicts and alerts have finished.
   *
   * Uses fetch + a ReadableStream rather than EventSource: EventSource
   * cannot send the Authorization header this API requires, and silently
   * reconnects on completion, which would re-run a paid Claude call every
   * time the stream closed.
   */
  streamRecommendation: async (
    onPick: (pick: PickSuggestion, pickNumber: number) => void,
  ): Promise<Recommendation> => {
    const res = await fetch(`${BASE}/api/recommend/pick/stream`, {
      headers: API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {},
    });
    if (!res.ok || !res.body) {
      throw new Error(`Recommendation stream failed (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let complete: Recommendation | null = null;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line. Keep any trailing partial
      // frame in the buffer for the next read.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const event = /^event: (.+)$/m.exec(frame)?.[1];
        const data = /^data: (.+)$/m.exec(frame)?.[1];
        if (!event || !data) continue;
        const payload = JSON.parse(data);
        if (event === "pick") {
          onPick(payload.main as PickSuggestion, payload.pick_number);
        } else if (event === "complete") {
          complete = payload as Recommendation;
        } else if (event === "error") {
          throw new Error(payload.detail ?? "Recommendation failed");
        }
      }
    }

    if (!complete) throw new Error("Recommendation stream ended without a result");
    return complete;
  },

  getScarcity: () => get<ScarcityAnalysis>("/api/recommend/scarcity"),

  getHandcuff: (playerId: number) =>
    get<Player>(`/api/recommend/handcuff?player_id=${playerId}`),

  // Sleeper sync
  startSync: (draftId: string) =>
    post<SyncStatus>("/api/sync/start", { draft_id: draftId }),

  stopSync: () => del<void>("/api/sync/stop"),

  getSyncStatus: () => get<SyncStatus>("/api/sync/status"),

  // Sleeper prefill — best-effort settings detection for setup
  getSleeperPrefill: (draftId: string, username?: string) =>
    get<SleeperPrefill>(
      `/api/sleeper/prefill?draft_id=${encodeURIComponent(draftId)}` +
      (username ? `&username=${encodeURIComponent(username)}` : "")
    ),
};
