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

export interface PickSuggestion {
  player_id: number;
  player_name: string;
  position: string;
  adp: number;
  reasoning: string;
}

export interface Recommendation {
  recommendation: PickSuggestion;
  alternatives: PickSuggestion[];
  alerts: string[];
  model: string;
  pick_number: number;
  is_my_turn: boolean;
  picks_until_my_turn: number;
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
