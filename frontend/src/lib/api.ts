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
  /**
   * Sleeper's designation — "IR", "Out", "PUP", "Suspended", "Questionable",
   * "Doubtful", "NA" — or null when healthy. Populated by
   * scripts/sync_sleeper_ids.py; see Player.injury_status in
   * backend/db/models.py. Optional here because the field post-dates this
   * client and a cached older response won't carry it.
   */
  injury_status?: string | null;
}

/**
 * One player's PlayerMetrics row — see backend/db/models.py.
 *
 * Every metric is optional and a missing one means "not known", never zero:
 * nflverse coverage is sparse by position and by player, so the drawer omits
 * rows it has no value for rather than rendering a confident 0.
 *
 * `season` / `through_week` / `games_played` are required because they're how
 * you judge whether the rest is signal — a 17-game row and a 3-game row look
 * identical otherwise.
 */
export interface PlayerMetrics {
  season: number;
  through_week: number;
  games_played: number;
  /** The team these numbers were earned with — not necessarily today's team. */
  team: string | null;

  targets_per_game: number | null;
  carries_per_game: number | null;
  red_zone_touches_per_game: number | null;
  /** 0-1 */
  snap_pct: number | null;
  /** 0-1 */
  target_share: number | null;
  /** 0-1 */
  carry_share: number | null;

  yards_per_target: number | null;
  yards_per_carry: number | null;
  yac_per_reception: number | null;
  racr: number | null;
  /** 0-1 */
  catch_rate: number | null;

  /** 0-1 */
  team_pass_rate: number | null;
  /** 1 = starter at the position */
  depth_chart_rank: number | null;

  /** PPR, per game */
  fantasy_points_avg: number | null;
  /** Week-to-week PPR standard deviation — the boom/bust measure. */
  fantasy_points_stdev: number | null;
  injury_report_appearances: number;
  games_missed: number;

  /** Last 3 weeks minus season average; positive = rising role. */
  target_share_trend: number | null;
  snap_pct_trend: number | null;
  /** Rank change over the last 3 weeks; NEGATIVE = moving up the chart. */
  depth_chart_trend: number | null;
  is_rookie_or_second_year: boolean;
}

/** Draft capital plus final-college-season production — see DraftProfile. */
export interface DraftProfile {
  draft_year: number;
  draft_round: number | null;
  draft_pick: number | null;
  draft_team: string | null;
  college: string | null;

  college_season: number | null;
  passing_yards: number | null;
  passing_td: number | null;
  interceptions_thrown: number | null;
  rushing_yards: number | null;
  rushing_td: number | null;
  carries: number | null;
  receiving_yards: number | null;
  receiving_td: number | null;
  receptions: number | null;
}

export interface ScheduleGame {
  week: number;
  opponent: string;
  is_home: boolean;
}

/**
 * Everything the app knows about one player, for the detail drawer.
 *
 * `metrics` is null for anyone with no NFL snaps (rookies, by construction);
 * `draft_profile` is null for undrafted players. Both are ordinary states, not
 * errors. `schedule` is empty when the schedule hasn't been ingested for the
 * inferred season — treat that as unknown, not as a bye.
 */
export interface PlayerDetail {
  player: Player;
  metrics: PlayerMetrics | null;
  draft_profile: DraftProfile | null;
  schedule: ScheduleGame[];
  season: number | null;
}

export interface Scarcity {
  QB: number;
  RB: number;
  WR: number;
  TE: number;
  DST: number;
  K: number;
}

export interface ModelChoice {
  model: string;       // "haiku" | "sonnet" | "custom"
  choices: string[];   // what setModel accepts — currently ["haiku", "sonnet"]
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
export type SectionTag = "main" | "best_available" | "needs" | "depth" | "opportunity";

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
   * The model's single synthesized pick — the primary entry backed by real
   * reasoning (tiers, opportunity cost, VOR, news). best_available/needs/
   * depth below are NOT model output: they're computed server-side straight
   * from the same board data ("cheapest by ADP" / "fills your open slot" is
   * a lookup, not a judgement call), specifically to keep generation cost
   * down. `opportunity` below is the one other model-authored field. See
   * the backend's RecommendationResult docstring.
   */
  main: PickSuggestion;
  /** Up to 2, cheapest by ADP regardless of roster need. */
  best_available: PickSuggestion[];
  /**
   * Up to 2, the realistic slate at your highest-priority open starting
   * slot. Empty once the skill lineup is filled, except it can resume
   * recommending DST/K once those become the last open slots — see the
   * backend's _needs docstring. Can be non-empty alongside `depth`.
   */
  needs: PickSuggestion[];
  /**
   * 0-2: one QB stash and one TE stash, once the skill starting lineup is
   * filled — one candidate per position, not just whichever of the two is
   * cheaper by ADP overall. Nowhere else does this app ever suggest a
   * second QB/TE. Can be non-empty alongside `needs`.
   */
  depth: PickSuggestion[];
  /**
   * 0-1: an RB/WR/TE whose real role this season looks bigger than his
   * ADP/tier reflects — usage trend, roster departures/arrivals, real
   * upcoming schedule, plus general football knowledge (depth-chart
   * competition, scheme/coaching changes, contract situations) `main` is
   * NOT allowed to use. Unlike best_available/needs/depth, this IS model
   * output — there's no formula for it — so it's empty whenever the AI is
   * unavailable or names someone invalid. See the backend's
   * RecommendationResult.opportunity docstring.
   */
  opportunity: PickSuggestion[];
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

  /**
   * Everything known about one player, in one round trip — see the backend's
   * PlayerDetailResponse. Backs the detail drawer, which opens on a click
   * mid-draft, so this deliberately isn't three separate fetches.
   */
  playerDetail: (playerId: number) =>
    get<PlayerDetail>(`/api/players/${playerId}/detail`),

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

  // AI model toggle — "haiku" | "sonnet" (or "custom" on GET, if
  // CLAUDE_MODEL was overridden to something neither option matches).
  // Switches live, no backend restart; persists across one while a draft
  // session is active. See backend/app/api/recommendations.py.
  getModel: () => get<ModelChoice>("/api/recommend/model"),

  setModel: (model: "haiku" | "sonnet") =>
    post<ModelChoice>("/api/recommend/model", { model }),

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
