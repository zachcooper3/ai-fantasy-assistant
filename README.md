# AI Fantasy Football Draft Assistant

AI-powered draft day co-pilot for Sleeper PPR leagues. Built with FastAPI (Python) + Next.js.

---

## Prerequisites

- Python 3.10+
- Node.js 18+

`data/` and `venv/` are both gitignored, so a fresh clone has neither — the
steps below create them. (If you're following this on a machine that
already has `venv/` set up, skip straight to step 2.)

---

## First-Time Setup

### 1. Create a virtual environment and install Python dependencies

```bash
py -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Fetch the initial player data

Pulls current PPR ADP from FantasyFootballCalculator, writes
`data/raw/fantasypros_adp.csv`, loads it into SQLite (`data/fantasy.db`),
and syncs Sleeper player IDs — all in one step:

```bash
py -m backend.ingestion.fetch_adp
```

(If you'd rather use a FantasyPros export you already have, see
"Refreshing Player Data" below instead.)

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Add your Anthropic API key (optional but recommended)

Copy `.env.example` to `.env` and fill in your key:

```bash
copy .env.example .env
```

Then edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Without a key the app still works — recommendations fall back to best-available-by-ADP.
The server prints a startup banner confirming whether a key was actually detected, so
you don't have to guess or dig through logs.

---

## Running the App

### Quick start

```bash
./dev.sh
```

Runs backend and frontend together in one terminal; `Ctrl+C` stops both. Backend runs without
auto-reload by default — see the note below before adding `--reload` back.

### Or, two terminals (useful if you want backend/frontend logs kept separate)

**Terminal 1 — Backend**

```bash
venv/Scripts/activate
uvicorn backend.app.main:app
```

API runs at `http://localhost:8000`
Interactive API docs at `http://localhost:8000/docs`

**A note on `--reload`:** both commands above intentionally omit it. On Windows, `--reload` runs
the app inside a child process spawned by uvicorn's reload supervisor — and the first real
ChromaDB query (your first "Get pick" of a session) makes ChromaDB's embedding backend try to
spawn its own worker process, a grandchild attaching across two layers of process supervision.
Confirmed live: this crashed with a Windows multiprocessing `WinError 87` right after a
recommendation. Since this app's real purpose is running live *during* a draft — not being
edited while a draft is in progress — `--reload`'s only value is while actively developing.
Add it back for that (`./dev.sh` via `DEV_RELOAD=1 ./dev.sh`, or `uvicorn backend.app.main:app
--reload` directly) and drop it again once you're just running the app.

**Terminal 2 — Frontend**

```bash
cd frontend
npm run dev
```

App runs at `http://localhost:3000`

---

## Draft Day Workflow

1. Open `http://localhost:3000`
2. Set your league size and draft slot in the setup screen
3. As each pick is made, click **Draft** next to the player on the big board
4. Click **Get pick** in the AI panel for a recommendation when you're on the clock
5. Use **Undo** in the status bar to reverse a mis-entered pick

---

## Refreshing Player Data

ADP data is fetched automatically from [FantasyFootballCalculator](https://fantasyfootballcalculator.com) on startup if the local CSV is older than 7 days. No manual action needed in most cases.

To force a manual refresh at any time:

```bash
py -m backend.ingestion.fetch_adp
```

This fetches fresh PPR ADP, overwrites `data/raw/fantasypros_adp.csv`, reloads the database,
and re-syncs Sleeper player IDs (a full reload wipes them, so this happens automatically —
you don't need a separate step). It also validates the response before writing: a fetch that
returns suspiciously few players is rejected rather than overwriting good data, and every run
logs an old-count → new-count diff so you can sanity-check the result at a glance. Options:

```bash
py -m backend.ingestion.fetch_adp --year 2026   # specific season
py -m backend.ingestion.fetch_adp --teams 10    # non-12-team league
py -m backend.ingestion.fetch_adp --no-ingest   # write CSV only, skip DB reload
```

**Note:** FantasyFootballCalculator typically publishes data starting in July/August once community drafts begin. Running before then will show a warning and keep existing data.

If you have a FantasyPros CSV you'd prefer to use instead, drop it into `data/raw/fantasypros_adp.csv` and run:

```bash
py -m backend.ingestion.ingest_players
py -m backend.ingestion.sync_sleeper_ids
py -m backend.db.metrics_repo
```

(Three steps here, not one — `ingest_players` alone doesn't know to re-sync Sleeper IDs or
relink PlayerMetrics the way `fetch_adp` does, since it's also used standalone for CSVs that
have nothing to do with Sleeper. The third step only matters if you've already run
`fetch_metrics`/`fetch_synthesis` — skip it on a fresh setup with no metrics yet. Don't skip it
on a repeat run, though: `ingest_players` reassigns every player's internal ID on each reload,
which silently detaches any existing PlayerMetrics rows from the right player until this step
re-links them.)

---

## Player Analytics & News

Recommendations are grounded in more than ADP — a three-stage pipeline builds a ChromaDB
layer of real player news and Claude-synthesized analysis, retrieved into the recommendation
prompt for whichever players are actually under consideration. None of this is required for
the app to run; every stage degrades gracefully if skipped or if a data source is unavailable.

**1. Compute player metrics** (opportunity/volume, efficiency, team context, consistency &
risk, forward-looking signals) from `nflreadpy` (nflverse's stats package):

```bash
py -m backend.ingestion.fetch_metrics            # current season
py -m backend.ingestion.fetch_metrics --season 2025
```

**2. Ingest "what happened"** — factual event reporting: Sleeper's `injury_status` field and
RotoWire's NFL news RSS feed (both explicitly free to reuse this way):

```bash
py -m backend.ingestion.chunker
```

**3. Generate "what it means"** — a short Claude-written scouting note per player, synthesized
only from that player's own computed metrics (never invented stats). Requires
`ANTHROPIC_API_KEY`; one Claude call per player, so `--limit` is worth using for a smoke test:

```bash
py -m backend.ingestion.fetch_synthesis --limit 5 --dry-run   # preview a few notes first
py -m backend.ingestion.fetch_synthesis                       # generate for everyone, write to Chroma
```

Once populated, `backend/app/services/ai_service.py` automatically retrieves both chunk types
for the top candidate players on every recommendation — no extra step needed. If ChromaDB is
empty or unavailable, recommendations just proceed without that section.

**Rookies** structurally can never have a `PlayerMetrics` row (no prior NFL season to compute
it from), so steps 4-6 below give them their own equivalent data instead: draft capital and
college production stand in for the metrics a veteran would have, and get their own
Claude-written note. Run these once per draft class (i.e. once a year, right after the NFL
draft) — same "doesn't change intra-day" reasoning as steps 1-3 don't run on every server boot.

**4. Fetch rookie draft capital** (round/pick/team/college) from `nflreadpy`:

```bash
py -m backend.ingestion.fetch_draft_profiles --dry-run   # preview matches first
py -m backend.ingestion.fetch_draft_profiles             # write to the database
```

**5. Enrich with final-college-season production** (passing/rushing/receiving) from
CollegeFootballData.com. Requires a free `CFBD_API_KEY` (see `.env.example` — no credit card,
just an email signup); no-ops with a clear message if unset. Must run *after* step 4 — it only
enriches players who already have a draft profile to attach stats to:

```bash
py -m backend.ingestion.fetch_college_stats --dry-run   # preview matches first
py -m backend.ingestion.fetch_college_stats             # write to the database
```

**6. Generate rookie-specific "what it means" notes** — same idea as step 3, but grounded only
in draft capital + college production instead of NFL metrics, for players who don't have a
`PlayerMetrics` row yet. Requires `ANTHROPIC_API_KEY`; one real Claude call per eligible rookie
— note that `--dry-run` here only skips the ChromaDB write, **not** the API call, so use
`--limit`/`--sleeper-id` for a cheap smoke test before running the full batch:

```bash
py -m backend.ingestion.fetch_rookie_synthesis --limit 3 --dry-run   # preview a few (still costs tokens)
py -m backend.ingestion.fetch_rookie_synthesis                       # generate for everyone, write to Chroma
```

Once a rookie's first real NFL season lands and `fetch_metrics.py`/`fetch_synthesis.py` produce
a real `PlayerMetrics` row and note for them, that note automatically takes over — both scripts
share the same dedupe key, so no manual "graduation" step is needed.

You can sanity-check what's actually in the collection with:

```bash
py -m backend.rag.vector_store
```

This re-ingests "what happened" chunks, then drops you into an interactive prompt. Ask about a
specific player (e.g. "Is Puka Nacua injured?"), not a ranking ("who's the best WR?") — the
collection holds narrow factual/analytical snippets, not leaderboards; ADP/SQL handles ranking.

**If you ever change what a chunk-producing function outputs** — a reworded template, a new
field, a removed source — run this once with `--reset` first:

```bash
py -m backend.rag.vector_store --reset
```

Ingestion only ever upserts what it's given; it never deletes chunks an old code path used to
produce. Without a reset, old- and new-format chunks sit side by side indefinitely and can
outcompete the current chunks in similarity search (this happened in practice: a pre-rewrite
collection had 382 leftover chunks in the original ADP-CSV-row format, and generic queries kept
matching those instead of the real Sleeper/RotoWire content).

---

## Project Structure

```
ai-fantasy-assistant/
├── backend/
│   ├── app/          # FastAPI routes, services, AI layer
│   ├── db/           # SQLite models and player repo
│   ├── ingestion/    # CSV → SQLite ingestion, Sleeper ID sync, metrics/rookie/RAG pipeline
│   └── rag/          # ChromaDB vector store, wired into ai_service.py's recommendation prompt
├── data/             # gitignored — created by first-time setup, not cloned
│   ├── raw/          # Source CSVs (fantasypros_adp.csv)
│   └── fantasy.db    # SQLite database (auto-created)
├── chroma_db/        # gitignored — local vector store, created by the analytics/news pipeline
├── frontend/         # Next.js draft room UI
├── dev.sh            # runs backend + frontend together
├── .env              # API keys (not committed)
├── .env.example
├── .gitattributes    # normalizes line endings to LF
├── requirements.txt
└── README.md
```

## Configuration

All runtime config is read from environment variables (see `.env.example` for the full list
and defaults) — nothing requires a code change to adjust, including for deployment:

- `ANTHROPIC_API_KEY` — Claude API key. Unset (or left as the `.env.example` placeholder) →
  recommendations fall back to best-available-by-ADP. The startup banner tells you which mode
  you're in.
- `CLAUDE_MODEL` — override the recommendation model (default: `claude-haiku-4-5-20251001`).
- `SYNTHESIS_MODEL` — override the model used for batch "what it means" note generation
  (`fetch_synthesis.py` and `fetch_rookie_synthesis.py`). Defaults to `CLAUDE_MODEL`; safe to
  point at a pricier model since it runs offline, not on the clock.
- `CFBD_API_KEY` — CollegeFootballData.com key for rookie college production
  (`fetch_college_stats.py`). Free, no credit card — request at
  https://collegefootballdata.com/key. Unset → that step no-ops with a clear message; rookies
  still get a note from draft capital alone.
- `DB_PATH` — SQLite file location (default: `data/fantasy.db`).
- `CORS_ORIGINS` — comma-separated allowed frontend origins. Defaults to the local dev ports;
  set this to your deployed frontend URL when you go live instead of editing `main.py`.
- `DEV_RELOAD` — set to `1` to have `./dev.sh` pass `--reload` to uvicorn (auto-restart on code
  changes). Off by default — see the `--reload` note under "Running the App" for why.
