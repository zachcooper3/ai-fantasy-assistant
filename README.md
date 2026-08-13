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

### Refresh everything (recommended)

One command runs every data source in dependency order — **except ADP**, which is
excluded on purpose (see below):

```bash
py -m backend.ingestion.refresh              # everything free, ADP untouched
py -m backend.ingestion.refresh --with-ai    # + the two Claude synthesis steps
```

The split is deliberate. Of the seven steps this runs, five hit free public sources
and can be re-run as often as you like; the two synthesis steps call the Claude API
once per player and cost real money, so they never run unless you ask for them by
name. During draft week the free refresh is the one you want daily — injuries move,
synthesis output doesn't change much day to day.

Order matters and the steps are not independent — they all key off the `Player` table
however it currently stands:

```
ids ─┬→ metrics ─┬→ synthesis   [Claude]
     │    draft ─┴→ college
     └→ news  ──────→ rookies   [Claude]
```

`ids` is critical: if it fails the run stops, because everything else keys off the
sleeper_id crosswalk it produces, and continuing would just write mismatched rows on
top of a broken foundation. Everything else is best-effort and the run continues
without it. Other options:

```bash
py -m backend.ingestion.refresh --dry-run              # print the plan, run nothing
py -m backend.ingestion.refresh --only metrics news    # re-run specific steps
py -m backend.ingestion.refresh --only adp             # ADP, by explicit name only
```

The summary at the end prints the exact retry command for anything that failed.

### ADP is manual, on purpose

ADP used to auto-refresh from FantasyFootballCalculator whenever the local CSV looked
stale — on every app startup, and as the first step of `refresh`. As of 2026-08-13 it
does neither: `data/raw/fantasypros_adp.csv` only changes when you explicitly ask for
it, because ADP is the one source people hand-curate (dropping in a real FantasyPros
export, see below), and having "refresh everything" or "restart the server" silently
overwrite that choice defeated the point of curating it. The startup banner still
prints how old the on-disk data is — a stale file is visible, it's just never silently
fixed for you.

To pull fresh PPR ADP from FantasyFootballCalculator:

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

If you'd rather use a real FantasyPros export (`fantasypros.com/nfl/adp/overall.php` →
Export to CSV), it needs converting first — the raw export packs name/team/bye into one
column and formats defenses differently, which `ingest_players.py` doesn't parse on its own:

```bash
py -m backend.ingestion.convert_fantasypros_export "FantasyPros_2026_Overall_ADP_Rankings.csv"
py -m backend.ingestion.reingest
```

The first command writes a converted `data/raw/fantasypros_adp.csv`; the second loads it
and re-syncs/relinks everything that depends on the `Player` table (Sleeper IDs,
PlayerMetrics, DraftProfile) — the same four steps `fetch_adp` runs after its own fetch,
just without fetching anything first.

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

## Schedule Data

The NFL schedule (opponent/week/home-away) for the season being drafted, from `nflreadpy` —
same nflverse data source `fetch_metrics` uses above, no new library or API key. Backs the
Opportunity recommendation section's matchup context, which otherwise has no way to know the
real schedule: it's published (typically mid-May) after Claude's reliable knowledge cutoff for
that season, so asking the model to recall it directly risks a confident, wrong answer rather
than an honest "I don't know."

```bash
py -m backend.ingestion.fetch_schedule            # current draft season
py -m backend.ingestion.fetch_schedule --season 2026
```

Optional, same as steps 1-6 above — recommendations work without it, just without real
opponent context. Run it once the season's schedule is out (mid-May) and it won't change again
except rare game-time moves, so there's no need to re-run this on a schedule of its own.

---

## Troubleshooting

Three read-only diagnostics. None of them writes to the database.

**A field is empty for every player, or a metric looks wrong**

```bash
py -m backend.tools.diagnose_ingestion
```

Loads every nflverse source, prints the real error for any that fail, and — for the
ones that succeed — checks every column the ingestion actually reads against what is
present in the data, dumping all columns and a sample row when a lookup misses.

Worth running after any nflverse update. Ingestion failures here are silent by design
(each source is wrapped so one flaky feed can't take down a draft-day refresh), and a
missing column is quieter still: an absent field reads as `0.0`, so a share computes
to 0/0 and stores as `None` — indistinguishable from "this player genuinely has no
data" at every layer above.

**"Why did it recommend X over Y?"**

```bash
py -m backend.tools.explain_players "Player One" "Player Two"
```

Prints DB-wide metric coverage, then per player: the exact line the prompt renders,
whether draft capital is being shown, an explicit list of what is *not* visible to the
model, and the retrieved ChromaDB chunks. Most surprising recommendations turn out to
be a data gap rather than a reasoning failure, and this is the fastest way to tell
them apart.

**Recommendations feel slow**

```bash
py -m backend.tools.profile_recommendation            # real API call
py -m backend.tools.profile_recommendation --no-api   # free, local work only
py -m backend.tools.profile_recommendation --repeat 3 # shows what caching saves
```

Times each stage separately — DB context build, ChromaDB retrieval, the Claude call,
parsing — and reports actual token counts and generation rate. Latency is normally
dominated by *output* tokens, since generation is sequential while prompt ingestion is
not; the profiler tells you whether to trim the response shape or look elsewhere,
rather than guessing.

---

## Project Structure

```
ai-fantasy-assistant/
├── backend/
│   ├── app/          # FastAPI routes, services, AI layer
│   ├── db/           # SQLite models and player repo
│   ├── ingestion/    # CSV → SQLite ingestion, Sleeper ID sync, metrics/rookie/RAG pipeline
│   │                 #   refresh.py runs all of it in dependency order
│   ├── rag/          # ChromaDB vector store, wired into ai_service.py's recommendation prompt
│   └── tools/        # read-only diagnostics (see Troubleshooting)
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
- `CLAUDE_MODEL` — override the recommendation model's INITIAL value on boot (default:
  `claude-haiku-4-5-20251001`). The AI panel has a live Haiku/Sonnet toggle
  (`GET`/`POST /api/recommend/model`) that switches without a restart and takes over from
  there — this env var only matters for what the toggle starts on.
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
