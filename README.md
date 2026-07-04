# AI Fantasy Football Draft Assistant

AI-powered draft day co-pilot for Sleeper PPR leagues. Built with FastAPI (Python) + Next.js.

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- A virtual environment already set up at `venv/`

---

## First-Time Setup

### 1. Install Python dependencies

```bash
venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Seed the player database

Downloads the FantasyPros ADP CSV into SQLite. Run this once, and again whenever you have a fresh CSV.

```bash
python -m backend.ingestion.ingest_players
```

This creates `data/fantasy.db` with all players loaded.

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

---

## Running the App

Open **two terminals**, both from the repo root (`ai-fantasy-assistant/`).

### Terminal 1 — Backend

```bash
venv/Scripts/activate
uvicorn backend.app.main:app --reload
```

API runs at `http://localhost:8000`
Interactive API docs at `http://localhost:8000/docs`

### Terminal 2 — Frontend

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
python -m backend.ingestion.fetch_adp
```

This fetches fresh PPR ADP, overwrites `data/raw/fantasypros_adp.csv`, and reloads the database. Options:

```bash
python -m backend.ingestion.fetch_adp --year 2026   # specific season
python -m backend.ingestion.fetch_adp --teams 10    # non-12-team league
python -m backend.ingestion.fetch_adp --no-ingest   # write CSV only, skip DB reload
```

**Note:** FantasyFootballCalculator typically publishes data starting in July/August once community drafts begin. Running before then will show a warning and keep existing data.

If you have a FantasyPros CSV you'd prefer to use instead, drop it into `data/raw/fantasypros_adp.csv` and run:

```bash
python -m backend.ingestion.ingest_players
```

---

## Project Structure

```
ai-fantasy-assistant/
├── backend/
│   ├── app/          # FastAPI routes, services, AI layer
│   ├── db/           # SQLite models and player repo
│   └── ingestion/    # CSV → SQLite + ChromaDB ingestion
├── data/
│   ├── raw/          # Source CSVs (fantasypros_adp.csv)
│   └── fantasy.db    # SQLite database (auto-created)
├── frontend/         # Next.js draft room UI
├── .env              # API keys (not committed)
├── requirements.txt
└── README.md
```
