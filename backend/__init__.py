"""
Loads .env into the process environment before any other backend module
reads os.getenv() for config (ANTHROPIC_API_KEY, DB_PATH, CORS_ORIGINS,
SYNTHESIS_MODEL, etc.).

This has to live in the top-level package's __init__.py specifically:
every entrypoint in this app — the FastAPI server (`uvicorn
backend.app.main:app`) and every standalone ingestion script (`py -m
backend.ingestion.*`) — imports something under the `backend` package
before it reads its first env var, so this file is guaranteed to run
first regardless of which one starts the process.

Without this, .env was never being loaded at all — confirmed no dotenv
import existed anywhere in the codebase. os.getenv() only ever saw real
OS environment variables, so a real ANTHROPIC_API_KEY set in .env (per
the README's setup steps) was silently ignored, and recommendations ran
in ADP-fallback mode with no indication why — the placeholder-key guard
in ai_service.py only catches the ".env.example wasn't edited" case, not
"the file was edited correctly but never got loaded" (this bug).

override=False (the default) so a real environment variable set outside
.env — e.g. by a hosting platform like Render at deploy time — always
wins over whatever .env says, and a missing .env file is a silent no-op
rather than an error (the app is meant to run without one).

Author: Zach Cooper
"""

from dotenv import load_dotenv

load_dotenv()
