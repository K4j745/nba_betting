# nba_betting/scripts/sync_nba_data.py
"""
Synchronizacja logów NBA z laptopa do repo (branch `data`).

stats.nba.com nie odpowiada runnerom GitHub Actions (probe 2026-10-30), więc dane
pobiera laptop, a Actions czyta je z brancha `data`.

Co robi:
  1. history.load_logs(refresh_current=True) — 4 zapytania (bieżący sezon P/T, RS/PO);
     zakończone sezony pobierane tylko, jeśli brakuje ich w data/raw.
  2. Buduje commit z samymi plikami data/raw/*.csv.gz (bez rodzica) i wypycha go
     z --force na refs/heads/data. Branch ma zawsze jeden commit → repo nie puchnie.
     Katalog roboczy, index i bieżący branch NIE są ruszane (plumbing gita).

Użycie:
  python scripts/sync_nba_data.py            # pobierz + wypchnij
  python scripts/sync_nba_data.py --no-push  # tylko pobierz (test)
Uruchamiane przez timer systemd: deploy/systemd/nba-data-sync.{service,timer}.
"""
import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from history import RAW_DIR, load_logs  # noqa: E402

logger = logging.getLogger("sync_nba_data")
DATA_BRANCH = "data"


def _git(*args: str, env: dict | None = None, input: str | None = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, env=env, input=input,
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def build_data_commit(message: str) -> str:
    """Commit (bez rodzica) z plikami data/raw w korzeniu drzewa. Zwraca SHA."""
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    if not files:
        raise RuntimeError(f"Brak plików w {RAW_DIR}")
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        for f in files:
            sha = _git("hash-object", "-w", str(f))
            _git("update-index", "--add", "--cacheinfo", f"100644,{sha},{f.name}", env=env)
        tree = _git("write-tree", env=env)
    return _git("commit-tree", tree, "-m", message)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    players, _ = load_logs(refresh_current=True)
    last = players["GAME_DATE"].max().date()
    logger.info("Logi: %d wierszy, ostatni mecz %s", len(players), last)

    if args.no_push:
        return

    commit = build_data_commit(f"data: logi NBA do {last} [skip ci]")
    _git("push", "--force", "origin", f"{commit}:refs/heads/{DATA_BRANCH}")
    logger.info("Wypchnięto %s → %s (ostatni mecz %s)", commit[:8], DATA_BRANCH, last)


if __name__ == "__main__":
    main()
