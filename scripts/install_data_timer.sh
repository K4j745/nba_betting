#!/usr/bin/env bash
# Instaluje timer systemd (użytkownika) synchronizujący logi NBA do brancha `data`.
# Użycie: scripts/install_data_timer.sh        (z katalogu repo albo dowolnego)
#         scripts/install_data_timer.sh --uninstall
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl --user disable --now nba-data-sync.timer || true
  rm -f "$UNIT_DIR/nba-data-sync.service" "$UNIT_DIR/nba-data-sync.timer"
  systemctl --user daemon-reload
  echo "Odinstalowano."
  exit 0
fi

[[ -x "$REPO/.venv/bin/python" ]] || { echo "Brak $REPO/.venv — utwórz venv i zainstaluj requirements.txt"; exit 1; }
mkdir -p "$UNIT_DIR"
sed "s|@REPO@|$REPO|g" "$REPO/deploy/systemd/nba-data-sync.service" > "$UNIT_DIR/nba-data-sync.service"
cp "$REPO/deploy/systemd/nba-data-sync.timer" "$UNIT_DIR/nba-data-sync.timer"
systemctl --user daemon-reload
systemctl --user enable --now nba-data-sync.timer
systemctl --user list-timers nba-data-sync.timer --no-pager
echo "Logi: journalctl --user -u nba-data-sync.service"
