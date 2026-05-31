# GitHub Actions — konfiguracja automatycznego pipeline

Pipeline działa na GitHub Actions: generuje kupon o 20:00 CEST, weryfikuje wyniki
o 09:00 CEST, i po każdym uruchomieniu deployuje dashboard na GitHub Pages.

**Dashboard URL**: `https://K4j745.github.io/nba_betting/dashboard.html`

---

## Krok 1 — Dodaj GitHub Secret: ODDS_API_KEY

1. Wejdź na `https://github.com/K4j745/nba_betting/settings/secrets/actions`
2. **New repository secret**
3. Name: `ODDS_API_KEY`
4. Value: twój klucz API z The Odds API
5. **Add secret**

---

## Krok 2 — Włącz GitHub Pages

1. Wejdź na `https://github.com/K4j745/nba_betting/settings/pages`
2. **Source** → `GitHub Actions` (nie "Deploy from a branch")
3. Zapisz

GitHub Pages zostanie automatycznie skonfigurowane po pierwszym uruchomieniu workflow.

> **Uwaga**: GitHub Pages działa bezpłatnie tylko dla **publicznych repozytoriów**.
> Dla prywatnych wymagany jest plan GitHub Pro/Team.

---

## Krok 3 — Wypchnij zmiany do repo

```bash
cd nba_betting

# Upewnij się że nba_betting.db i models/ są śledzone
git add -f nba_betting.db models/ dashboard.html dashboard_data.json .gitignore config.py
git add .github/workflows/pipeline.yml cron_setup.md main.py
git commit -m "feat: GitHub Actions pipeline + dashboard Pages"
git push origin main
```

---

## Harmonogram automatyczny

| Czas | Trigger GitHub Actions | Komenda Python |
|---|---|---|
| **20:00 CEST** (18:00 UTC) | `cron: '0 18 * * *'` | `python main.py --run-now` |
| **09:00 CEST** (07:00 UTC) | `cron: '0 7 * * *'` | `python main.py --verify-only` |

Po każdym uruchomieniu:
1. Wyniki commitowane do `main` (DB + modele + `dashboard_data.json`)
2. Dashboard deployowany na GitHub Pages automatycznie

---

## Uruchomienie manualne

W zakładce **Actions** → `NBA Betting Pipeline` → **Run workflow**:

- **pipeline** — generuje kupon na dziś (jak 20:00)
- **verify** — weryfikuje wczorajsze wyniki (jak 09:00)

---

## Persystencja danych między runami

Dane persystują przez commit do repo po każdym uruchomieniu:

| Plik | Zawartość | Commit? |
|---|---|---|
| `nba_betting.db` | SQLite — kursy, kupony, wyniki, cache API | ✅ tak |
| `models/*.pkl` | Wytrenowane modele XGBoost | ✅ tak |
| `coupon_YYYY-MM-DD.json` | Historyczne kupony | ✅ tak |
| `dashboard_data.json` | Dane dashboardu (generowane) | ✅ tak |
| `logs/` | Logi uruchomień | ❌ ignorowane |
| `backups/` | Backupy DB | ❌ ignorowane |

---

## Czas zimowy vs letni (CET/CEST)

GitHub Actions cron używa UTC. Workflow uruchamia się:
- **Lato** (CEST = UTC+2): `0 18 * * *` → 20:00 CEST ✅
- **Zima** (CET = UTC+1): `0 18 * * *` → 19:00 CET ❌ (godzina wcześniej)

Jeśli chcesz stałej godziny 20:00 przez cały rok, zaktualizuj cron w zimie:
```yaml
- cron: '0 19 * * *'   # 20:00 CET = 19:00 UTC (zima)
```

---

## Logi i debugowanie

```bash
# Sprawdź ostatnie uruchomienia
gh run list --repo K4j745/nba_betting --workflow pipeline.yml

# Logi konkretnego uruchomienia
gh run view <run-id> --log

# Uruchom pipeline teraz przez CLI
gh workflow run pipeline.yml --repo K4j745/nba_betting --field mode=pipeline
```

---

## Pierwsze uruchomienie (brak modelu)

Jeśli w repo nie ma jeszcze wytrenowanego modelu (`models/*.pkl`):

```bash
# Wytrenuj lokalnie
python train.py

# Wypchnij modele do repo
git add -f models/
git commit -m "feat: initial model v1"
git push
```

Bez modelu pipeline zakończy się ostrzeżeniem `Brak wytrenowanego modelu` i nie wygeneruje kuponu.
