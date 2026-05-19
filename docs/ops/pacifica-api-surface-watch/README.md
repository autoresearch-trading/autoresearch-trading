# Pacifica API Surface Watch

Checked at: 2026-05-18T13:00:20.289044+00:00
Verdict: UNCHANGED

Purpose: alert when Pacifica's public docs/API surface appears to expose new collectable market-data endpoints, websocket sources, or intervals. This is read-only and does not update the collector automatically.

## REST paths

Added:
- none

Removed:
- none

## WebSocket sources

Added:
- none

Removed:
- none

## Intervals

Added:
- none

Removed:
- none

## Current discovered surface

REST paths:
- `/funding/history`
- `/info`
- `/info/prices`
- `/kline`

WebSocket sources:
- none

Intervals:
- `12h`
- `15m`
- `1d`
- `1h`
- `1m`
- `2h`
- `30m`
- `3m`
- `4h`
- `5m`
- `8h`

Source URLs:
- `https://docs.pacifica.fi/api-documentation/api`
- `https://docs.pacifica.fi/api-documentation/changelog`
- `https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FO2lcakUmUFILzrKCX989%2Fimports%2FkkPsObvrXySwq95qP3NU%2Fopenapi.yaml?alt=media&token=061f1f51-e277-4eec-9d8c-6dc0320c4d40`

## Artifacts

- baseline_surface.json
- current_surface.json
- api_surface_diff.json

If verdict is CHANGED, manually inspect the docs/API, decide whether the new surface is public market data, add tests, then update the collector/silver layers and baseline in a separate reviewed change.
