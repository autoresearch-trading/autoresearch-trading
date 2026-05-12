# Pacifica external ops alert plan

Overall severity: `PAGE`

This artifact separates health-check classification from notification delivery. It does not send alerts and does not require or store delivery credentials.

No external delivery credentials should be committed. Wire delivery via Hermes cron/chat, Telegram, Discord, email, or another scheduler outside this repo/configured secrets path.

## Page/Warn conditions

| condition | severity | action | evidence |
| --- | --- | --- | --- |
| R2_REMOTE_FRESHNESS | PAGE | Investigate lifecycle upload/verify path if stale. | r2_latest_remote_age_min=471.20;threshold=60.00 |

## All conditions

| condition | severity | action | evidence |
| --- | --- | --- | --- |
| FLY_APP_STARTED | OK | No action. | fly_state=started |
| RAW_FRESHNESS | OK | No action. | newest_raw_age_min=0.00;threshold=15.00 |
| FREE_DISK_LOW | OK | No action. | free_gb=141.00;floor=50.00 |
| LIFECYCLE_DB_ERRORS | OK | No action. | rows_with_errors=0 |
| LIFECYCLE_UPLOAD_FAILURES | OK | No action. | lifecycle_upload_failed=0 |
| LIFECYCLE_VERIFY_FAILURES | OK | No action. | lifecycle_verify_failed=0 |
| R2_SIDECAR_MISMATCH | OK | No action. | r2_sidecar_mismatch_count=0 |
| R2_RAW_PREFIX_PRESENT | OK | No action. | r2_raw_present=True |
| R2_REMOTE_FRESHNESS | PAGE | Investigate lifecycle upload/verify path if stale. | r2_latest_remote_age_min=471.20;threshold=60.00 |
| WATCHDOG_STATUS_FRESH | OK | No action. | watchdog_status_age_min=25.90;watchdog_ok=True |
| API_SURFACE_CHANGED | OK | No action. | api_surface_changed=False |
| ARCHIVE_INVENTORY_FRESH | OK | No action. | archive_inventory_age_hours=8.45;threshold=24.00 |
| RESEARCH_REFRESH_OK | OK | No action. | research_refresh_ok=True |
| DELIVERY_CHANNEL_CONFIGURED | OK | No action. | delivery_channels=hermes_cron:e61c2f7c5593 |

## Thresholds

| raw_stale_after_min | free_disk_floor_gb | r2_remote_stale_after_min | watchdog_stale_after_min | archive_inventory_stale_after_hours |
| --- | --- | --- | --- | --- |
| 15 | 50 | 60 | 180 | 24 |

## Artifacts

- `alert_plan.csv`
- `summary.json`
- `thresholds.csv`
- `input_snapshot.json`
