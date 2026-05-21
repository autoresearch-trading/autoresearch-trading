# Next Session Handoff — Pacifica Full-Fidelity Paper Trading

Updated: 2026-05-21 14:20 UTC

## Current state

Active program: Hermes-only Pacifica full-fidelity, non-HFT paper-trading research. Do not use Claude Code, recreate `CLAUDE.md`, propose HFT/latency strategies, tune early toxicity thresholds, or claim edge from the current young archive.

Canonical active runtime/archive:

- Fly app: `pacifica-full-fidelity`
- Machine: `e2862502a76778`
- Active R2 archive prefix: `r2:pacifica-trading-data/raw/`
- Active local lifecycle DB on Fly: `/data/pacifica_full_fidelity_storage.sqlite`
- Local research raw cache: `data/pacifica_full_fidelity/` restored from R2 for research rebuilds
- Local silver output: `data/pacifica_silver_partitioned/`

## Latest 2026-05-21 Fly spool catch-up mode

Timestamp: `2026-05-21T14:20Z`.

Diego asked why so much data was sitting on Fly and approved catch-up mode to make Fly a bounded spool rather than a long-term archive. No raw local files or R2 objects were blindly deleted.

Live read-only evidence before the change:

```text
fly_app=pacifica-full-fidelity
fly_machine=e2862502a76778 state=started
fly_volume=pacifica_full_fidelity_data size=200GB
latest health log free_gb=100.81
latest health log unverified_gb=84.13
lifecycle_db_counts:
  pruned:   42,601 files / 21,071,688,481 bytes
  sealed:      857 files /  1,062,910,708 bytes
  uploaded: 158,110 files / 89,273,052,409 bytes
  verified:   1,500 files /    521,234,824 bytes
  rows_with_errors=0
old deployed cadence:
  PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500
  PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S=21600
  effective uploaded-backlog clearing estimate at old cadence: about 79 days before prune eligibility
```

Change deployed to Fly:

```text
config file: ops/fly/pacifica-full-fidelity/fly.toml
first deploy image:  pacifica-full-fidelity:deployment-01KS5C956KDPF90E2XAMT27MAY
second deploy image: pacifica-full-fidelity:deployment-01KS5DPARYSD6Y7ZVTPVZ576ZJ
machine instance after second deploy: 01KS5DQ4W6VQ86AQ2WHWA8G75Z
PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=5000
PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S=900
PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S=0
PACIFICA_FULL_FIDELITY_RETENTION_DAYS=1
PACIFICA_R2_PRUNE_EXECUTE=1
```

Why `FULL_SCAN_INTERVAL_S=0`: the first catch-up deploy reached the fresh lane, then appeared to block on a broad full-archive scan before verify/prune. The second deploy disables broad full scans during catch-up so the already-known `uploaded` backlog can be verified/pruned. Recent 12h scans remain enabled for newly closed chunks.

Verification run before deploy:

```text
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh && bash -n ops/fly/pacifica-full-fidelity/entrypoint.sh
uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_pacifica_full_fidelity_storage.py -q
  33 passed
git diff --check
  clean
flyctl deploy ... --app pacifica-full-fidelity --remote-only
  machine reached good state
```

Live post-deploy log evidence:

```text
2026-05-21T14:06:51Z lifecycle scan/upload/verify/prune start
2026-05-21T14:07:06Z {"scanned": 5902, "state_db": "/data/pacifica_full_fidelity_storage.sqlite"}
2026-05-21T14:07:19Z {"failed": 0, "skipped": 0, "uploaded": 67}
2026-05-21T14:07:39Z {"failed": 0, "sidecars_uploaded": 2000, "skipped": 0}
2026-05-21T14:07:39Z {"full_scan_skipped":true,"interval_s":0}
2026-05-21T14:07:39Z {"dry_run": false, "reset": 0, "skipped_missing": 0, "skipped_recent": 0}
```

Current monitoring implication:

- The catch-up verify/prune cycle was running after `14:07:39Z`; its `upload-verify`/`prune` summary had not appeared yet by `14:20Z`.
- Next check should use Fly logs/health output to confirm `verified` and then `pruned` files/bytes are increasing, `rows_with_errors` stays zero, and `free_gb` trends up or at least does not trend toward the 50GiB floor.
- If verify cycles take too long and delay fresh uploads, reduce `PACIFICA_FULL_FIDELITY_VERIFY_LIMIT` from `5000` to a smaller bounded value (for example `1000` or `2000`) while keeping `FULL_SCAN_INTERVAL_S=0` until backlog is under control.
- After the uploaded backlog is cleared/pruned, restore a slower safety cadence and re-enable a broad full scan interval if needed; do not leave aggressive catch-up settings permanent without checking R2 request spend and archive freshness.

## Latest 2026-05-19 post-promotion next actions

Timestamp: `2026-05-19T17:55Z`.

Diego asked to execute the next-action list after canonical promotion.

Completed:

```text
git_push: pushed local main commits through 777a1be to origin/main
bounded_ops_check:
  fly_app=pacifica-full-fidelity
  fly_machine=e2862502a76778 state=started
  recent_fly_watchdog_ok=true
  fly_free_gb=109.16
  db_error_rows=0
  live_symbols=66
  live_subscriptions=1651
  bounded_r2_freshness_ok=true
  latest_sampled_r2_payload_age_min=103.28
  sampled_r2_payloads=158
  sampled_r2_sidecars=158
  sampled_r2_missing_sidecars=0
  local_free_disk=83GiB
  local_raw_cache_size=93.865GiB
activity_gate_plan=docs/plans/pacifica-activity-gate-redesign-2026-05-19.md
```

Local cleanup note:

```text
Reviewed local temp cleanup candidates:
  data/pacifica_silver_partitioned_candidate_20260518T173526Z (~5.7G)
  data/ops/promotion-backups/pacifica-canonical-promotion-20260518T173526Z-20260518T221131Z (~11G)
  smaller candidate regime/eligibility/toxic/lineage dirs under data/ops
Attempted exact-path local deletion after push + green ops check, but the tool safety layer blocked the rm command and instructed not to retry. No local temp artifacts were pruned in this step.
```

Current stance:

- Keep collecting and monitoring; do not restart/resume/resize anything unless explicitly approved.
- Do not paper/live trade; canonical eligibility remains `0/66`.
- Treat reports as diagnostic only until at least 30 full distinct days; prefer 60+ days for serious validation.
- Do not tune toxicity/governor/activity thresholds on the 19-day sample.
- Next research task, if approved, is implementing the V2 activity-gate diagnostics from `docs/plans/pacifica-activity-gate-redesign-2026-05-19.md` with TDD and without changing final eligibility semantics.

## Latest 2026-05-18 canonical promotion + refreshed diagnostic reports

Timestamp: `2026-05-18T22:18Z`.

Diego approved canonical promotion after the bounded R2 rehydration, side-by-side silver/regime candidate build, duplicate-key fix, and green verifier.

Promotion run:

```text
run_id=20260518T173526Z
backup_root=data/ops/promotion-backups/pacifica-canonical-promotion-20260518T173526Z-20260518T221131Z
promoted_silver=data/pacifica_silver_partitioned
promoted_regime=docs/experiments/non-hft-regime-state
advanced_manifest=data/ops/pacifica-source-manifest/source_manifest_previous.csv
source_manifest=data/ops/pacifica-source-manifest/source_manifest_20260518T173526Z.csv
post_promotion_self_check=docs/ops/pacifica-incremental-refresh/post-promotion-self-check-20260518T173526Z
```

Post-promotion self-check:

```text
ok=True
failures=[]
canonical_old_regime_rows=1,110,785
promoted_regime_rows=1,225,259
row_delta=114,474
focused tests: tests/scripts/test_build_pacifica_full_fidelity_silver.py + tests/scripts/test_verify_pacifica_side_by_side_refresh.py => 30 passed
```

Refreshed canonical report state:

```text
docs/experiments/non-hft-regime-state/README.md
  bucket=1min rows=1,225,259 symbols=66
  status: useful regime/risk substrate, not an alpha claim

docs/experiments/paper-trading-eligibility/README.md
  verdict=INSUFFICIENT_SAMPLE_DIAGNOSTIC
  symbols_evaluated=66 eligible_symbols=0
  min_days gate=30; top preview symbols currently show up to n_days=19
  gate_counts: sample_gate_pass=0/66, liquidity_gate_pass=25/66, spread_cost_gate_pass=62/66, activity_gate_pass=0/66, stability_gate_pass=63/66, concentration_gate_pass=66/66, eligible=0/66

docs/experiments/toxic-regime-overlay/README.md
  verdict=INSUFFICIENT_SAMPLE_DIAGNOSTIC
  rows=1,225,259 symbols=66 distinct_dates=19
  minimum serious-validation gate=30 distinct days; fixed cutoffs remain [0.9, 0.8, 0.7]

docs/experiments/trade-activity-lineage/README.md
  verdict=LINEAGE_AUDIT_PASS_DIAGNOSTIC
  symbols_audited=10 raw/silver mismatches=0 silver/regime trade-count mismatches=0 unexplained_zero_medians=0 sparse_trade_zero_median_explanations=8
```

Interpretation:

- The local canonical silver/regime substrate is now current through the bounded refreshed local cache and passed side-by-side/post-promotion checks.
- This is a data-plumbing success only. It is not an alpha claim, paper-trading permission, or evidence of post-cost profitability.
- Paper-trading eligibility remains `0/66`; the hard blockers are still the 30-day sample gate and activity gate.
- The lineage audit indicates the activity failures are not caused by a raw → silver → regime trade-count break for audited symbols; the all-row median trade-notional metric is mostly zero because trade-active minutes remain sparse.
- Toxic overlay remains diagnostic: 19 distinct dates is still below the 30-day serious-validation gate, and thresholds must not be tuned on this sample.

Current recommendation:

1. Keep treating all outputs as diagnostic.
2. Do not tune toxicity/governor thresholds on this sample.
3. Do not paper/live trade; there are still zero eligible symbols.
4. Continue cost-aware collection/cache decisions first, then refresh reports only from bounded approved data.
5. Revisit provisional conclusions only after at least `30` full distinct days; prefer `60+` days for serious validation.

## Latest 2026-05-17 original-plan restart + billing usage check

Timestamp: `2026-05-17T16:06Z`.

Diego approved continuing with the original data-safe 24/7 full-fidelity plan after reviewing that the realistic budget is closer to `$50-60/month` across Fly + Cloudflare, not `<$20` each. The Fly collector was restarted.

```text
flyctl machine start e2862502a76778 -a pacifica-full-fidelity
  e2862502a76778 has been started

flyctl machine status e2862502a76778 -a pacifica-full-fidelity
  State: started
  HostStatus: ok
  Updated: 2026-05-17T16:06:11Z
  Image: pacifica-full-fidelity:deployment-01KRHKTRDN862V3YA274ZNE27J
  Volume: vol_vwn2mpw8mmgwx38v

flyctl status -a pacifica-full-fidelity
  app=e2862502a76778 version=30 region=iad state=started last_updated=2026-05-17T16:06:11Z

recent Fly logs
  /data mounted and resized to 214731587584 bytes
  ops watchdog run start
  lifecycle scan/upload/verify/prune start
  scanned=0 state_db=/data/pacifica_full_fidelity_storage.sqlite
```

Billing/usage check:

- Fly exact current invoice was not exposed through `flyctl`; `flyctl billing`/`flyctl invoices` do not exist in this installed CLI. Fly GraphQL exposed org billing status but not invoice line items.
- Fly org status via GraphQL: `billingStatus=CURRENT`, `billable=True`, `paidPlan=True`, `creditBalance_cents=0`.
- Live Fly billable resources: one shared-cpu-1x/1GB machine now `started`; one `200GB` volume `pacifica_full_fidelity_data` attached to `e2862502a76778`. Approximate ongoing burn from here is about `$1.20/day` while running (`~$1/day` volume + `~$0.20/day` compute), before snapshots/bandwidth. Known-resource MTD estimate from 100GB volume until `2026-05-09T23:44Z`, 200GB volume after that, and compute runtime through pause/restart is about `$14.63` before snapshots/bandwidth/dashboard adjustments.
- Cloudflare exact billing endpoints returned `403 Authentication error` with the current Wrangler OAuth token, so exact invoice also requires the dashboard or a token with billing scope.
- Cloudflare R2 GraphQL usage for `2026-05-01..2026-05-17` was available and is the best current programmatic signal.

Cloudflare R2 month-to-date usage from GraphQL:

```text
All R2 buckets latest storage on 2026-05-17:
  pacifica-trading-data Standard: object_count=3,091,964 payload_gb=116.915 metadata_gb=0.048
  pacifica-cache Standard:        object_count=4,003     payload_gb=1.012   metadata_gb=0.000
  pacifica-models Standard:       object_count=0         payload_gb=0.000   metadata_gb=0.000

pacifica-trading-data operations MTD:
  total_requests=11,816,974
  response_gb=98.508
  top actions:
    DeleteObject=6,634,278
    HeadObject=3,079,749
    ListObjects=980,745
    PutObject=449,364
    GetObject=351,224
    CopyObject=321,613

Rough R2 billable estimate from visible R2 usage only:
  class_a_requests≈1,751,727 -> about $3.38 after 1M free
  class_b_requests≈3,430,973 -> about $0.00 after 10M free
  latest Standard storage≈117.927GB -> about $1.62/month if constant after 10GB free
  visible-R2 subtotal estimate≈$5.00 before any non-R2 Cloudflare products/taxes/dashboard adjustments
```

Budget check cadence:

- Daily read-only budget checks were scheduled for the next 10 days.
- Cron job: `c2e30faff15a` (`Pacifica original-plan daily budget check`), `every 24h`, next run `2026-05-18T11:06:46-05:00`, delivery `origin`.
- The job is read-only: it must not restart/stop Fly, delete local files, delete R2 objects, or modify billing resources.

Operational stance after restart:

- Continue original full-fidelity plan, but watch cost daily.
- Treat exact provider dashboards as the source of truth for invoice totals; CLI/API estimates are resource/usage evidence only.
- If Fly or Cloudflare MTD usage approaches the agreed original-plan budget (`~$50-60/month` total), pause and redesign rather than resizing further.
- Avoid full-prefix local R2 mirrors; bounded rehydration only.

## Latest 2026-05-16 cost-control re-verification

Timestamp: `2026-05-16T20:29:51Z` live bounded checks; local cache partition scan at `2026-05-16T20:34:24Z`.

Diego asked to continue cost-control mode. No destructive action was run. The Fly collector remains stopped and must not be restarted unless Diego explicitly approves renewed spend.

Live checks:

```text
git status --short
  dirty working tree with existing modified generated docs/reports/code/tests, including docs/NEXT_SESSION_HANDOFF.md, non-hft-regime-state outputs, paper-trading-eligibility outputs, trade-activity-lineage outputs, pacifica-r2-archive-health outputs, scripts/verify_pacifica_side_by_side_refresh.py, tests/scripts/test_verify_pacifica_side_by_side_refresh.py, and untracked docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification/.

flyctl machine status e2862502a76778 -a pacifica-full-fidelity
  State: stopped
  HostStatus: ok
  Image: pacifica-full-fidelity:deployment-01KRHKTRDN862V3YA274ZNE27J
  Region: iad
  Volume: vol_vwn2mpw8mmgwx38v
  Updated: 2026-05-16T14:38:16Z
  event: exit_code=0, oom_killed=false, requested_stop=true

flyctl volumes list -a pacifica-full-fidelity
  vol_vwn2mpw8mmgwx38v  created  pacifica_full_fidelity_data  200GB  iad  attached_to=e2862502a76778  encrypted=true

rclone size r2:pacifica-trading-data/raw/pacifica/full_fidelity --json
  {"count":332586,"bytes":106353538565,"sizeless":0}

du -sh data/pacifica_full_fidelity data/pacifica_silver_partitioned docs/experiments/non-hft-regime-state data/ops
  90G   data/pacifica_full_fidelity
  5.3G  data/pacifica_silver_partitioned
  156M  docs/experiments/non-hft-regime-state
  1.4G  data/ops

process scan for rclone/build/verifier/refresh jobs
  matching_processes=0
```

Local laptop cache read-only partition summary:

```text
data/pacifica_full_fidelity total_files=302936 payloads=151462 sidecars=151474 total_gib=89.45
latest_payload_mtime_utc=2026-05-16T11:06:45.851718+00:00
latest_payload_path=data/pacifica_full_fidelity/channel=book/symbol=BP/date=2026-05-16/hour=10/run-20260513T212918Z.jsonl.gz
dates=2026-04-30..2026-05-16 count=17
keep last 5 local dates 2026-05-12..2026-05-16: keep 16.24 GiB, reclaim 73.22 GiB
keep last 3 local dates 2026-05-14..2026-05-16: keep 4.78 GiB, reclaim 84.67 GiB
keep last 7 local dates 2026-05-10..2026-05-16: keep 26.09 GiB, reclaim 63.37 GiB
```

Cost-control interpretation:

- Keep the Fly machine stopped. It is stopped now, so new collector ingestion, upload/verify request churn, and new R2 raw growth from this collector should remain paused.
- Fly volume storage is still a cost: `pacifica_full_fidelity_data` is still a 200GB attached volume. Do not destroy it yet unless Diego first decides the lifecycle DB/backlog state is no longer needed or exports/verifies enough replacement evidence.
- R2 storage is still a cost: active raw prefix is about 332.6k objects / 106.35GB decimal (~99.05 GiB). Do not delete R2 raw directly. First build a non-destructive retention/cold-compaction report and verify manifests/restores.
- Laptop local cache is not a cloud-cost driver but is too large for the laptop. Pruning by whole date partitions is the safest local-disk cleanup after explicit local-delete approval. Default recommendation: keep only the last 5 local dates and reclaim about 73.22 GiB.
- Replace full-prefix local R2 mirroring with bounded date/symbol/channel rehydration or a small rolling local cache.

Budget target from Diego: keep both Fly and Cloudflare invoices under `$20` each. Pricing checked from official docs on `2026-05-17T01:22:01Z`: Fly Volumes are `$0.15/GB-month` provisioned and charged even when the attached Machine is stopped; Cloudflare R2 Standard storage is `$0.015/GB-month` after 10GB free, with Class A operations at `$4.50/million` and Class B at `$0.36/million`. At 200GB provisioned, the Fly volume alone is about `$30/month` (`~$1/day` on a 30-day month), so keeping the current 200GB volume all month is incompatible with a `<$20` Fly invoice. At the current R2 raw size, R2 storage alone is only about `$1.45/month` after the 10GB free tier; Cloudflare risk is mainly request/operation churn from full mirrors, lifecycle scans, inventories, and collection uploads.

Root-cause note for the unexpected invoices: the original operational plan was data-loss-safe, not budget-fail-closed. It correctly specified compact raw payloads, a 50GiB free-disk guard, R2 copy/upload semantics, prune-after-remote-verified only, and R2 remote expiry only after cold-compaction/manifests. The failure was that these safety gates were scaled to 24/7 all-symbol/full-fidelity collection before a hard dollar budget gate existed. When verification/backlog lagged ingestion, Fly local prune stayed blocked (`uploaded` is not `verified`), the volume was expanded to 200GB to preserve data/freshness, and that provisioned volume alone became incompatible with a <$20 Fly invoice. Separately, local research refresh used a broad R2-to-laptop copy, creating a near-full local mirror and likely adding Cloudflare operation churn; future refreshes must use bounded rehydration instead.

Budget-constrained feasibility: the original architecture can still work under `<$20` each only with stricter parameters. With 24/7 shared-cpu-1x/1GB compute estimated around `$5.92/month`, Fly volume must stay below roughly 74-80GB if reserving `$2-3` for snapshots/overhead; 100GB is already about `$20.92/month` before snapshots and 200GB is impossible. Because the collector uses a 50GiB free-disk floor, full-universe/full-fidelity 24/7 collection is only feasible at an 80GB-class volume if verified/pruned throughput reliably stays within about one day of ingestion. Otherwise the system must reduce raw GB/day via fewer channels/symbols or scheduled capture windows. On R2, reserving `$5` for operations leaves about 1010GB Standard storage under a `$20` cap; that implies rough steady-state raw-retention limits of ~33.7GB/day for 30 days, ~22.4GB/day for 45 days, or ~16.8GB/day for 60 days before cold compaction/expiry.

Original-plan budget estimate: the data-safe 24/7 full-fidelity plan needs about `$25-40/month` if lifecycle verification/pruning stays healthy and the volume can stay 75-100GB, split roughly Fly `$18-26` plus Cloudflare `$6-15`. A robust/backlog-tolerant version using the current 200GB Fly volume is more like `$45-60+/month`, split roughly Fly `$36-45+` plus Cloudflare `$8-20+` depending on R2 operation churn. At the observed current R2 growth rate (~6.26GB/day across the current prefix), R2 Standard storage is cheap (`~$2.67/month` for 30d retention, `~$5.48/month` for 60d retention), but the earlier docs' rough 25GB/day design rate would be much tighter (`~$11.10/month` for 30d retention and `~$22.35/month` for 60d retention before operations). Therefore the original plan's realistic budget should have been at least about `$50/month` across Fly+Cloudflare, with a safer backlog-tolerant cap closer to `$60/month`; `<$20` each requires budget-mode scope/rate reductions.

Immediate no-risk next actions:

1. Leave `pacifica-full-fidelity` machine `e2862502a76778` stopped.
2. Avoid any full-prefix `rclone copy` from R2 to the laptop.
3. Build/read a report-only R2 retention/cold-compaction plan; no remote deletes or lifecycle expiry.
4. If doing local research before a new collection design, rehydrate only specific dates/symbols/channels into the local cache.

Requires explicit approval before running:

1. Local delete: prune `data/pacifica_full_fidelity/date=<old-date>` whole date partitions. Suggested policy is keep `2026-05-12` through `2026-05-16` and remove older local date partitions, reclaiming about 73.22 GiB. This saves laptop disk only, not invoices.
2. Fly destructive/irreversible action: destroy/downsize the 200GB volume or delete the app/machine. Do this only after deciding the lifecycle DB/backlog state is no longer needed or safely exported.
3. R2 destructive action: delete raw objects or enable lifecycle expiry. This requires verified cold compaction + manifest coverage + restore validation + explicit delete approval.
4. Restart collection: `flyctl machine start e2862502a76778 -a pacifica-full-fidelity` only if Diego explicitly accepts renewed spend.

## Latest 2026-05-16 >24h ops check + local research refresh

Timestamp: `2026-05-16T13:10Z`

Diego noted that more than 24h had passed since the last check, so a bounded, non-destructive live check was run before any local rebuild work.

Live ops/archive evidence:

```text
flyctl status --app pacifica-full-fidelity
  app=pacifica-full-fidelity
  machine=e2862502a76778
  state=started
  image=pacifica-full-fidelity:deployment-01KRHKTRDN862V3YA274ZNE27J
  last_updated=2026-05-13T21:29:17Z

uv run python scripts/check_pacifica_r2_freshness.py --stale-after-min 180 --timeout-s 120 --out data/ops/pacifica-r2-freshness-20260516T130108Z.json
  ok=True
  failures=[]
  latest_payload_age_min=114.4
  latest_payload=channel=book/symbol=ETH/date=2026-05-16/hour=10/run-20260513T212918Z.jsonl.gz
  payload_count=137
  sidecar_count=137
  sidecar_missing_count=0

Sampled latest R2 payload copied locally for verification:
  data/ops/r2-sample-verification-20260516T130108Z/run-20260513T212918Z.jsonl.gz
  sha256_match=True
  gzip_rows=3002
  bytes=1417062

uv run python scripts/watch_pacifica_api_surface.py --fail-on-change --out-dir docs/ops/pacifica-api-surface-watch
  changed=False
```

Local research raw cache was stale relative to R2 before refresh:

```text
data/pacifica_full_fidelity/
  payloads=145995
  sidecars=145995
  payload_gib=86.72
  dates=('2026-04-30', '2026-05-14')
  latest_local_mtime_utc=2026-05-14T11:04:49.850822+00:00
  latest_local_age_min=2995.81
  latest_local_path=data/pacifica_full_fidelity/channel=book/symbol=ANTHROPIC/date=2026-05-14/hour=10/run-20260513T212918Z.jsonl.gz
```

A safe non-destructive refresh was started to copy new R2 raw objects into the local research cache and then build side-by-side candidate artifacts only, but it was intentionally stopped after Diego flagged the local cache size. No side-by-side candidate was created from this run.

```text
Hermes background process: proc_2bf0bfb5b20f
OS wrapper process id: 18537
Observed child rclone pid while copy was in progress: 18565
Wrapper: data/ops/pacifica_24h_refresh_20260516T130108Z.sh
Run timestamp: 20260516T130108Z
Copy log: data/ops/pacifica-research-refresh-20260516T130108Z/r2_copy.log
Stop time: about 2026-05-16T13:21Z via `process kill proc_2bf0bfb5b20f`
Side-by-side driver log: not created
Candidate silver/regime dirs: not created
```

Post-stop local cache/retention check:

```text
data/pacifica_full_fidelity/ size: 91G by du; 89.45 GiB including sidecars by file sum
payloads=151462 sidecars=151474
latest_local_mtime_utc=2026-05-16T11:06:45.851718+00:00
no local rclone/rebuild process remained after stop

Local cache reclaim estimate if pruned by whole date partitions:
  keep 2026-05-14..2026-05-16 only: keep 4.78 GiB, reclaim 84.67 GiB
  keep 2026-05-12..2026-05-16 only: keep 16.24 GiB, reclaim 73.22 GiB
  keep 2026-05-10..2026-05-16 only: keep 26.09 GiB, reclaim 63.37 GiB
```

Fly/R2 pruning evidence checked at about `2026-05-16T13:24Z`:

```text
Fly volume: pacifica_full_fidelity_data, 200GB attached to e2862502a76778
fly machine exec ... df -h /data
  /data size=197G used=68G avail=120G use=37%

Latest Fly health log around 2026-05-16T12:56Z:
  ok=True failures=[] rows_with_errors=0
  db_counts.pruned:   35,603 files / 18,597,964,743 bytes
  db_counts.sealed:      789 files / 1,325,213,987 bytes
  db_counts.uploaded:124,271 files / 68,678,539,199 bytes
  db_counts.verified:  2,000 files / 732,505,126 bytes
  unverified_gb=65.2

Latest lifecycle logs show frequent fresh-lane upload/sidecar repair, but the 6h backlog lane was skipped in the 12:42Z cycle. Pruning is safety-correct on Fly (`PACIFICA_R2_PRUNE_EXECUTE=1`, retention_days=1), but most local Fly bytes are still `uploaded` rather than `verified`, so they are not yet prune-eligible.

R2 raw prefix size:
  rclone size r2:pacifica-trading-data/raw/pacifica/full_fidelity --json
  count=332090 bytes=106326359902 (~99.02 GiB)

R2 remote raw deletion/expiry is intentionally not active. `scripts/plan_pacifica_r2_retention.py` is report-only and requires verified cold compaction + manifest coverage + separate explicit approval before any remote raw expiry/delete.
```

Safety boundary: do not resume full-prefix R2 -> laptop copy until a local-cache policy is chosen. Prefer bounded date/symbol rehydration or prune the laptop cache by whole date partitions after explicit approval. Fly local prune is running safely but verification backlog is the bottleneck; R2 is intentionally append-only for now, not mis-pruned.

## Latest 2026-05-16 cost-control pause

Timestamp: `2026-05-16T14:39Z`

Diego reported Cloudflare and Fly invoices already around `$16` each. To stop further cloud growth immediately without deleting data, the Fly collector machine was paused after a no-response clarification timeout.

```text
flyctl machine stop e2862502a76778 -a pacifica-full-fidelity
  e2862502a76778 has been successfully stopped

flyctl machine status e2862502a76778 -a pacifica-full-fidelity
  State: stopped
  Updated: 2026-05-16T14:38:16Z
  exit_code=0, oom_killed=false, requested_stop=true
```

Current cost posture:

- New Pacifica raw collection is paused, so new R2 object growth and request volume from the collector should stop.
- Existing Fly volume `pacifica_full_fidelity_data` still exists at 200GB and will continue to incur storage cost until downsized/destroyed.
- Existing R2 raw archive remains at about `106,353,538,565` bytes / `99.05 GiB` and will continue to incur storage cost until a reviewed retention/cold-compaction delete plan is explicitly approved and executed.
- Do not destroy the Fly volume yet: latest Fly DB health showed most bytes in `uploaded` rather than `verified`, so the volume is still useful for verification/backlog state unless we first export/verify enough evidence.

Restart command if Diego explicitly accepts renewed cloud spend:

```bash
flyctl machine start e2862502a76778 -a pacifica-full-fidelity
```

Next safe cost-down steps:

1. Prune laptop cache by whole date partitions after explicit local-delete approval. This saves local disk, not invoices.
2. Build a non-destructive R2 cost report from inventory: object/request drivers, retained recent raw, and cold-compaction candidates.
3. Decide whether to keep a smaller future collector: fewer channels/symbols or scheduled short capture windows instead of 24/7 full universe.
4. Only after verified cold archive/manifest coverage and approval, apply R2 raw expiry/delete. No remote delete is approved yet.

## Latest 2026-05-13 research refresh continuation

Timestamp: `2026-05-13T23:32Z`

The refresh was resumed from the trade-activity lineage audit handoff. Ops/archive freshness was checked first with non-destructive live evidence:

```text
flyctl status --app pacifica-full-fidelity
  app=pacifica-full-fidelity
  machine=e2862502a76778
  version=30
  state=started
  image=pacifica-full-fidelity:deployment-01KRHKTRDN862V3YA274ZNE27J

uv run python scripts/check_pacifica_r2_freshness.py --stale-after-min 180 --timeout-s 120 --out data/ops/pacifica-r2-freshness-20260513T230359Z.json
  ok=True
  failures=[]
  latest_payload_age_min=95.01
  latest_payload=channel=book/symbol=ETH/date=2026-05-13/hour=21/run-20260513T205507Z.jsonl.gz
  payload_count=228
  sidecar_count=228
  sidecar_missing_count=0
```

The local research raw cache was stale before refresh (`data/pacifica_full_fidelity/` latest local payload was 2026-05-12T15:06:53Z, age about 1919 minutes), so a non-destructive R2 copy was started before building manifests or silver/regime artifacts:

```text
Hermes background process: proc_7085172d0c1d
Child rclone pid observed: 84762
Command:
  rclone copy r2:pacifica-trading-data/raw/pacifica/full_fidelity data/pacifica_full_fidelity --transfers 16 --checkers 32 --fast-list --stats 30s --stats-one-line
```

A second non-destructive background driver was started to continue only after the copy PID exits:

```text
Hermes background process: proc_40a907d748a7
Script: data/ops/pacifica_research_refresh_after_r2_copy.sh
Log: data/ops/pacifica-research-refresh-20260513T233058Z/refresh.log
Behavior:
  1. wait for rclone pid 84762 to exit;
  2. summarize local raw cache;
  3. run read-only `rclone check` source->local with `--one-way --size-only`;
  4. build `data/ops/pacifica-source-manifest/source_manifest_20260513T233058Z.csv` and matching incremental plan;
  5. build side-by-side candidate silver under `data/pacifica_silver_partitioned_candidate_20260513T233058Z`;
  6. build candidate regime delta/full snapshots under `data/ops/pacifica-regime-candidate-20260513T233058Z*`;
  7. run side-by-side verification into `docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification`.
```

Safety boundary: this background driver does not advance `source_manifest_previous.csv`, does not overwrite canonical `data/pacifica_silver_partitioned/`, does not overwrite canonical `docs/experiments/non-hft-regime-state/`, does not rebuild canonical eligibility, and does not mutate R2/raw. After it completes, manually inspect the verifier before any promotion or eligibility refresh.

At handoff update time, the local copy was still running and local payload inventory had advanced to about 58,806 payload files / 42.81 GiB with 2026-05-13 data present. Treat silver/regime/eligibility as stale until the background driver completes and a human/agent reviews the verification report.

## Latest 2026-05-14 research refresh continuation update

Timestamp: `2026-05-14T13:53Z`

The long R2 -> local copy completed with exit code 0 after retrying transient R2 `403`/XML parse errors. Local raw cache summary immediately after completion:

```text
payloads=145995
sidecars=145995
payload_gib=86.72
symbols=67
channels=['bbo', 'book', 'candle', 'mark_price_candle', 'pong', 'prices', 'subscribe', 'trades']
dates=('2026-04-30', '2026-05-14')
latest_local_mtime=2026-05-14T11:04:49.850822+00:00
latest_local_age_min=160.64
latest_local_path=data/pacifica_full_fidelity/channel=book/symbol=ANTHROPIC/date=2026-05-14/hour=10/run-20260513T212918Z.jsonl.gz
```

The first continuation driver `proc_40a907d748a7` then failed at the strict full-root `rclone check` step because the live R2 archive moved during/after the long copy. It found only 10 newest remote files missing locally, all under 2026-05-14 hour=11 for `pong/UNKNOWN` and trades symbols `2Z`, `CHIP`, `kPEPE`, and `WLFI` plus sidecars. This is a moving-snapshot mismatch, not evidence that copied local sealed files are corrupt. Do not promote canonical artifacts from this state without the side-by-side verification below.

A second local-snapshot side-by-side refresh ran from the sealed/checksum/gzip-verified local snapshot rather than a full-root check against a moving live archive. The original verifier implementation failed the run because it used an over-broad generic silver duplicate key and then later OOMed when pandas materialized full production channels:

```text
Hermes background process: proc_026e06e384d3
Script: data/ops/pacifica_research_refresh_from_local_snapshot.sh
Log: data/ops/pacifica-research-refresh-20260514T135053Z/refresh_from_local_snapshot.log
Original exit: 1
Original verifier verdict: ok=False
Original failures: ['candidate_silver_duplicate_keys']
```

Verifier fix completed on 2026-05-14T19:09Z:

```text
scripts/verify_pacifica_side_by_side_refresh.py
  - production silver metrics now use DuckDB/parquet-side aggregation one channel at a time instead of `read_silver_table`/pandas full-frame loads.
  - computes row counts, symbol/date coverage, required-key nulls, channel-specific semantic duplicate keys, and exact-payload duplicates excluding `source_key`, `source_path`, and `source_sha256`.
  - DuckDB spill dir defaults to `.tmp/duckdb-verifier-spill`; memory limit defaults to `PACIFICA_VERIFIER_DUCKDB_MEMORY_LIMIT` or `8GB`.

tests/scripts/test_verify_pacifica_side_by_side_refresh.py
  - keeps channel-specific duplicate semantics tests for `trades`, `bbo`, `candle`, and exact-payload duplicates.
  - adds a scale-safety regression proving production silver metrics do not call the pandas `read_silver_table` path.
```

The existing candidate `20260514T135053Z` was rerun through the fixed verifier and is now green:

```text
Command:
uv run python scripts/verify_pacifica_side_by_side_refresh.py \
  --canonical-silver-dir data/pacifica_silver_partitioned \
  --candidate-silver-dir data/pacifica_silver_partitioned_candidate_20260514T135053Z \
  --canonical-regime-dir docs/experiments/non-hft-regime-state \
  --candidate-regime-dir data/ops/pacifica-regime-candidate-20260514T135053Z \
  --out-dir docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification

Exit: 0
Verifier verdict: ok=True
Failures: []
Runtime: about 3 minutes wall clock in Hermes background process `proc_b4451ff25e3c`
```

Current green verifier artifacts:

```text
docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification/README.md
  ok=True
  failures=[]

docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification/summary.csv
  channels=prices,trades,bbo,book,candle,mark_price_candle

docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification/silver_row_counts.csv
  prices: canonical=1,039,338 candidate=5,252,223 delta=4,212,885
  trades: canonical=91,069 candidate=699,037 delta=607,968
  bbo: canonical=12,974,473 candidate=23,669,943 delta=10,695,470
  book: canonical=14,137,732 candidate=60,389,166 delta=46,251,434
  candle: canonical=1,915,776 candidate=8,152,538 delta=6,236,762
  mark_price_candle: canonical=22,446,290 candidate=158,241,147 delta=135,794,857

docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification/regime_row_counts.csv
  regime_state: canonical=519,903 candidate=1,110,785 delta=590,882
```

Status after Diego's explicit `approved` message on 2026-05-15: promotion was executed and verified. The former candidate is now canonical, `source_manifest_previous.csv` has been advanced to `source_manifest_20260514T135053Z.csv`, and backups are under `data/ops/pacifica-promotion-backups/20260515T145557Z`. The next sections record the post-promotion self-check, refreshed eligibility, and lineage audit.

## Latest 2026-05-15 approved promotion + refreshed eligibility/lineage

Timestamp: `2026-05-15T14:55Z`

Diego approved promotion after the fixed side-by-side verifier was green. Promotion was executed locally without deleting raw data:

```text
Promoted candidate run: 20260514T135053Z
Backup root: data/ops/pacifica-promotion-backups/20260515T145557Z
  pacifica_silver_partitioned_before/      # old canonical silver
  non-hft-regime-state_before/             # old canonical regime/report
  source_manifest_previous_before.MISSING  # no previous manifest existed before promotion
  promotion_record.json

Canonical after promotion:
  data/pacifica_silver_partitioned                         5.3G
  docs/experiments/non-hft-regime-state                    156M
  data/ops/pacifica-source-manifest/source_manifest_previous.csv
    line_count=139376 (same current manifest rows + header)
```

Candidate paths were moved into canonical paths as part of promotion:

```text
data/pacifica_silver_partitioned_candidate_20260514T135053Z -> data/pacifica_silver_partitioned
data/ops/pacifica-regime-candidate-20260514T135053Z -> docs/experiments/non-hft-regime-state
```

Post-promotion canonical self-check:

```text
uv run python scripts/verify_pacifica_side_by_side_refresh.py \
  --canonical-silver-dir data/pacifica_silver_partitioned \
  --candidate-silver-dir data/pacifica_silver_partitioned \
  --canonical-regime-dir docs/experiments/non-hft-regime-state \
  --candidate-regime-dir docs/experiments/non-hft-regime-state \
  --out-dir data/ops/pacifica-incremental-refresh-selfcheck-20260515T145557Z

Exit: 0
ok=True
failures=[]

Self-check rows:
  prices: 5,252,223
  trades: 699,037
  bbo: 23,669,943
  book: 60,389,166
  candle: 8,152,538
  mark_price_candle: 158,241,147
  regime_state: 1,110,785
```

Canonical eligibility refresh:

```text
uv run python scripts/build_pacifica_eligibility_gates.py

verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
symbols_evaluated: 66
eligible_symbols: 0
report: docs/experiments/paper-trading-eligibility/README.md

gate_counts:
  sample_gate_pass: 0 / 66
  liquidity_gate_pass: 25 / 66
  spread_cost_gate_pass: 62 / 66
  activity_gate_pass: 0 / 66
  stability_gate_pass: 63 / 66
  concentration_gate_pass: 65 / 66
  eligible: 0 / 66
```

Interpretation: no symbols are paper-trading eligible. This is expected diagnostic behavior for a young archive: only 15 distinct days are in the promoted regime state, below the fixed 30-day sample gate, and all symbols still fail the current activity gate. Do not loosen gates or treat this as an edge claim.

Trade-activity lineage rerun after promotion and eligibility refresh:

```text
uv run python scripts/audit_pacifica_trade_activity_lineage.py --max-symbols 10

verdict: LINEAGE_AUDIT_PASS_DIAGNOSTIC
symbols_audited: 10
sparse_trade_zero_median_explanations: 7
raw/silver mismatches: 0
silver/regime trade-count mismatches: 0
unexplained zero medians: 0
report: docs/experiments/trade-activity-lineage/README.md
```

Interpretation: the promoted raw -> silver -> regime -> eligibility trade-activity lineage is internally consistent for audited symbols. Seven audited symbols have zero all-row median trade notional because trade minutes are sparse, not because the pipeline dropped trades. This still does not authorize trading; it only turns the prior lineage failure into a diagnostic pass.

Next safe steps:

1. Keep collecting until at least the fixed 30-day eligibility sample gate can pass; 60+ days remains preferred for serious validation.
2. Do not weaken sample/activity gates from this diagnostic run.
3. If activity remains the dominant blocker after more days, pre-register a replacement metric before changing it: e.g. active-minute median plus active-minute share as a two-part gate.
4. Rerun eligibility and lineage after the next verified canonical refresh.

## Historical 2026-05-13 trade-activity lineage audit

Superseded by the 2026-05-15 approved promotion and post-promotion lineage pass above. This section is retained only to explain why the refresh/promotion work was needed.

Timestamp: `2026-05-13T22:18Z`

Implemented a TDD-tested diagnostic audit for the current zero-activity eligibility concern. This is plumbing validation only; it is not a strategy, alpha claim, or paper-trading permission.

Added:

```text
scripts/audit_pacifica_trade_activity_lineage.py
  - traces raw `channel=trades` JSONL.GZ -> silver `channel=trades` parquet -> regime `trade_count`/`trade_notional` -> eligibility activity metrics.
  - default target selection audits BTC/ETH/SOL plus the first symbols from eligibility, capped by `--max-symbols`.
  - writes `docs/experiments/trade-activity-lineage/README.md`, `symbol_summary.csv`, and `date_summary.csv`.
  - fails closed on missing required regime columns and non-empty silver trade files missing required columns.
  - reports raw/silver row and notional deltas, silver/regime trade-count and notional deltas, active trade bucket share, all-row vs active-row trade-notional medians, and diagnostic notes.

tests/scripts/test_audit_pacifica_trade_activity_lineage.py
  - RED/GREEN coverage for sparse-trade zero-median explanation, silver/regime mismatch failure, report artifact language, direct script execution, and missing-regime-column fail-closed behavior.

AGENTS.md
  - active pipeline now lists the trade-activity lineage audit.
```

Current local diagnostic run:

```text
uv run python scripts/audit_pacifica_trade_activity_lineage.py --max-symbols 10
verdict: LINEAGE_AUDIT_FAIL_DIAGNOSTIC
symbols_audited: 10
wrote report: docs/experiments/trade-activity-lineage/README.md

Audited symbols:
  BTC, ETH, SOL, WLFI, kPEPE, CRV, PUMP, XRP, 2Z, AVAX

Failure counters:
  raw/silver mismatches: 10 symbols
  silver/regime trade-count mismatches: 0 symbols
  sparse-trade zero-median explanations: 0 symbols (strict raw->silver->regime pass required)
  unexplained zero medians: 0 symbols
```

Interpretation:

```text
1. The current eligibility `median_trade_notional_per_min=0` is consistent with sparse trade minutes across all existing regime rows for audited symbols: active trade bucket share was only about 7-10%. The all-row median can be zero even when active trade minutes exist.
2. Silver -> regime trade aggregation lined up for audited symbols (`silver_regime_trade_count_delta=0`). That means the current regime table is not dropping current silver trade rows for those symbols.
3. Raw -> silver mismatches are still present because the local raw cache now contains trades beyond the current silver/regime/eligibility artifacts, and some overlapping early-date raw rows are not represented in current silver. Because raw -> silver is not clean, the audit intentionally does not mark zero medians as fully explained. Treat current eligibility as stale until silver/regime/eligibility are refreshed from the latest raw archive and the audit is rerun.
```

Next commands after ops freshness is confirmed:

```bash
# Follow the side-by-side flow in docs/ops/pacifica-incremental-refresh/README.md.
# Do not overwrite canonical silver/regime artifacts until candidate verification is green and manually reviewed.
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST_DIR=data/ops/pacifica-source-manifest
CANDIDATE_SILVER=data/pacifica_silver_partitioned_candidate_${RUN_TS}
CANDIDATE_REGIME=data/ops/pacifica-regime-candidate-${RUN_TS}
VERIFY_DIR=docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification
mkdir -p "$MANIFEST_DIR"

uv run python scripts/build_pacifica_source_manifest.py \
  --raw-dir data/pacifica_full_fidelity \
  --out "$MANIFEST_DIR/source_manifest_${RUN_TS}.csv" \
  --previous "$MANIFEST_DIR/source_manifest_previous.csv" \
  --verify-sha \
  --count-rows \
  --plan-out "$MANIFEST_DIR/incremental_plan_${RUN_TS}.csv"

uv run python scripts/build_pacifica_full_fidelity_silver.py \
  --layout incremental \
  --raw-dir data/pacifica_full_fidelity \
  --out-dir "$CANDIDATE_SILVER" \
  --source-manifest "$MANIFEST_DIR/source_manifest_${RUN_TS}.csv" \
  --previous-source-manifest "$MANIFEST_DIR/source_manifest_previous.csv"

uv run python scripts/build_non_hft_regime_state.py \
  --silver-dir "$CANDIDATE_SILVER" \
  --out-dir "$CANDIDATE_REGIME" \
  --bucket 1min

uv run python scripts/verify_pacifica_side_by_side_refresh.py \
  --canonical-silver-dir data/pacifica_silver_partitioned \
  --candidate-silver-dir "$CANDIDATE_SILVER" \
  --canonical-regime-dir docs/experiments/non-hft-regime-state \
  --candidate-regime-dir "$CANDIDATE_REGIME" \
  --out-dir "$VERIFY_DIR"

# After verified promotion / refreshed canonical eligibility, rerun:
uv run python scripts/audit_pacifica_trade_activity_lineage.py --max-symbols 10
```

Do not weaken the activity gate based on this diagnostic. First refresh silver/regime/eligibility, then use the lineage audit to decide whether the activity metric should stay as an all-row median, move to an active-minute metric, or become a two-part gate.

## Latest 2026-05-13 live ops remediation — R2 freshness + watchdog race

Timestamp: `2026-05-13T21:40Z`

Highest live blocker handled: bounded R2 freshness was green after the v28 slow-safety cycle, but a Fly-side watchdog later failed while overlapping the lifecycle upload/sidecar-repair window, and a post-deploy local check exposed the payload freshness SLO still had a narrow stale gap.

Live evidence before fixes:

```text
Fly status before changes:
  app=pacifica-full-fidelity
  machine=e2862502a76778
  version=28
  state=started
  image=pacifica-full-fidelity:deployment-01KRGTS1E1VQQ537ZGDQJ34V48

Fly health/log evidence before v29/v30:
  2026-05-13T20:20:51Z ops watchdog reported failures
  watchdog latest_status at 2026-05-13T20:20:44Z:
    ok=false
    operation=r2_freshness_check returncode=2
    failures=[R2_SIDECAR_MISSING]
    latest_payload_age_min=133.91
    sidecar_missing_count=4
  Local rerun at 2026-05-13T20:41:20Z:
    ok=true, failures=[], latest_payload_age_min=155.79, sidecar_missing_count=0

After deploying the retry-only watchdog fix, the R2 freshness SLO still crossed stale at 2026-05-13T21:07:14Z:
  ok=false
  failures=[R2_REMOTE_FRESHNESS_STALE]
  latest_payload=channel=bbo/symbol=BTC/date=2026-05-13/hour=17/run-20260513T141137Z.jsonl.gz
  latest_payload_age_min=181.69
  sidecar_missing_count=0
```

Root cause interpretation:

```text
1. The watchdog can run while the lifecycle fresh lane has copied payloads but before `repair-sidecars` finishes. That is a transient sidecar-lag race, not archive corruption.
2. `PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS=7200` was too tight for a 180-minute R2 payload freshness SLO because hourly chunks close several minutes after the hour and the lifecycle loop also scans/uploads/sleeps. A just-closed hour could miss one cycle and leave sampled payload mtime just past 180 minutes.
```

Implemented and pushed:

```text
e7a3443 fix(pacifica): retry transient R2 sidecar lag
  scripts/run_pacifica_fly_ops_watchdogs.py
    - retries only when bounded freshness fails solely with R2_SIDECAR_MISSING,
      latest payload is still fresh, sidecar_missing_count > 0, and listing_errors is empty.
    - does not retry stale/mixed/listing-error/non-numeric/missing-count failures.
    - default retry: one attempt after 300s; env overrides:
      PACIFICA_R2_FRESHNESS_SIDECAR_RETRY_ATTEMPTS
      PACIFICA_R2_FRESHNESS_SIDECAR_RETRY_DELAY_S
  tests/scripts/test_run_pacifica_fly_ops_watchdogs.py
    - regression for transient sidecar-lag retry.
    - negative regression that stale/mixed/listing-error/non-numeric/missing-count cases are not masked.

d48950e fix(pacifica): restore R2 freshness margin
  scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  ops/fly/pacifica-full-fidelity/entrypoint.sh
  ops/fly/pacifica-full-fidelity/fly.toml
    - lowers PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS from 7200 to 5400.
    - keeps `--skip-current-hour` scanning; still requires 90 minutes since file mtime before upload.
  tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
    - enforces lifecycle default, Fly runtime env, and entrypoint fallback all use 5400.
```

Verification:

```text
RED:
  uv run pytest tests/scripts/test_run_pacifica_fly_ops_watchdogs.py::test_run_once_retries_transient_sidecar_lag_before_failing_watchdog -q
  failed with missing WatchdogConfig retry fields.

GREEN for v29:
  python -m py_compile scripts/run_pacifica_fly_ops_watchdogs.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py
  uv run pytest tests/scripts/test_run_pacifica_fly_ops_watchdogs.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py -q
  20 passed

RED for v30:
  uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py::test_lifecycle_default_min_upload_age_has_freshness_margin tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py::test_fly_runtime_min_upload_age_has_freshness_margin -q
  failed because runtime/default min upload age was still 7200.

GREEN for v30:
  bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/entrypoint.sh
  python -m py_compile tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py tests/scripts/test_plan_pacifica_ops_alerts.py -q
  53 passed

Independent reviews:
  - v29 watchdog retry review passed; no security concerns or logic errors.
  - v30 min-upload-age review passed; no destructive raw/R2 behavior added and skip-current-hour + 90-minute mtime age was considered safe.
```

Deployments:

```text
v29 deploy:
  image=registry.fly.io/pacifica-full-fidelity:deployment-01KRHHW7B3HPH1M3K1CNH4HEHB
  machine=e2862502a76778
  version=29
  state=started
  last_updated=2026-05-13T20:55:06Z

v30 deploy:
  image=registry.fly.io/pacifica-full-fidelity:deployment-01KRHKTRDN862V3YA274ZNE27J
  machine=e2862502a76778
  version=30
  state=started
  last_updated=2026-05-13T21:29:17Z
```

Post-v30 evidence:

```text
Local bounded R2 freshness at 2026-05-13T21:34:50Z:
  ok=true
  failures=[]
  latest_payload=channel=book/symbol=ETH/date=2026-05-13/hour=18/run-20260513T141137Z.jsonl.gz
  latest_payload_modified=2026-05-13T19:07:16Z
  latest_payload_age_min=147.58
  payload_count=206
  sidecar_count=206
  sidecar_missing_count=0

Remote Fly-side watchdog artifact copied from R2 at 2026-05-13T21:36Z:
  checked_at=2026-05-13T21:35:58Z
  ok=true
  operation=r2_freshness_check returncode=0
  stderr_tail=retry_after_transient_sidecar_lag: attempt=1 delay_s=300
  latest_payload_age_min=147.53
  sidecar_missing_count=0

Fly status at 2026-05-13T21:36Z:
  version=30
  state=started
  image=pacifica-full-fidelity:deployment-01KRHKTRDN862V3YA274ZNE27J
```

Remaining watch items:

- v30 is green immediately after sidecar repair and the watchdog retry worked, but watch one more normal lifecycle/watchdog hour to confirm the 5400s upload age keeps bounded R2 freshness under 180 minutes without repeated retries.
- The uploaded/unverified backlog remains large (`unverified_gb` about 50.07 in the latest Fly health log). Slow safety-lane verify/prune did run earlier, but archive health is not fully recovered until verify/prune throughput stays positive across safety cycles and full/broader inventory checks are green.
- Do not treat the committed bounded R2 report as full-bucket proof. Full recursive live R2 listing from the laptop still timed out at 600s in this session.
- Direct Fly SSH should still not be retried in the denied form. Prefer Fly status/logs, Fly health output, R2 watchdog artifacts, and bounded/local R2 samples.
- Local working tree still had unrelated generated docs under `docs/ops/pacifica-api-surface-watch/` and `docs/ops/pacifica-r2-archive-health/` dirty before this handoff update; do not include them in lifecycle commits unless intentionally refreshing those reports.

## Latest 2026-05-13 walk-forward validation hardening

Timestamp: `2026-05-13T18:59Z`

Implemented the requested fail-closed, strategy-neutral walk-forward validation layer for future materialized Pacifica strategy-return streams. This is diagnostic infrastructure only; it makes no alpha claims and does not allow paper/live trading.

Updated code/report:

```text
scripts/run_pacifica_walk_forward_validation.py
  - builds chronological train / purge / OOS-test windows; `step_days < test_days` is rejected to avoid overlapping OOS windows.
  - verdict maturity is OOS-only: diagnostic/provisional/validation-grade labels use distinct OOS days, not full input/archive days.
  - random same-frequency controls are mandatory for pass-eligible verdicts and are sampled from the OOS population.
  - dumb baseline scorecard supports primary `baseline_return_bps` plus additional `*_baseline_return_bps` columns; verdicts fail unless the strategy beats every supplied baseline OOS.
  - day/symbol/hour concentration gates are tracked at OOS and per-window levels.
  - invalid timestamps, symbols, strategy returns, primary baseline returns, optional baseline returns, and malformed eligible flags are counted and fail closed.
  - `--allow-fail-diagnostic` only permits clean insufficient-sample diagnostics; it still exits nonzero on invalid fields or real provisional/validation failures.

docs/experiments/walk-forward-validation/README.md
  - diagnostic README explicitly says: no alpha claims, no threshold tuning on current sample, and PASS is not trade permission.
```

Current bootstrap diagnostic report:

```text
uv run python scripts/run_pacifica_walk_forward_validation.py --bootstrap-if-missing --allow-fail-diagnostic
verdict=INSUFFICIENT_SAMPLE_DIAGNOSTIC
failure_reasons=insufficient_oos_distinct_days;no_oos_validation_rows;no_purged_validation_windows
```

TDD / verification:

```text
RED:
  uv run pytest tests/scripts/test_run_pacifica_walk_forward_validation.py -q
  failed on OOS-only maturity, hour concentration, dumb baseline scorecard, optional-baseline invalid accounting, and missing diagnostic README artifact assertions.

GREEN / current verification:
  python -m py_compile scripts/run_pacifica_walk_forward_validation.py tests/scripts/test_run_pacifica_walk_forward_validation.py
  uv run pytest tests/scripts/test_run_pacifica_walk_forward_validation.py -q
  16 passed

Focused changed-test sweep:
  python -m py_compile scripts/build_pacifica_source_manifest.py scripts/build_pacifica_full_fidelity_silver.py scripts/build_non_hft_regime_state.py scripts/verify_pacifica_side_by_side_refresh.py scripts/build_pacifica_regime_governor.py scripts/run_pacifica_walk_forward_validation.py tests/scripts/test_build_pacifica_source_manifest.py tests/scripts/test_build_pacifica_full_fidelity_silver.py tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_verify_pacifica_side_by_side_refresh.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_run_pacifica_walk_forward_validation.py
  uv run pytest tests/scripts/test_build_pacifica_source_manifest.py tests/scripts/test_build_pacifica_full_fidelity_silver.py tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_verify_pacifica_side_by_side_refresh.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_run_pacifica_walk_forward_validation.py -q
  75 passed

Read-only canonical verifier self-check:
  uv run python scripts/verify_pacifica_side_by_side_refresh.py --canonical-silver-dir data/pacifica_silver_partitioned --candidate-silver-dir data/pacifica_silver_partitioned --canonical-regime-dir docs/experiments/non-hft-regime-state --candidate-regime-dir docs/experiments/non-hft-regime-state --out-dir data/ops/pacifica-incremental-refresh-selfcheck-20260513T190145Z
  ok=True, failures=[]

Full scripts-suite attempt:
  uv run pytest tests/scripts -q
  timed out at 600s after 80%+ progress with no failure output.

Independent review:
  - staged walk-forward code/test diff reviewed by a fresh subagent.
  - passed=true, security_concerns=[], logic_errors=[].
```

## Latest 2026-05-13 diagnostic regime governor v2

Timestamp: `2026-05-13T17:58Z`

Implemented a TDD-tested diagnostic-only no-trade governor over the latest local non-HFT regime-state schema. This layer is not a strategy and does not make trade signals.

Updated code/report:

```text
scripts/build_pacifica_regime_governor.py
  - fixed state set is now exactly:
    SKIP_STALE_DATA
    SKIP_WIDE_SPREAD
    SKIP_LOW_DEPTH
    SKIP_TOXIC_REGIME
    SKIP_MARK_ORACLE_DISLOCATION
    TRADABLE_DIAGNOSTIC
  - `TRADABLE_DIAGNOSTIC` means only "not blocked by this diagnostic governor"; it is not a trade signal, alpha claim, or permission to paper/live trade.
  - required latest-schema columns are:
    symbol, bucket_start_ms, bbo_updates, trade_count, trade_notional,
    price_updates, avg_spread_bps, top_depth_notional, toxicity_score,
    mark_oracle_basis_abs_bps.
  - fails closed on missing required columns, null keys, NaN/non-numeric safety metrics, stale BBO/trade/price feeds, and invalid/non-finite thresholds.

docs/experiments/regime-governor/README.md
docs/experiments/regime-governor/decision_summary.csv
docs/experiments/regime-governor/thresholds.csv
```

Current diagnostic report from `docs/experiments/non-hft-regime-state/regime_state.parquet`:

```text
rows=519903
threshold_version=pacifica_governor_v2_fixed_diagnostic
SKIP_STALE_DATA=506956
SKIP_WIDE_SPREAD=203
SKIP_LOW_DEPTH=1017
SKIP_TOXIC_REGIME=0
SKIP_MARK_ORACLE_DISLOCATION=0
TRADABLE_DIAGNOSTIC=11727
```

TDD / verification:

```text
RED:
  uv run pytest tests/scripts/test_build_pacifica_regime_governor.py -q
  failed with 23 failures against the old v1 labels/thresholds/schema handling.

GREEN:
  uv run pytest tests/scripts/test_build_pacifica_regime_governor.py -q
  23 passed

Report generation:
  uv run python scripts/build_pacifica_regime_governor.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/regime-governor
  verdict=DIAGNOSTIC_GOVERNOR_RULES_ONLY
  rows=519903

Pre-commit verification at `2026-05-13T18:31Z`:
  - `git diff --cached --check` passed.
  - focused py_compile + pytest for source-manifest/silver/regime/verifier/governor passed: `59 passed`.
  - read-only canonical verifier self-check passed: `ok=True`, `failures=[]`.
  - full `uv run pytest tests/scripts -q` was attempted with a 600s cap; it timed out after 80%+ progress with no failure output, so focused changed-test verification is the trusted signal for this commit.
  - independent pre-commit review passed with no security concerns and no logic errors.
```

## Latest 2026-05-13 source-object manifest + incremental side-by-side refresh

Timestamp: `2026-05-13T16:17:04Z`

Implemented a TDD-tested source-object manifest and incremental candidate refresh path for the Pacifica research pipeline. The new manifest key is:

```text
channel=<channel>/symbol=<symbol>/date=<date>/hour=<hour>/run=<run>
```

New/updated code:

```text
scripts/build_pacifica_source_manifest.py
  - builds local raw source-object manifests from `data/pacifica_full_fidelity/**/*.jsonl.gz`.
  - marks chunks processable only when a valid `.jsonl.gz.sha256` sidecar exists.
  - optional row counting and payload SHA verification.
  - diffs previous/current manifests and emits only new/changed sealed source objects.
  - incremental silver requires planned rows with verified SHA and readable gzip row checks.

scripts/build_pacifica_full_fidelity_silver.py
  - new `--layout incremental` mode.
  - writes deterministic per-source-object candidate parquet chunks under `channel=/symbol=/date=/hour=/run=/part.parquet`.
  - can seed a side-by-side candidate from canonical silver via `--base-silver-dir` only when the base uses partitioned/source-object layout; flat `<channel>.parquet` base seeds are refused fail-closed to avoid duplicate/stale affected rows.
  - writes `source_manifest.csv`, `incremental_plan.csv`, and candidate `quality_summary.csv` in the candidate out-dir.

scripts/build_non_hft_regime_state.py
  - now reads recursive hour/run silver chunk layouts.
  - new `--source-plan` mode processes only affected symbol/date partitions and writes `regime_state_delta.parquet`, not canonical `regime_state.parquet`.

scripts/verify_pacifica_side_by_side_refresh.py
  - compares canonical vs candidate silver/regime outputs.
  - checks row counts, symbol/date/channel coverage, missing/null keys, duplicate-key regressions, and report diffs.
  - fails closed on candidate regressions, missing required key columns, null keys, or duplicate-key regressions above canonical baseline.
```

Docs/runbook:

```text
docs/ops/pacifica-incremental-refresh/README.md
```

TDD / verification:

```text
RED:
  uv run pytest tests/scripts/test_build_pacifica_source_manifest.py -q
  failed with missing `scripts.build_pacifica_source_manifest`.

RED:
  uv run pytest tests/scripts/test_build_pacifica_full_fidelity_silver.py tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_verify_pacifica_side_by_side_refresh.py -q
  failed with missing incremental silver/regime/verifier functions.

GREEN / fail-closed hardening:
  python -m py_compile scripts/build_pacifica_source_manifest.py scripts/build_pacifica_full_fidelity_silver.py scripts/build_non_hft_regime_state.py scripts/verify_pacifica_side_by_side_refresh.py scripts/build_pacifica_regime_governor.py tests/scripts/test_build_pacifica_source_manifest.py tests/scripts/test_build_pacifica_full_fidelity_silver.py tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_verify_pacifica_side_by_side_refresh.py tests/scripts/test_build_pacifica_regime_governor.py
  uv run pytest tests/scripts/test_build_pacifica_source_manifest.py tests/scripts/test_build_pacifica_full_fidelity_silver.py tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_verify_pacifica_side_by_side_refresh.py tests/scripts/test_build_pacifica_regime_governor.py -q
  59 passed

Hardening added after independent review:
  - side-by-side verifier now fails on missing required key columns in non-empty candidate silver/regime tables.
  - incremental silver now refuses flat `<channel>.parquet` base seeds because affected rows cannot be safely removed before source-object chunks are added.
  - both CLIs now run correctly as `uv run python scripts/...` from the repo root.
  - read-only canonical self-check succeeded against current local canonical silver/regime outputs:
    `data/ops/pacifica-incremental-refresh-selfcheck-20260513T181133Z`, ok=True, failures=[].

Side-by-side smoke fixture:
  processed_source_objects=1
  planned_source_objects=1
  delta_rows=1
  verification_ok=True
  verification_failures=[]
```

Promotion rule: do not overwrite canonical `data/pacifica_silver_partitioned/` or canonical non-HFT regime-state outputs until a candidate side-by-side refresh has a green verifier report and Diego explicitly approves promotion. Candidate data/manifests under `data/` remain gitignored and should not be committed.

## Latest 2026-05-13 R2 archive-health parser/report upgrade

Timestamp: `2026-05-13T15:12Z`

Implemented and committed a safer R2 archive-health report path for raw payload/sidecar diagnostics:

```text
scripts/pacifica_r2_inventory.py
  - added line-oriented `rclone lsf --format pst --separator ';'` parser helpers.
  - added streaming `write_inventory_csv_from_lsf_stream(...)` so large listings can be written line-by-line instead of captured/truncated through tool stdout.
  - CLI now accepts `--lsf`, `--stream-lsf`, and `--key-prefix raw/pacifica/full_fidelity` in addition to old `--lsjson`.

scripts/check_pacifica_r2_archive_health.py
  - reports payload/`.sha256` pairing and orphan sidecars.
  - reports latest payload freshness age vs `--stale-after-min`.
  - reports current UTC-hour payload leakage.
  - reports channel/date/symbol coverage plus channel/date coverage CSVs.
  - supports read-only remote gzip sample verification via `rclone cat`: payload SHA-256 must match sibling `.sha256`, and gzip must decompress locally.
  - parses `rclone lsf` naive timestamps as process-local before UTC conversion, matching the known local-time rclone behavior.
  - writes local Markdown/CSV artifacts only; no R2 write/delete path was added.
```

Generated report artifacts:

```text
docs/ops/pacifica-r2-archive-health/README.md
docs/ops/pacifica-r2-archive-health/prefix_summary.csv
docs/ops/pacifica-r2-archive-health/channel_coverage.csv
docs/ops/pacifica-r2-archive-health/date_coverage.csv
docs/ops/pacifica-r2-archive-health/channel_date_symbol_coverage.csv
docs/ops/pacifica-r2-archive-health/missing_sidecars.csv
docs/ops/pacifica-r2-archive-health/orphan_sidecars.csv
docs/ops/pacifica-r2-archive-health/active_hour_objects.csv
docs/ops/pacifica-r2-archive-health/latest_remote_objects.csv
docs/ops/pacifica-r2-archive-health/remote_gzip_sample_verification.csv
docs/ops/pacifica-r2-archive-health/gzip_integrity_audit.csv
```

Committed report run scope: bounded live sample, not full-bucket proof. The committed CSV was built from line-oriented listings for six high-signal prefixes over today/yesterday: `bbo/BTC`, `book/ETH`, `trades/BTC`, `mark_price_candle/ICP`, `prices/BTC`, and `candle/BTC`. Full recursive live listing attempts were intentionally not used after they exceeded the foreground cap / remained too slow; partial files under `data/ops/pacifica-r2-archive-health/*.partial_*` are gitignored local diagnostics only and must not be interpreted as evidence.

Bounded live report result at `2026-05-13T15:19:05Z`:

```text
ok=False
failures=['R2_GZIP_SAMPLE_VERIFICATION_FAILED', 'R2_SIDECAR_MISSING']
payload_objects=276
sidecar_objects=272
missing_sidecars=4
orphan_sidecars=0
active_current_hour_payload_objects=0
latest_payload=raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz
latest_payload_mod_time=2026-05-13T13:03:10+00:00
latest_payload_age_min=135.73
stale_after_min=180.0
latest_payload_freshness_ok=True
distinct_channels=6
distinct_dates=2
distinct_symbols=3
remote_gzip_sample_ok=5/8
remote_gzip_sample_bad=3
write_or_delete_executed=False
```

Interpretation: bounded freshness is green, but the newest sampled `run-20260513T122511Z` payloads exposed sidecar lag. This should be rechecked after the fresh sidecar-repair lane runs; it is not evidence that R2 payload bytes are corrupt, because the failures are sidecar-missing verifier failures rather than bad gzip bytes.

Use for future full inventory without truncation:

```bash
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR=data/ops/pacifica-r2-archive-health
mkdir -p "$OUT_DIR"
rclone lsf r2:pacifica-trading-data/raw/pacifica/full_fidelity \
  --recursive --files-only --format pst --separator ';' \
  > "$OUT_DIR/r2_raw_inventory_${RUN_TS}.lsf"
uv run python scripts/pacifica_r2_inventory.py \
  --lsf "$OUT_DIR/r2_raw_inventory_${RUN_TS}.lsf" \
  --stream-lsf \
  --key-prefix raw/pacifica/full_fidelity \
  --out-csv "$OUT_DIR/r2_raw_inventory_${RUN_TS}.csv"
uv run python scripts/check_pacifica_r2_archive_health.py \
  --inventory-csv "$OUT_DIR/r2_raw_inventory_${RUN_TS}.csv" \
  --out-dir docs/ops/pacifica-r2-archive-health \
  --raw-prefix raw/pacifica/full_fidelity/ \
  --stale-after-min 180 \
  --remote-base r2:pacifica-trading-data \
  --gzip-sample-size 8
```

TDD / verification:

```text
RED:
  uv run pytest tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_check_pacifica_r2_archive_health.py -q
  failed with ImportError for missing streaming inventory helpers and gzip verifier.

GREEN:
  uv run pytest tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_check_pacifica_r2_archive_health.py -q
  12 passed

Focused regression:
  python -m py_compile scripts/pacifica_r2_inventory.py scripts/check_pacifica_r2_archive_health.py tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_check_pacifica_r2_archive_health.py
  uv run pytest tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_check_pacifica_r2_archive_health.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py -q
  23 passed
  git diff --check passed

Commit/push:
  806d84e feat(pacifica): harden R2 archive health report
  pushed to origin/main
```

Remaining caveat: this does not replace the lifecycle DB health counters. The report proves the bounded sampled R2 prefixes have payload/sidecar/gzip health and freshness at report time; a future full-bucket inventory still needs to complete cleanly before claiming archive-wide pairing/orphan coverage.

## Latest 2026-05-13 v28 lifecycle health counter exposure and bounded status check

Timestamp: `2026-05-13T14:30Z`

Reason: direct Fly SSH lifecycle DB inspection was blocked/denied earlier in the session, and should not be retried in the same form. To expose direct lifecycle DB facts safely through the normal Fly health path, version 28 adds read-only lifecycle health counters to `scripts/check_pacifica_full_fidelity_health.py`:

```text
- opens the lifecycle SQLite DB read-only (`mode=ro`) when present;
- preserves existing status counts by `archive_files.status`;
- adds `db_error_counts` with rows/bytes by status plus top status/error groups from `archive_files.error`;
- adds `recent_activity` over a configurable window (`--recent-window-min`, default 60) for first_seen, last_seen, uploaded, verified, and pruned files/bytes;
- marks health failed if lifecycle DB error rows are present;
- performs no remote/local delete and no lifecycle mutation.
```

TDD / verification:

```text
RED:
  uv run pytest tests/scripts/test_check_pacifica_full_fidelity_health.py -q
  failed with ImportError for missing db_error_counts/db_recent_activity

GREEN / post-commit verification:
  python -m py_compile scripts/check_pacifica_full_fidelity_health.py tests/scripts/test_check_pacifica_full_fidelity_health.py
  uv run pytest tests/scripts/test_check_pacifica_full_fidelity_health.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
  50 passed
  git diff --check passed
```

Commit/push/deploy:

```text
8661995 feat(pacifica): expose lifecycle health counters
pushed to origin/main
flyctl deploy . -c ops/fly/pacifica-full-fidelity/fly.toml --dockerfile ops/fly/pacifica-full-fidelity/Dockerfile --app pacifica-full-fidelity --remote-only
image=registry.fly.io/pacifica-full-fidelity:deployment-01KRGTS1E1VQQ537ZGDQJ34V48
machine=e2862502a76778
version=28
state=started
last_updated=2026-05-13T14:11:36Z
```

Post-deploy direct lifecycle DB counts exposed by Fly health output at `2026-05-13T14:19:33Z`:

```text
status counts:
  pruned:   30,279 files, 16,108,306,996 bytes
  sealed:   24,251 files, 13,442,444,800 bytes
  uploaded: 73,172 files, 38,871,497,457 bytes
  verified:  2,824 files,  1,388,265,113 bytes

error counts:
  rows_with_errors=0
  bytes_with_errors=0
  by_status={}
  top_errors=[]

recent_activity window=3600s:
  first_seen: 1,322 files, 554,246,631 bytes
  last_seen:  6,479 files, 2,988,398,876 bytes
  uploaded:   6,000 files, 4,560,126,427 bytes
  verified:       0 files, 0 bytes
  pruned:         0 files, 0 bytes

health:
  ok=true
  failures=[]
  free_gb=136.14
  unverified_gb=48.72
```

Recent lifecycle cycle evidence after deploy:

```text
2026-05-13T14:11:36Z lifecycle start
2026-05-13T14:12:03Z recent scan scanned=6016
2026-05-13T14:16:59Z fresh upload uploaded=2000 failed=0 skipped=0
2026-05-13T14:19:32Z sidecar repair sidecars_uploaded=2000 failed=0 skipped=0
2026-05-13T14:19:32Z full_scan_skipped reason=backlog_lane_not_due interval_s=21600
2026-05-13T14:19:32Z backlog_lane_skipped interval_s=21600
2026-05-13T14:19:32Z lifecycle complete
```

Cycle duration: about `7m56s`. In that cycle, fresh upload throughput was `2,000` files, sidecar repair throughput was `2,000` sidecars, verify throughput was `0`, and prune throughput was `0` because the slow safety/backlog lane was not due.

Backlog direction from the latest comparable direct health snapshots:

```text
2026-05-13T13:04:28Z -> 2026-05-13T14:19:33Z
sealed:   28,929 -> 24,251 files  (shrinking by 4,678 files)
uploaded: 67,172 -> 73,172 files  (growing by 6,000 files)
verified:  2,824 ->  2,824 files  (flat)
pruned:   30,279 -> 30,279 files  (flat)
```

Interpretation: fresh upload catch-up is shrinking sealed backlog, but uploaded/unverified backlog is still growing while the safety verify/prune lane is skipped. This is progress, not full recovery.

Bounded R2 freshness at `2026-05-13T14:28:56Z`:

```text
ok=true
failures=[]
latest_payload=channel=book/symbol=ETH/date=2026-05-13/hour=11/run-20260513T000011Z.jsonl.gz
latest_payload_modified=2026-05-13T12:09:24Z
latest_payload_age_min=139.54
payload_count=167
sidecar_count=167
sidecar_missing_count=0
```

Remaining watch items:

- R2 bounded freshness is currently green, but the uploaded/unverified backlog is still growing until the slow safety lane verifies/prunes. Do not call archive health fully recovered until verify/prune throughput is positive and R2 freshness survives a slow safety cycle.
- The next real slow safety lane still needs verification: logs should show `post_safety_fresh_catchup=true`, a second recent scan/upload/repair after the safety lane, and bounded R2 freshness remaining under 180 minutes.
- Direct Fly SSH should still not be retried in the denied form. Prefer the new Fly health output, Fly status/logs, uploaded watchdog reports, and bounded R2 samples.
- `docs/ops/pacifica-api-surface-watch/README.md` had an unrelated cron timestamp-only dirty change from the 08:04 local API watcher; do not accidentally include it in Pacifica lifecycle commits unless intentionally refreshing that artifact.

## Latest 2026-05-13 v27 post-safety fresh catch-up remediation

Timestamp: `2026-05-13T12:35Z`

Version 26 fixed missing `.sha256` sidecars in the frequent fresh lane, but R2 freshness still went stale after the next slow safety lane. Live bounded check at `2026-05-13T12:20:09Z` failed:

```text
ok=false
failure=R2_REMOTE_FRESHNESS_STALE
latest_payload=channel=bbo/symbol=BTC/date=2026-05-13/hour=07/run-20260513T000011Z.jsonl.gz
latest_payload_modified=2026-05-13T08:05:17Z
latest_payload_age_min=254.88
sidecar_missing_count=0
```

Root cause evidence:

```text
v26 Fly logs:
2026-05-13T10:37:14Z lifecycle start
2026-05-13T10:38:47Z recent scan scanned=5517
2026-05-13T10:46:38Z fresh upload uploaded=2000 failed=0
2026-05-13T10:48:26Z sidecar repair sidecars_uploaded=2000 failed=0
2026-05-13T11:09:55Z broad full scan scanned=99304
2026-05-13T12:04:24Z upload-verify uploaded=250 verified=500 failed=0
2026-05-13T12:04:54Z lifecycle complete
2026-05-13T12:10:38Z local health output completed
```

Interpretation: the slow safety lane was correctly gated to every 6h, but when it did run it still blocked the next fresh upload for ~1h16m, then the entrypoint ran a multi-minute local health check and slept. With `PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS=7200`, that is enough to push sampled R2 object freshness past the 180-minute SLO even though uploads and sidecar repair had no errors.

Implemented:

```text
scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - factored recent scan + fresh upload-batch + repair-sidecars into run_fresh_lane().
  - normal lifecycle still starts with the fresh lane.
  - if the slow safety lane runs, it now marks the safety marker and then runs a second post-safety fresh catch-up before returning to entrypoint health-check/sleep.
  - no remote/local deletion behavior changed.
  - no concurrent lifecycle writers added; the catch-up runs sequentially in the same lifecycle process and SQLite connection pattern.

tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  - regression: safety-lane cycles must run a second recent scan/upload-batch/repair-sidecars after prune and before returning.
```

Verification:

```text
RED:
  uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py::test_lifecycle_runs_post_safety_fresh_catchup_before_returning -q
  failed: expected 2 fresh scans, got 1

GREEN/focused:
  bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  python -m py_compile tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
  48 passed
  git diff --check passed

Post-commit hook reran formatting; post-commit verification repeated and stayed green:
  48 passed
```

Commit/push:

```text
5f1e260 fix(pacifica): refresh after slow safety lane
pushed to origin/main
```

Deployment:

```text
flyctl deploy . -c ops/fly/pacifica-full-fidelity/fly.toml --dockerfile ops/fly/pacifica-full-fidelity/Dockerfile --app pacifica-full-fidelity --remote-only
image=registry.fly.io/pacifica-full-fidelity:deployment-01KRGMPSPBMFZ8QC7G3282SQW6
machine=e2862502a76778
version=27
state=started
last_updated=2026-05-13T12:25:10Z
```

Post-deploy evidence:

```text
v27 first lifecycle cycle:
2026-05-13T12:25:10Z lifecycle start
2026-05-13T12:27:01Z recent scan scanned=5489
2026-05-13T12:30:48Z fresh upload uploaded=2000 failed=0
2026-05-13T12:32:39Z sidecar repair sidecars_uploaded=2000 failed=0
2026-05-13T12:32:39Z full_scan_skipped reason=backlog_lane_not_due
2026-05-13T12:32:39Z backlog_lane_skipped interval_s=21600
2026-05-13T12:32:39Z lifecycle complete

Bounded local R2 freshness at 2026-05-13T12:33:16Z:
  ok=true
  failures=[]
  latest_payload=channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=09/run-20260513T000011Z.jsonl.gz
  latest_payload_modified=2026-05-13T10:03:53Z
  latest_payload_age_min=149.40
  payload_count=159
  sidecar_count=159
  sidecar_missing_count=0
```

Remaining watch item:

- v27 proves normal fresh uploads recover the bounded R2 checker and has a regression test for post-safety catch-up. The next real slow safety lane is not expected until roughly 6h after the last marker (`~2026-05-13T18:04Z` from observed v26 safety completion). After that cycle, verify logs show `post_safety_fresh_catchup=true`, a second recent scan/upload/repair, and bounded R2 freshness still under 180 minutes.
- Fly SSH diagnostics were blocked/denied in this session; do not retry the denied SSH form. Continue using Fly status/logs, bounded R2 samples, uploaded watchdog reports, and local repo tests unless explicit approval is given.

## Superseded 2026-05-13 v26 sidecar repair fresh-lane remediation

Timestamp: `2026-05-13T00:05Z`

Version 25 fixed the broad full-scan/safety-lane blocker, but archive-health still had a separate sidecar risk: a payload could be marked/uploaded while a sibling `.sha256` sidecar was absent, and the old repair path only reset later mismatch errors during the slow safety lane. Implemented and deployed version 26 so the frequent fresh lane also repairs sidecars for already-uploaded rows without reuploading payloads.

Implemented:

```text
scripts/pacifica_full_fidelity_storage.py
  - added repair_uploaded_sidecars_batch(...).
  - added CLI subcommand: repair-sidecars.
  - regenerates `.sha256` sidecars from lifecycle DB checksums for rows with status='uploaded'.
  - copies only sidecar files via one bounded parallel `rclone copy`.
  - does not delete R2 objects, does not reupload payloads, and does not mark rows verified.

scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - runs repair-sidecars immediately after the frequent fresh upload lane.
  - defaults PACIFICA_FULL_FIDELITY_SIDECAR_REPAIR_LIMIT to the fresh upload limit.
  - keeps broad full scan / reset / upload-verify / prune gated behind the slow safety-lane due decision.

tests/scripts/test_pacifica_full_fidelity_storage.py
  - regression: repair_uploaded_sidecars_batch copies only sidecars for uploaded rows and leaves sealed rows alone.

tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  - regression: fresh cycle runs repair-sidecars even when the safety lane is skipped.
  - regression: repair-sidecars runs after upload-batch.
```

Verification:

```text
RED check before implementation:
  uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py::test_repair_uploaded_sidecars_batch_copies_only_sidecars_for_uploaded_rows ...
  failed with ImportError: cannot import name 'repair_uploaded_sidecars_batch'

Post-implementation focused checks:
  bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  python -m py_compile scripts/pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
  47 passed
  git diff --check
  passed

Post-commit hook reformatted files, then post-commit verification was rerun:
  47 passed
  git diff --check passed
```

Commit/push:

```text
c3a8701 fix(pacifica): repair uploaded sidecars in fresh lane
pushed to origin/main
```

Deployment:

```text
flyctl deploy . -c ops/fly/pacifica-full-fidelity/fly.toml --dockerfile ops/fly/pacifica-full-fidelity/Dockerfile --app pacifica-full-fidelity --remote-only
image=registry.fly.io/pacifica-full-fidelity:deployment-01KRFA2Y96NYVVDDQ7B4HD1C4W
machine=e2862502a76778
version=26
state=started
last_updated=2026-05-13T00:00:10Z
```

Immediate post-deploy evidence:

```text
Fly status at 2026-05-13T00:00:43Z:
  version=26
  state=started
  image=pacifica-full-fidelity:deployment-01KRFA2Y96NYVVDDQ7B4HD1C4W

Bounded local R2 freshness at 2026-05-13T00:00:46Z:
  ok=true
  failures=[]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=21/run-20260512T204313Z.jsonl.gz
  latest_payload_modified=2026-05-12T21:20:36Z
  latest_payload_age_min=160.18
  payload_count=105
  sidecar_count=105
  sidecar_missing_count=0
```

Remaining watch item:

- v26 is green immediately after deploy, but the latest sampled payload was still only ~160 minutes old against a 180-minute threshold. One-shot cron job `eb48017f12b6` is scheduled for ~2026-05-13T00:28Z to re-run the bounded checker across the boundary; do not treat the immediate green check alone as final proof.
- Backlog verification/pruning is still separate from fresh payload/sidecar repair. Do not start manual lifecycle writers while the Fly app lifecycle may be active; use bounded read-only checks first.

## Superseded 2026-05-12 v25 full-scan slow-lane gating remediation

Timestamp: `2026-05-12T21:30Z`

Version 24 was not sufficient: logs showed the lifecycle still ran a broad full archive scan after the fresh upload even when the backlog lane was skipped. That full scan delayed the next fresh cycle:

```text
2026-05-12T20:43:41Z recent scan scanned=5854
2026-05-12T20:47:15Z fresh upload uploaded=2000 failed=0
2026-05-12T21:06:34Z broad full scan scanned=93457
2026-05-12T21:06:37Z backlog_lane_skipped interval_s=21600
```

Implemented and deployed version 25:

```text
scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - compute BACKLOG_LANE_IS_DUE once after the fresh upload.
  - gate the broad full scan behind the same slow safety-lane due decision.
  - when the safety lane is not due, skip both broad full scan and reset/upload-verify/prune.
  - fresh recent scan + newest-first upload still run every lifecycle cycle.

tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  - added regression: missing full-scan marker must not trigger a broad scan when safety lane is not due.
```

Verification:

```text
uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py -q
3 passed

bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
46 passed

git diff --check
# passed
```

Commit/push:

```text
5ebd22e fix(pacifica): gate full scan behind safety lane
pushed to origin/main
```

Deployment:

```text
flyctl deploy . -c ops/fly/pacifica-full-fidelity/fly.toml --dockerfile ops/fly/pacifica-full-fidelity/Dockerfile --app pacifica-full-fidelity --remote-only
image=registry.fly.io/pacifica-full-fidelity:deployment-01KRF0Z6M2B8J4XWXPVS7G97G2
machine=e2862502a76778
version=25
state=started
last_updated=2026-05-12T21:20:49Z
```

Note: running `flyctl deploy --app pacifica-full-fidelity --remote-only` from the repo root picked up/resolved the configured Dockerfile path as stale `/Users/diego/ops/fly/pacifica-full-fidelity/Dockerfile`. Use the explicit deploy command above from the repo root, or fix the Fly config path before relying on bare deploy commands.

Post-deploy evidence:

```text
Fly status after deploy:
  version=25
  state=started
  image=pacifica-full-fidelity:deployment-01KRF0Z6M2B8J4XWXPVS7G97G2

v25 lifecycle logs:
  2026-05-12T21:20:49Z lifecycle start
  2026-05-12T21:21:05Z recent scan scanned=6453
  2026-05-12T21:25:30Z fresh upload uploaded=2000 failed=0
  2026-05-12T21:25:31Z full_scan_skipped reason=backlog_lane_not_due interval_s=21600
  2026-05-12T21:25:31Z backlog_lane_skipped interval_s=21600
  2026-05-12T21:25:31Z lifecycle complete

Bounded local R2 freshness at 2026-05-12T21:25:46Z:
  ok=true
  failures=[]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=18/run-20260512T170641Z.jsonl.gz
  latest_payload_modified=2026-05-12T19:09:14Z
  latest_payload_age_min=136.54
  payload_count=197
  sidecar_count=197
  sidecar_missing_count=0
```

Remaining watch item:

- v25 proves the current lifecycle cycle no longer blocks on a broad full scan when the safety lane is not due. It still needs a later check across the next 180-minute freshness boundary to prove the repeated cadence stays green.
- Backlog verification/pruning and the known historical missing sidecar remain separate archive-health work; do not start manual lifecycle writers while the Fly app lifecycle may be active.

## Superseded 2026-05-12 v24 safety-lane gating remediation

Timestamp: `2026-05-12T20:55Z`

After the diagnostic research refresh, R2 freshness had gone stale again even though Fly was running version 23. Root cause interpretation: v23 made the fresh upload itself fast, but the single sequential lifecycle loop could still spend hours in the slow full/backlog/verify/prune safety lane before starting the next fresh cycle.

Implemented and deployed version 24:

```text
scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - added PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S gating.
  - added PACIFICA_FULL_FIDELITY_BACKLOG_LANE_RUN_ON_MISSING_MARKER.
  - fresh scan/upload still runs every lifecycle cycle.
  - slow reset/upload-verify/prune safety lane can now be skipped until due.

ops/fly/pacifica-full-fidelity/entrypoint.sh and fly.toml
  - set PACIFICA_FULL_FIDELITY_BACKLOG_LANE_INTERVAL_S=21600.
  - set PACIFICA_FULL_FIDELITY_BACKLOG_LANE_RUN_ON_MISSING_MARKER=0 so the first v24 cycle resumes fresh uploads instead of immediately blocking on backlog work.

tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
  - regression tests for skipping the safety lane when not due and marking it when due.
```

Verification before deploy:

```text
uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py -q
2 passed

uv run pytest tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
38 passed

bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/entrypoint.sh
python -m py_compile scripts/pacifica_full_fidelity_storage.py scripts/check_pacifica_r2_freshness.py scripts/run_pacifica_fly_ops_watchdogs.py tests/scripts/test_run_pacifica_full_fidelity_r2_lifecycle.py
git diff --check
# passed
```

Deployment:

```text
flyctl deploy -c ops/fly/pacifica-full-fidelity/fly.toml --ha=false
image=registry.fly.io/pacifica-full-fidelity:deployment-01KREYSBXK6GE9F0PRNAVPR9TP
machine=e2862502a76778
version=24
state=started
last_updated=2026-05-12T20:43:12Z
```

Post-deploy evidence:

```text
Fly status at 2026-05-12T20:43:37Z:
  version=24
  state=started
  image=registry.fly.io/pacifica-full-fidelity:deployment-01KREYSBXK6GE9F0PRNAVPR9TP

Immediate R2 freshness at 2026-05-12T20:43:39Z was still stale, as expected before the new cycle finished.
R2 freshness after waiting for the v24 fresh lane at 2026-05-12T20:48:55Z:
  ok=true
  failures=[]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=17/run-20260512T170641Z.jsonl.gz
  latest_payload_modified=2026-05-12T18:05:30Z
  latest_payload_age_min=163.43
  payload_count=179
  sidecar_count=179
  sidecar_missing_count=0
```

Caveats:

- This verifies sampled R2 freshness recovered after v24; it does not prove full backlog verification/pruning is caught up.
- A direct spot check found one historical sidecar absent at `raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-11/hour=02/run-20260511T023605Z.jsonl.gz.sha256`. The later bounded freshness sample did not include it and returned green. Treat this as a separate archive-health repair item, not evidence that current fresh upload is broken.
- The slow safety lane is now less able to block freshness, but because it is gated to every 6h, backlog verification/pruning needs explicit monitoring.

## Superseded 2026-05-12 diagnostic research refresh + renewed R2 freshness blocker

Timestamp: `2026-05-12T20:40Z`

Diego said `continue`; the previous R2-to-local-cache refresh process `proc_815de4056c85` had completed successfully with exit code `0` and no stdout.

Diagnostic research refresh completed from the refreshed R2 cache without claiming edge:

```text
Raw local cache: data/pacifica_full_fidelity/
  files=88,782
  size=35.846 GiB
  channels=8 public/raw channels plus UNKNOWN control/status rows
  symbols=67
  dates=13: 2026-04-30 through 2026-05-12

Refreshed silver output, written to timestamped side directory:
  data/pacifica_silver_partitioned_refresh_20260512T1738Z/
  files=2,987
  size=1.609 GiB
  channels=bbo,book,candle,mark_price_candle,prices,trades
  symbols=66
  dates=13
  rows by channel:
    bbo=18,032,394
    book=34,973,130
    candle=2,265,103
    mark_price_candle=32,923,547
    prices=1,345,256
    trades=216,365

Refreshed 1-minute regime state:
  docs/experiments/non-hft-regime-state-refresh-20260512T1738Z/
  rows=848,863
  symbols=66
  dates=13
  min_bucket=2026-04-30 21:21 UTC
  max_bucket=2026-05-12 14:59 UTC

Fixed toxic overlay probe from refreshed regime state:
  docs/experiments/toxic-regime-overlay-refresh-20260512T1738Z/
  verdict=INSUFFICIENT_SAMPLE_DIAGNOSTIC
  distinct_dates=13
  default horizons/cutoffs unchanged: horizons=[5,15,30,60], cutoffs=[0.9,0.8,0.7]
  Do not tune cutoffs or claim edge from this diagnostic run.

Pre-trade eligibility gates from refreshed regime state:
  docs/experiments/paper-trading-eligibility-refresh-20260512T1738Z/
  verdict=INSUFFICIENT_SAMPLE_DIAGNOSTIC
  symbols_evaluated=66
  eligible_symbols=0
  gate counts:
    sample_gate_pass=0/66
    liquidity_gate_pass=24/66
    spread_cost_gate_pass=62/66
    activity_gate_pass=0/66
    stability_gate_pass=63/66
    concentration_gate_pass=65/66
```

Important caveats:

- The refreshed outputs above are timestamped diagnostic side artifacts. They were left untracked at handoff time and include large parquet files; do not blindly commit them.
- The canonical default `data/pacifica_silver_partitioned/` and default docs experiment directories were not swapped/overwritten during this pass. A direct swap command was blocked earlier; do not retry that exact destructive command form without explicit approval.
- The silver refresh exposed a next-level architecture gap: full raw rescans over ~35.8 GiB / ~89.8M normalized rows are slow. Add an incremental silver/regime refresh keyed by source object manifest or `(channel,symbol,date,hour,run)` before routine daily research rebuilds.

Renewed live ops blocker discovered after research refresh:

```text
Fly status at 2026-05-12T20:34Z:
  app=pacifica-full-fidelity
  machine=e2862502a76778
  version=23
  state=started
  image=registry.fly.io/pacifica-full-fidelity:deployment-01KREJD02600997WH8F9H7C53Z

Local bounded R2 freshness at 2026-05-12T20:34:23Z:
  ok=false
  failures=[R2_REMOTE_FRESHNESS_STALE]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=16/run-20260512T111943Z.jsonl.gz
  latest_payload_modified=2026-05-12T17:06:24Z
  latest_payload_age_min=208.0
  payload_count=159
  sidecar_count=159
  sidecar_missing_count=0

Uploaded watchdog artifact copied from R2:
  checked_at=2026-05-12T19:08:28Z
  ok=false
  failures=[R2_REMOTE_FRESHNESS_STALE]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=15/run-20260512T111943Z.jsonl.gz
  latest_payload_age_min=183.39
```

Interpretation: version 23 proved that batch fresh upload can upload a fresh batch quickly inside a lifecycle cycle, but R2 freshness still went stale later. The likely architecture issue is that the single sequential lifecycle loop runs fresh upload and then the slower full/backlog/verify/prune safety lane before it can start the next fresh cycle. The next ops fix should decouple or timebox the slow safety lane so a newest eligible fresh upload cycle can run reliably within the 180-minute threshold. Do not start competing manual lifecycle writers against the active SQLite DB.

## Latest 2026-05-12 v23 batch fresh-upload remediation

Timestamp: `2026-05-12T17:27Z`

Diego said to proceed with everything. Non-destructive work completed; destructive legacy R2 cleanup remains blocked unless Diego separately approves exact scope.

Live evidence before the fix:

```text
Fly status: pacifica-full-fidelity machine e2862502a76778 was started on version 22.
Bounded local R2 freshness at 2026-05-12T16:42:46Z:
  ok=false
  failures=[R2_REMOTE_FRESHNESS_STALE]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=12/run-20260512T111943Z.jsonl.gz
  latest_payload_modified=2026-05-12T13:06:41Z
  latest_payload_age_min=216.09
  payload_count=100
  sidecar_count=100
  sidecar_missing_count=0
Fly logs showed the version-22 15:26 lifecycle cycle had only reached recent scan by 15:27:25Z; no fresh upload result was visible by 17:00Z.
```

Root cause / interpretation: the recent scan was fast, but the fresh upload lane still launched one rclone process per payload and one per sidecar. A 2,000-object fresh batch could take long enough that sampled remote freshness crossed the 180-minute threshold while the app was still progressing safely. The issue was upload architecture/throughput, not disk capacity.

Implemented and deployed version 23:

```text
scripts/pacifica_full_fidelity_storage.py
  - added upload_pending_files_batch and CLI command upload-batch.
  - selects the same DB-safe upload candidates as the per-file uploader.
  - writes a payload --files-from list and a temporary sidecar mirror.
  - runs bounded parallel rclone copy for payloads and sidecars.
  - marks rows uploaded only after both batch copies succeed; failed batch rows stay unverified and get DB errors for retry.

scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - fresh lane now uses upload-batch with --transfers/--checkers.
  - backlog upload/verify remains on the conservative old path.

ops/fly/pacifica-full-fidelity/{entrypoint.sh,fly.toml}
  - added PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_TRANSFERS=16
  - added PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_CHECKERS=32
```

Deployment:

```text
flyctl deploy -c ops/fly/pacifica-full-fidelity/fly.toml --ha=false
image=registry.fly.io/pacifica-full-fidelity:deployment-01KREJD02600997WH8F9H7C53Z
machine=e2862502a76778
version=23
state=started
last_updated=2026-05-12T17:06:40Z
```

Post-deploy evidence:

```text
2026-05-12T17:06:40Z lifecycle scan/upload/verify/prune start
2026-05-12T17:07:30Z recent scan scanned=5442
2026-05-12T17:13:22Z batch fresh upload uploaded=1869 skipped=131 failed=0
2026-05-12T17:13:29Z reset-mismatch-errors reset=0 skipped_missing=0 skipped_recent=0

Bounded local R2 freshness at 2026-05-12T17:17:14Z:
  ok=true
  failures=[]
  latest_payload=channel=bbo/symbol=BTC/date=2026-05-12/hour=14/run-20260512T111943Z.jsonl.gz
  latest_payload_modified=2026-05-12T15:06:27Z
  latest_payload_age_min=130.8
  payload_count=123
  sidecar_count=123
  sidecar_missing_count=0
```

Verification:

```text
uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py -q
25 passed

python -m py_compile scripts/pacifica_full_fidelity_storage.py scripts/run_pacifica_fly_ops_watchdogs.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/entrypoint.sh
git diff --check
# passed

uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
36 passed
```

Caveats / next checks:

- The fresh upload lane is green again, but the version-23 lifecycle was still in the slower backlog upload/verify/prune lane after `17:13:29Z` in the latest sampled logs. Check for the next `upload-verify`, `prune`, and `lifecycle complete` lines.
- The uploaded hourly watchdog may remain stale until its next due run; the immediate local bounded checker is the current green evidence.
- A local R2-to-research-cache refresh was started in the current Hermes session as background process `proc_815de4056c85`: `rclone copy r2:pacifica-trading-data/raw/pacifica/full_fidelity data/pacifica_full_fidelity --transfers 16 --checkers 32 --fast-list --stats 30s`. It had no stdout yet at `2026-05-12T17:27Z`; poll it before assuming local research cache dates advanced beyond the old 2026-05-07 snapshot.
- Research artifacts are still from the old local cache snapshot (`8` dates through `2026-05-07`) until that cache refresh finishes and silver/regime/toxic/eligibility are rebuilt diagnostically.

## Latest 2026-05-12 R2 freshness follow-up

Timestamp: `2026-05-12T15:50Z`

Current read-only evidence:

```text
Fly status: started, version 22, image deployment-01KRDYHQ7A79GFCM2RGR08NEXM
Lifecycle evidence from logs:
  2026-05-12T13:51:22Z fresh upload uploaded=2000 failed=0 skipped=0
  2026-05-12T15:06:28Z backlog upload uploaded=250 failed=0 skipped=0; verify verified=500 failed=0 skipped=0
  2026-05-12T15:06:30Z lifecycle complete
  2026-05-12T15:26:13Z next lifecycle scan/upload/verify/prune started
Bounded local R2 freshness check after timezone parser fix:
  checked_at=2026-05-12T15:48:37Z
  ok=true
  failures=[]
  latest_payload=channel=book/symbol=ETH/date=2026-05-12/hour=12/run-20260512T111943Z.jsonl.gz
  latest_payload_modified=2026-05-12T13:06:41Z
  latest_payload_age_min=161.94
  payload_count=95
  sidecar_count=95
  sidecar_missing_count=0
```

Important parser fix: `rclone lsf --format t` renders timestamps in the caller's local timezone without an offset. The local laptop is `EST -0500`, so treating the string as UTC made the local bounded checker falsely report ~7.5h stale when the same object was ~2.7h old in UTC. `scripts/check_pacifica_r2_freshness.py` now parses rclone timestamps as process-local time and converts to UTC before freshness math.

Caveats:

- The uploaded watchdog artifact at `ops/pacifica/full_fidelity/watchdogs/latest/pacifica-r2-freshness/latest_status.json` was still stale at `2026-05-12T15:23:50Z`, before the latest observed hour=12 sample was visible locally.
- A direct Fly-side ad hoc SSH check was attempted but not completed because the shell-wrapped command was blocked; do not retry that exact command form. The local `TZ=UTC` run matched the expected Fly/UTC interpretation and returned `ok=true`.
- Do not start competing manual lifecycle upload/verify writers. Let the scheduled lifecycle continue and let the next hourly ops watchdog confirm the recovered R2 freshness.

Next exact check:

```text
uv run python scripts/check_pacifica_r2_freshness.py --remote-base r2:pacifica-trading-data --r2-prefix raw/pacifica/full_fidelity --stale-after-min 180 --timeout-s 45
rclone copyto r2:pacifica-trading-data/ops/pacifica/full_fidelity/watchdogs/latest/pacifica-r2-freshness/latest_status.json /tmp/pacifica-r2-freshness-latest.json
python -m json.tool /tmp/pacifica-r2-freshness-latest.json
```

Expected: local checker should remain `ok=true`; uploaded watchdog should flip to `ok=true` on its next hourly run if the 15:26 lifecycle upload path stays healthy.

## Latest 2026-05-12 bounded freshness-lane/watchdog update

Timestamp: `2026-05-12T11:45Z`

Deployed Fly image/version:

```text
Image: pacifica-full-fidelity:deployment-01KRDYHQ7A79GFCM2RGR08NEXM
Machine: e2862502a76778
Version: 22
State: started
Last updated: 2026-05-12T11:19:43Z
```

Why this deployment exists: the previous newest-first lane still required a broad lifecycle scan before the upload phase and could let too-recent sealed rows consume upload limits. R2 freshness remained stale even though local collection and lifecycle progress were otherwise healthy.

What changed:

- `scan_archive_files(..., recent_hours=...)` and CLI `scan --recent-hours N` now scan only bounded recent UTC hour partitions for the freshness lane.
- The lifecycle script now runs a fast recent scan and newest-first fresh upload before the slower full-scan/backlog/verify/prune safety lane.
- Full archive scans are marker-gated by `PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S=21600` so each 15m cycle does not restat/hash the entire archive.
- Upload selection now filters rows younger than `PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS=7200` in SQL, so too-recent chunks do not spend the upload limit.
- Backlog verification remains non-destructive and bounded separately with `PACIFICA_FULL_FIDELITY_BACKLOG_UPLOAD_LIMIT=250` and `PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500`.
- Added `scripts/check_pacifica_r2_freshness.py`, a bounded read-only R2 sample checker that verifies latest sampled payload freshness and payload/`.sha256` pairing.
- The Fly ops watchdog now runs the bounded R2 freshness checker hourly and writes `/data/ops/pacifica-r2-freshness/latest_status.json` before durable report upload.

Current Fly env additions in `ops/fly/pacifica-full-fidelity/fly.toml`:

```text
PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_LIMIT=2000
PACIFICA_FULL_FIDELITY_BACKLOG_UPLOAD_LIMIT=250
PACIFICA_FULL_FIDELITY_RECENT_SCAN_HOURS=12
PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S=21600
PACIFICA_FULL_FIDELITY_UPLOAD_ORDER=newest-first
PACIFICA_R2_FRESHNESS_CHECK_INTERVAL_S=3600
PACIFICA_R2_FRESHNESS_STALE_AFTER_MIN=180
PACIFICA_R2_FRESHNESS_SAMPLE_PREFIXES=channel=bbo/symbol=BTC,channel=book/symbol=ETH,channel=trades/symbol=BTC,channel=mark_price_candle/symbol=ICP
```

Verification before/after deploy:

```text
uv run pytest tests/scripts/test_build_pacifica_event_risk_calendar.py tests/scripts/test_build_pacifica_paper_ledger.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_validate_pacifica_idea_registry.py tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
124 passed in 1.94s

uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py tests/scripts/test_check_pacifica_r2_freshness.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
33 passed in 0.50s

python -m py_compile scripts/build_pacifica_event_risk_calendar.py scripts/build_pacifica_paper_ledger.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_regime_governor.py scripts/build_pacifica_symbol_lifecycle.py scripts/check_pacifica_feature_parity.py scripts/check_pacifica_r2_freshness.py scripts/plan_pacifica_ops_alerts.py scripts/run_pacifica_walk_forward_validation.py scripts/simulate_pacifica_execution.py scripts/validate_pacifica_idea_registry.py scripts/pacifica_full_fidelity_storage.py scripts/pacifica_r2_inventory.py scripts/run_pacifica_fly_ops_watchdogs.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/entrypoint.sh
git diff --check
# all passed
```

Caveat: `uv run pytest tests/scripts -q` timed out after 600s at about 69% progress in this working tree. The targeted new/changed test set above passed; do not report the full scripts suite as green until the slow/hanging remainder is isolated or run with a longer budget.

Post-deploy observations:

```text
Fly status at 2026-05-12T11:28Z: started, version 22, image deployment-01KRDYHQ7A79GFCM2RGR08NEXM
New lifecycle started: 2026-05-12T11:19:43Z
New ops watchdog started: 2026-05-12T11:19:43Z
Ops watchdog reported failures at 2026-05-12T11:20:03Z (expected while R2 freshness is still stale)
Recent bounded lifecycle scan completed: 2026-05-12T11:21:20Z scanned=4752
No post-version-22 fresh-upload/verify completion was visible in bounded logs by 2026-05-12T11:45Z.
```

Bounded R2 freshness check from local read-only script at `2026-05-12T11:28:54Z`:

```text
ok=false
failures=[R2_REMOTE_FRESHNESS_STALE]
latest_payload=channel=bbo/symbol=BTC/date=2026-05-12/hour=08/run-20260511T150931Z.jsonl.gz
latest_payload_modified=2026-05-12T04:03:57Z
latest_payload_age_min=444.95
payload_count=79
sidecar_count=79
sidecar_missing_count=0
listing_errors=[]
```

Interpretation: version 22 deployed and started cleanly; the new bounded recent scan is active and much smaller than the previous full scan, but R2 archive freshness has not recovered yet. Keep alert severity fail-closed until a later bounded R2 freshness check is under the 180-minute threshold and a lifecycle upload/verify line confirms the version-22 cycle reached upload completion. Do not start competing manual upload/verify writers against the active lifecycle SQLite DB.

## Latest 2026-05-11 lifecycle/freshness-lane update

Timestamp: `2026-05-11T15:15Z`

Deployed Fly image/version:

```text
Image: pacifica-full-fidelity:deployment-01KRBS9KFXCN3G9XDBE5XQJ520
Machine: e2862502a76778
Version: 21
State: started
Last updated: 2026-05-11T15:09:30Z
```

Why this deployment exists: a 12h follow-up showed the collector/lifecycle was healthy but R2 remained stale for bounded May 11 BTC prefixes. The earlier `newest-first` upload lane still ordered by `last_seen_at`, which is refreshed for all rows on every scan; when all rows shared the same scan timestamp, object-key tie-breaking could still walk lexicographic backlog instead of truly newest sealed chunks.

What changed:

- `archive_files` now has a `modified_at` column, auto-added by SQLite migration if missing.
- `scan_archive_files` records local payload mtime and reuses the saved SHA-256 for unchanged `(size_bytes, modified_at)` rows instead of rehashing every file every cycle.
- `--upload-order newest-first` now orders by `coalesce(modified_at, last_seen_at) desc`, preserving errored-uploaded repair priority before new sealed rows.
- `upload-verify` now supports split upload/verify limits, and Fly is configured with:
  - `PACIFICA_FULL_FIDELITY_UPLOAD_LIMIT=2000`
  - `PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500`
- Rationale for split limits: prioritize newest sealed payload visibility in R2 while reducing per-cycle verification read/metadata calls. Verification/prune still runs every cycle, just at a lower cap than upload catch-up.
- Tests prove these behaviors:
  - newest-first chooses the newest mtime even when object-key order would choose an older `symbol=ZZZ` file;
  - unchanged files are not rehashed on rescan;
  - upload and verify limits can differ, e.g. upload 3 rows while verifying only 1.

Verification before deploy:

```text
uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py -q
21 passed
python -m py_compile scripts/pacifica_full_fidelity_storage.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
bash -n ops/fly/pacifica-full-fidelity/entrypoint.sh
git diff --check -- scripts/pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_full_fidelity_storage.py scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/fly.toml ops/fly/pacifica-full-fidelity/entrypoint.sh
```

Post-deploy observations:

```text
Fly status: started, version 21, image deployment-01KRBS9KFXCN3G9XDBE5XQJ520
New lifecycle started: 2026-05-11T15:09:30Z
Runtime env verified over Fly SSH:
  PACIFICA_FULL_FIDELITY_BATCH_LIMIT=2000
  PACIFICA_FULL_FIDELITY_UPLOAD_LIMIT=2000
  PACIFICA_FULL_FIDELITY_VERIFY_LIMIT=500
  PACIFICA_FULL_FIDELITY_UPLOAD_ORDER=newest-first
Previous version-20 first migration scan:
  start 2026-05-11T12:39:30Z
  scanned=86666 at 2026-05-11T14:39:43Z
  reset=0 at 2026-05-11T14:39:49Z
SQLite schema includes modified_at column.
DB counts right after version-20 migration check:
  pruned   17779  7852094488 bytes
  sealed   76027  43408115236 bytes
  uploaded 1207   15834297 bytes
  verified 8000   5677461625 bytes
```

Caveat: the first post-migration scan was expensive because existing DB rows started with `modified_at = null`; it populated mtimes during the version-20 cycle. Subsequent scans should avoid full rehashing unchanged rows and should make lifecycle effective cadence closer to the configured interval. Version 21 intentionally reduces verification from 2000 to 500 rows per cycle while keeping upload at 2000 rows per cycle; this should improve R2 freshness and reduce Cloudflare read/metadata calls, but verified/pruned backlog will catch up more slowly. Do not claim R2 freshness recovered until a bounded R2 sample shows current sealed May 11 payloads and sidecars; the pre-deploy bounded BTC date=2026-05-11 sample returned zero objects for BTC bbo/mark_price_candle/trades.

Preserve these unless Diego explicitly approves deletion:

- `pacifica-full-fidelity`
- `r2:pacifica-trading-data/raw/`
- `/data/pacifica_full_fidelity_storage.sqlite`
- local research artifacts unless intentionally refreshing

## Latest operational check

Timestamp: `2026-05-08T01:55Z`

Fly status:

```text
App: pacifica-full-fidelity
Machine: e2862502a76778
Region: iad
State: started
Image: pacifica-full-fidelity:deployment-01KQW7TZFH27DWBGD0HHF6PSW6
Last updated: 2026-05-05T14:15:54Z
```

Latest observed Fly health JSON from logs:

```text
checked_at=2026-05-08T01:51:05.214052+00:00
ok=true
failures=[]
free_gb=56.45
unverified_gb=34.69
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=BP/date=2026-05-08/hour=01/run-20260505T141555Z.jsonl.gz
newest_raw_age_min=-3.19
```

Latest lifecycle DB counts from health logs:

```text
pruned|10537|3249105572
sealed|57343|37251568677
verified|1681|706023406
rows_with_errors: not directly queried in this pass; recent lifecycle logs showed upload failed=0 and verify failed=0.
```

Latest lifecycle evidence:

```text
2026-05-08T01:31:08Z scanned=59159 state_db=/data/pacifica_full_fidelity_storage.sqlite
2026-05-08T01:31:13Z reset=0 skipped_missing=0 skipped_recent=0 dry_run=false
2026-05-08T01:50:59Z upload failed=0 skipped=66 uploaded=134; verify failed=0 skipped=0 verified=134
2026-05-08T01:51:04Z lifecycle complete
```

Ops watchdog evidence:

```text
2026-05-08T00:56:54Z ops watchdog run start
2026-05-08T00:57:10Z ops watchdog run complete
```

Latest uploaded watchdog status read from R2:

```text
checked_at=2026-05-07T23:56:49.935913+00:00
ok=true
operation=noop_not_due
returncode=0
stdout_tail=No watchdog operation due.
```

Interpretation: the active Fly collector is running, lifecycle upload/verify/prune is healthy in recent logs, free disk remains above Diego's 50 GiB floor, and recent raw freshness is good. A transient WebSocket reconnect (`no close frame received or sent`) appeared at 2026-05-08T00:04Z, but subsequent lifecycle/health output remained `ok=true` with fresh raw files.

## R2 raw archive smoke check

Top-level R2 prefixes at `2026-05-08T00:29Z`:

```text
app/
funding/
ops/
raw/
```

Active raw prefix channel dirs at `r2:pacifica-trading-data/raw/pacifica/full_fidelity`:

```text
channel=bbo/
channel=book/
channel=candle/
channel=mark_price_candle/
channel=pong/
channel=prices/
channel=subscribe/
channel=trades/
rest/
```

Bounded R2 sample:

```text
r2:pacifica-trading-data/raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-07
count=42 objects
bytes=62866838
latest sampled payload=hour=20/run-20260505T141555Z.jsonl.gz
sample bytes=2252576
sha256=3f55663f4d69e34ae7d540697587b950b63d8ee3fd44b34fdb7180c4a49a0eb7
sidecar matched the payload hash
sample gzip decompressed/read successfully for at least 10 JSONL rows
```

Interpretation: this is a bounded R2 smoke check, not full-bucket proof. It independently confirms active raw payload + sidecar presence and gzip readability for a recent BTC BBO object.

## Legacy cleanup status

Done:

- Legacy Fly app `pacifica-collector` destroyed.
- Legacy GitHub Action `.github/workflows/daily_sync.yml` removed.
- Legacy helper scripts removed:
  - `scripts/sync_cloud_data.sh`
  - `scripts/sync_launch.sh`
  - `scripts/sync_remote.py`
- Removal committed as `c80a3f0 chore: remove legacy Pacifica collector GitHub Action`.
- Local laptop legacy dirs verified missing:
  - `data/prices`
  - `data/orderbook`
  - `data/trades`
  - `data/funding`
  - `data/app`

R2 legacy purge status:

- Top-level R2 prefixes at `2026-05-08T00:29Z` still included `app/` and `funding/`.
- `prices/`, `orderbook/`, and `trades/` remained absent from the top-level prefix listing.
- A bounded legacy `funding/` listing timed out in this pass, so do not infer its object count/size.

Interpretation:

```text
prices/    cleared
orderbook/ cleared
trades/    cleared
app/       still present
funding/   still present
raw/       preserved active archive
ops/       preserved/non-target
```

Do not claim R2 legacy cleanup complete until `app/` + `funding/` are verified gone/empty. Any new destructive purge needs explicit Diego approval and must preserve `raw/` and `ops/`.

## R2 raw archive / local research cache snapshot

Refreshed local research raw cache from active R2 raw prefix on 2026-05-08:

```text
rclone copy r2:pacifica-trading-data/raw/pacifica/full_fidelity data/pacifica_full_fidelity --transfers 16 --checkers 32 --fast-list --stats 30s
```

Local cache inventory after refresh:

```json
{
  "path": "data/pacifica_full_fidelity",
  "files": 34834,
  "payloads": 17417,
  "sha256_sidecars": 17417,
  "gib": 21.605,
  "symbols": 66,
  "dates": ["2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07"]
}
```

Earlier read-only raw archive reports remain at:

- `docs/ops/pacifica-r2-raw-health-latest/README.md`
- `docs/ops/pacifica-r2-raw-health-latest/summary.json`

A long `rclone lsf ... --recursive` inventory process from an earlier session was killed after the local health summary was produced; do not rely on partial `data/pacifica_r2_raw_health/raw_lsf_pst.txt` as final inventory.

## Research refresh completed

Restored current raw archive from R2 to local `data/pacifica_full_fidelity/`, then rebuilt silver/regime/toxic/eligibility artifacts without changing thresholds or gates.

Command chain completed successfully:

```text
uv run python scripts/build_pacifica_full_fidelity_silver.py --raw-dir data/pacifica_full_fidelity --out-dir data/pacifica_silver_partitioned --layout partitioned --chunk-size 250000
uv run python scripts/build_non_hft_regime_state.py --silver-dir data/pacifica_silver_partitioned --out-dir docs/experiments/non-hft-regime-state
uv run python scripts/non_hft_toxic_overlay_probe.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/toxic-regime-overlay
uv run python scripts/build_pacifica_eligibility_gates.py --state docs/experiments/non-hft-regime-state/regime_state.parquet --out-dir docs/experiments/paper-trading-eligibility
```

Output:

```text
bbo: 12335891 rows
book: 12738965 rows
candle: 1723416 rows
mark_price_candle: 21521071 rows
prices: 998593 rows
trades: 90446 rows
wrote silver tables to data/pacifica_silver_partitioned
wrote 519903 regime-state rows to docs/experiments/non-hft-regime-state
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
wrote report: docs/experiments/toxic-regime-overlay/README.md
verdict: INSUFFICIENT_SAMPLE_DIAGNOSTIC
symbols_evaluated: 65
eligible_symbols: 0
wrote report: docs/experiments/paper-trading-eligibility/README.md
```

Current research reports:

- `docs/experiments/non-hft-regime-state/README.md`
- `docs/experiments/toxic-regime-overlay/README.md`
- `docs/experiments/paper-trading-eligibility/README.md`
- `docs/experiments/paper-trading-economics-baselines/README.md`
- `docs/experiments/execution-simulator/README.md`
- `docs/experiments/paper-ledger/README.md`
- `docs/experiments/regime-governor/README.md`
- `docs/experiments/feature-parity/README.md`

System level-up plan:

- `docs/plans/2026-05-08-pacifica-system-level-up.md`

Interpretation discipline:

- Archive has 8 distinct dates, still diagnostic.
- Toxic probe verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Eligibility verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Eligible symbols: `0`, mainly due to sample/activity gates.
- Do not tune toxicity thresholds on this sample.
- Do not treat the diagnostic toxicity/probe output as an edge claim.

## Economics/baselines contract added

Existing report:

- `docs/experiments/paper-trading-economics-baselines/README.md`

Locked baseline assumptions before strategy work:

- Taker fee: 4 bps per side.
- Maker fee: 1.5 bps per side.
- Taker/taker round trip before slippage: 8 bps.
- Maker/maker round trip before adverse selection: 3 bps.
- Taker/maker round trip before slippage/adverse selection: 5.5 bps.
- Every backtest/paper run must include fees, slippage/adverse selection, funding, dumb baselines, random same-frequency controls, drawdown, Sortino, trade/day/symbol concentration, and post-cost PnL.

## Tests/checks

Completed after the 2026-05-08 research refresh:

```text
uv run pytest tests/scripts/test_build_non_hft_regime_state.py tests/scripts/test_non_hft_toxic_overlay_probe.py tests/scripts/test_build_pacifica_eligibility_gates.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
21 passed in 0.28s

python -m py_compile scripts/build_pacifica_full_fidelity_silver.py scripts/build_non_hft_regime_state.py scripts/non_hft_toxic_overlay_probe.py scripts/build_pacifica_eligibility_gates.py scripts/run_pacifica_fly_ops_watchdogs.py
# passed
```

## Watchdog R2 inventory fix

Completed on 2026-05-08:

- Replaced due watchdog raw inventory from full recursive `rclone lsjson` with line-oriented `rclone lsf --recursive --files-only --format pst --separator ';'` streamed to `/data/ops/r2_inventory.lsf`.
- Added `rclone_lsf_to_inventory` / `write_inventory_csv_from_lsf` conversion with LF-normalized CSV output.
- Updated watchdog summary output to record both `r2_inventory.lsf` and `r2_inventory.csv`.
- Added regression tests for line-oriented parsing, LF CSV output, and `run_once` command selection so the watchdog does not regress to `lsjson`.
- Added `pandas` to the Fly image because the existing retention planner imports pandas.
- Deployed to Fly image `pacifica-full-fidelity:deployment-01KR2RTJZGAMGB2S9NB2KXEBWF`; machine `e2862502a76778` reached `started` / good state at `2026-05-08T03:08:07Z`.
- Post-deploy logs showed `ops watchdog run start` at `2026-05-08T03:08:07Z` and `ops watchdog run complete` at `2026-05-08T03:08:15Z`.
- Latest uploaded R2 watchdog status after deploy was `ok=true` with `operation=noop_not_due` at `2026-05-08T03:08:08.535213+00:00`.

Verification:

```text
uv run pytest tests/scripts/test_pacifica_r2_inventory.py tests/scripts/test_plan_pacifica_r2_retention.py tests/scripts/test_run_pacifica_fly_ops_watchdogs.py -q
13 passed in 0.40s

python -m py_compile scripts/pacifica_r2_inventory.py scripts/plan_pacifica_r2_retention.py scripts/run_pacifica_fly_ops_watchdogs.py
# passed

git diff --check
# passed

Local bounded smoke using BTC/BBO 2026-05-07 R2 prefix:
r2_inventory_lsf ok=true returncode=0 wrote 3082 bytes
r2_inventory_csv ok=true returncode=0
r2_retention_plan ok=true returncode=0 objects=46 eligible_for_review=0 delete_command_written=False
```

Note: the first deploy attempt `deployment-01KR2RNZY91HA5TNF6NH8JDYJG` exposed a Fly image dependency gap (`pandas` missing for retention planning). The second deploy fixed it. Do not remove pandas from the Fly image unless the retention planner is rewritten to stdlib.

## 12h Fly/R2 health check — 2026-05-08T12:31Z

Live status:

- Fly app `pacifica-full-fidelity` is running image `pacifica-full-fidelity:deployment-01KR2RTJZGAMGB2S9NB2KXEBWF`.
- Machine `e2862502a76778` is `started` in `iad`, version `17`, last updated `2026-05-08T03:08:07Z`.
- Latest uploaded watchdog status copied from R2: `checked_at=2026-05-08T12:12:20.493277+00:00`, `ok=true`.
- Latest watchdog due operation was `api_surface_watch`, `ok=true`, `returncode=0`, `changed=False`.
- Watchdog logs continued hourly after the deploy fix: latest observed run start `2026-05-08T12:11:46Z`, complete `2026-05-08T12:12:36Z`.
- Latest complete lifecycle cycle observed in logs finished at `2026-05-08T10:52:32Z` with `upload.failed=0`, `upload.uploaded=142`, `verify.failed=0`, `verify.verified=142`.
- Latest health JSON observed after that lifecycle: `checked_at=2026-05-08T10:52:33.377594+00:00`, `ok=true`, `free_gb=54.31`, `newest_raw_age_min=-2.71`, `failures=[]`, `sealed.files=61393`, `pruned.files=11146`, `verified.files=1659`, `unverified_gb=36.99`.
- A newer lifecycle cycle had started at `2026-05-08T11:25:20Z`; by `2026-05-08T12:37:05Z` it had scanned `63836` rows and reset `0` rows, but upload/verify completion had not yet appeared in the sampled log tail.

R2 checks:

- Top-level R2 prefixes remain: `app/`, `funding/`, `ops/`, `raw/`. Legacy `app/` and `funding/` still exist; no destructive cleanup was run.
- Raw prefix dirs include `channel=bbo/`, `channel=book/`, `channel=candle/`, `channel=mark_price_candle/`, `channel=pong/`, `channel=prices/`, `channel=subscribe/`, `channel=trades/`, and `rest/`.
- Ops prefix still has `watchdogs/latest/`.
- Bounded R2 sample check on `raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-08` found `18` objects: `9` payloads and `9` `.sha256` sidecars.
- Latest sampled BTC/BBO payload: `channel=bbo/symbol=BTC/date=2026-05-08/hour=07/run-20260508T030808Z.jsonl.gz`; `.sha256` matched and gzip JSONL read 10 rows successfully.
- A full recursive local `rclone lsf` inventory of `raw/pacifica/full_fidelity` timed out at 240s during this check, so the health evidence above intentionally used bounded prefix checks plus Fly logs and uploaded watchdog status. Do not treat the timeout as data loss; treat it as confirmation that full-bucket inventories need generous watchdog timeouts or bounded/partitioned checks.

## Sample maturity / trading-readiness answer — 2026-05-08T13:07Z

Diego asked whether the data is still insufficient. Answer: yes, for edge validation and paper trading it is still insufficient.

Current refreshed research artifacts show:

- Regime-state report: `519903` 1-minute rows across `65` symbols.
- Toxic overlay report verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Toxic overlay sample: `8` distinct dates, `65` symbols.
- Toxic overlay serious-validation gate: `30` distinct days and at least `100` removed high-toxicity observations.
- Paper-trading eligibility verdict: `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- Symbols evaluated: `65`; eligible symbols: `0`.
- Eligibility gates: `sample_gate_pass=0/65`, `activity_gate_pass=0/65`, `eligible=0/65`.

Interpretation:

- The data is useful for plumbing diagnostics and early sanity checks, not for claiming edge.
- Current sample is `8` days; need roughly `2` more days for 10-day early sanity, `6` more days for 14-day early sanity, `22` more days for 30-day provisional validation, and `52` more days for 60-day preferred validation.
- Do not tune toxicity cutoffs on this diagnostic sample.
- Do not launch paper trading yet; all symbols remain blocked by eligibility gates.
- Next research action is to keep collecting, then rerun the fixed silver/regime/toxic/eligibility pipeline at 10-14 days for early sanity and at 30+ days for provisional validation.

## System level-up foundation — 2026-05-08

Diego approved working on the full system level-up track. Added:

- `docs/plans/2026-05-08-pacifica-system-level-up.md` — phased implementation plan covering execution economics, paper ledger, no-trade governor, feature parity, walk-forward validation, symbol lifecycle, reference-market context, alerting, event risk, and research idea registry.
- `scripts/simulate_pacifica_execution.py` + `tests/scripts/test_simulate_pacifica_execution.py` — strategy-neutral execution-cost simulator with fees, slippage, adverse selection, funding, Markdown/CSV report output.
- `docs/experiments/execution-simulator/README.md`, `assumptions.csv`, `example_round_trips.csv`.
- `scripts/build_pacifica_paper_ledger.py` + `tests/scripts/test_build_pacifica_paper_ledger.py` — strategy-neutral paper-ledger spine with fills, positions, fees, funding, realized PnL, equity curve, drawdown, and ineligible-symbol refusal when diagnostic override is disabled.
- `docs/experiments/paper-ledger/README.md`, `fills.csv`, `positions.csv`, `equity_curve.csv`, `summary.csv`.

## System level-up Phase 3 — no-trade regime governor — 2026-05-08; superseded by v2 on 2026-05-13

Added:

- `scripts/build_pacifica_regime_governor.py` + `tests/scripts/test_build_pacifica_regime_governor.py`.
- `docs/experiments/regime-governor/README.md`, `governor_decisions.csv`, `decision_summary.csv`, `thresholds.csv`.

Historical note: the original 2026-05-08 v1 governor included size-reduction and forced-flow labels. It is superseded by the 2026-05-13 v2 diagnostic-only state set documented near the top of this handoff:

```text
SKIP_STALE_DATA
SKIP_WIDE_SPREAD
SKIP_LOW_DEPTH
SKIP_TOXIC_REGIME
SKIP_MARK_ORACLE_DISLOCATION
TRADABLE_DIAGNOSTIC
```

Current v2 generated governor summary over the 519,903-row regime table:

```text
SKIP_STALE_DATA                506,956 rows
SKIP_WIDE_SPREAD                   203 rows
SKIP_LOW_DEPTH                   1,017 rows
SKIP_TOXIC_REGIME                    0 rows
SKIP_MARK_ORACLE_DISLOCATION         0 rows
TRADABLE_DIAGNOSTIC             11,727 rows
```

Audit note: independent review flagged potential fail-open behavior. The v2 governor requires all latest-schema safety columns, rejects null keys, fills NaN/non-numeric safety metrics with conservative skip-triggering values, makes stale BBO/trade/price feeds fail closed to `SKIP_STALE_DATA`, validates thresholds before classification, and keeps `TRADABLE_DIAGNOSTIC` explicitly diagnostic-only.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_regime_governor.py -q
23 passed
python -m py_compile scripts/build_pacifica_regime_governor.py tests/scripts/test_build_pacifica_regime_governor.py
passed
```

Interpretation:

- These artifacts are accounting/validation/governor infrastructure only.
- They do not authorize live trading.
- `TRADABLE_DIAGNOSTIC` is not a trade signal.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is online/offline feature parity before any live microbatch feature drives decisions.

## System level-up Phase 4 — online/offline feature parity — 2026-05-08

Added:

- `scripts/check_pacifica_feature_parity.py` + `tests/scripts/test_check_pacifica_feature_parity.py`.
- `docs/experiments/feature-parity/README.md`, `summary.csv`, `mismatches.csv`, `missing_keys.csv`, `version_mismatches.csv`, `metadata_mismatches.csv`, `invalid_metadata.csv`, `invalid_features.csv`, `invalid_keys.csv`, `duplicate_keys.csv`, `feature_columns.csv`.
- Bootstrap input artifacts under `docs/experiments/feature-parity/`: `offline_bootstrap_features.parquet`, `live_bootstrap_features.csv`.

Current generated parity verdict:

```text
PARITY_FAIL_DIAGNOSTIC
failure_reasons=missing_required_columns
missing_metadata_columns=offline.available_ts;offline.computed_at;offline.watermark_ts;offline.feature_version;offline.provisional_final_flag;live.available_ts;live.computed_at;live.watermark_ts;live.feature_version;live.provisional_final_flag
```

Interpretation: the parity harness is implemented and intentionally fails closed on the current bootstrap/current-style feature inputs because the offline/live feature artifacts do not yet carry required metadata (`available_ts`, `computed_at`, `watermark_ts`, `feature_version`, `provisional_final_flag`). This blocks live microbatch feature use until the builders emit parity-ready metadata.

Audit note: independent reviews found fail-open cases during development. Fixed before handoff: parity now fails for missing metadata, metadata mismatches/nulls, empty overlaps, missing keys, duplicate keys, invalid/blank keys, feature-version mismatches, nonnumeric/null/non-finite feature values, empty feature-column configuration, and invalid tolerance (`nan`, `inf`, negative). CLI exits nonzero on `PARITY_FAIL_DIAGNOSTIC` unless `--allow-fail-diagnostic` is explicitly passed for report generation.

Verification:

```text
uv run pytest tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
33 passed
python -m py_compile scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are parity-gate infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is walk-forward validation.

## System level-up Phase 5 — walk-forward validation — 2026-05-08

Added:

- `scripts/run_pacifica_walk_forward_validation.py` + `tests/scripts/test_run_pacifica_walk_forward_validation.py`.
- `docs/experiments/walk-forward-validation/README.md`, `summary.csv`, `config.csv`, `windows.csv`, `window_scorecard.csv`, `random_controls.csv`.
- Bootstrap input artifact: `docs/experiments/walk-forward-validation/bootstrap_strategy_returns.csv`.

Current generated walk-forward verdict:

```text
INSUFFICIENT_SAMPLE_DIAGNOSTIC
failure_reasons=insufficient_distinct_days;no_purged_validation_windows
```

Interpretation: the walk-forward harness is implemented, but the bootstrap artifact is intentionally diagnostic only. Future strategy-return inputs must pass purged chronological OOS windows, sample maturity, day/symbol concentration, post-cost PnL, baseline comparison, and random same-frequency controls before any result can be discussed as evidence.

Audit note: independent review found fail-open cases during development. Fixed before handoff: CLI `--allow-fail-diagnostic` only allows clean insufficient-sample diagnostics, not real provisional/validation failures or invalid required fields; random controls cannot be disabled for a passing verdict; invalid timestamps/symbols/eligible flags and nonnumeric/non-finite returns fail closed; string `eligible=False` is parsed as false; overlapping test windows are rejected; no-purge configs cannot pass; OOS maturity/concentration and feasible per-window concentration are enforced.

Verification:

```text
uv run pytest tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
45 passed
python -m py_compile scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are validation-gate infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is symbol lifecycle promotion/demotion.

## System level-up Phase 6 — symbol lifecycle promotion/demotion — 2026-05-08

Added:

- `scripts/build_pacifica_symbol_lifecycle.py` + `tests/scripts/test_build_pacifica_symbol_lifecycle.py`.
- `docs/experiments/symbol-lifecycle/README.md`, `symbol_lifecycle.csv`, `state_counts.csv`, `transitions.csv`, `config.csv`.

Current generated lifecycle verdict:

```text
NO_ELIGIBLE_SYMBOLS_DIAGNOSTIC
ELIGIBLE=0
DISABLED=65
```

Interpretation: all currently evaluated symbols remain disabled by the diagnostic lifecycle, mostly because the archive is still young and activity/sample gates have not matured. `paper_trading_allowed_diagnostic=False` for every current symbol. This is expected and does not authorize paper/live trading.

Audit note: independent reviews found fail-open cases during development. Fixed before handoff: lifecycle now rejects missing required columns, duplicate/null/dirty symbols, invalid booleans, invalid numeric counts, unknown/duplicate previous lifecycle states, and dirty baseline scorecards; sticky `RETIRED` cannot be silently unretired; `eligible=True` only promotes when all upstream gates are consistent; inconsistent eligibility snapshots disable instead of promote; missing or bad post-cost baseline disables promotable symbols when a baseline scorecard is supplied, including an explicitly empty scorecard.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
56 passed
python -m py_compile scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are lifecycle-gate infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.

## System level-up Phase 7 — cross-venue/reference market context — 2026-05-08

Added:

- `scripts/build_pacifica_reference_context.py` + `tests/scripts/test_build_pacifica_reference_context.py`.
- `docs/experiments/reference-market-context/README.md`, `reference_context.csv`, `risk_state_summary.csv`, `symbol_reference_summary.csv`, `config.csv`.

Current generated reference-context verdict:

```text
NO_ROWS_DIAGNOSTIC
reference_available_rows=0
```

Interpretation: the builder/report layer exists, but no production external reference feed has been wired yet. It starts from pluggable local CSV/parquet inputs and intentionally does not hardwire paid APIs. Missing reference rows are flagged rather than imputed. This is context infrastructure only, not a trade signal and not permission to paper/live trade.

Audit note: independent review found fail-open/misleading cases during development. Fixed before handoff: duplicate canonical keys after numeric coercion fail closed; fractional `bucket_start_ms` keys fail closed; negative volatility fails closed; high-vol positive reference returns are labeled `HIGH_VOL_RISK_ON` instead of risk-off; partial/missing BTC/ETH major-reference beta proxy coverage is explicitly labeled `PARTIAL_MAJOR_REFERENCE` or `MISSING_MAJOR_REFERENCE` rather than silently imputed/blank.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
63 passed
python -m py_compile scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open issues found.

Interpretation:

- These artifacts are reference-context infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is external ops alerting.

## System level-up Phase 8 — external ops alerting — 2026-05-08

Added:

- `scripts/plan_pacifica_ops_alerts.py` + `tests/scripts/test_plan_pacifica_ops_alerts.py`.
- `docs/ops/pacifica-alerting/README.md`, `alert_plan.csv`, `summary.json`, `thresholds.csv`, `input_snapshot.json`.

Current generated alert-plan verdict:

```text
WARN
PAGE=0
WARN=1
OK=13
```

Interpretation: this is an alert-classification planner, not actual notification delivery. The bootstrap diagnostic snapshot is intentionally not a live check and only warns because no external delivery channel is configured in the artifact. Health facts and delivery are kept separate; no delivery credentials are stored or committed.

Alert conditions classified:

- Fly app not started.
- Raw freshness stale above 15 minutes.
- Free disk below Diego's 50 GiB floor.
- Lifecycle DB errors, upload failures, or verify failures.
- R2 raw prefix missing, remote freshness stale, or sidecar mismatch.
- Watchdog status stale or not OK.
- API surface changed.
- Archive inventory stale/timeout as WARN, not PAGE when otherwise healthy.
- Research refresh failed as WARN.
- Missing delivery channel as WARN.

Audit note: independent review passed. The planner fails closed to `PAGE` on missing or invalid required status signals, treats empty/unknown alert frames as `PAGE`, keeps inventory timeout as WARN-only unless other health facts page, and does not claim alerts have actually been delivered.

Verification:

```text
uv run pytest tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
70 passed
python -m py_compile scripts/plan_pacifica_ops_alerts.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open or misleading-delivery issues found.

Interpretation:

- These artifacts are alert-planning infrastructure only.
- They do not replace live Fly/R2 health checks.
- They do not send notifications until wired through Hermes cron/chat or another external delivery path.
- Next implementation phase is event/calendar risk.

## System level-up Phase 9 — event/calendar risk layer — 2026-05-08

Added:

- `scripts/build_pacifica_event_risk_calendar.py` + `tests/scripts/test_build_pacifica_event_risk_calendar.py`.
- `docs/experiments/event-risk-calendar/README.md`, `event_risk_rows.csv`, `event_risk_summary.csv`, `symbol_event_risk_summary.csv`, `config.csv`.

Current generated event-risk verdict:

```text
NO_EVENTS_CONFIGURED_DIAGNOSTIC
rows=519903
event_risk_rows=0
```

Interpretation: the event-risk builder/report layer exists and was run over the current 519,903-row regime table, but no production local event calendar is configured yet. All rows are marked `NO_KNOWN_EVENT_RISK` only because no event calendar was supplied; this does not mean the market is safe. This is context/governor infrastructure only, not a trade signal.

Input contract:

- Local CSV/parquet only; no hidden external API.
- Required event columns: `event_timestamp`, `event_type`, `pre_window_minutes`, `post_window_minutes`, `severity`, `source_note`.
- Supported severity: `LOW`, `MEDIUM`, `HIGH`.

Audit note: independent review found fail-open/misleading cases during development. Fixed before handoff: event text rejects semicolons/newlines/control characters that would corrupt semicolon-joined report fields; empty calendars produce `NO_EVENTS_CONFIGURED_DIAGNOSTIC` instead of conflating with configured-but-inactive event windows; empty state inputs now reach `NO_STATE_ROWS_DIAGNOSTIC` instead of crashing. The layer also fails closed on missing columns, dirty symbols, invalid/fractional bucket timestamps, invalid event timestamps, negative/fractional windows, noncanonical severity, blank/dirty event text, and preserves overlapping event types while taking the highest severity.

Verification:

```text
uv run pytest tests/scripts/test_build_pacifica_event_risk_calendar.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
79 passed
python -m py_compile scripts/build_pacifica_event_risk_calendar.py scripts/plan_pacifica_ops_alerts.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Final independent audit after fixes: `PASS`, no blocking fail-open or misleading-report issues found.

Interpretation:

- These artifacts are event-risk context infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- Next implementation phase is the research idea registry. (Completed below as Phase 10.)

## System level-up Phase 10 — research idea registry — 2026-05-08

Added:

- `docs/research/pacifica-idea-registry.md`.
- `scripts/validate_pacifica_idea_registry.py` + `tests/scripts/test_validate_pacifica_idea_registry.py`.
- `docs/research/pacifica-idea-registry-validation/README.md`, `idea_registry_validation.csv`, `summary.json`.

Current generated registry validation:

```text
registry_validation_verdict=PASS
ideas=3
errors=0
```

Registered diagnostic/pending ideas:

- `IDEA-001` toxic regime no-trade overlay — `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
- `IDEA-002` event-risk no-trade overlay — `PENDING_DIAGNOSTIC`.
- `IDEA-003` reference-market dislocation governor — `PENDING_DIAGNOSTIC`.

Interpretation: the registry is a pre-registration/schema gate only. A registry schema `PASS` means the idea is falsifiable enough to test; it is not evidence of alpha, not permission to paper/live trade, and does not override the current sample/eligibility gates.

Fail-closed validator coverage:

- Requires hypothesis, mechanical label, trade/risk action, cost model, validation window, frozen parameters, kill criteria, OOS plan, and result/verdict.
- Rejects missing fields, duplicate IDs, placeholders, qualitative/visual mechanical labels, discretionary actions, missing measurable/comparison language, missing cost/OOS/kill/frozen-parameter semantics, and edge/proven-alpha claims.
- Rejects negated controls such as no/without fees, costs not modeled, no failure gates, continue retuning, parameters not fixed/can be retuned, no OOS, and `PASS; edge is proven`.
- Report columns distinguish `registration_schema_verdict` from `research_result_verdict` to avoid reading schema PASS as research PASS.

Verification:

```text
uv run pytest tests/scripts/test_validate_pacifica_idea_registry.py tests/scripts/test_build_pacifica_event_risk_calendar.py tests/scripts/test_plan_pacifica_ops_alerts.py tests/scripts/test_build_pacifica_reference_context.py tests/scripts/test_build_pacifica_symbol_lifecycle.py tests/scripts/test_run_pacifica_walk_forward_validation.py tests/scripts/test_check_pacifica_feature_parity.py tests/scripts/test_build_pacifica_regime_governor.py tests/scripts/test_simulate_pacifica_execution.py tests/scripts/test_build_pacifica_paper_ledger.py -q
86 passed
python -m py_compile scripts/validate_pacifica_idea_registry.py scripts/build_pacifica_event_risk_calendar.py scripts/plan_pacifica_ops_alerts.py scripts/build_pacifica_reference_context.py scripts/build_pacifica_symbol_lifecycle.py scripts/run_pacifica_walk_forward_validation.py scripts/check_pacifica_feature_parity.py scripts/build_pacifica_regime_governor.py scripts/simulate_pacifica_execution.py scripts/build_pacifica_paper_ledger.py
passed
git diff --check
passed
```

Independent audit: first audit found fail-open cases around negated cost/OOS/kill/frozen controls, qualitative labels with keywords, and misleading PASS wording. Fixed and re-audited; final audit `PASS`.

Interpretation:

- These artifacts are idea-governance infrastructure only.
- They do not authorize live trading.
- They do not change the current `INSUFFICIENT_SAMPLE_DIAGNOSTIC` maturity verdict.
- The system level-up spine Phases 1-10 are now implemented.

## Continuation ops check — 2026-05-08 18:29 UTC

Read-only live checks run after Phase 10 completion.

Fly status:

```text
checked_at=2026-05-08T18:28:57Z
App: pacifica-full-fidelity
Machine: e2862502a76778
Region: iad
State: started
Image: pacifica-full-fidelity:deployment-01KR2RTJZGAMGB2S9NB2KXEBWF
Last updated: 2026-05-08T03:08:07Z
```

Latest completed lifecycle evidence from logs:

```text
2026-05-08T17:45:08Z upload failed=0 skipped=0 uploaded=200; verify failed=0 skipped=0 verified=200
2026-05-08T17:45:12Z lifecycle complete
```

Latest observed health JSON from logs:

```text
checked_at=2026-05-08T17:45:13.135039+00:00
ok=true
failures=[]
free_gb=52.81
unverified_gb=38.31
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=kBONK/date=2026-05-08/hour=17/run-20260508T030808Z.jsonl.gz
newest_raw_age_min=-2.85
db_counts: pruned=11593 files / 3678491035 bytes; sealed=63266 files / 41140405212 bytes; verified=1691 files / 602515107 bytes
```

Current caveat: a new lifecycle cycle started at `2026-05-08T18:18:10Z`; the bounded log check had not yet observed its upload/verify completion. Do not treat that as failure; use the next log/status check to confirm the next `lifecycle complete` line.

Ops watchdog evidence:

```text
2026-05-08T16:44:17Z ops watchdog run reported failures
2026-05-08T17:44:17Z ops watchdog run start
2026-05-08T17:44:41Z ops watchdog run complete
uploaded latest_status checked_at=2026-05-08T17:44:35.693259+00:00 ok=true operation=noop_not_due
```

Interpretation: the earlier watchdog failure was followed by an OK uploaded watchdog status and a later complete run. Keep watching, but latest watchdog artifact is OK.

API-surface watchdog artifact:

```text
api_surface_diff changed=false
rest_paths added=[] removed=[]
ws_sources added=[] removed=[]
intervals added=[] removed=[]
```

R2 bounded checks:

```text
top-level prefixes: app/, funding/, ops/, raw/
active raw channel dirs: channel=bbo/, channel=book/, channel=candle/, channel=mark_price_candle/, channel=pong/, channel=prices/, channel=subscribe/, channel=trades/, rest/
BTC BBO sample: raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-07/hour=23/run-20260505T141555Z.jsonl.gz
sha256 sidecar matched: b923514f6439bb47e68f4423e53d790ea3699ba98c7648fbb3e8c1f4924499d7
gzip JSONL readability: read 20 rows successfully
```

R2 legacy cleanup caveat remains: `app/` and `funding/` still exist at top level. Do not destructively purge without Diego's explicit approval. Preserve `raw/` and `ops/`.

## Alert planner refresh — 2026-05-08 18:38 UTC

Refreshed `docs/ops/pacifica-alerting/` from the latest observed non-secret health facts.

```text
overall_severity=WARN
PAGE=0
WARN=1
OK=13
```

Only WARN condition: `DELIVERY_CHANNEL_CONFIGURED` because no external notification delivery channel is configured in the repo artifact. All collector/R2/watchdog/API/research snapshot signals were classified OK using the latest bounded evidence. This remains an alert-classification artifact only; it does not send notifications.

Caveat: the refreshed snapshot uses the latest observed completed lifecycle at `2026-05-08T17:45:12Z`; the newer `2026-05-08T18:18:10Z` lifecycle cycle was still in progress/not yet complete in bounded logs.

## Continuation poll — 2026-05-08 19:23 UTC

Checked Fly logs again for the `2026-05-08T18:18:10Z` lifecycle cycle.

```text
checked_at=2026-05-08T19:23:40Z
2026-05-08T18:18:10Z lifecycle scan/upload/verify/prune start
```

No later upload/verify summary or `lifecycle complete` line was visible in that bounded log poll. The app still reported `started`, and a later ops watchdog run completed:

```text
2026-05-08T18:44:42Z ops watchdog run start
2026-05-08T18:45:08Z ops watchdog run complete
uploaded latest_status checked_at=2026-05-08T18:44:56.101339+00:00 ok=true operation=noop_not_due
```

Direct remote health probing caveat: one `flyctl ssh console ... python /app/scripts/check_pacifica_full_fidelity_health.py ...` attempt timed out after 180s, likely because it recursively scans the raw tree. A later shell-wrapped SSH diagnostic form was blocked by user approval policy (`BLOCKED: User denied. Do NOT retry.`). Do not retry that exact denied command form; use logs, bounded R2 checks, uploaded watchdog artifacts, or a safer already-deployed health/status artifact instead.

## Continuation poll — 2026-05-08 21:16 UTC

The previously pending `18:18` lifecycle cycle completed successfully in later logs:

```text
checked_at=2026-05-08T21:16:44Z
2026-05-08T18:18:10Z lifecycle scan/upload/verify/prune start
2026-05-08T19:32:28Z scanned=66133 state_db=/data/pacifica_full_fidelity_storage.sqlite
2026-05-08T19:32:34Z reset=0 skipped_missing=0 skipped_recent=0 dry_run=false
2026-05-08T19:52:47Z upload failed=0 skipped=63 uploaded=137; verify failed=0 skipped=0 verified=137
2026-05-08T19:52:54Z lifecycle complete
```

Latest observed health JSON after that cycle:

```text
checked_at=2026-05-08T19:52:55.933725+00:00
ok=true
failures=[]
free_gb=52.41
unverified_gb=38.93
newest_raw_file=/data/pacifica_full_fidelity/channel=mark_price_candle/symbol=ADA/date=2026-05-08/hour=19/run-20260508T030808Z.jsonl.gz
newest_raw_age_min=-3.5
db_counts: pruned=11795 files / 3785207612 bytes; sealed=64305 files / 41803780248 bytes; verified=1626 files / 559524101 bytes
```

A newer lifecycle cycle started at `2026-05-08T20:26:32Z`; bounded logs through `2026-05-08T21:16:44Z` had not yet shown its upload/verify completion. Latest uploaded watchdog artifact remained OK:

```text
watchdog latest_status checked_at=2026-05-08T20:45:57.374570+00:00 ok=true operation=noop_not_due
```

Interpretation: the earlier pending `18:18` cycle is now resolved healthy. Next follow-up should check completion of the `20:26` lifecycle cycle, not the already-completed `18:18` cycle.

## Continuation ops/alert refresh — 2026-05-09 15:24 UTC

Diego asked to proceed sequentially through the next ops items. Read-only checks were run without destructive R2 cleanup.

Direct Fly status/log/lifecycle probing caveat: an attempted combined Fly status/log command in this session was blocked by approval policy (`BLOCKED: User denied. Do NOT retry.`). Do not retry that exact form. The current refresh therefore uses bounded R2/watchdog/API artifacts and intentionally classifies unknown lifecycle/Fly/disk facts fail-closed rather than pretending they are OK.

Bounded R2/watchdog/API evidence:

```text
checked_at=2026-05-09T15:24:04Z
R2 top-level prefixes: app/, funding/, ops/, raw/
active raw channel dirs: channel=bbo/, channel=book/, channel=candle/, channel=mark_price_candle/, channel=pong/, channel=prices/, channel=subscribe/, channel=trades/, rest/
watchdog latest_status checked_at=2026-05-09T14:53:15.480756+00:00 ok=true operation=noop_not_due
api_surface_diff changed=false
latest sampled R2 raw payload across bounded current-day channel/symbol scan: channel=bbo/symbol=STRK/date=2026-05-09/hour=11/run-20260508T030808Z.jsonl.gz
latest sampled R2 raw modified=2026-05-09T06:24:13Z
latest sampled R2 raw age≈538.7 minutes at check time
latest sampled sidecar hash prefix matched local sha256; gzip JSONL readability read 20 rows successfully
```

Interpretation: bounded R2 evidence proves some raw uploads occurred after the previously pending `2026-05-08T20:26:32Z` lifecycle cycle start, but current bounded R2 raw freshness is stale by the alert thresholds and direct lifecycle/Fly DB health could not be verified in this pass. Treat this as `PAGE` until a safe Fly-side/status artifact confirms collector/lifecycle health or uploads resume.

Alert planner refreshed:

```text
docs/ops/pacifica-alerting/summary.json
overall_severity=PAGE
PAGE=5
WARN=1
OK=6
```

The PAGE state is intentionally conservative: invalid/unknown lifecycle counts, unknown Fly state, stale sampled raw freshness, unknown free disk, and stale sampled R2 remote freshness. `DELIVERY_CHANNEL_CONFIGURED` is now OK because a Hermes cron alert bridge exists.

Created external alert bridge:

```text
script=/Users/diego/.hermes/scripts/pacifica_ops_watchdog_alert.py
cron_job_id=e61c2f7c5593
name=pacifica-r2-watchdog-alert-bridge
schedule=every 30m
deliver=origin
mode=no_agent
```

The bridge is read-only and emits only on problems. It checks bounded R2 raw freshness, uploaded watchdog status, and API-surface diff. It does not perform Fly SSH/log lifecycle DB checks and does not delete anything. A manual run emitted a PAGE because the newest sampled R2 raw payload was ~538.7 minutes old.

Maturity decision: do not rerun the research pipeline yet. Local restored raw cache still has 8 distinct dates (`2026-04-30` through `2026-05-07`) and current artifacts remain diagnostic. First resolve/understand the collector or R2 upload freshness PAGE, then wait for at least 10-14 distinct days for early sanity reruns; keep 30+ days as provisional and 60+ days preferred serious validation.

## Continuation remediation — 2026-05-10 00:05 UTC

Diego asked to address everything. The prior PAGE was diagnosed and partially remediated.

Root cause found from safe Fly logs: the app was `started`, but collector writes were repeatedly blocked by Diego's 50 GiB free-disk guard:

```text
websocket collection error; reconnecting in 5s: free disk below safety floor: 50.0 GiB available under /data/pacifica_full_fidelity, requires at least 50.0 GiB
REST snapshot failed for /info: free disk below safety floor: 50.0 GiB available under /data/pacifica_full_fidelity, requires at least 50.0 GiB
```

Fly-side DB/disk state before remediation showed `/data` effectively at the 50 GiB floor and a large sealed backlog:

```text
volume before: 100GB
df before: /dev/vdc 98G size, 43G used, 50G avail
archive_files: pruned=13,558; sealed=71,105 (~43.8GB); verified=2,221 (~1.37GB); rows_with_errors=0
```

Actions taken:

```text
flyctl volumes extend vol_vwn2mpw8mmgwx38v -a pacifica-full-fidelity --size 200 --yes
flyctl deploy -c ops/fly/pacifica-full-fidelity/fly.toml --ha=false
```

Deployment/config now active:

```text
image=pacifica-full-fidelity:deployment-01KR7JQA20E7TFVKX6CANZ7H37
machine=e2862502a76778 version=18 state=started last_updated=2026-05-09T23:58:44Z
volume=vol_vwn2mpw8mmgwx38v size=200GB
PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S=900
PACIFICA_FULL_FIDELITY_BATCH_LIMIT=2000
PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB=50
```

Verification after remediation:

```text
df after: /dev/vdc 197G size, 43G used, 145G avail
local collector fresh at 2026-05-10T00:04:38Z:
  BTC bbo current-hour file age≈0.05 min
  BTC book current-hour file age≈0.01 min
  BTC mark_price_candle current-hour file age≈0.07 min
archive_files: pruned=13,558; sealed=70,993; uploaded=112; verified=2,221; rows_with_errors=0
```

Important caveat: R2 remote freshness is still PAGE/stale until lifecycle upload/verify catches up. The alert planner was refreshed from post-remediation facts and now has only one PAGE condition:

```text
docs/ops/pacifica-alerting/alert_plan.csv
R2_REMOTE_FRESHNESS=PAGE
all other conditions=OK
```

A manual high-limit upload/verify attempt (`proc_f16ad5bb672f`) was started while the old lifecycle scan held the DB lock. It repeatedly failed with `sqlite3.OperationalError: database is locked` and exited with signal/exit code `-15` before deploy, after only a small number of rows advanced (`uploaded=112`). A post-deploy read-only DB check at `2026-05-10T00:07:32Z` showed `/data` healthy (`197G` size, `43G` used, `145G` available), `archive_files: pruned=13,558; sealed=70,993; uploaded=112; verified=2,221; rows_with_errors=0`. The redeployed lifecycle should now run every 15 minutes with batch limit 2000, but R2 catch-up still needs follow-up monitoring. Do not treat local collector freshness as proof that R2 archival has caught up.

## 24h follow-up — 2026-05-10 22:05 UTC

Diego noted it had been more than 24h since the last check. Current status remains partially healthy but not fully resolved.

Verified current facts:

```text
checked_at=2026-05-10T22:05Z
fly_state=started
image=pacifica-full-fidelity:deployment-01KR7JQA20E7TFVKX6CANZ7H37
machine=e2862502a76778 version=18
/data=/dev/vdc 197G size, 47G used, 141G available
latest local raw files fresh/current-hour around 2026-05-10T22:02-22:04Z for bbo/book/trades
rows_with_errors=0
latest watchdog status checked_at=2026-05-10T21:39:35Z ok=true age≈25.9m
api_surface_changed=false
```

Lifecycle DB direct read at the same follow-up:

```text
pruned|15,449|5,436,657,952 bytes
sealed|74,672|42,227,364,191 bytes
uploaded|421|160,694,162 bytes
verified|6,330|6,128,931,801 bytes
rows_with_errors=0
```

Change versus the post-remediation `2026-05-10T00:07Z` check:

```text
pruned_files +1,891 / pruned_bytes +1.04GB
verified_files +4,109 / verified_bytes +4.76GB
uploaded_files +309 / uploaded_bytes +114.8MB
sealed_files +3,679 / sealed_bytes -1.51GB
```

Interpretation: lifecycle is making safe non-destructive progress and verification/pruning increased materially, but the backlog is not caught up. `sealed` files net-increased because live collection kept adding sealed chunks faster than the backlog cleared in file-count terms, though sealed bytes decreased. R2 freshness is still stale by alert threshold.

Bounded R2 evidence:

```text
latest sampled R2 raw payload modified≈2026-05-10T14:09Z
latest sampled payload age≈471.2 minutes at 2026-05-10T22:00Z
sample sidecar hash matched and gzip read 20 rows
raw prefix present=true
```

Alert planner refreshed at `docs/ops/pacifica-alerting/`:

```text
overall_severity=PAGE
PAGE=1
WARN=0
OK=13
only PAGE condition: R2_REMOTE_FRESHNESS
```

Do not manually start another competing upload/verify loop while the lifecycle process is active. Continue monitoring direct DB counts and R2 freshness. Local collection and disk are healthy, but R2 archival freshness remains the unresolved PAGE.

## Remaining work

1. Monitor the deployed version-23 batch fresh-upload lane (`pacifica-full-fidelity:deployment-01KREJD02600997WH8F9H7C53Z`). Immediate local bounded R2 freshness was green at `2026-05-12T17:17:14Z`, but the hourly uploaded watchdog artifact may not flip until its next due run.
2. Check completion of the current version-23 lifecycle cycle after the `17:13:29Z` reset-mismatch line. Expected next healthy lines are backlog `upload-verify`, prune, `lifecycle complete`, and a health JSON. Do not start competing manual lifecycle writers.
3. Poll local background process `proc_815de4056c85`. If the R2-to-local-cache refresh completes successfully, inventory `data/pacifica_full_fidelity/`; only then rerun silver/regime/toxic/eligibility as diagnostic/early-sanity artifacts.
4. If the cache refresh remains slow or no-output, use bounded/date/channel R2 copies or improved inventory tooling instead of assuming the local cache is current.
5. Keep alert severity fail-closed on any future stale R2 sample, sidecar mismatch, lifecycle failure, low disk, or stale watchdog artifact. Current green evidence is the local bounded checker after the v23 deploy, not yet a completed hourly watchdog cycle.
6. Do not rerun research for edge claims. Reruns are OK only as plumbing diagnostics/early sanity until distinct-day maturity reaches 30+ days for provisional validation and 60+ for preferred serious validation.
7. Do not claim edge or paper-trade until the 30-day sample gate and symbol eligibility gates pass. Existing local artifacts remain based on the old 8-date cache through `2026-05-07` and previously showed `0/65` eligible.
8. Verify top-level R2 prefixes after any approved legacy purge. `app/` and `funding/` were still present in the latest non-destructive top-level check; `raw/` and `ops/` must remain preserved.
9. Destructive legacy cleanup remains blocked unless Diego explicitly approves exact scope. If approved, target only remaining legacy `app/` and `funding/` prefixes. Do not touch `raw/` or `ops/`.
10. System level-up Phases 1-10 plus the v23 ops freshness lane are implemented. Keep future strategy work behind the registry, eligibility, economics, governor, parity, walk-forward, baseline, random-control, and concentration gates.

## Lifecycle freshness-lane remediation — 2026-05-11 02:36 UTC

Diego said to proceed with the next step: diagnose why lifecycle progress was not restoring R2 freshness. Root cause found:

```text
The lifecycle interval was set to 900s, but the lifecycle cycle itself was taking hours.
Observed cycle:
  2026-05-10T19:52:20Z lifecycle start
  2026-05-10T21:30:03Z scan completed, scanned=81,423
  2026-05-11T02:24:27Z upload/verify completed, uploaded=2,000 verified=2,000
  2026-05-11T02:24:46Z lifecycle complete
```

Why R2 stayed stale despite safe progress:

```text
1. scan hashes the full archive every cycle before uploading, which took ~1h38m in the observed cycle.
2. upload/verify is sequential per object/sidecar/size/cat and took ~4h54m for the 2,000-file batch.
3. upload candidate ordering was object_key ascending, so backlog catch-up walked old lexicographic partitions first instead of sending the newest eligible sealed files. This kept bounded R2 freshness stale even while verification/pruning progressed.
```

Implemented and deployed a non-destructive freshness-lane change:

```text
scripts/pacifica_full_fidelity_storage.py
  - added --upload-order {object-key,newest-first,oldest-first}
  - newest-first orders sealed upload candidates by last_seen_at desc, object_key desc
  - errored uploaded rows still remain first for repair safety

scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
  - passes PACIFICA_FULL_FIDELITY_UPLOAD_ORDER to upload-verify

ops/fly/pacifica-full-fidelity/fly.toml
  - PACIFICA_FULL_FIDELITY_UPLOAD_ORDER="newest-first"
```

Deployment:

```text
flyctl deploy -c ops/fly/pacifica-full-fidelity/fly.toml --ha=false
image=pacifica-full-fidelity:deployment-01KRAE6VQCAMV0P61R75YCD20Q
machine=e2862502a76778 version=19 state=started last_updated=2026-05-11T02:36:04Z
```

Post-deploy verification:

```text
upload_order=newest-first
python scripts/pacifica_full_fidelity_storage.py --help includes --upload-order
/data=/dev/vdc 197G size, 48G used, 141G available
archive_files: pruned=15,779; sealed=73,045; uploaded=48; verified=8,000; rows_with_errors=0
lifecycle scan/upload/verify/prune start at 2026-05-11T02:36:04Z
```

Tests/verification:

```text
uv run pytest tests/scripts/test_pacifica_full_fidelity_storage.py -q
19 passed

python -m py_compile scripts/pacifica_full_fidelity_storage.py
bash -n scripts/run_pacifica_full_fidelity_r2_lifecycle.sh
git diff --check -- scripts/pacifica_full_fidelity_storage.py tests/scripts/test_pacifica_full_fidelity_storage.py scripts/run_pacifica_full_fidelity_r2_lifecycle.sh ops/fly/pacifica-full-fidelity/fly.toml
all passed
```

Important caveat: this changes which eligible files are uploaded first; it does not make the full scan or sequential rclone verification fast. R2 freshness should improve after the current post-deploy lifecycle reaches its upload phase. If freshness is still stale after that cycle, the next root-cause target is scan/verification architecture, not disk capacity.

## Blocked/avoid

- Do not retry exact shell/Python diagnostic forms that previously returned `BLOCKED: User denied. Do NOT retry.`
- Do not run destructive `rm -rf` cleanup chains or destructive R2 cleanup without explicit approval.
- Do not upload current/live time partitions from append-style raw collectors to R2/S3; lifecycle upload should continue to skip recent/current chunks.
- Do not claim an edge while reports remain `INSUFFICIENT_SAMPLE_DIAGNOSTIC`.
