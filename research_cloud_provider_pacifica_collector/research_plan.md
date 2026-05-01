# Research plan: cloud provider feasibility for Pacifica full-fidelity collector

Main question: Which always-on cloud provider is feasible/cost-effective for running the Pacifica full-fidelity raw collector plus R2 upload/prune lifecycle, and is Fly.io free capacity enough?

Assumptions from repo context:
- Collector estimated raw ingest rate: ~23-25 GiB/day before further stream-volume reductions.
- Durable archive target: Cloudflare R2; cloud VM/volume is only spool/cache.
- Need always-on process, not batch-only, because laptop may be off.
- Current Fly template uses a small VM plus persistent volume at /data and prunes verified files.
- Need enough CPU/RAM/network for Python websocket collector + rclone lifecycle.

Subtopics:
1. Fly.io free/low-cost capacity and constraints for always-on collector: free allowance, Machines pricing, volume pricing, egress/bandwidth, whether always-on + volumes are free enough.
2. Alternative low-cost always-on providers: Hetzner, DigitalOcean, Vultr, Oracle Cloud Always Free, AWS Lightsail/EC2, GCP/Azure basics; compare compute, storage, reliability, and operational fit.
3. R2/storage economics and network-cost implications: Cloudflare R2 storage/request/free-tier, provider egress to R2, whether inbound to R2 is free, pitfalls.
4. Recommendation for this workload: estimated monthly cost under 25 GiB/day raw ingest with 1-3 day local spool, and provider ranking.

Synthesis target:
- Short go/no-go on Fly free feasibility.
- Recommended provider for V1.
- Expected monthly cost bands.
- Risks and next steps.
