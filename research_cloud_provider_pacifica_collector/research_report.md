# Cloud provider research: Pacifica full-fidelity always-on collector

Date: 2026-05-01

## Question

Can the Pacifica full-fidelity collector run cheaply/reliably in the cloud while the laptop is off? Is Fly.io free capacity enough?

## Workload assumption

- Always-on Python websocket collector.
- Raw ingest currently estimated around 23-25 GiB/day before further stream-volume reductions.
- Cloudflare R2 is durable archive.
- Cloud VM/volume is only a local spool/cache.
- Desired spool: roughly 50-100 GiB, depending retention and upload failures.
- Upload traffic from collector host to R2: roughly 750 GiB/month at 25 GiB/day.
- Local verified pruning should keep VM disk bounded.

## Main answer

Fly.io is feasible technically, but Fly free capacity is not enough for this workload if we need a 50-100 GiB persistent volume. Treat Fly as a paid deployment around $10-20/month for a tiny always-on Machine plus 50-100 GiB volume, before any extra bandwidth/snapshot costs.

Best paid provider from the research pass: Hetzner Cloud.

Best zero-cost-on-paper provider: Oracle Cloud Always Free, but with more operational risk around capacity/account/free-tier reliability.

## Fly.io

Sources:
- https://fly.io/docs/about/pricing/
- https://fly.io/docs/volumes/overview/

Findings:
- Fly Machines can run this workload: a tiny shared CPU machine is enough for Python websocket + rclone lifecycle unless memory proves otherwise.
- Fly Volumes are local persistent disks attached to the Machine, suitable as spool but not durable archive.
- Volume pricing is the blocker for free usage. A 50-100 GiB volume exceeds free-scale assumptions.
- Small 24/7 machine estimates from current pricing page/subagent scrape:
  - shared-cpu-1x 256MB: about $1.94/month
  - shared-cpu-1x 512MB: about $3.19/month
  - shared-cpu-1x 1GB: about $5.70/month
- Volume estimate: about $0.15/GB-month.
  - 50GB: about $7.50/month
  - 100GB: about $15/month

Estimated Fly monthly cost:
- 512MB + 50GB: about $10.69/month
- 512MB + 100GB: about $18.19/month
- 1GB + 50GB: about $13.20/month
- 1GB + 100GB: about $20.70/month

Verdict:
- Fly free: no, not enough for 50-100GB spool.
- Fly paid: yes, feasible and operationally convenient.
- Use Fly if we value simple deployment and already have a Fly workflow.

## Cloudflare R2 economics

Source:
- https://developers.cloudflare.com/r2/pricing/
- https://developers.cloudflare.com/r2/buckets/object-lifecycles/

Current R2 pricing page facts observed:
- Standard storage: $0.015 / GB-month.
- Class A operations: $4.50 / million requests.
- Class B operations: $0.36 / million requests.
- Egress to Internet: Free.

Cost at 25 GiB/day:
- 25 GiB/day * 30 days = 750 GiB/month added.
- 750 GiB is about 805 decimal GB.
- Monthly storage run-rate per retained monthly tranche: about $11-12/month.
- First month from empty averages roughly half that, about $5.50-6.00.
- No lifecycle expiry: storage run-rate grows linearly.
  - 6 months retained: roughly 4.5 TiB, about $67-72/month run-rate.
  - 12 months retained: roughly 9 TiB, about $135-145/month run-rate.

Request cost:
- Hourly or minute-ish chunking should stay inside the free request tier or be negligible.
- Sidecar `.sha256` files double object count and PUTs, but still likely cheap at hourly/minute cadence.
- Avoid per-second or tiny-object layouts because request/list overhead grows quickly.

Network caveat:
- R2 does not charge ingress or egress, but the VPS/Fly provider may charge outbound transfer to R2.
- At 25 GiB/day, the collector host sends roughly 750 GiB/month to R2.
- Provider egress pricing/allowance matters more than R2 network pricing.

## Provider comparison

### Hetzner Cloud

Sources:
- https://www.hetzner.com/cloud/
- https://www.hetzner.com/cloud/volumes/
- https://docs.hetzner.com/cloud/billing/faq/

Fit:
- Best paid option.
- Cheap VPS and cheap volumes.
- Very large included traffic allowances compared with 750 GiB/month workload.
- Simple VPS model.

Expected cost:
- Roughly €5-8/month for small instance plus enough extra disk, depending exact VM/volume choice.

Verdict:
- Recommended if we want lowest paid cost and can operate a normal VPS.

### Oracle Cloud Always Free

Sources:
- https://www.oracle.com/cloud/free/
- https://www.oracle.com/cloud/free/faq/
- https://www.oracle.com/cloud/networking/pricing/
- https://www.oracle.com/cloud/storage/block-volumes/pricing/

Fit:
- Best free-on-paper option.
- Always Free includes Arm Ampere A1 compute resources and block volume allocation sufficient for this workload, subject to availability and account limits.
- Oracle networking free allowance is generous enough on paper for ~750 GiB/month.

Risks:
- Capacity can be unavailable.
- Account/free-tier operational friction is higher.
- Not ideal as sole production collector unless we add monitoring/failover.

Verdict:
- Good experimental/free option, but not my first recommendation for reliable unattended market-data collection.

### DigitalOcean

Sources:
- https://www.digitalocean.com/pricing/droplets
- https://www.digitalocean.com/pricing/volumes
- https://docs.digitalocean.com/products/billing/bandwidth/

Fit:
- Simple and reliable.
- $6/month class has 1GB RAM, 25GB SSD, 1TB transfer based on page text observed.
- Need block volume for 50-100GB spool.
- 1TB transfer fits 750 GiB/month but leaves limited margin.

Expected cost:
- About $11-16/month depending added volume.

Verdict:
- Good fallback; easier ops than some, but worse bandwidth/storage economics than Hetzner.

### Vultr

Sources:
- https://www.vultr.com/pricing/
- https://www.vultr.com/products/block-storage/
- https://docs.vultr.com/vultr-billing-faq

Fit:
- Similar to DigitalOcean.
- Common low-end plan class has around 25GB SSD and ~1TB bandwidth.
- Add block storage for spool.

Expected cost:
- About $10-15/month depending disk.

Verdict:
- Good fallback, not best primary.

### AWS Lightsail

Sources:
- https://aws.amazon.com/lightsail/pricing/
- https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-understanding-data-transfer-charges.html

Fit:
- Simpler than EC2.
- Bundled transfer can handle 750 GiB/month on low plans.
- Add disk if needed.

Expected cost:
- About $9-11/month for a small instance plus enough extra disk.

Verdict:
- Viable if we want AWS simplicity, but Hetzner is cheaper.

### AWS EC2 / GCP / Azure

Sources:
- https://aws.amazon.com/ec2/pricing/on-demand/
- https://aws.amazon.com/ebs/pricing/
- https://aws.amazon.com/ec2/pricing/on-demand/#Data_Transfer
- https://cloud.google.com/free
- https://cloud.google.com/vpc/network-pricing
- https://azure.microsoft.com/free/
- https://azure.microsoft.com/pricing/details/bandwidth/

Fit:
- Reliable but poor cost fit for a small collector pushing ~750 GiB/month to R2.
- Egress and persistent disk costs can dominate.

Verdict:
- Avoid unless there is a separate reason to be on that cloud.

## Recommendation

V1 practical recommendation:
1. Use Hetzner Cloud small VPS + 80-100GB disk/volume if lowest cost and reliability matter.
2. Use Fly.io paid if we prefer the current Fly-based deployment template and simpler app deployment; expect roughly $10-20/month, not free.
3. Try Oracle Always Free only as a secondary/free experiment, not as the only collector unless we accept capacity/account risk.

For the current repo state, since Fly deployment files already exist, it is reasonable to either:
- proceed with Fly paid as V1 for speed, or
- port the same container/entrypoint to Hetzner Docker Compose/systemd for lower recurring cost.

My call:
- Fly free is not feasible.
- Fly paid is feasible.
- Hetzner is likely the best long-term always-on host.
- R2 cost is acceptable at first but grows with permanent retention; at 25 GiB/day, 12 months retained is about 9 TiB and ~$135-145/month storage run-rate, so lifecycle/compression/volume control matters.
