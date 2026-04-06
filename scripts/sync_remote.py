"""Sync parquet files to R2 and purge old data. Runs on the Fly.io instance.

Usage:
    python3 sync_remote.py 2026-04-05     # sync files for that date
    python3 sync_remote.py --purge-only   # purge only

Uses date partitions in the directory structure (date=YYYY-MM-DD).
Skips files already in R2 (by key+size) using targeted prefix listing.
"""

import os
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

data_dir = pathlib.Path("/app/data")
bucket = os.environ["S3_BUCKET_NAME"]
date_pattern = re.compile(r"date=(\d{4}-\d{2}-\d{2})")


def list_existing(s3, target_date):
    """List R2 objects for a specific date using targeted prefix scans.

    Discovers data types and symbols from local dirs, then lists R2
    with precise prefixes like 'trades/symbol=BTC/date=2026-04-06/'.
    Much faster than scanning the entire bucket.
    """
    existing = {}  # key -> size
    prefixes_checked = 0

    for dtype_dir in sorted(data_dir.iterdir()):
        if not dtype_dir.is_dir():
            continue
        dtype = dtype_dir.name  # e.g., 'trades', 'orderbook'

        for sym_dir in sorted(dtype_dir.iterdir()):
            if not sym_dir.is_dir() or not sym_dir.name.startswith("symbol="):
                continue

            prefix = f"{dtype}/{sym_dir.name}/date={target_date}/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    existing[obj["Key"]] = obj["Size"]
            prefixes_checked += 1

    return existing, prefixes_checked


def sync_date(s3, target_date):
    """Upload parquet files for a date, skipping those already in R2."""
    # Find local files for this date
    local_files = [
        f
        for f in data_dir.rglob("*.parquet")
        if (m := date_pattern.search(str(f))) and m.group(1) == target_date
    ]
    print(f"[{target_date}] {len(local_files)} local files", flush=True)

    if not local_files:
        return 0, 0

    # Check what's already in R2
    t0 = time.time()
    existing, n_prefixes = list_existing(s3, target_date)
    elapsed = time.time() - t0
    print(
        f"[{target_date}] {len(existing)} already in R2 ({n_prefixes} prefixes, {elapsed:.0f}s)",
        flush=True,
    )

    # Filter to only new/changed files
    files = []
    for f in local_files:
        key = str(f.relative_to(data_dir))
        local_size = f.stat().st_size
        if key not in existing or existing[key] != local_size:
            files.append(f)

    print(f"[{target_date}] {len(files)} to upload", flush=True)
    if not files:
        return 0, 0

    def upload(f):
        key = str(f.relative_to(data_dir))
        s3.upload_file(str(f), bucket, key)
        return key

    uploaded = 0
    failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(upload, f): f for f in files}
        for fut in as_completed(futures):
            try:
                fut.result()
                uploaded += 1
            except Exception as e:
                failed += 1
                print(f"  FAIL {futures[fut]}: {e}", file=sys.stderr, flush=True)
            if uploaded % 1000 == 0 and uploaded > 0:
                elapsed = time.time() - t0
                print(
                    f"  [{target_date}] {uploaded}/{len(files)} ({elapsed:.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"[{target_date}] Uploaded {uploaded}, failed {failed} ({elapsed:.0f}s)",
        flush=True,
    )
    return uploaded, failed


def purge():
    """Delete files older than 2 days from the Fly volume."""
    print("Purging files older than 2 days...", flush=True)
    result = subprocess.run(
        [
            "find",
            "/app/data",
            "-type",
            "f",
            "-name",
            "*.parquet",
            "-mtime",
            "+1",
            "-print",
            "-delete",
        ],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        purged = result.stdout.strip().count("\n") + 1
        print(f"Purged {purged} files.", flush=True)
    else:
        print("Nothing to purge.", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "--purge-only":
        purge()
        sys.exit(0)

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    uploaded, failed = sync_date(s3, arg)
    s3._endpoint.http_session.close()

    if failed:
        sys.exit(1)
