"""Sync parquet files to R2 and purge old data. Runs on the Fly.io instance."""

import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

data_dir = pathlib.Path("/app/data")
bucket = os.environ["S3_BUCKET_NAME"]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

cutoff = time.time() - 86400  # last 24h
files = [f for f in data_dir.rglob("*.parquet") if f.stat().st_mtime >= cutoff]
print(f"Found {len(files)} files to upload", flush=True)


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
            print(f"FAIL {futures[fut]}: {e}", file=sys.stderr, flush=True)
        if uploaded % 500 == 0 and uploaded > 0:
            elapsed = time.time() - t0
            print(f"  {uploaded}/{len(files)} uploaded ({elapsed:.0f}s)", flush=True)

elapsed = time.time() - t0
print(f"Uploaded {uploaded} files to s3://{bucket} in {elapsed:.0f}s", flush=True)
if failed:
    print(f"WARNING: {failed} uploads failed", flush=True)

s3._endpoint.http_session.close()

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
    print(result.stdout, flush=True)
print("Purge complete.", flush=True)
