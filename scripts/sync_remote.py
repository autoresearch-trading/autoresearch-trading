"""Sync parquet files to R2 and purge old data. Runs on the Fly.io instance."""

import os
import pathlib
import subprocess
import time

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
uploaded = 0
for f in data_dir.rglob("*.parquet"):
    if f.stat().st_mtime < cutoff:
        continue
    key = str(f.relative_to(data_dir))
    s3.upload_file(str(f), bucket, key)
    uploaded += 1

print(f"Uploaded {uploaded} files to s3://{bucket}")

print("Purging files older than 2 days...")
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
    print(result.stdout)
print("Purge complete.")
