"""Download a size-capped subset of shards from a public GCS prefix.

The MT3 training buckets (gs://magentadata/..., gs://mt3/...) are public and
support anonymous XML listing, so individual shards can be selected and
pulled without gsutil/gcloud auth. This script lists every object under a
prefix, picks shards (in listed order, i.e. shard index order for the
`*.tfrecord-NNNNN-of-MMMMM` naming scheme) until a byte cap is reached, and
downloads only those. See ../../../docs/TRAINING_DATA.md for the dataset paths,
sizes, and why you would or would not want to use this.

Usage:
    python gcs_capped_download.py <bucket> <prefix> <out_dir> --cap-gb 50

Example (grab up to 20GB of guitarset):
    python gcs_capped_download.py mt3 data/datasets/guitarset/ ./guitarset --cap-gb 20

Dry run (list what would be pulled, download nothing):
    python gcs_capped_download.py mt3 data/datasets/musicnet_em/ ./out --cap-gb 5 --dry-run
"""
import argparse
import os
import urllib.request
import xml.etree.ElementTree as ET

NS = "{http://doc.s3.amazonaws.com/2006-03-01}"


def list_objects(bucket, prefix, timeout=25):
    """Yield (key, size) for every object under prefix, in listing order."""
    marker = ""
    while True:
        url = (f"https://storage.googleapis.com/{bucket}"
               f"?prefix={prefix}&marker={marker}&max-keys=1000")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
        keys = []
        for c in root.findall(f"{NS}Contents"):
            key = c.find(f"{NS}Key").text
            size = int(c.find(f"{NS}Size").text)
            if size > 0:  # skip the zero-byte "directory marker" object
                yield key, size
            keys.append(key)
        is_trunc = root.find(f"{NS}IsTruncated")
        if is_trunc is not None and is_trunc.text == "true" and keys:
            marker = keys[-1]
        else:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bucket")
    ap.add_argument("prefix")
    ap.add_argument("out_dir")
    ap.add_argument("--cap-gb", type=float, required=True,
                     help="Stop once selected shards reach this many GB.")
    ap.add_argument("--dry-run", action="store_true",
                     help="List what would be downloaded; don't fetch.")
    args = ap.parse_args()

    cap_bytes = args.cap_gb * 1e9
    selected = []
    total = 0
    for key, size in list_objects(args.bucket, args.prefix):
        if total + size > cap_bytes and selected:
            break
        selected.append((key, size))
        total += size
        if total >= cap_bytes:
            break

    print(f"Selected {len(selected)} objects, {total/1e9:.2f} GB "
          f"(cap {args.cap_gb} GB)")
    for key, size in selected:
        print(f"  {size/1e6:8.1f} MB  {key}")

    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for key, size in selected:
        url = f"https://storage.googleapis.com/{args.bucket}/{key}"
        dest = os.path.join(args.out_dir, os.path.basename(key))
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"skip (exists): {dest}")
            continue
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    main()
