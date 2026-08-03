"""Download and validate the public multi-instrument MT3 checkpoint."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request


MODEL_ID = 'official-multitrack-v1'
BUCKET = 'mt3'
PREFIX = 'checkpoints/mt3/'
LIST_URL = (
    f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?'
    + urllib.parse.urlencode({
        'prefix': PREFIX,
        'fields': 'items(name,size,md5Hash),nextPageToken',
    }))


def _objects() -> list[dict[str, str]]:
  with urllib.request.urlopen(LIST_URL) as response:
    payload = json.load(response)
  if payload.get('nextPageToken'):
    raise RuntimeError('checkpoint listing is paginated; update downloader.')
  return [item for item in payload.get('items', []) if item['name'] != PREFIX]


def _matches(destination: Path, item: dict[str, str]) -> bool:
  if not destination.is_file() or destination.stat().st_size != int(item['size']):
    return False
  expected_md5 = item.get('md5Hash')
  if not expected_md5:
    return True
  hasher = hashlib.md5()  # nosec B324 - GCS integrity metadata
  with destination.open('rb') as input_file:
    for block in iter(lambda: input_file.read(1024 * 1024), b''):
      hasher.update(block)
  digest = hasher.digest()
  return base64.b64encode(digest).decode('ascii') == expected_md5


def download_model(output_dir: Path) -> Path:
  """Downloads ``official-multitrack-v1`` and returns its checkpoint directory."""
  output_dir = output_dir.expanduser().resolve()
  checkpoint_dir = output_dir / 'checkpoint_0'
  objects = _objects()
  total_bytes = sum(int(item['size']) for item in objects)
  print(f'Downloading {MODEL_ID}: {len(objects)} files ({total_bytes / 1024 / 1024:.1f} MiB)')
  for index, item in enumerate(objects, 1):
    # The bucket's own objects have no `checkpoint_0/` prefix -- this
    # directory is our own on-disk layout, matching every documented
    # `--checkpoint .../checkpoint_0` invocation, not part of the bucket.
    destination = checkpoint_dir / item['name'].removeprefix(PREFIX)
    if _matches(destination, item):
      continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_name = urllib.parse.quote(item['name'], safe='')
    url = f'https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/{encoded_name}?alt=media'
    temporary = destination.with_name(destination.name + f'.partial.{os.getpid()}')
    with urllib.request.urlopen(url) as response, temporary.open('wb') as output:
      shutil.copyfileobj(response, output)
    if not _matches(temporary, item):
      temporary.unlink(missing_ok=True)
      raise RuntimeError(f'Integrity check failed for {item["name"]}')
    temporary.replace(destination)
    if index % 25 == 0 or index == len(objects):
      print(f'  {index}/{len(objects)} files')
  if not checkpoint_dir.is_dir():
    raise RuntimeError(f'Downloaded model is missing checkpoint directory: {checkpoint_dir}')
  return checkpoint_dir


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', choices=[MODEL_ID], default=MODEL_ID)
  parser.add_argument('--output-dir', type=Path, required=True,
                      help='Directory that will contain checkpoint_0.')
  parser.add_argument('--json', action='store_true', help='Write the checkpoint path as JSON.')
  args = parser.parse_args(argv)
  checkpoint = download_model(args.output_dir)
  if args.json:
    print(json.dumps({'model': args.model, 'checkpoint': str(checkpoint)}, sort_keys=True))
  else:
    print(f'Model ready: {checkpoint}')
  return 0
