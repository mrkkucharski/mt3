"""Download Magenta's public multi-instrument MT3 checkpoint.

Example, from the MT3 repository:

  uv run python mt3/scripts/download_official_mt3_checkpoint.py \
      --output-dir ../model/mt3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request


BUCKET = 'mt3'
PREFIX = 'checkpoints/mt3/'
LIST_URL = (
    f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?'
    + urllib.parse.urlencode({
        'prefix': PREFIX,
        'fields': 'items(name,size),nextPageToken',
    }))


def _objects() -> list[dict[str, str]]:
  with urllib.request.urlopen(LIST_URL) as response:
    payload = json.load(response)
  if payload.get('nextPageToken'):
    raise RuntimeError('checkpoint listing is paginated; update downloader.')
  return [item for item in payload.get('items', []) if item['name'] != PREFIX]


def download(output_dir: Path) -> None:
  objects = _objects()
  total_bytes = sum(int(item['size']) for item in objects)
  print(f'Downloading {len(objects)} files ({total_bytes / 1024 / 1024:.1f} MiB) to {output_dir}')
  for index, item in enumerate(objects, 1):
    relative_path = item['name'].removeprefix(PREFIX)
    destination = output_dir / relative_path
    expected_size = int(item['size'])
    if destination.is_file() and destination.stat().st_size == expected_size:
      continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_name = urllib.parse.quote(item['name'], safe='')
    url = f'https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/{encoded_name}?alt=media'
    temporary = destination.with_name(destination.name + f'.partial.{os.getpid()}')
    with urllib.request.urlopen(url) as response, temporary.open('wb') as output:
      shutil.copyfileobj(response, output)
    # A previous interrupted invocation can still be finishing in the
    # background. If it won the race, accept its complete destination file.
    if not temporary.is_file():
      if destination.is_file() and destination.stat().st_size == expected_size:
        continue
      raise RuntimeError(f'Download temporary file disappeared for {relative_path}')
    if temporary.stat().st_size != expected_size:
      temporary.unlink(missing_ok=True)
      raise RuntimeError(f'Incomplete download for {relative_path}')
    temporary.replace(destination)
    if index % 25 == 0 or index == len(objects):
      print(f'  {index}/{len(objects)} files')


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  download(args.output_dir.resolve())
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
