"""Download the pinned guitar_pilot MT3 fine-tune checkpoint from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import snapshot_download


MODEL_ID = 'guitar-pilot-v1'
HF_REPO_ID = 'mrkkucharski/mt3-guitar-pilot'
# The commit this checkpoint was uploaded in (see
# `huggingface_hub.HfApi().list_repo_commits(HF_REPO_ID)`), pinned the same
# way vgt pins this fork's own commit (MT3_PINNED_COMMIT). Bump this together
# with HF_CHECKPOINT_DIR whenever a new fine-tune is pushed to HF_REPO_ID.
HF_REVISION = '0f1b4d0bece94d0a7458b7acbfd7c685f202d75f'
# The checkpoint-step folder within HF_REPO_ID@HF_REVISION holding the latest
# fine-tune. It's an Orbax/OCDBT checkpoint from this fork's own training
# (see Transcriber._restore's use_orbax=True path), not the legacy flat
# layout `rebuild_checkpoint_0_view` reconstructs below.
HF_CHECKPOINT_DIR = 'checkpoint_1002000'


def download_model(output_dir: Path) -> Path:
  """Downloads the pinned guitar_pilot checkpoint and returns its directory.

  The downloaded directory is always materialized as ``checkpoint_0`` under
  ``output_dir``, matching every documented ``--checkpoint .../checkpoint_0``
  invocation, regardless of the step number in its name on the HF repo.
  """
  output_dir = output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_dir = output_dir / 'checkpoint_0'
  print(f'Downloading {MODEL_ID} from {HF_REPO_ID}@{HF_REVISION[:12]}')
  with tempfile.TemporaryDirectory(dir=output_dir) as snapshot_dir:
    snapshot_root = Path(snapshot_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        allow_patterns=f'{HF_CHECKPOINT_DIR}/*',
        local_dir=snapshot_dir,
    ))
    downloaded = snapshot_root / HF_CHECKPOINT_DIR
    if not downloaded.is_dir():
      raise RuntimeError(
          f'{HF_REPO_ID}@{HF_REVISION} has no {HF_CHECKPOINT_DIR}/ directory')
    if checkpoint_dir.exists():
      shutil.rmtree(checkpoint_dir)
    shutil.move(str(downloaded), str(checkpoint_dir))
  return checkpoint_dir


def rebuild_checkpoint_0_view(model_dir: Path) -> Path:
  """Rebuilds the legacy-checkpoint compatibility view under ``checkpoint_0``.

  An earlier downloader version wrote the checkpoint's flat files (the
  `checkpoint` index and one directory per variable) straight into
  ``model_dir`` instead of ``model_dir/checkpoint_0``; every documented
  ``--checkpoint .../checkpoint_0`` invocation expects the latter. Rather
  than re-download, this recreates ``checkpoint_0`` as a directory of
  symlinks back to those flat files, matching what the local Mac checkpoint
  directory already has by hand. It is idempotent and safe to call
  unconditionally: a volume upload (e.g. onto Modal) may not carry relative
  symlinks faithfully, so callers should rebuild this view themselves after
  mounting rather than trust whatever came over the wire.
  """
  model_dir = model_dir.expanduser().resolve()
  checkpoint_dir = model_dir / 'checkpoint_0'
  checkpoint_dir.mkdir(exist_ok=True)
  for entry in model_dir.iterdir():
    if entry.name == 'checkpoint_0':
      continue
    link = checkpoint_dir / entry.name
    if link.is_symlink() or link.exists():
      link.unlink()
    link.symlink_to(Path('..') / entry.name, target_is_directory=entry.is_dir())
  return checkpoint_dir


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', choices=[MODEL_ID], default=MODEL_ID)
  parser.add_argument('--output-dir', type=Path,
                      help='Directory that will contain checkpoint_0.')
  parser.add_argument('--json', action='store_true', help='Write the checkpoint path as JSON.')
  parser.add_argument('--rebuild-checkpoint-0-view', type=Path, default=None,
                      help='Skip downloading; only rebuild checkpoint_0 under this directory.')
  args = parser.parse_args(argv)
  if args.rebuild_checkpoint_0_view is not None:
    checkpoint = rebuild_checkpoint_0_view(args.rebuild_checkpoint_0_view)
  else:
    if args.output_dir is None:
      parser.error('--output-dir is required unless --rebuild-checkpoint-0-view is given.')
    checkpoint = download_model(args.output_dir)
  if args.json:
    print(json.dumps({'model': args.model, 'checkpoint': str(checkpoint)}, sort_keys=True))
  else:
    print(f'Model ready: {checkpoint}')
  return 0
