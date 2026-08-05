# Copyright 2026 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for model_download."""

from pathlib import Path

import pytest

from mt3 import model_download


def _flat_checkpoint_dir(tmp_path: Path) -> Path:
  model_dir = tmp_path / 'mt3'
  model_dir.mkdir()
  (model_dir / 'checkpoint').write_bytes(b'index-bytes')
  (model_dir / 'target.decoder.layers_0.mlp.wo.kernel').mkdir()
  (model_dir / 'target.decoder.layers_0.mlp.wo.kernel' / 'arr_0').write_bytes(b'weights')
  return model_dir


def test_rebuild_creates_symlinks_to_flat_files(tmp_path):
  model_dir = _flat_checkpoint_dir(tmp_path)

  checkpoint_dir = model_download.rebuild_checkpoint_0_view(model_dir)

  assert checkpoint_dir == model_dir / 'checkpoint_0'
  link = checkpoint_dir / 'checkpoint'
  assert link.is_symlink()
  assert link.read_bytes() == b'index-bytes'
  nested = checkpoint_dir / 'target.decoder.layers_0.mlp.wo.kernel' / 'arr_0'
  assert nested.read_bytes() == b'weights'


def test_rebuild_is_idempotent_and_replaces_stale_links(tmp_path):
  model_dir = _flat_checkpoint_dir(tmp_path)

  model_download.rebuild_checkpoint_0_view(model_dir)
  # Simulate drift: a stale link left over from a previous, different layout.
  stale = model_dir / 'checkpoint_0' / 'stale'
  stale.symlink_to(Path('..') / 'checkpoint')
  checkpoint_dir = model_download.rebuild_checkpoint_0_view(model_dir)

  assert (checkpoint_dir / 'checkpoint').read_bytes() == b'index-bytes'
  assert (checkpoint_dir / 'stale').exists()  # untouched: rebuild only adds/refreshes


def test_rebuild_does_not_recurse_into_checkpoint_0_itself(tmp_path):
  model_dir = _flat_checkpoint_dir(tmp_path)

  model_download.rebuild_checkpoint_0_view(model_dir)
  checkpoint_dir = model_download.rebuild_checkpoint_0_view(model_dir)

  assert not (checkpoint_dir / 'checkpoint_0').exists()


def _fake_snapshot_download(files: dict[str, bytes]):
  def _fake(*, repo_id, revision, allow_patterns, local_dir):
    del repo_id, revision, allow_patterns  # unused by the fake
    for relative_path, contents in files.items():
      destination = Path(local_dir) / model_download.HF_CHECKPOINT_DIR / relative_path
      destination.parent.mkdir(parents=True, exist_ok=True)
      destination.write_bytes(contents)
    return local_dir
  return _fake


def test_download_model_renames_checkpoint_step_dir_to_checkpoint_0(tmp_path, monkeypatch):
  monkeypatch.setattr(
      model_download, 'snapshot_download',
      _fake_snapshot_download({'manifest.ocdbt': b'ocdbt-bytes'}))
  output_dir = tmp_path / 'models'

  checkpoint_dir = model_download.download_model(output_dir)

  assert checkpoint_dir == output_dir / 'checkpoint_0'
  assert (checkpoint_dir / 'manifest.ocdbt').read_bytes() == b'ocdbt-bytes'


def test_download_model_is_idempotent(tmp_path, monkeypatch):
  monkeypatch.setattr(
      model_download, 'snapshot_download',
      _fake_snapshot_download({'manifest.ocdbt': b'ocdbt-bytes'}))
  output_dir = tmp_path / 'models'

  model_download.download_model(output_dir)
  checkpoint_dir = model_download.download_model(output_dir)

  assert (checkpoint_dir / 'manifest.ocdbt').read_bytes() == b'ocdbt-bytes'


def test_download_model_raises_if_checkpoint_step_dir_missing(tmp_path, monkeypatch):
  monkeypatch.setattr(model_download, 'snapshot_download', _fake_snapshot_download({}))

  with pytest.raises(RuntimeError, match=model_download.HF_CHECKPOINT_DIR):
    model_download.download_model(tmp_path / 'models')
