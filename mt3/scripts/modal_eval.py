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

"""Modal one-off evaluation entrypoint: score a fixed checkpoint against
whatever eval TFRecord currently sits on the `mt3-guitar-data` volume.

Built to answer "how does an old checkpoint (trained on a smaller corpus)
compare to new ones on today's held-out split" -- `modal_train.py`'s
periodic infer_eval only ever scores the checkpoint it is actively
producing, against whatever was on the data volume *at that time*. This
lets any checkpoint still present on `mt3-guitar-runs` (e.g. the pre-rename
`guitar_pilot_finetune_v1/checkpoint_1004000`) be scored against the
*current* eval set for a true apples-to-apples number.

Uses `t5x.train` with `relative_steps=1` rather than the more obviously-
named `t5x.eval` (mt3/gin/eval.gin): `t5x.eval`'s `InferenceEvaluator`
passes `log_dir` explicitly to `seqio.Evaluator.__init__` but never passes
`logger_cls`, and that parameter's gin-configured default (bound to match
train.gin's identical setting) is never actually injected at that call
site -- confirmed via `seqio.Evaluator`'s own warning, "'logger_cls' is
empty so seqio.Evaluator will not log its results", even though the
operative gin config dump showed the binding was recorded. Net effect: a
real `t5x.eval` run completes successfully but writes no metrics anywhere
(no JSONL, no TensorBoard events, no stdout summary) -- silently useless
despite exit code 0. `t5x.train`'s periodic infer_eval doesn't share this
bug (every fine-tune leg this session has produced a correct
`inference_eval/*.jsonl`), so this resumes the target checkpoint into a
disposable probe run directory for exactly one training step -- enough to
trigger that periodic eval once -- rather than actually continuing to
train it.

Deliberately self-contained (image/volumes duplicated from `modal_train.py`
rather than imported from it): `modal run` loads this file's module-level
code locally first, in whatever bare Python environment the `modal` CLI
itself runs under, which does not have the `mt3` package installed -- a
`from mt3.scripts.modal_train import ...` at module level fails there with
`ModuleNotFoundError: No module named 'mt3.scripts'` even though the same
import works fine once inside the built container. `modal_train.py` avoids
this by deferring its own `mt3.*` imports to inside the function body,
which only ever runs remotely; this script follows the same rule instead of
depending on that module.

Usage (from the transcription repo root):

  modal run mt3/mt3/scripts/modal_eval.py \
      --checkpoint-path guitar_pilot_finetune_v1/checkpoint_1004000 \
      --probe-model-dir eval_probe_checkpoint_1004000
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import modal


def _find_fork_root(start: Path) -> Path:
  """Walks up from `start` to the directory containing pyproject.toml. See
  `modal_train.py`'s identical helper for why this must tolerate a shallow
  in-container mount."""
  current = start.resolve()
  for candidate in (current, *current.parents):
    if (candidate / 'pyproject.toml').is_file():
      return candidate
  return current


FORK_ROOT = _find_fork_root(Path(__file__).parent)

app = modal.App('mt3-guitar-eval')

data_volume = modal.Volume.from_name('mt3-guitar-data')
model_volume = modal.Volume.from_name('mt3-guitar-model')
runs_volume = modal.Volume.from_name('mt3-guitar-runs')

VOLUMES = {
    '/workspace/data': data_volume,
    '/workspace/model': model_volume,
    '/workspace/runs': runs_volume,
}

CUDA_TAG = '12.8.1-devel-ubuntu24.04'

# Kept identical to modal_train.py's own image -- see that file's comments
# for why each step is here (uv version pin, non-darwin tensorflow-cpu
# swap, JAX CUDA build).
image = (
    modal.Image.from_registry(f'nvidia/cuda:{CUDA_TAG}', add_python='3.11')
    .entrypoint([])
    .apt_install('git', 'libsndfile1', 'libfluidsynth3', 'fluid-soundfont-gm', 'ffmpeg')
    .pip_install('uv>=0.9.20,<0.10')
    .add_local_dir(
        FORK_ROOT,
        remote_path='/workspace/mt3',
        copy=True,
        ignore=['.venv', '.git', 'build', 'dist', '*.egg-info', '__pycache__'],
    )
    .run_commands(
        'cd /workspace/mt3 && uv pip install --system .',
        'TF_VERSION=$(uv pip show tensorflow | grep "^Version:" | awk \'{print $2}\') '
        '&& uv pip uninstall --system tensorflow '
        '&& uv pip install --system --reinstall --no-deps "tensorflow-cpu==$TF_VERSION"',
        "uv pip install --system --upgrade 'jax[cuda12]==0.10.2'",
    )
)


def _ensure_cuda_library_path() -> None:
  """See `modal_train.py`'s identical helper for why this is needed."""
  import glob
  import os
  import sysconfig

  site_packages = sysconfig.get_paths()['purelib']
  cuda_dirs = sorted(glob.glob(f'{site_packages}/nvidia/*/lib'))
  if not cuda_dirs:
    return
  existing = os.environ.get('LD_LIBRARY_PATH', '')
  os.environ['LD_LIBRARY_PATH'] = ':'.join(cuda_dirs + ([existing] if existing else []))


def _t5x_probe_command(checkpoint_path: str, probe_model_dir: str) -> list[str]:
  return [
      'python', '-m', 't5x.train',
      '--gin_search_paths=/workspace/mt3',
      '--gin_file=mt3/gin/model.gin',
      '--gin_file=mt3/gin/train.gin',
      '--gin_file=mt3/gin/guitar_pilot_finetune_modal.gin',
      # subprocess.run receives this as a list, so there is no shell to strip
      # a layer of quoting -- each argv element is passed to t5x.train's
      # flag parser exactly as written here. Gin parses override values via
      # ast.literal_eval, so a *single* pair of quote characters must be
      # present in the value to mark it as a Python string literal.
      f"--gin.MODEL_DIR='/workspace/runs/{probe_model_dir}'",
      f"--gin.CHECKPOINT_PATH='/workspace/runs/{checkpoint_path}'",
      # One step is enough to trigger the periodic infer_eval once; this
      # isn't meant to actually continue training the probed checkpoint.
      '--gin.train.relative_steps=1',
      '--gin.utils.SaveCheckpointConfig.period=1',
      # Must be a multiple of the checkpoint period (see modal_train.py's
      # identical comment on this exact t5x.train constraint).
      '--gin.train.eval_period=1',
      '--alsologtostderr',
  ]


@app.function(image=image, gpu='A10G', timeout=30 * 60, volumes=VOLUMES)
def run_eval(checkpoint_path: str, probe_model_dir: str) -> None:
  """Resumes `checkpoint_path` (relative to the runs volume root) into the
  disposable `probe_model_dir` (also relative to the runs volume root) for
  one step, to trigger its infer_eval against the eval task's current
  dataset (relative to the data volume root). Metrics land under
  `probe_model_dir/inference_eval/`, same layout as a real training run."""
  from mt3.model_download import rebuild_checkpoint_0_view

  rebuild_checkpoint_0_view(Path('/workspace/model/mt3'))
  _ensure_cuda_library_path()

  result = subprocess.run(
      _t5x_probe_command(checkpoint_path, probe_model_dir), cwd='/workspace/mt3')
  runs_volume.commit()
  result.check_returncode()


@app.local_entrypoint()
def main(checkpoint_path: str, probe_model_dir: str) -> None:
  run_eval.remote(checkpoint_path=checkpoint_path, probe_model_dir=probe_model_dir)
