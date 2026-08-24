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

"""Modal training entrypoint (MODAL_TRAINING_SETUP_PLAN.md).

Runs the pretrained, full-size guitar_pilot fine-tune on a CUDA GPU, against
volumes created ahead of time with:

  modal volume create mt3-guitar-data
  modal volume create mt3-guitar-model
  modal volume create mt3-guitar-runs

and populated with:

  modal volume put mt3-guitar-data data/pilot /pilot
  modal volume put mt3-guitar-model model/mt3 /mt3

Usage (from the transcription repo root):

  modal run mt3/mt3/scripts/modal_train.py --train-steps 1        # preflight
  modal run mt3/mt3/scripts/modal_train.py --train-steps 100      # pilot leg 1
  modal run mt3/mt3/scripts/modal_train.py --train-steps 400      # resume to 500
  modal run mt3/mt3/scripts/modal_train.py::check_gpu             # GPU-only smoke check
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import modal

def _find_fork_root(start: Path) -> Path:
  """Walks up from `start` to the directory containing pyproject.toml.

  Modal re-imports this module inside the remote container to look up
  functions by name (the default, non-serialized invocation path), where
  it may be mounted at a much shallower path with no such ancestor -- a
  fixed `parents[2]` raised IndexError there. That's harmless: by the time
  a container runs, the image (and its add_local_dir call) is already
  built, so FORK_ROOT is only ever load-bearing during the local `modal
  run` invocation that constructs the image in the first place.
  """
  current = start.resolve()
  for candidate in (current, *current.parents):
    if (candidate / 'pyproject.toml').is_file():
      return candidate
  return current


FORK_ROOT = _find_fork_root(Path(__file__).parent)

app = modal.App('mt3-guitar-pilot')

data_volume = modal.Volume.from_name('mt3-guitar-data')
model_volume = modal.Volume.from_name('mt3-guitar-model')
runs_volume = modal.Volume.from_name('mt3-guitar-runs')

VOLUMES = {
    '/workspace/data': data_volume,
    '/workspace/model': model_volume,
    '/workspace/runs': runs_volume,
}

CUDA_TAG = '12.8.1-devel-ubuntu24.04'

# pyproject.toml's own lockfile (uv.lock) is resolved for macOS/arm64 only
# (see [tool.uv].environments) and cannot be reused here. `uv pip install`
# re-resolves the same dependency spec -- including its git sources and its
# non-darwin overrides (tensorflow-cpu, seqio-nightly, tfds-nightly) -- for
# this image's platform instead. JAX is swapped for a CUDA build afterward;
# nothing else in this project depends on JAX's CPU/GPU wheel choice.
image = (
    modal.Image.from_registry(f'nvidia/cuda:{CUDA_TAG}', add_python='3.11')
    .entrypoint([])
    .apt_install('git', 'libsndfile1', 'libfluidsynth3', 'fluid-soundfont-gm', 'ffmpeg')
    # Pinned to match pyproject.toml's own [tool.uv].required-version --
    # a bare `pip install uv` pulled 0.12.1, which that pin rejects outright.
    .pip_install('uv>=0.9.20,<0.10')
    .add_local_dir(
        FORK_ROOT,
        remote_path='/workspace/mt3',
        copy=True,
        ignore=['.venv', '.git', 'build', 'dist', '*.egg-info', '__pycache__'],
    )
    .run_commands(
        # `tool.uv.sources` (the git pins for flax/note-seq/seqio/t5x) and
        # `tool.uv.override-dependencies` (the non-darwin tensorflow-cpu /
        # seqio-nightly / tfds-nightly swap) are only honored when uv treats
        # this as installing the project itself -- i.e. `uv pip install .`
        # run from inside /workspace/mt3, not a path reference from outside.
        'cd /workspace/mt3 && uv pip install --system .',
        # `tensorflow` and the override's `tensorflow-cpu` both write into
        # the same `tensorflow/` import namespace -- uv installs both
        # (override-dependencies adds a constraint, it does not substitute
        # the package name), and whichever's files land last wins, leaving
        # a mix of two builds' compiled .so files. Observed directly: this
        # produced `undefined symbol: ...pywrap_tensorflow_lite_metrics...`
        # on one rebuild where install order changed.
        #
        # An earlier version of this fix captured `TF_VERSION` dynamically
        # from whatever `tensorflow` resolved to, specifically to avoid
        # `--reinstall tensorflow-cpu` drifting ahead to whatever's newest
        # on PyPI. That stopped working once `tensorflow` itself drifted to
        # 2.21.0 (unpinned in pyproject.toml) -- version-consistent with
        # `tensorflow-cpu`/`tensorflow-text`, so the file-mixing bug above
        # didn't recur, but 2.21.0's own `tensorflow-text` wheel is broken
        # independent of that: `undefined symbol: icudt77_dat` in its
        # compiled `.so`, confirmed by reinstalling `tensorflow-text` alone
        # at the matching version (no fix) versus reinstalling both
        # `tensorflow-cpu` and `tensorflow-text` pinned to 2.20.0 (fixed).
        # So this now hardcodes a known-good version rather than matching
        # whatever `tensorflow` resolved to -- a deliberate downgrade, not
        # driven by pyproject.toml's own (still unpinned) `tensorflow` dep.
        # `tensorflow-text` must be reinstalled too, not just `tensorflow-cpu`,
        # or it stays linked against the original (broken) 2.21.0 install.
        # `--no-deps` on the tensorflow-cpu reinstall drops `tensorboard`
        # too -- 2.20.0 depends on it (2.21.0 didn't), and `t5x.checkpoints`
        # imports it directly, so it needs an explicit (dep-resolving)
        # install of its own or t5x fails at import with an unrelated-looking
        # `ModuleNotFoundError: No module named 'tensorboard'`.
        'TF_VERSION=2.20.0 '
        '&& uv pip uninstall --system tensorflow '
        '&& uv pip install --system --reinstall --no-deps "tensorflow-cpu==$TF_VERSION" '
        '&& uv pip install --system --reinstall --no-deps "tensorflow-text==$TF_VERSION" '
        '&& uv pip install --system "tensorboard==$TF_VERSION"',
        "uv pip install --system --upgrade 'jax[cuda12]==0.10.2'",
    )
)


def _ensure_cuda_library_path() -> None:
  """Points the dynamic linker at pip-installed `nvidia-*-cu12` wheels.

  Those wheels drop their `.so` files under `site-packages/nvidia/<name>/lib`
  rather than a system library directory. Without LD_LIBRARY_PATH covering
  it, JAX's CUDA plugin fails to dlopen them (e.g. "cuSPARSE library was
  not found") and silently falls back to CPU instead of erroring loudly.
  Must run before `import jax` (in-process) or before launching t5x.train
  (as a subprocess, which inherits the mutated environment).
  """
  import glob
  import os
  import sysconfig

  site_packages = sysconfig.get_paths()['purelib']
  cuda_dirs = sorted(glob.glob(f'{site_packages}/nvidia/*/lib'))
  if not cuda_dirs:
    return
  existing = os.environ.get('LD_LIBRARY_PATH', '')
  os.environ['LD_LIBRARY_PATH'] = ':'.join(cuda_dirs + ([existing] if existing else []))


# Where a 4 s run writes. Deliberately NOT the 2 s MODEL_DIR baked into
# guitar_pilot_finetune_modal.gin: two window lengths writing checkpoints into
# one directory would interleave incompatible runs in the same step sequence.
#
# Named as the 2 s run's tag plus a `_4s` suffix, so the pair reads as one
# experiment: guitar_pilot_finetune_64ex_it2 is the 2 s baseline and this is
# the same corpus and recipe at the longer window. Both restore from
# checkpoint_0, so `_4s` denotes the window, not a continuation of it2.
CONTEXT_4S_MODEL_DIR = '/workspace/runs/guitar_pilot_finetune_64ex_it2_4s'


def _t5x_train_command(train_steps: int,
                       save_period: int,
                       context_4s: bool = False,
                       model_dir: str | None = None) -> list[str]:
  """Builds the t5x.train argv.

  `context_4s` appends gin/context_4s.gin, which rebinds TASK_FEATURE_LENGTHS
  to the ~4 s window (512 input frames, 2048 target tokens). Gin applies files
  in order and last-write-wins, so the overlay must come AFTER
  guitar_pilot_finetune_modal.gin to override its 256/1024.

  Setting `context_4s` also forces MODEL_DIR away from the 2 s default, so a
  4 s run cannot silently write into the 2 s run's directory. Pass `model_dir`
  to choose the destination explicitly; the flag is not usable in a way that
  reuses the 2 s directory by accident.
  """
  gin_files = [
      '--gin_file=mt3/gin/model.gin',
      '--gin_file=mt3/gin/train.gin',
      '--gin_file=mt3/gin/guitar_pilot_finetune_modal.gin',
  ]
  overrides = []
  if context_4s:
    gin_files.append('--gin_file=mt3/gin/context_4s.gin')
    overrides.append(
        # Quoted because gin parses a binding's value as a Python literal;
        # bare /workspace/... is not one. subprocess gets an argv list rather
        # than a shell string, so these quotes reach gin intact.
        f"--gin.MODEL_DIR='{model_dir or CONTEXT_4S_MODEL_DIR}'")
  elif model_dir:
    overrides.append(f"--gin.MODEL_DIR='{model_dir}'")

  return [
      'python', '-m', 't5x.train',
      '--gin_search_paths=/workspace/mt3',
      *gin_files,
      *overrides,
      # NOT --gin.TRAIN_STEPS: t5x.train's `relative_steps` (an int step
      # count, not a boolean) unconditionally overrides TRAIN_STEPS/
      # total_steps whenever it's set -- see guitar_pilot_finetune_modal.gin.
      # NOT `train_script.train...` either: that alias only resolves inside
      # a .gin file's own text (dynamic_registration import aliasing is
      # file-scoped); CLI --gin.<selector> bindings are matched against the
      # real registered name, which for a bare `gin.configurable(train)` is
      # just `train`.
      f'--gin.train.relative_steps={train_steps}',
      f'--gin.utils.SaveCheckpointConfig.period={save_period}',
      # train.gin hardcodes eval_period=5000; t5x.train requires the
      # checkpoint period, eval period, and GC period (always 0 here) to all
      # be multiples of each other, so a save_period that doesn't divide
      # 5000 (e.g. 2000) fails fast with "Checkpoint period (N), eval period
      # (5000), and GC period (0) must all be multiples of each other."
      # Pinning eval_period to save_period keeps them trivially compatible.
      f'--gin.train.eval_period={save_period}',
      '--alsologtostderr',
  ]


@app.function(image=image, gpu='A10G', timeout=4 * 60 * 60, volumes=VOLUMES)
def run_training(train_steps: int = 1,
                 save_period: int = 25,
                 context_4s: bool = False,
                 model_dir: str | None = None) -> None:
  """Runs one t5x.train invocation and commits the run-artifacts volume.

  `train_steps` is relative to wherever the restored checkpoint currently
  sits (see guitar_pilot_finetune_modal.gin) -- pass 1 for the step 8
  preflight, then 100 and later 400 to resume to a cumulative 500 (step 9).
  T5X itself runs as a subprocess, not in this function's own process.

  `context_4s` switches the run to the ~4 s window and redirects MODEL_DIR
  accordingly; see _t5x_train_command.
  """
  from mt3.model_download import rebuild_checkpoint_0_view

  rebuild_checkpoint_0_view(Path('/workspace/model/mt3'))
  _ensure_cuda_library_path()

  command = _t5x_train_command(train_steps, save_period, context_4s, model_dir)
  print('t5x command:', ' '.join(command), flush=True)
  result = subprocess.run(command, cwd='/workspace/mt3')
  runs_volume.commit()
  result.check_returncode()


@app.function(image=image, gpu='A10G', timeout=300)
def check_gpu() -> None:
  """Fast standalone check that JAX sees the requested NVIDIA GPU.

  Runs the actual `import jax` as a subprocess, not in this function's own
  process: glibc's dynamic linker parses LD_LIBRARY_PATH once at process
  startup, so mutating os.environ afterward (_ensure_cuda_library_path)
  has no effect on this same process's later dlopen calls -- only a
  freshly exec'd child picks up the corrected environment. Confirmed by
  instrumenting the in-process version: the CUDA library dirs and the
  final LD_LIBRARY_PATH were both correct, and `import jax` still fell
  back to CPU with "cuSPARSE library was not found".
  """
  _ensure_cuda_library_path()
  probe = (
      'import jax\n'
      'devices = jax.devices()\n'
      'print(f"jax.devices(): {devices}")\n'
      'raise SystemExit(0 if any(d.platform == "gpu" for d in devices) else 1)\n'
  )
  result = subprocess.run(['python', '-c', probe])
  if result.returncode != 0:
    raise RuntimeError(f'JAX did not report a GPU device (exit code {result.returncode})')


@app.local_entrypoint()
def main(train_steps: int = 1,
         save_period: int = 25,
         context_4s: bool = False,
         model_dir: str | None = None) -> None:
  run_training.remote(train_steps=train_steps, save_period=save_period,
                      context_4s=context_4s, model_dir=model_dir)
