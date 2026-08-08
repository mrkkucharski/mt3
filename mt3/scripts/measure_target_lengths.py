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

"""Measures the target-token length distribution for a given input window.

WHY THIS EXISTS. Extending the acoustic window from 256 frames (~2.048 s) to
512 frames (~4.096 s) roughly doubles the number of note events the decoder
must emit per window, but TASK_FEATURE_LENGTHS['targets'] does not scale
automatically. mt3/tasks.py ends the training pipeline with

    functools.partial(preprocessors.handle_too_long, skip=skip_too_long)

and `skip_too_long` defaults to False (tasks.py), which selects the
`assert_not_too_long` branch -- a tf.debugging.assert_less_equal that RAISES
rather than dropping the example. An undersized `targets` therefore does not
degrade quietly; it kills the run with 'Value for "targets" field exceeds
maximum length', potentially deep into a paid GPU job.

This script answers "what targets length does a 4 s window actually need?"
BEFORE that happens, by running the real training pipeline with `targets` set
deliberately high (--probe_targets, default 4096) so handle_too_long stays
permissive, then histogramming the true lengths.

SAMPLING CAVEAT. The training task applies t5.data.preprocessors
.select_random_chunk(uniform_random_start=True) and mixing
.mix_transcription_examples, so each pass sees a random chunk of each
underlying example and a random mixture. The result is a SAMPLE, not an
exhaustive bound. Raise --max_examples (and re-run with --seed varied) until
the reported maximum stops climbing. The eval task cannot be used for this
measurement at all: it calls preprocessors.add_dummy_targets, so its targets
carry no real event content.

Typical use -- compare the current window against the proposed one:

    uv run python mt3/scripts/measure_target_lengths.py \
        --task=guitar_pilot_notes_ties_vb1_train --inputs=256
    uv run python mt3/scripts/measure_target_lengths.py \
        --task=guitar_pilot_notes_ties_vb1_train --inputs=512
"""

import numpy as np

from absl import app
from absl import flags

import mt3.tasks  # pylint: disable=unused-import

import seqio
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("task", None, "A registered Task.")
flags.DEFINE_integer("inputs", 512,
                     "Input window in spectrogram frames (125 frames = 1 s).")
flags.DEFINE_integer("probe_targets", 4096,
                     "Oversized targets length used only to keep "
                     "handle_too_long permissive while measuring.")
flags.DEFINE_integer("proposed_targets", 1024,
                     "The targets length you intend to ship; reported as an "
                     "overflow rate.")
flags.DEFINE_integer("max_examples", 500, "Number of examples to sample.")
flags.DEFINE_string("split", "train", "Which split to sample.")


def main(_):
  frames_per_second = 125.0  # 16000 Hz / hop_width 128; see spectrograms.py
  window_seconds = FLAGS.inputs / frames_per_second

  if FLAGS.probe_targets <= FLAGS.proposed_targets:
    raise ValueError(
        f"--probe_targets ({FLAGS.probe_targets}) must exceed "
        f"--proposed_targets ({FLAGS.proposed_targets}); otherwise "
        "handle_too_long trips during measurement and the overflow rate "
        "cannot be observed.")

  task = seqio.get_mixture_or_task(FLAGS.task)
  ds = task.get_dataset(
      sequence_length={"inputs": FLAGS.inputs,
                       "targets": FLAGS.probe_targets},
      split=FLAGS.split,
      use_cached=False,
      shuffle=False)

  lengths = []
  for ex in ds.take(FLAGS.max_examples):
    lengths.append(int(tf.shape(ex["targets"])[0]))

  if not lengths:
    raise ValueError(f"Task {FLAGS.task!r} split {FLAGS.split!r} yielded no "
                     "examples; check the TFRecord paths in datasets.py.")

  lengths = np.array(lengths)
  # handle_too_long subtracts 1 from the budget when the feature adds EOS,
  # so the effective ceiling is proposed_targets - 1. Mirror that exactly.
  add_eos = task.output_features["targets"].add_eos
  ceiling = FLAGS.proposed_targets - (1 if add_eos else 0)
  overflow = int((lengths > ceiling).sum())

  print(f"task              {FLAGS.task}")
  print(f"split             {FLAGS.split}")
  print(f"inputs            {FLAGS.inputs} frames  (~{window_seconds:.3f} s)")
  print(f"examples sampled  {len(lengths)}")
  print("")
  print("target token length")
  print(f"  min             {lengths.min()}")
  print(f"  median          {int(np.median(lengths))}")
  print(f"  p95             {int(np.percentile(lengths, 95))}")
  print(f"  p99             {int(np.percentile(lengths, 99))}")
  print(f"  max             {lengths.max()}")
  print("")
  print(f"proposed targets  {FLAGS.proposed_targets} "
        f"(effective ceiling {ceiling}, add_eos={add_eos})")
  print(f"  would overflow  {overflow} / {len(lengths)} "
        f"({100.0 * overflow / len(lengths):.2f}%)")
  if overflow:
    print("  VERDICT         TOO SMALL -- training would raise "
          "'Value for \"targets\" field exceeds maximum length'.")
    print(f"                  Smallest safe value seen here: "
          f"{int(lengths.max()) + (1 if add_eos else 0)}")
  else:
    print(f"  VERDICT         fits, with {ceiling - int(lengths.max())} "
          "tokens of headroom on this sample.")


if __name__ == "__main__":
  flags.mark_flags_as_required(["task"])

  app.run(main)
