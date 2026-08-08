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

"""Tests for preprocessors.split_tokens_strided.

split_tokens_strided is the encoder-window segmenter for the head-crop
overlap feature (see MT3_HEADCROP_OVERLAP_PLAN.md): at hop_tokens ==
window_tokens it must reproduce t5.data.preprocessors.split_tokens exactly,
since Transcriber(..., lookahead_frames=0) relies on that equivalence to
leave today's baseline behavior unchanged.
"""

from mt3 import preprocessors

import numpy as np
import tensorflow as tf


def _make_example(n, feature_dim=5, seed=0):
  rng = np.random.RandomState(seed)
  inputs = rng.rand(n, feature_dim).astype(np.float32)
  times = (np.arange(n) / 125.0).astype(np.float32)
  return inputs, times


def _reference_non_overlapping_split(inputs, times, length):
  """Manual reference matching t5.data.preprocessors.split_tokens at length==window."""
  n = len(inputs)
  num_segments = -(-n // length) if n else 0
  segments = []
  for i in range(num_segments):
    start, end = i * length, min((i + 1) * length, n)
    segments.append((inputs[start:end], times[start:end]))
  return segments


# (n, window, hop) triples covering: exact multiples, off-by-one, hop that
# does not evenly divide window, and window larger than the whole example
# (every window in that case is short, not just the last one).
_GEOMETRIES = [
    (1, 256, 256), (63, 256, 64), (64, 256, 64), (65, 256, 64),
    (255, 256, 256), (256, 256, 256), (257, 256, 256),
    (256, 256, 192), (256, 256, 128), (256, 256, 64),
    (511, 512, 512), (512, 512, 512), (513, 512, 512),
    (512, 512, 384), (512, 512, 256), (512, 512, 448),
    (2560, 512, 384), (2563, 256, 128),
    (511, 562, 15),  # window > n: every window is short, not only the last.
]


class SplitTokensStridedTest(tf.test.TestCase):

  def test_equivalent_to_non_overlapping_split_when_hop_equals_window(self):
    for n, window, _ in _GEOMETRIES:
      inputs, times = _make_example(n)
      ds = tf.data.Dataset.from_tensors({'inputs': inputs, 'input_times': times})
      windows = list(
          preprocessors.split_tokens_strided(
              ds, window_tokens=window, hop_tokens=window,
              feature_key='inputs',
              additional_feature_keys=['input_times']).as_numpy_iterator())
      reference = _reference_non_overlapping_split(inputs, times, window)
      self.assertLen(windows, len(reference), f'n={n} window={window}')
      for got, (ref_inputs, ref_times) in zip(windows, reference):
        self.assertAllClose(got['inputs'], ref_inputs, msg=f'n={n} window={window}')
        self.assertAllEqual(got['input_times'], ref_times, msg=f'n={n} window={window}')

  def test_kept_regions_tile_the_example_with_no_gap_or_overlap(self):
    for n, window, hop in _GEOMETRIES:
      inputs, times = _make_example(n)
      ds = tf.data.Dataset.from_tensors({'inputs': inputs, 'input_times': times})
      windows = list(
          preprocessors.split_tokens_strided(
              ds, window_tokens=window, hop_tokens=hop,
              feature_key='inputs',
              additional_feature_keys=['input_times']).as_numpy_iterator())
      num_windows = len(windows)
      kept_end = 0
      for i in range(num_windows):
        kept_start = i * hop
        kept_stop = (i + 1) * hop if i < num_windows - 1 else n
        self.assertEqual(kept_start, kept_end,
                         f'gap/overlap at window {i}, n={n} window={window} hop={hop}')
        kept_end = kept_stop
      self.assertEqual(kept_end, n, f'incomplete coverage, n={n} window={window} hop={hop}')

  def test_window_content_and_orig_length_are_correct(self):
    for n, window, hop in _GEOMETRIES:
      inputs, times = _make_example(n)
      ds = tf.data.Dataset.from_tensors({'inputs': inputs, 'input_times': times})
      windows = list(
          preprocessors.split_tokens_strided(
              ds, window_tokens=window, hop_tokens=hop,
              feature_key='inputs',
              additional_feature_keys=['input_times']).as_numpy_iterator())
      for i, w in enumerate(windows):
        start = i * hop
        end = min(start + window, n)
        self.assertAllClose(w['inputs'], inputs[start:end],
                            msg=f'n={n} window={window} hop={hop} i={i}')
        self.assertAllEqual(w['input_times'], times[start:end],
                            msg=f'n={n} window={window} hop={hop} i={i}')

  def test_passthrough_features_are_replicated_unchanged(self):
    inputs, times = _make_example(600)
    unique_id = np.array([42], dtype=np.int32)
    ds = tf.data.Dataset.from_tensors({
        'inputs': inputs, 'input_times': times, 'unique_id': unique_id,
    })
    windows = list(
        preprocessors.split_tokens_strided(
            ds, window_tokens=256, hop_tokens=128,
            feature_key='inputs', additional_feature_keys=['input_times'],
            passthrough_feature_keys=['unique_id']).as_numpy_iterator())
    self.assertLen(windows, 5)
    for w in windows:
      self.assertAllEqual(w['unique_id'], unique_id)

  def test_empty_example_yields_no_windows(self):
    ds = tf.data.Dataset.from_tensors({
        'inputs': np.zeros((0, 5), dtype=np.float32),
        'input_times': np.zeros((0,), dtype=np.float32),
    })
    windows = list(
        preprocessors.split_tokens_strided(
            ds, window_tokens=256, hop_tokens=128,
            additional_feature_keys=['input_times']).as_numpy_iterator())
    self.assertEmpty(windows)

  def test_hop_tokens_out_of_range_raises(self):
    for bad_hop in (0, -1, 300):
      with self.assertRaisesRegex(ValueError, 'hop_tokens must be in'):
        preprocessors.split_tokens_strided(
            tf.data.Dataset.from_tensors({'inputs': np.zeros((10, 5), np.float32)}),
            window_tokens=256, hop_tokens=bad_hop)

  def test_overlapping_split_and_passthrough_keys_raises(self):
    with self.assertRaisesRegex(ValueError, 'also included in passthrough keys'):
      preprocessors.split_tokens_strided(
          tf.data.Dataset.from_tensors({
              'inputs': np.zeros((10, 5), np.float32),
              'input_times': np.zeros((10,), np.float32),
          }),
          window_tokens=256, hop_tokens=128,
          additional_feature_keys=['input_times'],
          passthrough_feature_keys=['input_times'])


if __name__ == '__main__':
  tf.test.main()
