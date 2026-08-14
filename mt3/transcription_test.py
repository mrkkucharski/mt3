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

"""Tests for transcription.WindowGeometry and the Transcriber constructor's

validation of it. WindowGeometry describes an encoder window as
[lookback | keep | lookahead]; lookback_frames=0 must always reproduce the
pre-lookback (lookahead-only) geometry exactly.
"""

import tempfile

import tensorflow as tf
from mt3 import transcription


class WindowGeometryTest(tf.test.TestCase):

  def test_baseline_no_overlap(self):
    g = transcription.WindowGeometry(window_frames=256)
    self.assertEqual(g.lookback_frames, 0)
    self.assertEqual(g.lookahead_frames, 0)
    self.assertEqual(g.keep_frames, 256)
    self.assertEqual(g.hop_frames, 256)
    self.assertEqual(g.cost_multiplier, 1.0)

  def test_lookahead_only_matches_pre_lookback_formula(self):
    # This is the compatibility anchor: with lookback_frames=0, keep_frames
    # and cost_multiplier must equal the pre-lookback
    # (window - lookahead) / (window / (window - lookahead)) formulas.
    for window, lookahead in [(256, 0), (256, 64), (512, 256), (512, 137)]:
      g = transcription.WindowGeometry(
          window_frames=window, lookahead_frames=lookahead)
      self.assertEqual(g.keep_frames, window - lookahead, msg=(window, lookahead))
      self.assertEqual(g.hop_frames, window - lookahead, msg=(window, lookahead))
      self.assertAlmostEqual(
          g.cost_multiplier, window / (window - lookahead), msg=(window, lookahead))

  def test_four_requested_geometries(self):
    # 4s window (500 frames at 125 fps): 1s/2s/1s, 1.5s/1s/1.5s,
    # 1s/2.5s/0.5s, 0s/4s/0s (frame counts at 125 frames/s).
    cases = [
        # (window, lookback, lookahead, expected_keep)
        (500, 125, 125, 250),
        (500, 188, 188, 124),
        (500, 125, 63, 312),
        (500, 0, 0, 500),
    ]
    for window, lookback, lookahead, expected_keep in cases:
      g = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      self.assertEqual(g.keep_frames, expected_keep, msg=(window, lookback, lookahead))
      self.assertGreater(g.keep_frames, 0)
      self.assertAlmostEqual(g.cost_multiplier, window / expected_keep)

  def test_cost_multiplier_grows_with_either_side(self):
    # 1s/2s/1s on a 4s (500-frame) window costs the same as 0s/2s/2s: both
    # leave a 250-frame kept region.
    lookback_and_lookahead = transcription.WindowGeometry(
        window_frames=500, lookback_frames=125, lookahead_frames=125)
    lookahead_only = transcription.WindowGeometry(
        window_frames=500, lookahead_frames=250)
    self.assertEqual(
        lookback_and_lookahead.cost_multiplier, lookahead_only.cost_multiplier)

  def test_seconds_conversion(self):
    class _FakeSpectrogramConfig:
      frames_per_second = 125.0
    g = transcription.WindowGeometry(window_frames=500, lookback_frames=125)
    self.assertAlmostEqual(g.seconds(125, _FakeSpectrogramConfig()), 1.0)

  def test_negative_lookback_frames_raises(self):
    with self.assertRaisesRegex(ValueError, 'lookback_frames must be >= 0'):
      transcription.WindowGeometry(window_frames=256, lookback_frames=-1)

  def test_negative_lookahead_frames_raises(self):
    with self.assertRaisesRegex(ValueError, 'lookahead_frames must be >= 0'):
      transcription.WindowGeometry(window_frames=256, lookahead_frames=-1)

  def test_no_kept_region_raises_with_full_geometry_in_message(self):
    with self.assertRaisesRegex(
        ValueError,
        r'window_frames=512 leaves no kept region: lookback_frames=256 \+ '
        r'lookahead_frames=256 >= 512'):
      transcription.WindowGeometry(
          window_frames=512, lookback_frames=256, lookahead_frames=256)

  def test_lookback_plus_lookahead_exactly_equal_to_window_raises(self):
    with self.assertRaises(ValueError):
      transcription.WindowGeometry(
          window_frames=256, lookback_frames=128, lookahead_frames=128)

  def test_one_frame_kept_region_is_accepted(self):
    g = transcription.WindowGeometry(
        window_frames=256, lookback_frames=127, lookahead_frames=128)
    self.assertEqual(g.keep_frames, 1)


class TranscriberGeometryValidationTest(tf.test.TestCase):
  """Tests the parts of Transcriber.__init__ that run before any checkpoint is loaded.

  An existing-but-empty directory is enough for these, since geometry
  validation happens before model construction / restore.
  """

  def test_invalid_geometry_raises_before_loading_a_checkpoint(self):
    with tempfile.TemporaryDirectory() as ckpt:
      with self.assertRaisesRegex(ValueError, 'leaves no kept region'):
        transcription.Transcriber(
            ckpt, input_length=256, lookahead_frames=256)

  def test_nonzero_lookback_frames_raises_not_implemented(self):
    with tempfile.TemporaryDirectory() as ckpt:
      with self.assertRaisesRegex(
          NotImplementedError, 'lookback_frames is not yet wired'):
        transcription.Transcriber(ckpt, input_length=256, lookback_frames=64)

  def test_missing_checkpoint_raises_before_geometry_validation(self):
    with self.assertRaises(FileNotFoundError):
      transcription.Transcriber(
          '/nonexistent/path/for/mt3/tests',
          input_length=256, lookahead_frames=999999)


if __name__ == '__main__':
  tf.test.main()
