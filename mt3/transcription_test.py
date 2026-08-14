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

"""Tests for transcription.WindowGeometry, _windowed_input_dataset, and the

Transcriber constructor's validation of them. WindowGeometry describes an
encoder window as [lookback | keep | lookahead]; lookback_frames=0 must
always reproduce the pre-lookback (lookahead-only) geometry exactly.
"""

import math
import tempfile

import numpy as np
import tensorflow as tf
from mt3 import preprocessors
from mt3 import spectrograms
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

  def test_missing_checkpoint_raises_before_geometry_validation(self):
    with self.assertRaises(FileNotFoundError):
      transcription.Transcriber(
          '/nonexistent/path/for/mt3/tests',
          input_length=256, lookahead_frames=999999)


# Small integer analogs of the four requested real-world geometries, using
# the default hop_width=128 frame grid; kept proportionally similar in
# shape (symmetric overlap, larger-overlap-than-keep, asymmetric overlap,
# no overlap) rather than literal second values, so tests build small
# audio arrays and run fast.
_TEST_GEOMETRIES = (
    # (window_frames, lookback_frames, lookahead_frames)  -- like 1s/2s/1s
    (20, 5, 5),
    # like 1.5s/1s/1.5s: overlap on each side larger than the kept region
    (20, 8, 8),
    # like 1s/2.5s/0.5s: asymmetric, lookback > lookahead
    (20, 5, 2),
    # like 0s/4s/0s: no overlap at all
    (20, 0, 0),
)


def _make_ramp_audio(num_frames: int, hop: int) -> np.ndarray:
  """A distinguishable-from-silence signal: frame i's samples are all i+1."""
  return np.repeat(np.arange(1, num_frames + 1, dtype=np.float32), hop)


class WindowedInputDatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.hop = self.spectrogram_config.hop_width
    self.fps = self.spectrogram_config.frames_per_second

  def test_lookback_zero_matches_pre_lookback_baseline(self):
    # The compatibility anchor: lookback_frames=0 must reproduce exactly
    # what a plain (non-left-padded) split_tokens_strided call produces.
    for window, lookahead in [(20, 0), (20, 5), (16, 15)]:
      for num_orig_frames in (7, 20, 41):
        geometry = transcription.WindowGeometry(
            window_frames=window, lookahead_frames=lookahead)
        audio = _make_ramp_audio(num_orig_frames, self.hop)
        got = list(
            transcription._windowed_input_dataset(
                audio, self.spectrogram_config, geometry
            ).as_numpy_iterator())

        frame_times = np.arange(num_orig_frames) / self.fps
        ref_ds = tf.data.Dataset.from_tensors({
            'inputs': audio.reshape(num_orig_frames, self.hop),
            'input_times': frame_times,
        })
        want = list(
            preprocessors.split_tokens_strided(
                ref_ds, window_tokens=window, hop_tokens=geometry.keep_frames,
                feature_key='inputs',
                additional_feature_keys=['input_times']
            ).as_numpy_iterator())

        self.assertLen(got, len(want), msg=(window, lookahead, num_orig_frames))
        for g, w in zip(got, want):
          self.assertAllClose(g['inputs'], w['inputs'])
          self.assertAllClose(g['input_times'], w['input_times'])


  def test_window_zero_lookback_region_is_silent_and_negative_time(self):
    for window, lookback, lookahead in _TEST_GEOMETRIES:
      if lookback == 0:
        continue
      geometry = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      audio = _make_ramp_audio(50, self.hop)
      windows = list(
          transcription._windowed_input_dataset(
              audio, self.spectrogram_config, geometry
          ).as_numpy_iterator())
      first = windows[0]
      self.assertAllEqual(
          first['inputs'][:lookback], np.zeros((lookback, self.hop), np.float32),
          msg=(window, lookback, lookahead))
      expected_times = (np.arange(lookback) - lookback) / self.fps
      self.assertAllClose(
          first['input_times'][:lookback], expected_times,
          msg=(window, lookback, lookahead))
      self.assertLess(first['input_times'][0], 0)

  def test_kept_regions_tile_original_audio_with_no_gap_or_overlap(self):
    for window, lookback, lookahead in _TEST_GEOMETRIES:
      geometry = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      for num_orig_frames in (
          geometry.keep_frames,  # exact multiple (1x)
          geometry.keep_frames * 3,  # exact multiple (3x)
          geometry.keep_frames * 2 + 3,  # not a multiple
          3,  # shorter than one kept region
      ):
        audio = _make_ramp_audio(num_orig_frames, self.hop)
        windows = list(
            transcription._windowed_input_dataset(
                audio, self.spectrogram_config, geometry
            ).as_numpy_iterator())

        expected_num_windows = math.ceil(num_orig_frames / geometry.keep_frames)
        self.assertLen(
            windows, expected_num_windows,
            msg=(window, lookback, lookahead, num_orig_frames))

        for i, w in enumerate(windows):
          kept_start_frame = round(w['input_times'][0] * self.fps) + lookback
          self.assertEqual(
              kept_start_frame, i * geometry.keep_frames,
              msg=(window, lookback, lookahead, num_orig_frames, i))

  def test_kept_region_content_matches_original_audio(self):
    # The frames inside a window's kept region (after skipping its own
    # lookback prefix) must be exactly the corresponding slice of the
    # original, un-padded audio -- not the left-pad, not another window's
    # content.
    window, lookback, lookahead = 20, 5, 2
    geometry = transcription.WindowGeometry(
        window_frames=window, lookback_frames=lookback,
        lookahead_frames=lookahead)
    num_orig_frames = 47
    audio = _make_ramp_audio(num_orig_frames, self.hop)
    windows = list(
        transcription._windowed_input_dataset(
            audio, self.spectrogram_config, geometry
        ).as_numpy_iterator())
    orig_frames = audio.reshape(num_orig_frames, self.hop)
    for i, w in enumerate(windows):
      kept = w['inputs'][lookback:lookback + geometry.keep_frames]
      start = i * geometry.keep_frames
      end = min(start + geometry.keep_frames, num_orig_frames)
      want = orig_frames[start:end]
      self.assertAllClose(kept[:len(want)], want, msg=i)


class _FakeCodec:

  def __init__(self, steps_per_second):
    self.steps_per_second = steps_per_second


class _FakeSpectrogramConfig:

  def __init__(self, frames_per_second, sample_rate=16000):
    self.frames_per_second = frames_per_second
    self.sample_rate = sample_rate


class QuantizeTest(tf.test.TestCase):

  def test_floors_onto_the_codec_step_grid(self):
    codec = _FakeCodec(steps_per_second=100)  # step = 0.01s
    self.assertAlmostEqual(transcription._quantize(0.017, codec), 0.01)
    self.assertAlmostEqual(transcription._quantize(0.02, codec), 0.02)
    self.assertAlmostEqual(transcription._quantize(0.0, codec), 0.0)

  def test_exact_multiples_are_not_floored_down_a_whole_extra_step(self):
    # Regression test: `t - t % step` looks equivalent but isn't -- `5.0 %
    # 0.01` in floating point is ~0.00999999999999990, not 0.0, since 0.01
    # has no exact binary representation. That floors 5.0 down to 4.99.
    codec = _FakeCodec(steps_per_second=100)
    for t in (5.0, 4.5, 6.5, 8.5, 9.0):
      self.assertEqual(transcription._quantize(t, codec), t, msg=t)


class MinDecodeTimeTest(tf.test.TestCase):

  def test_zero_lookback_returns_none_not_start_time(self):
    # None (not start_time) is what makes lookback_frames=0 a true no-op:
    # a non-None min_time changes decode_events' max_time boundary test
    # from the legacy inclusive `>` to the half-open `>=`, even if it can
    # never actually suppress anything on its own. See the docstring.
    geometry = transcription.WindowGeometry(window_frames=256)
    codec = _FakeCodec(steps_per_second=100)
    spec = _FakeSpectrogramConfig(frames_per_second=125)
    self.assertIsNone(
        transcription._min_decode_time(0.32, geometry, codec, spec))

  def test_adds_and_quantizes_lookback_seconds(self):
    # lookback_frames=125 at 125 fps -> 1.0s of lookback.
    geometry = transcription.WindowGeometry(
        window_frames=500, lookback_frames=125, lookahead_frames=125)
    codec = _FakeCodec(steps_per_second=100)  # step = 0.01s
    spec = _FakeSpectrogramConfig(frames_per_second=125)
    self.assertAlmostEqual(
        transcription._min_decode_time(0.32, geometry, codec, spec), 1.32)

  def test_negative_start_time_from_window_zero_padding(self):
    # Window 0 with lookback has a negative start_time (its own first
    # frame is in the artificial silence pad); min_decode_time should land
    # at or near true audio time 0.
    geometry = transcription.WindowGeometry(
        window_frames=500, lookback_frames=125, lookahead_frames=125)
    codec = _FakeCodec(steps_per_second=100)
    spec = _FakeSpectrogramConfig(frames_per_second=125)
    self.assertAlmostEqual(
        transcription._min_decode_time(-1.0, geometry, codec, spec), 0.0)


class ShouldCapLastSegmentTailTest(tf.test.TestCase):

  def test_false_for_the_pure_baseline(self):
    # The compatibility anchor: 0/0 must not gain new behavior relative to
    # the pre-overlap baseline.
    geometry = transcription.WindowGeometry(window_frames=256)
    self.assertFalse(transcription._should_cap_last_segment_tail(geometry))

  def test_true_for_lookahead_only(self):
    geometry = transcription.WindowGeometry(window_frames=256, lookahead_frames=64)
    self.assertTrue(transcription._should_cap_last_segment_tail(geometry))

  def test_true_for_lookback_only(self):
    geometry = transcription.WindowGeometry(window_frames=256, lookback_frames=64)
    self.assertTrue(transcription._should_cap_last_segment_tail(geometry))

  def test_true_for_both(self):
    geometry = transcription.WindowGeometry(
        window_frames=256, lookback_frames=32, lookahead_frames=32)
    self.assertTrue(transcription._should_cap_last_segment_tail(geometry))


class CapLastSegmentTailTest(tf.test.TestCase):

  def test_caps_the_chronologically_last_prediction(self):
    predictions = [
        {'start_time': 0.0},
        {'start_time': 2.0},
        {'start_time': 1.0},
    ]
    transcription._cap_last_segment_tail(predictions, audio_duration_seconds=2.5)
    self.assertNotIn('max_decode_time', predictions[0])
    self.assertNotIn('max_decode_time', predictions[2])
    self.assertAlmostEqual(predictions[1]['max_decode_time'], 2.5, delta=1e-5)

  def test_cap_is_nudged_past_audio_duration_not_exactly_on_it(self):
    # Regression test: a note whose true offset lands exactly at
    # audio_duration_seconds must not be silently dropped. If the last
    # segment also has its own min_decode_time (lookback_frames > 0),
    # run_length_encoding.decode_events treats max_time as the EXCLUSIVE
    # end of a half-open interval -- correct when max_time is the *next*
    # segment's min_decode_time (nothing left over, since that segment's
    # min_time picks it up instead), wrong here since there is no next
    # segment to compensate.
    predictions = [{'start_time': 0.0}]
    transcription._cap_last_segment_tail(predictions, audio_duration_seconds=9.0)
    self.assertGreater(predictions[0]['max_decode_time'], 9.0)
    # ...but only a hair past it: a genuinely hallucinated event from the
    # padding tail needs at least one full codec step (0.01s at the
    # defaults) past audio_duration_seconds to be encoded at all, and must
    # still be excluded.
    self.assertLess(predictions[0]['max_decode_time'], 9.005)

  def test_empty_predictions_is_a_noop(self):
    predictions = []
    transcription._cap_last_segment_tail(predictions, audio_duration_seconds=2.5)
    self.assertEqual(predictions, [])


if __name__ == '__main__':
  tf.test.main()
