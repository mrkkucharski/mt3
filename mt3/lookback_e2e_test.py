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

"""End-to-end segmentation-invariance and baseline-equivalence tests for lookback.

For a fixed reference NoteSequence, _simulate_and_decode() drives the real
production decode path (metrics_utils.event_predictions_to_ns,
NoteEncodingWithTiesSpec, and the Transcriber crop-bound helpers
_min_decode_time/_should_cap_last_segment_tail/_cap_last_segment_tail) with
a *perfect* simulated model: for each window (segmented exactly as
transcription._windowed_input_dataset would, per WindowGeometry), it reports
the true events within that window's own observed real-audio range, plus a
correct tie section for whatever is already active at the window's first
real frame. This exercises the full multi-window segmentation-and-recombine
pipeline without needing a checkpoint or a real forward pass.

The property under test: for a fixed source recording, the decoded output
must be identical no matter how the encoder window is sliced up -- lookback,
lookahead, both, or neither.
"""

import math

from mt3 import event_codec
from mt3 import metrics_utils
from mt3 import note_sequences
from mt3 import run_length_encoding
from mt3 import transcription
from mt3 import vocabularies

import note_seq
import numpy as np
import tensorflow as tf

CODEC = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=1))
FPS = 125  # Spectrogram frame rate; matches WindowGeometry's own frame unit.


class _FakeSpectrogramConfig:
  frames_per_second = FPS


def _notes_key(ns: note_seq.NoteSequence):
  """(pitch, program, is_drum, start, end) tuple set, quantized to codec steps."""
  step = 1.0 / CODEC.steps_per_second
  return {
      (note.pitch, note.program, note.is_drum,
       round(note.start_time / step) * step,
       round(note.end_time / step) * step)
      for note in ns.notes
  }


def _build_reference_ns() -> note_seq.NoteSequence:
  """A 9s recording with notes chosen to stress every edge case this feature

  touches: a note starting at t=0 (window 0's lookback-padding edge), notes
  spanning what would be a window boundary under several different
  geometries, a very short note near a boundary, and a note ending exactly
  at the recording's end (the last-window tail-cap edge).
  """
  notes = [
      (60, 0, 0.0, 0.6),
      (62, 25, 1.8, 3.4),
      (64, 25, 2.0, 2.05),
      (67, 30, 5.0, 5.5),
      (72, 0, 4.5, 6.5),
      (69, 30, 8.5, 9.0),
  ]
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  for pitch, program, start, end in notes:
    ns.notes.add(start_time=start, end_time=end, pitch=pitch, velocity=100,
                program=program, is_drum=False)
    ns.total_time = max(ns.total_time, end)
  return ns


_AUDIO_DURATION = 9.0

# (window_frames, lookback_frames, lookahead_frames), all multiples of 25
# frames (0.2s at 125fps == 20 codec steps at 100 steps/s) so every derived
# time boundary lands on a clean decimal, matching the four requested
# geometry shapes.
_TEST_GEOMETRIES = {
    '0s/4s/0s (baseline)': (500, 0, 0),
    '1s/2s/1s': (500, 125, 125),
    '1.4s/1s/1.4s': (475, 175, 175),
    '1s/2.4s/0.6s': (500, 125, 75),
    # Lookback and lookahead in isolation: the module docstring claims this
    # holds "lookback, lookahead, both, or neither" -- the four geometries
    # above only ever combine the two (or use neither), never exactly one.
    '1s/3s/0s (lookback only)': (500, 125, 0),
    '0s/3s/1s (lookahead only)': (500, 0, 125),
}


def _simulate_and_decode(ns: note_seq.NoteSequence, geometry: transcription.WindowGeometry,
                         audio_duration: float) -> dict:
  """Simulates a perfect multi-window transcription of `ns`, decoded back through

  the real production combine path. See module docstring.
  """
  # Integer-indexed rather than np.arange(..., step=0.01) directly: the
  # latter accumulates floating-point drift over ~10s at 0.01s resolution,
  # so frame_times[k] can land a hair below k*0.01 -- and _quantize's
  # floor-based modulo is sensitive enough to that to floor down a whole
  # extra codec step for values just below a clean boundary. The margin
  # covers the largest a window's own nominal end can reach beyond
  # audio_duration (its full window_frames, in the extreme case a single
  # window covers the whole clip).
  margin_seconds = geometry.window_frames / FPS + 1.0
  num_frame_times = round((audio_duration + margin_seconds) * 100) + 1
  frame_times = np.arange(num_frame_times) / 100.0
  times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
  (tokens, event_start_indices, _, state_tokens, state_event_indices) = (
      run_length_encoding.encode_and_index_events(
          state=note_sequences.NoteEncodingState(),
          event_times=times, event_values=values,
          encode_event_fn=note_sequences.note_event_data_to_events,
          codec=CODEC, frame_times=frame_times,
          encoding_state_to_events_fn=note_sequences.note_encoding_state_to_events))
  tie_event_id = CODEC.encode_event(event_codec.Event('tie', 0))

  def _frame_idx(t):
    # round(), not np.searchsorted(frame_times, t): window_start_time is
    # computed via float subtraction (e.g. 2.0 - 1.4 == 0.6000000000000001,
    # not exactly 0.6), so it can land a hair above its true grid point --
    # searchsorted would then return the NEXT index (frame_times[61]==0.61
    # instead of frame_times[60]==0.6), silently misaligning this window's
    # sliced content by a full frame relative to its own reported
    # start_time. Rounding is robust to that noise; frame_times is built at
    # exactly the same 0.01s resolution these times are meant to land on.
    return int(np.clip(round(t * 100), 0, len(frame_times) - 1))

  def _compress_shifts(raw_tokens):
    ds = tf.data.Dataset.from_tensors({'targets': raw_tokens})
    ds = run_length_encoding.run_length_encode_shifts_fn(
        CODEC, feature_key='targets')(ds)
    return np.asarray(next(iter(ds))['targets'].numpy(), np.int32)

  keep_seconds = geometry.keep_frames / FPS
  lookback_seconds = geometry.lookback_frames / FPS
  window_seconds = geometry.window_frames / FPS
  num_windows = math.ceil(audio_duration * FPS / geometry.keep_frames)

  shift_one = CODEC.encode_event(event_codec.Event('shift', 1))

  predictions = []
  for k in range(num_windows):
    kept_start_time = k * keep_seconds
    if kept_start_time >= audio_duration:
      # Matches _windowed_input_dataset's own trailing-window trim: a window
      # whose kept region starts at or past the end of the real audio
      # exists only because left-padding grows the total frame count.
      continue
    window_start_time = kept_start_time - lookback_seconds
    window_end_time = window_start_time + window_seconds
    # Real audio (and hence real events) only exists in [0, audio_duration);
    # a window's own nominal start/end can extend into the artificial
    # silence padding on either side of that (window 0's lookback prefix,
    # or the last window's lookahead tail).
    content_start_time = max(0.0, window_start_time)

    start_idx = _frame_idx(content_start_time)
    # Sliced a frame past the window's own nominal end, NOT clipped to
    # audio_duration: real audio has no events past audio_duration anyway
    # (nothing to spuriously include), and clipping the slice here would
    # cut off a real event landing exactly at audio_duration before
    # decode_events' own max_decode_time crop (_cap_last_segment_tail) ever
    # gets a chance to make that call -- exactly matching how a real
    # window's audio isn't pre-truncated either. The extra frame matters
    # whenever window_end_time lands exactly on a real event's own time
    # (always true for the last window under a lookahead=0 geometry, since
    # then window_end_time == kept_end == audio_duration exactly):
    # event_start_indices[f] indexes events strictly before frame_times[f],
    # so slicing only up to window_end_time's own frame would exclude an
    # event sitting exactly on that boundary. A real window has the same
    # slack for free, via tf.signal.frame's own pad_end=True framing.
    end_idx = min(len(frame_times) - 1, _frame_idx(window_end_time) + 1)
    body_tokens = list(
        tokens[event_start_indices[start_idx]:event_start_indices[end_idx]])

    # The tie section reflects state as of content_start_time, which is
    # identical to state as of window_start_time whenever they differ (the
    # gap between them is pure silence, so nothing changes across it).
    snapshot_start = state_event_indices[start_idx]
    remaining = list(state_tokens[snapshot_start:])
    tie_section_tokens = remaining[:remaining.index(tie_event_id) + 1]

    # A real window's own token stream (and hence its reported start_time)
    # begins at window_start_time, not content_start_time -- window 0 (or
    # any window whose lookback exceeds its own preceding kept region) can
    # have a negative start_time, exactly like _windowed_input_dataset's
    # real left-padding. Explicit shift-by-1 tokens stand in for the silent
    # lead-in a real model would encode identically, so decode_events' own
    # shift-accumulation lands cur_time on the right absolute times.
    num_padding_steps = round(
        (content_start_time - window_start_time) * CODEC.steps_per_second)
    padding_tokens = [shift_one] * num_padding_steps

    est_tokens = _compress_shifts(
        tie_section_tokens + padding_tokens + body_tokens)
    start_time = transcription._quantize(window_start_time, CODEC)
    predictions.append({
        'est_tokens': est_tokens,
        'start_time': start_time,
        'min_decode_time': transcription._min_decode_time(
            start_time, geometry, CODEC, _FakeSpectrogramConfig()),
        'raw_inputs': [],
    })

  if transcription._should_cap_last_segment_tail(geometry):
    transcription._cap_last_segment_tail(predictions, audio_duration)

  return metrics_utils.event_predictions_to_ns(
      predictions, codec=CODEC, encoding_spec=note_sequences.NoteEncodingWithTiesSpec)


class LookbackEndToEndTest(tf.test.TestCase):

  def test_all_requested_geometries_round_trip_exactly(self):
    ns = _build_reference_ns()
    reference_key = _notes_key(ns)
    for name, (window, lookback, lookahead) in _TEST_GEOMETRIES.items():
      geometry = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      result = _simulate_and_decode(ns, geometry, _AUDIO_DURATION)
      decoded = result['est_ns']
      # Count, not just the deduplicated set: an exact duplicate note would
      # otherwise collapse into the same set entry and hide the bug.
      self.assertLen(decoded.notes, len(ns.notes), msg=name)
      self.assertEqual(_notes_key(decoded), reference_key, msg=name)
      self.assertEqual(result['est_invalid_events'], 0, msg=name)

  def test_geometry_invariance_all_four_produce_the_same_decoded_output(self):
    # Not just each equal to the source (the previous test) -- equal to
    # EACH OTHER, confirming segmentation choice alone changes nothing.
    ns = _build_reference_ns()
    keys = []
    for window, lookback, lookahead in _TEST_GEOMETRIES.values():
      geometry = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      result = _simulate_and_decode(ns, geometry, _AUDIO_DURATION)
      keys.append(_notes_key(result['est_ns']))
    for key in keys[1:]:
      self.assertEqual(key, keys[0])

  def test_note_held_across_a_window_boundary_is_not_truncated_or_duplicated(self):
    # (72, 0, 4.5, 6.5) crosses at least one window boundary under every
    # geometry above EXCEPT the baseline (keep_seconds=4.0 puts it entirely
    # inside the single kept window [4, 8)); explicitly isolate it. The
    # baseline iteration below is still a valid (if less interesting)
    # same-window round-trip check for this note.
    ns = _build_reference_ns()
    for name, (window, lookback, lookahead) in _TEST_GEOMETRIES.items():
      geometry = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      result = _simulate_and_decode(ns, geometry, _AUDIO_DURATION)
      matches = [n for n in result['est_ns'].notes if n.pitch == 72]
      self.assertLen(matches, 1, msg=name)
      self.assertAlmostEqual(matches[0].start_time, 4.5, delta=0.02, msg=name)
      self.assertAlmostEqual(matches[0].end_time, 6.5, delta=0.02, msg=name)

  def test_note_at_the_very_start_and_end_of_the_recording(self):
    # (60, 0, 0.0, 0.6) exercises window 0's lookback-padding edge case;
    # (69, 30, 8.5, 9.0) ends exactly at the recording's end, exercising
    # the last-window tail cap.
    ns = _build_reference_ns()
    for name, (window, lookback, lookahead) in _TEST_GEOMETRIES.items():
      geometry = transcription.WindowGeometry(
          window_frames=window, lookback_frames=lookback,
          lookahead_frames=lookahead)
      result = _simulate_and_decode(ns, geometry, _AUDIO_DURATION)
      first = [n for n in result['est_ns'].notes if n.pitch == 60]
      last = [n for n in result['est_ns'].notes if n.pitch == 69]
      self.assertLen(first, 1, msg=name)
      self.assertLen(last, 1, msg=name)
      self.assertAlmostEqual(first[0].start_time, 0.0, delta=0.02, msg=name)
      self.assertAlmostEqual(last[0].end_time, 9.0, delta=0.02, msg=name)

  def test_baseline_geometry_matches_a_single_unsplit_window(self):
    # The compatibility anchor, checked at the full-pipeline level: 0/0
    # decoded across several windows must equal decoding the whole
    # recording as a single window (a window large enough that
    # num_windows == 1).
    ns = _build_reference_ns()
    split_geometry = transcription.WindowGeometry(window_frames=500)
    split_result = _simulate_and_decode(ns, split_geometry, _AUDIO_DURATION)

    whole_geometry = transcription.WindowGeometry(
        window_frames=math.ceil(_AUDIO_DURATION * FPS) + FPS)
    whole_result = _simulate_and_decode(ns, whole_geometry, _AUDIO_DURATION)

    self.assertEqual(_notes_key(split_result['est_ns']), _notes_key(ns))
    self.assertEqual(_notes_key(whole_result['est_ns']), _notes_key(ns))
    self.assertEqual(
        _notes_key(split_result['est_ns']), _notes_key(whole_result['est_ns']))


if __name__ == '__main__':
  tf.test.main()
