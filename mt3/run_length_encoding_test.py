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

"""Tests for run_length_encoding."""

from mt3 import event_codec
from mt3 import run_length_encoding

import note_seq
import numpy as np
import seqio
import tensorflow as tf

assert_dataset = seqio.test_utils.assert_dataset
codec = event_codec.Codec(
    max_shift_steps=100,
    steps_per_second=100,
    event_ranges=[
        event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                               note_seq.MAX_MIDI_PITCH),
        event_codec.EventRange('velocity', 0, 127),
        event_codec.EventRange('drum', note_seq.MIN_MIDI_PITCH,
                               note_seq.MAX_MIDI_PITCH),
        event_codec.EventRange('program', note_seq.MIN_MIDI_PROGRAM,
                               note_seq.MAX_MIDI_PROGRAM),
        event_codec.EventRange('tie', 0, 0)
    ])
run_length_encode_shifts = run_length_encoding.run_length_encode_shifts_fn(
    codec=codec)


class RunLengthEncodingTest(tf.test.TestCase):

  def test_remove_redundant_state_changes(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [3, 525, 356, 161, 2, 525, 356, 161, 355, 394]
    })

    assert_dataset(
        run_length_encoding.remove_redundant_state_changes_fn(
            codec=codec,
            state_change_event_types=['velocity', 'program'])(og_dataset),
        {
            'targets': [3, 525, 356, 161, 2, 161, 355, 394],
        })

  def test_run_length_encode_shifts(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1, 1, 1, 161, 1, 1, 1, 162, 1, 1, 1]
    })

    assert_dataset(
        run_length_encode_shifts(og_dataset),
        {
            'targets': [3, 161, 6, 162],
        })

  def test_run_length_encode_shifts_beyond_max_length(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1] * 202 + [161, 1, 1, 1]
    })

    assert_dataset(
        run_length_encode_shifts(og_dataset),
        {
            'targets': [100, 100, 2, 161],
        })

  def test_run_length_encode_shifts_simultaneous(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1, 1, 1, 161, 162, 1, 1, 1]
    })

    assert_dataset(
        run_length_encode_shifts(og_dataset),
        {
            'targets': [3, 161, 162],
        })

  def test_merge_run_length_encoded_targets(self):
    # pylint: disable=bad-whitespace
    targets = np.array([
        [  3, 161, 162,   5, 163],
        [160, 164,   3, 165,   0]
    ])
    # pylint: enable=bad-whitespace
    merged_targets = run_length_encoding.merge_run_length_encoded_targets(
        targets=targets, codec=codec)
    expected_merged_targets = [
        160, 164, 3, 161, 162, 165, 5, 163
    ]
    np.testing.assert_array_equal(expected_merged_targets, merged_targets)


class _FakeDecodingState:
  """A minimal decoding state for exercising decode_events() without note_sequences.py.

  decode_events' min_time/suppress contract is generic (it signals via a
  plain `state.suppress` attribute, not anything note-specific), so it can
  and should be tested without NoteDecodingState.
  """

  def __init__(self):
    self.suppress = False
    # (cur_time, event.value, suppress-flag-at-call-time) for every non-shift
    # event that reached decode_event_fn.
    self.log = []


def _fake_decode_event_fn(state, time, event, codec):
  del codec  # unused
  state.log.append((time, event.value, state.suppress))


def _fake_decode_event_fn_raising_for_value(bad_value):
  """A decode_event_fn that raises ValueError for one specific event value."""
  def fn(state, time, event, codec):
    del codec  # unused
    if event.value == bad_value:
      raise ValueError(f'invalid event value {event.value}')
    state.log.append((time, event.value, state.suppress))
  return fn


_rle_codec = event_codec.Codec(
    max_shift_steps=100,
    steps_per_second=100,
    event_ranges=[event_codec.EventRange('note', 0, 127)])


def _shift(steps):
  return _rle_codec.encode_event(event_codec.Event('shift', steps))


def _note(value):
  return _rle_codec.encode_event(event_codec.Event('note', value))


class DecodeEventsTest(tf.test.TestCase):

  def test_min_time_none_preserves_legacy_half_open_at_max_time(self):
    # Legacy (min_time=None) semantics: `cur_time > max_time` drops, so an
    # event exactly at max_time is KEPT -- this must not change.
    state = _FakeDecodingState()
    tokens = [_shift(100), _note(1)]  # cur_time reaches exactly 1.0s
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=tokens, start_time=0, max_time=1.0,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn)
    self.assertEqual(state.log, [(1.0, 1, False)])
    self.assertEqual((invalid, dropped, suppressed), (0, 0, 0))

  def test_max_time_zero_drops_rather_than_being_ignored(self):
    # Regression test for the pre-lookback `if max_time and ...` truthiness
    # bug: max_time=0.0 is falsy but must still be honored as a real bound.
    state = _FakeDecodingState()
    tokens = [_shift(5), _note(1)]  # cur_time reaches 0.05s > 0.0
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=tokens, start_time=0, max_time=0.0,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn)
    self.assertEqual(state.log, [])
    self.assertEqual(invalid, 0)
    self.assertGreater(dropped, 0)
    self.assertEqual(suppressed, 0)

  def test_min_time_suppresses_prefix_and_counts_it(self):
    state = _FakeDecodingState()
    # cur_steps resets to 0 after every non-shift event, so reaching 1.3s
    # for the second note (after the reset following the first note) takes
    # two shift tokens: max_shift_steps=100 caps a single one at 1.0s.
    tokens = [
        _shift(50), _note(1),               # cur_time=0.5 < 1.0: suppressed
        _shift(100), _shift(30), _note(2),  # cur_time=1.3 >= 1.0: kept
    ]
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=tokens, start_time=0, max_time=None,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn,
        min_time=1.0)
    self.assertEqual(state.log, [(0.5, 1, True), (1.3, 2, False)])
    self.assertEqual((invalid, dropped, suppressed), (0, 0, 1))

  def test_suppressed_events_is_zero_when_min_time_is_none(self):
    state = _FakeDecodingState()
    tokens = [_shift(50), _note(1), _shift(80), _note(2)]
    _, _, suppressed = run_length_encoding.decode_events(
        state=state, tokens=tokens, start_time=0, max_time=None,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn)
    self.assertEqual(suppressed, 0)
    # state.suppress is untouched (never set) when min_time is None.
    self.assertEqual([s for _, _, s in state.log], [False, False])

  def test_invalid_event_before_min_time_counts_only_as_invalid(self):
    # An event that is both before min_time and invalid must not be
    # double-counted in both invalid_events and suppressed_events.
    state = _FakeDecodingState()
    tokens = [_shift(50), _note(99)]  # cur_time=0.5 < min_time=1.0
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=tokens, start_time=0, max_time=None,
        codec=_rle_codec,
        decode_event_fn=_fake_decode_event_fn_raising_for_value(99),
        min_time=1.0)
    self.assertEqual(state.log, [])
    self.assertEqual((invalid, dropped, suppressed), (1, 0, 0))

  def test_min_time_and_max_time_form_half_open_kept_interval(self):
    # [min_time, max_time): an event exactly at min_time is kept, an event
    # exactly at max_time is dropped -- the opposite of the min_time=None
    # legacy boundary, which is exactly why this only applies when min_time
    # is not None.
    state = _FakeDecodingState()
    # cur_steps resets after the first note, so reaching 2.0s for the
    # second note takes two more shift tokens (max_shift_steps=100 caps a
    # single one at 1.0s).
    tokens = [
        _shift(100), _note(1),               # cur_time=1.0 == min_time: kept
        _shift(100), _shift(100), _note(2),  # cur_time=2.0 == max_time: dropped
    ]
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=tokens, start_time=0, max_time=2.0,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn,
        min_time=1.0)
    self.assertEqual(state.log, [(1.0, 1, False)])
    self.assertEqual(invalid, 0)
    self.assertGreater(dropped, 0)
    self.assertEqual(suppressed, 0)

  def test_suppress_does_not_leak_stale_into_a_later_min_time_none_call(self):
    # A state reused across a min_time=X call (whose last event ends
    # suppressed) and then a min_time=None call must not carry state.suppress
    # over as a stale True -- every event in the second call must decode
    # normally.
    state = _FakeDecodingState()
    run_length_encoding.decode_events(
        state=state, tokens=[_note(1)], start_time=0, max_time=None,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn,
        min_time=1.0)  # cur_time=0.0 < min_time=1.0: ends suppressed=True.
    self.assertTrue(state.suppress)

    state.log.clear()
    run_length_encoding.decode_events(
        state=state, tokens=[_note(2)], start_time=5.0, max_time=None,
        codec=_rle_codec, decode_event_fn=_fake_decode_event_fn)
    self.assertEqual(state.log, [(5.0, 2, False)])


if __name__ == '__main__':
  tf.test.main()
