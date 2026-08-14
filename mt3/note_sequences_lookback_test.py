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

"""Tests for decode_note_event's handling of a lookback (min_time) head crop.

Simulates two overlapping encoder windows sharing one persistent
NoteDecodingState, the same way metrics_utils.decode_and_combine_predictions
processes real prediction windows: window A is decoded first (establishing
ground truth in `active_pitches`/`note_sequence`), then window B is decoded
against the same state with `min_time` set to window B's kept-region start.
Window B's own tie section and any events before `min_time` are exactly what
a real overlapping window would produce -- the model re-observing (and
possibly mis-perceiving, since it has no way to know the true onset of a
note that started before its own audio) whatever is already active from its
own point of view.
"""

from mt3 import event_codec
from mt3 import note_sequences
from mt3 import run_length_encoding
from mt3 import vocabularies

import tensorflow as tf

CODEC = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=1))


def _shift(steps):
  return CODEC.encode_event(event_codec.Event('shift', steps))


def _pitch(value):
  return CODEC.encode_event(event_codec.Event('pitch', value))


def _velocity(bin_value):
  return CODEC.encode_event(event_codec.Event('velocity', bin_value))


def _program(value):
  return CODEC.encode_event(event_codec.Event('program', value))


def _tie():
  return CODEC.encode_event(event_codec.Event('tie', 0))


class LookbackTieSectionTest(tf.test.TestCase):

  def test_held_note_across_boundary_keeps_original_onset_and_is_not_duplicated(self):
    state = note_sequences.NoteDecodingState()

    # Window A: real audio [0.0, ...); kept region ends at 1.0 (this
    # window's max_time == the next window's min_time, matching
    # decode_and_combine_predictions). Note 60 onsets here and is still
    # open when window A's kept region ends.
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_pitch(60)], start_time=0.0, max_time=1.0,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))
    self.assertIn((60, 0, False), state.active_pitches)
    self.assertEqual(state.active_pitches[(60, 0, False)][0], 0.0)

    # Window B: real audio starts at 0.5 (its lookback covers [0.5, 1.0)),
    # kept region starts at min_time=1.0. Its own tie section re-declares
    # pitch 60 as already active -- exactly what the model would emit,
    # since from window B's perspective the note is sounding at its first
    # frame too -- entirely inside the suppressed prefix. The real note-off
    # arrives after min_time, inside the kept region.
    note_sequences.begin_tied_pitches_section(state)
    window_b_tokens = [
        _pitch(60), _tie(),                    # suppressed tie section
        _shift(130), _velocity(0), _pitch(60),  # kept: note-off at t=1.8
    ]
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=window_b_tokens, start_time=0.5, max_time=None,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))
    self.assertGreater(suppressed, 0)

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    note = ns.notes[0]
    self.assertEqual(note.pitch, 60)
    # Original onset from window A is retained, not window B's suppressed
    # re-declaration and not the boundary time.
    self.assertEqual(note.start_time, 0.0)
    # Not truncated at the window boundary (1.0): the true offset, decoded
    # by window B after min_time, wins.
    self.assertEqual(note.end_time, 1.8)

  def test_missing_tie_closes_carried_note_at_kept_boundary(self):
    state = note_sequences.NoteDecodingState()

    # Window A leaves the note active at its output boundary.
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_pitch(60)], start_time=0.0, max_time=1.0,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))

    # Window B does not consider the note active at its own start.  There is
    # no kept non-shift event to trigger reconciliation, so this also verifies
    # decode_events' end-of-stream reconciliation path.
    note_sequences.begin_tied_pitches_section(state)
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_tie(), _shift(130)], start_time=0.5,
        max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    self.assertEqual(ns.notes[0].start_time, 0.0)
    self.assertEqual(ns.notes[0].end_time, 1.0)

  def test_program_disagreement_does_not_leave_old_key_open(self):
    state = note_sequences.NoteDecodingState()

    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_program(30), _pitch(63)], start_time=0.0,
        max_time=1.0, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))

    # The next window hears the same pitch as program 31 rather than 30.
    # Its prefix-only key must not synthesize a new output note, while the
    # stale committed program-30 key must be closed at the kept boundary.
    note_sequences.begin_tied_pitches_section(state)
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state,
        tokens=[_program(31), _pitch(63), _tie(), _shift(130)],
        start_time=0.5, max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    self.assertEqual(ns.notes[0].program, 30)
    self.assertEqual(ns.notes[0].end_time, 1.0)

  def test_note_off_seen_only_in_lookback_closes_carried_note(self):
    state = note_sequences.NoteDecodingState()

    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_pitch(64)], start_time=0.0, max_time=1.0,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))

    note_sequences.begin_tied_pitches_section(state)
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state,
        tokens=[_pitch(64), _tie(), _shift(30), _velocity(0), _pitch(64)],
        start_time=0.5, max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    # The previous kept region remains authoritative for timing, so evidence
    # from the overlap closes the note at the boundary, not retroactively at
    # the shadow decoder's 0.8-second offset.
    self.assertEqual(ns.notes[0].end_time, 1.0)

  def test_reonset_in_lookback_does_not_continue_stale_original_onset(self):
    state = note_sequences.NoteDecodingState()

    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_pitch(65)], start_time=0.0, max_time=1.0,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))

    note_sequences.begin_tied_pitches_section(state)
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state,
        tokens=[
            _pitch(65), _tie(),
            _shift(20), _velocity(0), _pitch(65),
            _shift(30), _velocity(1), _pitch(65),
        ],
        start_time=0.5, max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    self.assertEqual(ns.notes[0].end_time, 1.0)

  def test_note_entirely_inside_lookback_region_is_not_duplicated(self):
    state = note_sequences.NoteDecodingState()

    # Window A: a short, fully-closed note entirely within its kept region.
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state,
        tokens=[_pitch(61), _shift(20), _velocity(0), _pitch(61)],
        start_time=0.0, max_time=1.0, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))
    self.assertLen(state.note_sequence.notes, 1)

    # Window B re-observes (and re-declares) the same already-closed note
    # entirely inside its own suppressed prefix -- a hallucinated re-onset
    # and re-offset, both before min_time.
    note_sequences.begin_tied_pitches_section(state)
    window_b_tokens = [
        _pitch(61), _shift(10), _velocity(0), _pitch(61),  # both suppressed
    ]
    invalid, dropped, suppressed = run_length_encoding.decode_events(
        state=state, tokens=window_b_tokens, start_time=0.5, max_time=None,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))
    self.assertGreater(suppressed, 0)

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)  # not duplicated by window B.

  def test_tie_section_for_an_already_closed_pitch_does_not_raise(self):
    state = note_sequences.NoteDecodingState()

    # Window A: pitch 62 onsets and closes entirely within window A.
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state,
        tokens=[_pitch(62), _shift(20), _velocity(0), _pitch(62)],
        start_time=0.0, max_time=1.0, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))
    self.assertNotIn((62, 0, False), state.active_pitches)

    # Window B's tie section (mis-)declares pitch 62 as still tied -- were
    # this not suppressed, decode_note_event would raise "inactive
    # pitch/program/rhythm in tie section" (it is no longer in
    # active_pitches). Entirely inside the suppressed prefix, it must not
    # raise and must not be counted invalid.
    note_sequences.begin_tied_pitches_section(state)
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_pitch(62), _tie()], start_time=0.5,
        max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))

  def test_sticky_program_state_carries_through_suppressed_prefix(self):
    state = note_sequences.NoteDecodingState()
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_program(30), _pitch(63)], start_time=0.0,
        max_time=1.0, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event, min_time=0.0)
    self.assertEqual((invalid, dropped), (0, 0))

    # Window B's suppressed prefix re-declares the program change (as the
    # model would, to establish modal state for the kept region that
    # follows) before any kept event arrives.
    note_sequences.begin_tied_pitches_section(state)
    window_b_tokens = [
        _program(30),                            # suppressed, but applied
        _shift(130), _velocity(0), _pitch(63),    # kept: note-off, program 30
    ]
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=window_b_tokens, start_time=0.5, max_time=None,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event,
        min_time=1.0)
    self.assertEqual((invalid, dropped), (0, 0))

    ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    self.assertEqual(ns.notes[0].program, 30)

  def test_flush_warns_about_an_implausibly_long_held_note(self):
    # Even with boundary reconciliation, a model can consistently claim that
    # a note remains active through every later window.  If none ever closes
    # it, flush force-closes it at `current_time`; keep that visible.
    state = note_sequences.NoteDecodingState()
    invalid, dropped, _ = run_length_encoding.decode_events(
        state=state, tokens=[_pitch(64)], start_time=0.0, max_time=None,
        codec=CODEC, decode_event_fn=note_sequences.decode_note_event)
    self.assertEqual((invalid, dropped), (0, 0))
    state.current_time = note_sequences.LONG_HELD_NOTE_WARNING_SECONDS + 1.0

    with self.assertLogs(level='WARNING') as logs:
      ns = note_sequences.flush_note_decoding_state(state)
    self.assertLen(ns.notes, 1)
    self.assertTrue(any('dropped note-off' in m for m in logs.output))


if __name__ == '__main__':
  tf.test.main()
