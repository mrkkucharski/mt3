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

"""Round-trip tests proving NoteSequence -> tokens -> NoteSequence preserves
both `program` and `rhythm` (MT3_guitar_transcription_plan.md, Phase 0,
"Round-trip tests").
"""

from mt3 import event_codec
from mt3 import note_sequences
from mt3 import run_length_encoding
from mt3 import vocabularies

import note_seq
import numpy as np
import tensorflow as tf

CODEC = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=1))

# Every General MIDI guitar program (DATA_CONTRACT.md: rhythm is guitar-only).
GUITAR_PROGRAMS = range(24, 32)


def _notes_key(ns: note_seq.NoteSequence):
  """(pitch, program, rhythm, start, end) set, quantized to codec steps."""
  step = 1.0 / CODEC.steps_per_second
  rhythm_by_instrument = note_sequences.instrument_rhythms(ns)
  return {
      (note.pitch, note.program,
       rhythm_by_instrument.get(note.instrument, False),
       round(note.start_time / step) * step,
       round(note.end_time / step) * step)
      for note in ns.notes
  }


def _compress_shifts(tokens):
  """Applies the same shift run-length compression the real target pipeline
  uses (run_length_encode_shifts_fn) -- decode_events expects cumulative,
  not per-step, shift tokens.
  """
  ds = tf.data.Dataset.from_tensors({'targets': tokens})
  ds = run_length_encoding.run_length_encode_shifts_fn(
      CODEC, feature_key='targets')(ds)
  return next(iter(ds))['targets'].numpy()


def _round_trip(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
  """Encodes `ns` to tokens (no ties) and decodes them back to a NoteSequence."""
  frame_times = np.arange(0, ns.total_time + 0.02, step=0.001)
  times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
  tokens, _, _, _, _ = run_length_encoding.encode_and_index_events(
      state=None, event_times=times, event_values=values,
      encode_event_fn=note_sequences.note_event_data_to_events,
      codec=CODEC, frame_times=frame_times)
  tokens = _compress_shifts(tokens)

  decoding_state = note_sequences.NoteDecodingState()
  invalid_ids, dropped_events = run_length_encoding.decode_events(
      state=decoding_state, tokens=tokens, start_time=0, max_time=None,
      codec=CODEC, decode_event_fn=note_sequences.decode_note_event)
  assert invalid_ids == 0, f'{invalid_ids} invalid ids decoding {tokens}'
  assert dropped_events == 0, f'{dropped_events} dropped events decoding {tokens}'
  return note_sequences.flush_note_decoding_state(decoding_state)


def _build_ns(notes, rhythms):
  """notes: list of (pitch, program, start, end). rhythms: parallel bools."""
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  for pitch, program, start, end in notes:
    ns.notes.add(start_time=start, end_time=end, pitch=pitch, velocity=100,
                program=program, is_drum=False)
    ns.total_time = max(ns.total_time, end)
  note_sequences.assign_instruments(ns, note_rhythms=list(rhythms))
  return ns


class RhythmRoundTripTest(tf.test.TestCase):

  def test_every_guitar_program_in_both_roles(self):
    notes = []
    rhythms = []
    t = 0.0
    for program in GUITAR_PROGRAMS:
      for rhythm in (False, True):
        notes.append((60, program, t, t + 0.5))
        rhythms.append(rhythm)
        t += 0.5
    ns = _build_ns(notes, rhythms)

    decoded = _round_trip(ns)
    self.assertEqual(_notes_key(ns), _notes_key(decoded))
    # Sanity: this actually covers every guitar program in both roles.
    self.assertLen(_notes_key(ns), len(GUITAR_PROGRAMS) * 2)

  def test_simultaneous_notes_same_program_different_rhythm(self):
    ns = _build_ns(
        notes=[(60, 30, 0.0, 1.0), (64, 30, 0.0, 1.0)],
        rhythms=[False, True])
    decoded = _round_trip(ns)
    self.assertEqual(_notes_key(ns), _notes_key(decoded))
    self.assertLen(_notes_key(decoded), 2)

  def test_simultaneous_notes_same_rhythm_different_program(self):
    ns = _build_ns(
        notes=[(60, 24, 0.0, 1.0), (60, 29, 0.0, 1.0)],
        rhythms=[True, True])
    decoded = _round_trip(ns)
    self.assertEqual(_notes_key(ns), _notes_key(decoded))
    self.assertLen(_notes_key(decoded), 2)

  def test_overlapping_same_pitch_notes_differing_only_in_rhythm(self):
    # The fused (program-only) scheme could not express this at all: same
    # pitch and program, active at the same time, distinguished only by role.
    ns = _build_ns(
        notes=[(60, 30, 0.0, 0.5), (60, 30, 0.1, 0.4)],
        rhythms=[False, True])
    decoded = _round_trip(ns)
    self.assertEqual(_notes_key(ns), _notes_key(decoded))
    self.assertLen(_notes_key(decoded), 2)

  def test_absence_of_rhythm_annotation_is_preserved(self):
    ns = _build_ns(
        notes=[(60, 0, 0.0, 0.5), (64, 71, 0.5, 1.0)],
        rhythms=[False, False])
    decoded = _round_trip(ns)
    self.assertEqual(_notes_key(ns), _notes_key(decoded))
    self.assertTrue(all(not rhythm for _, _, rhythm, _, _ in _notes_key(decoded)))

    # No token in the stream ever encodes rhythm value 1.
    times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
    frame_times = np.arange(0, ns.total_time + 0.02, step=0.001)
    tokens, _, _, _, _ = run_length_encoding.encode_and_index_events(
        state=None, event_times=times, event_values=values,
        encode_event_fn=note_sequences.note_event_data_to_events,
        codec=CODEC, frame_times=frame_times)
    rhythm_lo, rhythm_hi = CODEC.event_type_range('rhythm')
    rhythm_values = [t - rhythm_lo for t in tokens if rhythm_lo <= t <= rhythm_hi]
    self.assertTrue(all(v == 0 for v in rhythm_values))

  def test_tied_note_across_segment_boundary_retains_rhythm(self):
    """A rhythm-flagged note spanning a chunk boundary must decode as one note
    with rhythm still set, using the tie-section state exactly as the real
    training pipeline resumes a chunk (see NoteEncodingWithTiesSpec).
    """
    ns = _build_ns(notes=[(60, 30, 0.0, 2.0)], rhythms=[True])

    # Encode the whole recording with the tie-aware spec so we get both the
    # regular token stream and, per frame, the "state as of this frame"
    # stream that a real chunk boundary would prepend.
    frame_times = np.arange(0, 2.02, step=0.001)
    times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
    (tokens, event_start_indices, _, state_tokens, state_event_indices) = (
        run_length_encoding.encode_and_index_events(
            state=note_sequences.NoteEncodingState(),
            event_times=times, event_values=values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=CODEC, frame_times=frame_times,
            encoding_state_to_events_fn=note_sequences.note_encoding_state_to_events))

    # Split at t=1.0: "segment 1" is everything before the boundary, decoded
    # normally (establishing the note as active). "Segment 2" is decoded
    # against the *same* persistent state, exactly like
    # metrics_utils.decode_and_combine_predictions does across real
    # prediction windows: begin_segment_fn (here, begin_tied_pitches_section)
    # runs once per segment, and the tie section only needs to *confirm*
    # notes the state already knows are active -- it does not (re)populate
    # active_pitches itself, since decoding a segment in isolation has no
    # other way to know what's still sounding.
    boundary_frame = int(np.searchsorted(frame_times, 1.0))
    # Raw shift tokens here are relative to their segment (chunk-local), same
    # as a real training chunk; compress them the same way the pipeline does
    # before decode_events, which expects cumulative-within-segment shifts.
    segment1_tokens = _compress_shifts(tokens[:event_start_indices[boundary_frame]])
    segment2_tokens = _compress_shifts(tokens[event_start_indices[boundary_frame]:])
    # state_event_indices[frame] marks where the *currently relevant*
    # snapshot begins in the concatenated state_tokens array (each snapshot
    # itself ends with its own closing 'tie' token), not where it ends -- so
    # find that snapshot's own closing tie searching forward from there.
    snapshot_start = state_event_indices[boundary_frame]
    tie_event_id = CODEC.encode_event(event_codec.Event('tie', 0))
    remaining = list(state_tokens[snapshot_start:])
    tie_section_tokens = remaining[:remaining.index(tie_event_id) + 1]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=segment1_tokens, start_time=0.0,
        max_time=1.0, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event)
    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)

    note_sequences.begin_tied_pitches_section(decoding_state)
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=tie_section_tokens, start_time=1.0,
        max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event)
    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=segment2_tokens, start_time=1.0,
        max_time=None, codec=CODEC,
        decode_event_fn=note_sequences.decode_note_event)
    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    decoded = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertLen(decoded.notes, 1)
    rhythm_by_instrument = note_sequences.instrument_rhythms(decoded)
    note = decoded.notes[0]
    self.assertEqual(note.pitch, 60)
    self.assertEqual(note.program, 30)
    self.assertTrue(rhythm_by_instrument.get(note.instrument, False))
    self.assertAlmostEqual(note.start_time, 0.0, places=2)
    self.assertAlmostEqual(note.end_time, 2.0, places=2)


if __name__ == '__main__':
  tf.test.main()
