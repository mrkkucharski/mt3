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

"""Tests for the optional rhythm vocabulary (VocabularyConfig.include_rhythm).

The counterpart of note_sequences_rhythm_test.py: there, `rhythm` survives the
NoteSequence -> tokens -> NoteSequence round trip; here, a codec built without
it makes a rhythm-labelled corpus encode exactly like an unlabelled one, so
training on `data/pilot` never sees the distinction and no TFRecord has to be
rebuilt to opt out of it.
"""

from mt3 import event_codec
from mt3 import metrics
from mt3 import note_sequences
from mt3 import run_length_encoding
from mt3 import vocabularies

import note_seq
import numpy as np
import tensorflow as tf

RHYTHM_CONFIG = vocabularies.VocabularyConfig(num_velocity_bins=1)
NO_RHYTHM_CONFIG = vocabularies.VocabularyConfig(num_velocity_bins=1,
                                                 include_rhythm=False)
RHYTHM_CODEC = vocabularies.build_codec(RHYTHM_CONFIG)
NO_RHYTHM_CODEC = vocabularies.build_codec(NO_RHYTHM_CONFIG)

# Notes shared by several tests: two guitar notes on one program, one of them
# chordal accompaniment, plus a piano note that never carries the flag.
NOTES = [(60, 30, 0.0, 0.5), (64, 30, 0.5, 1.0), (67, 0, 1.0, 1.5)]
RHYTHMS = [True, False, False]


def _build_ns(notes, rhythms):
  """notes: list of (pitch, program, start, end). rhythms: parallel bools."""
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  for pitch, program, start, end in notes:
    ns.notes.add(start_time=start, end_time=end, pitch=pitch, velocity=100,
                 program=program, is_drum=False)
    ns.total_time = max(ns.total_time, end)
  note_sequences.assign_instruments(ns, note_rhythms=list(rhythms))
  return ns


def _encode(ns, codec):
  """Encodes `ns` to a raw (uncompressed-shift) token array.

  Reads the rhythm labels exactly as the training pipeline does: only when the
  codec has somewhere to put them (see preprocessors.py).
  """
  frame_times = np.arange(0, ns.total_time + 0.02, step=0.001)
  times, values = (
      note_sequences.note_sequence_to_onsets_and_offsets_and_programs(
          ns, include_rhythm=codec.has_event_type('rhythm')))
  tokens, _, _, _, _ = run_length_encoding.encode_and_index_events(
      state=None, event_times=times, event_values=values,
      encode_event_fn=note_sequences.note_event_data_to_events,
      codec=codec, frame_times=frame_times)
  return np.asarray(tokens)


def _decode(tokens, codec):
  """Decodes a token array back into a NoteSequence."""
  ds = tf.data.Dataset.from_tensors({'targets': tokens})
  ds = run_length_encoding.run_length_encode_shifts_fn(
      codec, feature_key='targets')(ds)
  tokens = next(iter(ds))['targets'].numpy()

  state = note_sequences.NoteDecodingState()
  invalid_ids, dropped_events, _ = run_length_encoding.decode_events(
      state=state, tokens=tokens, start_time=0, max_time=None, codec=codec,
      decode_event_fn=note_sequences.decode_note_event)
  assert invalid_ids == 0, f'{invalid_ids} invalid ids'
  assert dropped_events == 0, f'{dropped_events} dropped events'
  return note_sequences.flush_note_decoding_state(state)


class CodecCompatibilityTest(tf.test.TestCase):
  """The two codecs must stay checkpoint-compatible with each other."""

  def test_rhythm_range_is_present_only_when_configured(self):
    self.assertTrue(RHYTHM_CODEC.has_event_type('rhythm'))
    self.assertFalse(NO_RHYTHM_CODEC.has_event_type('rhythm'))
    with self.assertRaises(ValueError):
      NO_RHYTHM_CODEC.event_type_range('rhythm')

  def test_every_other_event_keeps_its_token_ids(self):
    # This is what lets a rhythm-free run restore from checkpoint_0, or from a
    # rhythm-trained checkpoint of this project: `rhythm` is the last event
    # range, so removing it shifts nothing before it.
    for event_type in ('shift', 'pitch', 'velocity', 'tie', 'program', 'drum'):
      self.assertEqual(RHYTHM_CODEC.event_type_range(event_type),
                       NO_RHYTHM_CODEC.event_type_range(event_type),
                       msg=f'{event_type} ids moved')
    self.assertEqual(RHYTHM_CODEC.num_classes - 2, NO_RHYTHM_CODEC.num_classes)

  def test_padded_embedding_size_is_unchanged(self):
    # The decoder's output layer is sized by num_embeddings, which rounds up to
    # a multiple of 128 -- the two-class difference disappears in the padding.
    self.assertEqual(
        vocabularies.num_embeddings(
            vocabularies.vocabulary_from_codec(RHYTHM_CODEC)),
        vocabularies.num_embeddings(
            vocabularies.vocabulary_from_codec(NO_RHYTHM_CODEC)))

  def test_abbrev_str_distinguishes_the_configs(self):
    # seqio task names are built from this, so the two modes cannot share a
    # registered task by accident (see tasks.construct_task_name).
    self.assertEqual('vb1', RHYTHM_CONFIG.abbrev_str)
    self.assertEqual('vb1nr', NO_RHYTHM_CONFIG.abbrev_str)

  def test_flat_granularity_drops_programs_without_a_rhythm_range(self):
    tokens = _encode(_build_ns(NOTES, RHYTHMS), NO_RHYTHM_CODEC)
    kept = vocabularies.drop_programs(tokens, NO_RHYTHM_CODEC)
    min_program_id, max_program_id = NO_RHYTHM_CODEC.event_type_range('program')
    self.assertNotEmpty(kept)
    self.assertFalse(np.any((kept >= min_program_id) &
                            (kept <= max_program_id)))


class NoRhythmEncodingTest(tf.test.TestCase):

  def test_rhythm_labels_encode_exactly_like_unlabelled_notes(self):
    labelled = _encode(_build_ns(NOTES, RHYTHMS), NO_RHYTHM_CODEC)
    unlabelled = _encode(_build_ns(NOTES, [False] * len(NOTES)),
                         NO_RHYTHM_CODEC)
    self.assertAllEqual(labelled, unlabelled)

    # Simultaneous notes of one program are the case where the labels could
    # still leak through: they are part of the sort key that orders events at
    # the same timestamp, so reading them at all would reorder this pair's
    # pitches relative to an unlabelled corpus even with no rhythm token to
    # show for it.
    simultaneous = [(64, 30, 0.0, 0.5), (60, 30, 0.0, 0.5)]
    self.assertAllEqual(
        _encode(_build_ns(simultaneous, [False, True]), NO_RHYTHM_CODEC),
        _encode(_build_ns(simultaneous, [False, False]), NO_RHYTHM_CODEC))

    # ...and the fixture really does carry rhythm: the rhythm-aware codec
    # tells the two sequences apart.
    self.assertNotAllEqual(
        _encode(_build_ns(NOTES, RHYTHMS), RHYTHM_CODEC),
        _encode(_build_ns(NOTES, [False] * len(NOTES)), RHYTHM_CODEC))

  def test_no_token_decodes_to_a_rhythm_event(self):
    tokens = _encode(_build_ns(NOTES, RHYTHMS), NO_RHYTHM_CODEC)
    types = {NO_RHYTHM_CODEC.decode_event_index(int(t)).type for t in tokens}
    self.assertNotIn('rhythm', types)
    self.assertContainsSubset({'shift', 'program', 'velocity', 'pitch'}, types)

  def test_round_trip_preserves_notes_and_drops_the_rhythm_split(self):
    ns = _build_ns(NOTES, RHYTHMS)
    decoded = _decode(_encode(ns, NO_RHYTHM_CODEC), NO_RHYTHM_CODEC)

    step = 1.0 / NO_RHYTHM_CODEC.steps_per_second
    def _key(sequence):
      return {(note.pitch, note.program,
               round(note.start_time / step), round(note.end_time / step))
              for note in sequence.notes}
    self.assertEqual(_key(ns), _key(decoded))

    # One instrument per program, none of them flagged.
    rhythms = note_sequences.instrument_rhythms(decoded)
    self.assertFalse(any(rhythms.values()))
    self.assertLen(decoded.instrument_infos, 2)

  def test_tie_section_omits_rhythm_events(self):
    # The tie section is encoded from the encoding state rather than from note
    # data, so it needs its own guarantee (a stray Event('rhythm', 0) here
    # would raise "Unknown event type" from codec.encode_event).
    state = note_sequences.NoteEncodingState()
    value = note_sequences.NoteEventData(
        pitch=60, velocity=100, program=30, is_drum=False, rhythm=True)
    note_sequences.note_event_data_to_events(state, value, NO_RHYTHM_CODEC)

    events = note_sequences.note_encoding_state_to_events(
        state, NO_RHYTHM_CODEC)
    self.assertEqual(['program', 'pitch', 'tie'], [e.type for e in events])
    for event in events:
      NO_RHYTHM_CODEC.encode_event(event)

    # The rhythm-aware codec still emits the flag, unchanged.
    rhythm_state = note_sequences.NoteEncodingState()
    note_sequences.note_event_data_to_events(rhythm_state, value, RHYTHM_CODEC)
    self.assertEqual(
        ['program', 'rhythm', 'pitch', 'tie'],
        [e.type for e in note_sequences.note_encoding_state_to_events(
            rhythm_state, RHYTHM_CODEC)])
    self.assertIn(event_codec.Event('rhythm', 1),
                  note_sequences.note_encoding_state_to_events(
                      rhythm_state, RHYTHM_CODEC))


class NoRhythmMetricsTest(tf.test.TestCase):
  """Scoring must collapse the split on the REFERENCE too.

  The reference NoteSequence comes straight from the corpus, which keeps its
  `:rhythm` track names however the model was trained; scoring a rhythm-free
  estimate against it program-and-rhythm-aware would count every accompaniment
  note as both a miss and a false positive.
  """

  F1 = 'Onset + offset + program F1 (full)'

  def test_rhythm_free_estimate_matches_a_rhythm_labelled_reference(self):
    ref_ns = _build_ns(NOTES, RHYTHMS)
    est_ns = _build_ns(NOTES, [False] * len(NOTES))

    collapsed = metrics._program_aware_note_scores(
        ref_ns, est_ns, granularity_type='full', include_rhythm=False)
    self.assertAlmostEqual(1.0, collapsed[self.F1])

    # Without the collapse the same pair scores as a partial miss, which is
    # exactly the failure this parameter exists to prevent.
    self.assertLess(
        metrics._program_aware_note_scores(
            ref_ns, est_ns, granularity_type='full')[self.F1],
        1.0)


if __name__ == '__main__':
  tf.test.main()
