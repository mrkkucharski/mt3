# Copyright 2026 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for the optional fixed-range pitch-bend vocabulary."""

from mt3 import event_codec
from mt3 import note_sequences
from mt3 import vocabularies

import note_seq
import tensorflow as tf


class PitchBendCodecTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.legacy_rhythm = vocabularies.build_codec(
        vocabularies.VocabularyConfig(num_velocity_bins=1))
    self.legacy_no_rhythm = vocabularies.build_codec(
        vocabularies.VocabularyConfig(
            num_velocity_bins=1, include_rhythm=False))
    self.bend_rhythm = vocabularies.build_codec(
        vocabularies.VocabularyConfig(
            num_velocity_bins=1, include_pitch_bends=True))
    self.bend_no_rhythm = vocabularies.build_codec(
        vocabularies.VocabularyConfig(
            num_velocity_bins=1, include_rhythm=False,
            include_pitch_bends=True))

  def test_legacy_ids_do_not_move(self):
    for event_type in ('shift', 'pitch', 'velocity', 'tie', 'program', 'drum'):
      expected = self.legacy_rhythm.event_type_range(event_type)
      self.assertEqual(expected, self.bend_rhythm.event_type_range(event_type))
      self.assertEqual(
          self.legacy_no_rhythm.event_type_range(event_type),
          self.bend_no_rhythm.event_type_range(event_type))
    self.assertEqual(
        self.legacy_rhythm.event_type_range('rhythm'),
        self.bend_rhythm.event_type_range('rhythm'))

  def test_bend_ids_are_identical_when_rhythm_is_toggled(self):
    self.assertTrue(self.bend_rhythm.has_event_type('rhythm'))
    self.assertFalse(self.bend_no_rhythm.has_event_type('rhythm'))
    self.assertEqual(
        self.bend_rhythm.event_type_range('pitch_bend'),
        self.bend_no_rhythm.event_type_range('pitch_bend'))
    for value in range(-12, 13):
      event = event_codec.Event('pitch_bend', value)
      self.assertEqual(self.bend_rhythm.encode_event(event),
                       self.bend_no_rhythm.encode_event(event))

  def test_bend_values_are_bounded(self):
    for value in (-12, -1, 0, 1, 12):
      token = self.bend_no_rhythm.encode_event(
          event_codec.Event('pitch_bend', value))
      self.assertEqual(event_codec.Event('pitch_bend', value),
                       self.bend_no_rhythm.decode_event_index(token))
    with self.assertRaises(ValueError):
      self.bend_no_rhythm.encode_event(event_codec.Event('pitch_bend', -13))
    with self.assertRaises(ValueError):
      self.bend_no_rhythm.encode_event(event_codec.Event('pitch_bend', 13))

  def test_checkpoint_geometry_remains_1536(self):
    for codec in (self.bend_rhythm, self.bend_no_rhythm):
      self.assertEqual(
          1536,
          vocabularies.num_embeddings(
              vocabularies.vocabulary_from_codec(codec)))

  def test_note_sequence_bend_round_trip_preserves_rhythm_track(self):
    source = note_seq.NoteSequence(ticks_per_quarter=220)
    source.instrument_infos.add(instrument=3, name='distortion-guitar:rhythm')
    source.notes.add(
        start_time=0.0, end_time=1.0, pitch=64, velocity=100,
        program=30, instrument=3)
    source.pitch_bends.add(
        time=0.1, bend=4096, program=30, instrument=3)

    times, values = (
        note_sequences.note_sequence_to_onsets_and_offsets_and_programs(
            source, include_rhythm=True, include_pitch_bends=True))
    encoder_state = note_sequences.NoteEncodingState()
    decoder_state = note_sequences.NoteDecodingState()
    note_sequences.begin_tied_pitches_section(decoder_state)
    note_sequences.decode_note_event(
        decoder_state, 0.0, event_codec.Event('tie', 0), self.bend_rhythm)
    for time, value in sorted(zip(times, values), key=lambda item: item[0]):
      for event in note_sequences.note_event_data_to_events(
          encoder_state, value, self.bend_rhythm):
        note_sequences.decode_note_event(
            decoder_state, time, event, self.bend_rhythm)
    decoded = note_sequences.flush_note_decoding_state(decoder_state)

    self.assertLen(decoded.notes, 1)
    self.assertEqual('distortion-guitar:rhythm', decoded.instrument_infos[0].name)
    self.assertEqual(decoded.notes[0].instrument, decoded.pitch_bends[0].instrument)
    # +6 semitones at the fixed +/-12 range, followed by a center reset.
    self.assertEqual([4096, 0], [bend.bend for bend in decoded.pitch_bends])

  def test_raw_wheel_quantization_covers_asymmetric_midi_endpoints(self):
    self.assertEqual(-12, note_sequences.midi_pitch_bend_to_semitones(-8192))
    self.assertEqual(12, note_sequences.midi_pitch_bend_to_semitones(8191))
    self.assertEqual(-8192, note_sequences.semitones_to_midi_pitch_bend(-12))
    self.assertEqual(8191, note_sequences.semitones_to_midi_pitch_bend(12))

  def test_missing_bend_in_next_tie_section_centers_carried_state(self):
    state = note_sequences.NoteDecodingState(current_time=1.0)
    state.current_bends[(30, False)] = 6
    note_sequences.begin_tied_pitches_section(state)
    note_sequences.decode_note_event(
        state, 1.0, event_codec.Event('tie', 0), self.bend_rhythm)
    self.assertEmpty(state.current_bends)
    self.assertEqual([0], [bend.bend for bend in state.note_sequence.pitch_bends])


if __name__ == '__main__':
  tf.test.main()
