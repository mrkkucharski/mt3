"""Tests for full multi-instrument MIDI export."""

import mido
import note_seq

from mt3.scripts import transcribe_mt3


def test_write_multitrack_midi_retains_all_programs_and_drums(tmp_path):
  sequence = note_seq.NoteSequence()
  for instrument, pitch, program, is_drum in (
      (0, 60, 0, False), (1, 45, 33, False), (2, 36, 0, True)):
    note = sequence.notes.add()
    note.start_time = 0.0
    note.end_time = 0.5
    note.pitch = pitch
    note.velocity = 100
    note.program = program
    note.is_drum = is_drum
    note.instrument = instrument
  output = tmp_path / 'multitrack.mid'

  count = transcribe_mt3.write_multitrack_midi(sequence, output)
  midi_file = mido.MidiFile(output)

  assert count == 3
  assert sum(message.type == 'note_on' and message.velocity > 0
             for track in midi_file.tracks for message in track) == 3
  assert any(message.type == 'program_change' and message.program == 0
             for track in midi_file.tracks for message in track)
  assert any(message.type == 'program_change' and message.program == 33
             for track in midi_file.tracks for message in track)
