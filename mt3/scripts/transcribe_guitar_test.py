"""Tests for local guitar MIDI export."""

import mido
import note_seq

from mt3.scripts import transcribe_guitar


def test_write_guitar_midi_creates_all_fixed_lanes(tmp_path):
  sequence = note_seq.NoteSequence()
  note = sequence.notes.add()
  note.start_time = 0.25
  note.end_time = 0.75
  note.pitch = 64
  note.velocity = 90
  note.program = 27
  note = sequence.notes.add()
  note.start_time = 0.5
  note.end_time = 1.0
  note.pitch = 40
  note.velocity = 100
  note.program = 30
  output = tmp_path / 'transcription.mid'

  counts = transcribe_guitar.write_guitar_midi(sequence, output)
  midi_file = mido.MidiFile(output)

  assert counts == {
      'clean-rhythm': 0,
      'clean-lead': 1,
      'distorted-rhythm': 0,
      'distorted-lead': 1,
  }
  assert [next(message.name for message in track if message.type == 'track_name')
          for track in midi_file.tracks[1:]] == [name for name, _ in transcribe_guitar.LANES]
  assert [sum(1 for message in track if message.type == 'note_on' and message.velocity > 0)
          for track in midi_file.tracks[1:]] == [0, 1, 0, 1]
