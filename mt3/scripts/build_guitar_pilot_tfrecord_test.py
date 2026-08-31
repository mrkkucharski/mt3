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

"""Tests for build_guitar_pilot_tfrecord."""

import json
from pathlib import Path
import tempfile
import wave

import mido
import note_seq
import pytest
import tensorflow as tf

from mt3 import note_sequences
from mt3.scripts import build_guitar_pilot_tfrecord as build_tfrecord


def _write_wav(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with wave.open(str(path), 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(44100)
    wav_file.writeframes(b'\x01\x00' * 33075)


def _write_midi(path: Path) -> None:
  """One rhythm guitar part and one plain (non-guitar) part."""
  path.parent.mkdir(parents=True, exist_ok=True)
  midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
  conductor = mido.MidiTrack()
  conductor.append(mido.MetaMessage('track_name', name='conductor', time=0))
  conductor.append(mido.MetaMessage('set_tempo', tempo=468750, time=0))
  midi_file.tracks.append(conductor)

  # distortion-guitar:rhythm, program 30.
  rhythm_track = mido.MidiTrack()
  rhythm_track.append(
      mido.MetaMessage('track_name', name='distortion-guitar:rhythm', time=0))
  rhythm_track.append(mido.Message('program_change', channel=0, program=30, time=0))
  rhythm_track.append(mido.Message('note_on', channel=0, note=40, velocity=100, time=0))
  rhythm_track.append(mido.Message('note_off', channel=0, note=40, velocity=0, time=480))
  midi_file.tracks.append(rhythm_track)

  # acoustic-grand-piano, program 0, no rhythm annotation (not a guitar).
  piano_track = mido.MidiTrack()
  piano_track.append(
      mido.MetaMessage('track_name', name='acoustic-grand-piano', time=0))
  piano_track.append(mido.Message('program_change', channel=1, program=0, time=0))
  piano_track.append(mido.Message('note_on', channel=1, note=64, velocity=90, time=0))
  piano_track.append(mido.Message('note_off', channel=1, note=64, velocity=0, time=480))
  midi_file.tracks.append(piano_track)

  midi_file.save(str(path))


def _write_fixture(dataset_dir: Path, include_pitch_bends: bool = False) -> None:
  _write_wav(dataset_dir / 'audio/train/ex_0001.wav')
  _write_midi(dataset_dir / 'midi/train/ex_0001.mid')

  record = {
      'id': 'ex_0001',
      'split': 'train',
      'audio_path': 'audio/train/ex_0001.wav',
      'midi_path': 'midi/train/ex_0001.mid',
      'sample_rate_hz': 44100,
      'channels': 1,
      'source_midi_id': 'source_0001',
      'source_midi_name': 'fixture.mid',
      'soundfont_library_root': '/Users/Shared/Soundfonts',
      'parts': [
          {
              'track_name': 'distortion-guitar:rhythm',
              'program': 30,
              'is_drum': False,
              'rhythm': True,
              'note_count': 1,
          },
          {
              'track_name': 'acoustic-grand-piano',
              'program': 0,
              'is_drum': False,
              'rhythm': None,
              'note_count': 1,
          },
      ],
      'renderer': 'test renderer',
      'render_seed': 1,
      'normalization': 'peak:-1dBFS',
  }
  if include_pitch_bends:
    rpp = dataset_dir / 'source.RPP'
    rpp.write_text(
        '<TRACK\n'
        '  NAME "distortion-guitar:rhythm"\n'
        '  <ITEM\n'
        '    POSITION 0\n'
        '    <SOURCE MIDI\n'
        '      HASDATA 1 480 QN\n'
        '      E 0 b0 65 00\n'
        '      E 0 b0 64 00\n'
        '      E 0 b0 06 0c\n'
        '      E 0 e0 00 60\n'
        '      E 480 e0 00 40\n'
        '    >\n'
        '  >\n'
        '>\n')
    record['source_project'] = str(rpp)
  (dataset_dir / 'manifest.jsonl').write_text(json.dumps(record) + '\n')


def test_build_dataset_preserves_program_and_rhythm():
  with tempfile.TemporaryDirectory() as directory:
    dataset_dir = Path(directory) / 'dataset'
    _write_fixture(dataset_dir)
    outputs = build_tfrecord.build_dataset(dataset_dir, dataset_dir / 'tfrecord')
    serialized = next(tf.compat.v1.io.tf_record_iterator(str(outputs['train'])))

  example = tf.train.Example.FromString(serialized)
  assert example.features.feature['id'].bytes_list.value[0] == b'ex_0001'

  sequence = note_seq.NoteSequence.FromString(
      example.features.feature['sequence'].bytes_list.value[0])
  assert len(sequence.notes) == 2

  rhythm_by_instrument = note_sequences.instrument_rhythms(sequence)
  parts = {
      (note.program, rhythm_by_instrument.get(note.instrument, False))
      for note in sequence.notes
  }
  assert parts == {(30, True), (0, False)}


def test_build_dataset_rejects_manifest_midi_mismatch():
  with tempfile.TemporaryDirectory() as directory:
    dataset_dir = Path(directory) / 'dataset'
    _write_fixture(dataset_dir)
    manifest_path = dataset_dir / 'manifest.jsonl'
    record = json.loads(manifest_path.read_text())
    # Manifest claims the guitar part is NOT rhythm; the MIDI track name says
    # otherwise. The converter must catch this rather than trust the MIDI.
    record['parts'][0]['rhythm'] = None
    manifest_path.write_text(json.dumps(record) + '\n')

    with pytest.raises(ValueError, match='do not match'):
      build_tfrecord.build_dataset(dataset_dir, dataset_dir / 'tfrecord')


def test_build_dataset_adds_pitch_bends_from_source_rpp():
  with tempfile.TemporaryDirectory() as directory:
    dataset_dir = Path(directory) / 'dataset'
    _write_fixture(dataset_dir, include_pitch_bends=True)
    outputs = build_tfrecord.build_dataset(dataset_dir, dataset_dir / 'tfrecord')
    serialized = next(tf.compat.v1.io.tf_record_iterator(str(outputs['train'])))

  example = tf.train.Example.FromString(serialized)
  sequence = note_seq.NoteSequence.FromString(
      example.features.feature['sequence'].bytes_list.value[0])
  assert [bend.bend for bend in sequence.pitch_bends] == [4096, 0]
  assert sequence.pitch_bends[0].instrument == sequence.notes[0].instrument
  assert sequence.pitch_bends[0].program == 30
  assert sequence.pitch_bends[1].time == pytest.approx(0.46875)
