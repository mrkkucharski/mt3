"""Tests for build_guitar_tfrecord."""

import json
from pathlib import Path
import tempfile
import wave

import mido
import note_seq
import tensorflow as tf

from mt3.scripts import build_guitar_tfrecord


def _write_fixture(dataset_dir: Path) -> None:
  (dataset_dir / 'audio/train').mkdir(parents=True)
  (dataset_dir / 'midi/train').mkdir(parents=True)
  with wave.open(str(dataset_dir / 'audio/train/gtr_0001.wav'), 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(44100)
    wav_file.writeframes(b'\x01\x00' * 33075)

  midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
  conductor = mido.MidiTrack()
  conductor.append(mido.MetaMessage('track_name', name='conductor', time=0))
  conductor.append(mido.MetaMessage('set_tempo', tempo=468750, time=0))
  conductor.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
  midi_file.tracks.append(conductor)
  for lane_name in build_guitar_tfrecord.LANE_TO_PROGRAM:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=lane_name, time=0))
    if lane_name in ('clean-rhythm', 'distorted-lead'):
      pitch = 40 if lane_name == 'clean-rhythm' else 64
      track.append(mido.Message('note_on', channel=0, note=pitch, velocity=100, time=0))
      track.append(mido.Message('note_off', channel=0, note=pitch, velocity=0, time=480))
      # A zero-length Reaper-style event must not prevent conversion or become
      # an MT3 target note.
      track.append(mido.Message('note_on', channel=0, note=pitch, velocity=100, time=0))
      track.append(mido.Message('note_off', channel=0, note=pitch, velocity=0, time=0))
    midi_file.tracks.append(track)
  midi_file.save(dataset_dir / 'midi/train/gtr_0001.mid')

  record = {
      'id': 'gtr_0001', 'split': 'train',
      'audio_path': 'audio/train/gtr_0001.wav',
      'midi_path': 'midi/train/gtr_0001.mid',
      'sample_rate_hz': 44100, 'channels': 1, 'bit_depth': 16,
      'source_midi_id': 'source_0001',
      'lanes': [
          {'name': name, 'is_empty': name not in ('clean-rhythm', 'distorted-lead'),
           'tuning': 'standard' if name in ('clean-rhythm', 'distorted-lead') else None}
          for name in build_guitar_tfrecord.LANE_TO_PROGRAM
      ],
      'renderer': 'test renderer', 'preset': 'test preset',
      'effects_chain': 'none', 'render_seed': 1,
      'normalization': 'peak:-1dBFS',
      'duration_seconds': 0.75, 'midi_end_seconds': 0.46875,
      'render_tail_seconds': 0.28125,
  }
  (dataset_dir / 'manifest.jsonl').write_text(json.dumps(record) + '\n')


def test_build_dataset_writes_programs_for_each_nonempty_lane():
  with tempfile.TemporaryDirectory() as directory:
    dataset_dir = Path(directory) / 'dataset'
    _write_fixture(dataset_dir)
    outputs = build_guitar_tfrecord.build_dataset(dataset_dir, dataset_dir / 'tfrecord')
    serialized = next(tf.compat.v1.io.tf_record_iterator(str(outputs['train'])))

  example = tf.train.Example.FromString(serialized)
  sequence = note_seq.NoteSequence.FromString(
      example.features.feature['sequence'].bytes_list.value[0])
  assert example.features.feature['id'].bytes_list.value[0] == b'gtr_0001'
  assert sorted({note.program for note in sequence.notes}) == [26, 30]
  assert len(sequence.notes) == 2
  assert sequence.total_time == 0.46875
