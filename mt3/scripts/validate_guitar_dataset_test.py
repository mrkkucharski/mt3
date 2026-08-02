"""Tests for validate_guitar_dataset."""

import json
from pathlib import Path
import tempfile
import wave

import mido

from mt3.scripts import validate_guitar_dataset


def _write_fixture(dataset_dir: Path, include_all_lanes: bool = True) -> None:
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
  conductor.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
  midi_file.tracks.append(conductor)
  lane_names = validate_guitar_dataset.REQUIRED_LANES
  if not include_all_lanes:
    lane_names = lane_names[:-1]
  for lane_name in lane_names:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=lane_name, time=0))
    if lane_name == 'clean-rhythm':
      track.append(mido.Message('note_on', channel=0, note=40, velocity=100, time=0))
      track.append(mido.Message('note_off', channel=0, note=40, velocity=0, time=480))
    midi_file.tracks.append(track)
  midi_file.save(dataset_dir / 'midi/train/gtr_0001.mid')

  record = {
      'id': 'gtr_0001', 'split': 'train',
      'audio_path': 'audio/train/gtr_0001.wav',
      'midi_path': 'midi/train/gtr_0001.mid',
      'sample_rate_hz': 44100, 'channels': 1, 'bit_depth': 16,
      'source_midi_id': 'source_0001',
      'lanes': [
          {'name': name, 'is_empty': name != 'clean-rhythm',
           'tuning': 'standard' if name == 'clean-rhythm' else None}
          for name in validate_guitar_dataset.REQUIRED_LANES
      ],
      'renderer': 'test renderer', 'preset': 'test preset',
      'effects_chain': 'none', 'render_seed': 1,
      'normalization': 'peak:-1dBFS',
      'duration_seconds': 0.75, 'midi_end_seconds': 0.5,
      'render_tail_seconds': 0.25,
  }
  (dataset_dir / 'manifest.jsonl').write_text(json.dumps(record) + '\n')


def test_valid_dataset_passes():
  with tempfile.TemporaryDirectory() as directory:
    dataset_dir = Path(directory)
    _write_fixture(dataset_dir)
    result = validate_guitar_dataset.validate_dataset(dataset_dir)
  assert result.ok
  assert not result.warnings
  assert result.records_checked == 1


def test_missing_lane_fails():
  with tempfile.TemporaryDirectory() as directory:
    dataset_dir = Path(directory)
    _write_fixture(dataset_dir, include_all_lanes=False)
    result = validate_guitar_dataset.validate_dataset(dataset_dir)
  assert not result.ok
  assert any('missing required lanes' in error for error in result.errors)
