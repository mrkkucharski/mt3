"""Validate a guitar-transcription dataset against the Phase 1 contract.

Run from the MT3 repository, for example:

  uv run python mt3/scripts/validate_guitar_dataset.py --dataset ../data/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
import wave

import mido
import pretty_midi


REQUIRED_LANES = (
    'clean-rhythm',
    'clean-lead',
    'distorted-rhythm',
    'distorted-lead',
)
ALLOWED_TUNINGS = {'standard', 'eb-standard', 'drop-d'}
MIN_PITCH = 38
MAX_PITCH = 91
EXPECTED_SAMPLE_RATE_HZ = 44100
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2
TIMING_TOLERANCE_SECONDS = 0.002


@dataclass
class ValidationResult:
  """Collected validation messages."""

  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)
  records_checked: int = 0

  @property
  def ok(self) -> bool:
    return not self.errors


def _error(result: ValidationResult, example_id: str, message: str) -> None:
  result.errors.append(f'{example_id}: {message}')


def _warning(result: ValidationResult, example_id: str, message: str) -> None:
  result.warnings.append(f'{example_id}: {message}')


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _dataset_file(dataset_dir: Path, relative_path: object) -> Path | None:
  if not isinstance(relative_path, str) or not relative_path:
    return None
  candidate = (dataset_dir / relative_path).resolve()
  try:
    candidate.relative_to(dataset_dir.resolve())
  except ValueError:
    return None
  return candidate


def _load_manifest(manifest_path: Path, result: ValidationResult) -> list[dict]:
  records = []
  if not manifest_path.is_file():
    result.errors.append(f'Manifest does not exist: {manifest_path}')
    return records
  for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
    if not line.strip():
      result.errors.append(f'Manifest line {line_number} is empty.')
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as error:
      result.errors.append(f'Manifest line {line_number} is invalid JSON: {error}.')
      continue
    if not isinstance(record, dict):
      result.errors.append(f'Manifest line {line_number} must be a JSON object.')
      continue
    records.append(record)
  if not records and not result.errors:
    result.errors.append('Manifest contains no records.')
  return records


def _validate_wav(path: Path, record: dict, result: ValidationResult,
                  example_id: str) -> float | None:
  try:
    with wave.open(str(path), 'rb') as wav_file:
      channels = wav_file.getnchannels()
      sample_rate = wav_file.getframerate()
      sample_width = wav_file.getsampwidth()
      frame_count = wav_file.getnframes()
      compression = wav_file.getcomptype()
  except (wave.Error, EOFError) as error:
    _error(result, example_id, f'cannot read WAV: {error}')
    return None

  if compression != 'NONE':
    _error(result, example_id, f'WAV must be uncompressed PCM, got {compression}.')
  if channels != EXPECTED_CHANNELS:
    _error(result, example_id, f'WAV must be mono, got {channels} channels.')
  if sample_rate != EXPECTED_SAMPLE_RATE_HZ:
    _error(result, example_id,
           f'WAV sample rate must be {EXPECTED_SAMPLE_RATE_HZ}, got {sample_rate}.')
  if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
    _error(result, example_id,
           f'WAV must be 16-bit PCM, got {sample_width * 8}-bit.')
  if frame_count == 0:
    _error(result, example_id, 'WAV contains no audio frames.')
  if record.get('sample_rate_hz') != sample_rate:
    _error(result, example_id, 'manifest sample_rate_hz does not match WAV.')
  if record.get('channels') != channels:
    _error(result, example_id, 'manifest channels does not match WAV.')
  if record.get('bit_depth') != sample_width * 8:
    _error(result, example_id, 'manifest bit_depth does not match WAV.')
  return frame_count / sample_rate


def _validate_midi(path: Path, record: dict, result: ValidationResult,
                   example_id: str) -> float | None:
  try:
    midi_file = mido.MidiFile(path)
  except (OSError, EOFError, ValueError) as error:
    _error(result, example_id, f'cannot read MIDI: {error}')
    return None

  if midi_file.type != 1:
    _warning(result, example_id, f'MIDI type {midi_file.type}; type 1 is preferred.')

  lane_tracks: dict[str, tuple[int, list[mido.Message]]] = {}
  for track_index, track in enumerate(midi_file.tracks):
    track_names = [message.name for message in track if message.type == 'track_name']
    note_ons = [message for message in track
                if message.type == 'note_on' and message.velocity > 0]
    lane_names = [name for name in track_names if name in REQUIRED_LANES]
    if note_ons and not lane_names:
      _error(result, example_id,
             f'MIDI track {track_index} contains notes but has no required lane name.')
    for lane_name in lane_names:
      if lane_name in lane_tracks:
        _error(result, example_id, f'MIDI lane {lane_name!r} occurs more than once.')
      else:
        lane_tracks[lane_name] = (track_index, note_ons)

  missing_lanes = [lane for lane in REQUIRED_LANES if lane not in lane_tracks]
  if missing_lanes:
    _error(result, example_id, f'MIDI is missing required lanes: {missing_lanes}.')

  lanes = record.get('lanes')
  if not isinstance(lanes, list) or [lane.get('name') if isinstance(lane, dict) else None
                                    for lane in lanes] != list(REQUIRED_LANES):
    _error(result, example_id, 'manifest lanes must list the four required lanes in order.')
  elif not missing_lanes:
    for lane in lanes:
      lane_name = lane['name']
      expected_empty = lane.get('is_empty')
      observed_empty = not lane_tracks[lane_name][1]
      if not isinstance(expected_empty, bool):
        _error(result, example_id, f'{lane_name}: is_empty must be boolean.')
      elif expected_empty != observed_empty:
        _error(result, example_id,
               f'{lane_name}: manifest is_empty does not match MIDI notes.')
      tuning = lane.get('tuning')
      if observed_empty and tuning is not None:
        _error(result, example_id, f'{lane_name}: empty lanes must have null tuning.')
      if not observed_empty and tuning not in ALLOWED_TUNINGS:
        _error(result, example_id,
               f'{lane_name}: tuning must be one of {sorted(ALLOWED_TUNINGS)}.')

  for lane_name, (_, notes) in lane_tracks.items():
    for note in notes:
      if not MIN_PITCH <= note.note <= MAX_PITCH:
        _error(result, example_id,
               f'{lane_name}: MIDI note {note.note} is outside {MIN_PITCH}-{MAX_PITCH}.')

  try:
    pretty_midi_file = pretty_midi.PrettyMIDI(str(path))
  except (OSError, ValueError) as error:
    _error(result, example_id, f'cannot calculate MIDI timing: {error}')
    return None
  return pretty_midi_file.get_end_time()


def _validate_metadata(record: dict, result: ValidationResult,
                       example_id: str) -> None:
  required_keys = (
      'id', 'split', 'audio_path', 'midi_path', 'sample_rate_hz', 'channels',
      'bit_depth', 'source_midi_id', 'lanes', 'renderer', 'preset',
      'effects_chain', 'render_seed', 'normalization')
  for key in required_keys:
    if key not in record:
      _error(result, example_id, f'manifest is missing required field {key!r}.')
  if record.get('split') not in {'train', 'test'}:
    _error(result, example_id, 'split must be train or test.')
  for key in ('renderer', 'preset', 'effects_chain', 'normalization'):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
      _error(result, example_id, f'{key} must be a non-empty string.')
    elif 'not-recorded' in value.lower():
      _warning(result, example_id, f'{key} is marked not-recorded.')


def _validate_record(dataset_dir: Path, record: dict,
                     result: ValidationResult) -> None:
  example_id = record.get('id') if isinstance(record.get('id'), str) else '<unknown-id>'
  _validate_metadata(record, result, example_id)

  audio_path = _dataset_file(dataset_dir, record.get('audio_path'))
  midi_path = _dataset_file(dataset_dir, record.get('midi_path'))
  if audio_path is None:
    _error(result, example_id, 'audio_path must be a non-empty relative dataset path.')
  elif not audio_path.is_file():
    _error(result, example_id, f'audio file does not exist: {audio_path}.')
  if midi_path is None:
    _error(result, example_id, 'midi_path must be a non-empty relative dataset path.')
  elif not midi_path.is_file():
    _error(result, example_id, f'MIDI file does not exist: {midi_path}.')
  if audio_path is None or midi_path is None or not audio_path.is_file() or not midi_path.is_file():
    return

  wav_duration = _validate_wav(audio_path, record, result, example_id)
  midi_end = _validate_midi(midi_path, record, result, example_id)
  if 'audio_sha256' in record and record['audio_sha256'] != _sha256(audio_path):
    _error(result, example_id, 'audio_sha256 does not match the audio file.')
  if 'midi_sha256' in record and record['midi_sha256'] != _sha256(midi_path):
    _error(result, example_id, 'midi_sha256 does not match the MIDI file.')

  if wav_duration is not None and isinstance(record.get('duration_seconds'), (int, float)):
    if abs(wav_duration - record['duration_seconds']) > TIMING_TOLERANCE_SECONDS:
      _error(result, example_id, 'manifest duration_seconds does not match WAV.')
  if midi_end is not None and isinstance(record.get('midi_end_seconds'), (int, float)):
    if abs(midi_end - record['midi_end_seconds']) > TIMING_TOLERANCE_SECONDS:
      _error(result, example_id, 'manifest midi_end_seconds does not match MIDI.')
  if (wav_duration is not None and midi_end is not None and
      isinstance(record.get('render_tail_seconds'), (int, float))):
    if abs((wav_duration - midi_end) - record['render_tail_seconds']) > TIMING_TOLERANCE_SECONDS:
      _error(result, example_id, 'manifest render_tail_seconds does not match files.')


def validate_dataset(dataset_dir: str | Path) -> ValidationResult:
  """Validates all records in ``dataset_dir/manifest.jsonl``."""
  dataset_path = Path(dataset_dir).resolve()
  result = ValidationResult()
  records = _load_manifest(dataset_path / 'manifest.jsonl', result)
  seen_ids: set[str] = set()
  source_splits: dict[str, set[str]] = {}
  for record in records:
    example_id = record.get('id') if isinstance(record.get('id'), str) else '<unknown-id>'
    if example_id in seen_ids:
      _error(result, example_id, 'example ID occurs more than once.')
    seen_ids.add(example_id)
    source_id = record.get('source_midi_id')
    split = record.get('split')
    if isinstance(source_id, str) and isinstance(split, str):
      source_splits.setdefault(source_id, set()).add(split)
    _validate_record(dataset_path, record, result)
    result.records_checked += 1
  for source_id, splits in source_splits.items():
    if len(splits) > 1:
      result.errors.append(
          f'source_midi_id {source_id!r} occurs in multiple splits: {sorted(splits)}.')
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True,
                      help='Path to a dataset directory containing manifest.jsonl.')
  parser.add_argument('--strict', action='store_true',
                      help='Treat warnings, such as not-recorded metadata, as failures.')
  args = parser.parse_args(argv)
  result = validate_dataset(args.dataset)
  for message in result.errors:
    print(f'ERROR: {message}', file=sys.stderr)
  for message in result.warnings:
    print(f'WARNING: {message}', file=sys.stderr)
  print(f'Checked {result.records_checked} manifest record(s): '
        f'{len(result.errors)} error(s), {len(result.warnings)} warning(s).')
  return 0 if result.ok and (not args.strict or not result.warnings) else 1


if __name__ == '__main__':
  raise SystemExit(main())
