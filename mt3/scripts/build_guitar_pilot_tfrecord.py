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

"""Build MT3 TFRecords from the guitar-pilot corpus.

The corpus (`DATA_CONTRACT.md`) is produced and validated by the `reaper2mt3`
tool, which writes Standard MIDI Files whose track names are already the
canonical `<slug>[:rhythm]` strings and whose per-note `program` already
matches its part's declared program. This script therefore does no relabeling
of its own: it reads each MIDI file with note_seq's standard reader (which
populates `program`, `is_drum` and `instrument_infos` from those tracks
directly), pairs it with its WAV, and writes one MT3-compatible TFRecord
example per manifest record.

Run from the MT3 repository:

  uv run python -m mt3.scripts.build_guitar_pilot_tfrecord \
      --dataset ../data/pilot
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys

import mido
import note_seq
import tensorflow as tf

from mt3 import note_sequences


_RPP_EVENT = re.compile(
    r'^[Ee]\s+(\d+)\s+([0-9a-fA-F]{2})\s+'
    r'([0-9a-fA-F]{2})\s+([0-9a-fA-F]{2})$')


def _tick_to_seconds_fn(midi_path: Path):
  """Return (ticks_per_beat, absolute-tick to seconds) from corpus MIDI."""
  midi = mido.MidiFile(midi_path)
  tempo_events = [(0, 500_000)]
  for track in midi.tracks:
    tick = 0
    for message in track:
      tick += message.time
      if message.type == 'set_tempo':
        tempo_events.append((tick, message.tempo))
  tempo_events.sort(key=lambda event: event[0])

  def tick_to_seconds(target_tick: int) -> float:
    seconds = 0.0
    previous_tick = 0
    tempo = 500_000
    for tick, next_tempo in tempo_events:
      if tick > target_tick:
        break
      seconds += mido.tick2second(
          tick - previous_tick, midi.ticks_per_beat, tempo)
      previous_tick = tick
      tempo = next_tempo
    return seconds + mido.tick2second(
        target_tick - previous_tick, midi.ticks_per_beat, tempo)

  return midi.ticks_per_beat, tick_to_seconds


def _read_rpp_pitch_bends(rpp_path: Path) -> tuple[int, list[tuple[str, int, int]]]:
  """Read (track name, absolute tick, raw wheel value) from an RPP.

  The pilot projects contain one project-length MIDI item per corpus track,
  all at POSITION 0. The converted slide gestures explicitly declare RPN 0,0
  as +/-12 semitones before their wheel messages; reject any other layout or
  range instead of silently assigning the wrong token meaning.
  """
  lines = rpp_path.read_text(encoding='utf-8', errors='replace').splitlines()
  track_name = None
  item_position = None
  tick = 0
  ppq = None
  active_source = False
  rpn_msb = {}
  rpn_lsb = {}
  bend_range = {}
  bends = []

  for line in lines:
    stripped = line.strip()
    if stripped.startswith('<TRACK'):
      track_name = None
      item_position = None
      active_source = False
    elif track_name is None and stripped.startswith('NAME '):
      name = stripped[5:].strip()
      track_name = (name[1:-1] if len(name) >= 2 and
                    name[0] == name[-1] and name[0] in '"\'`' else name)
    elif stripped.startswith('POSITION '):
      item_position = float(stripped.split()[1])
    elif stripped.startswith('HASDATA '):
      bits = stripped.split()
      if len(bits) < 3 or not bits[2].isdigit():
        raise ValueError(f'{rpp_path}: malformed MIDI HASDATA line: {stripped}')
      source_ppq = int(bits[2])
      if ppq is None:
        ppq = source_ppq
      elif ppq != source_ppq:
        raise ValueError(f'{rpp_path}: mixed MIDI PPQ values {ppq} and {source_ppq}')
      if item_position not in (None, 0.0):
        raise ValueError(
            f'{rpp_path}: pitch-bend import requires POSITION 0 MIDI items; '
            f'found {item_position}')
      tick = 0
      active_source = True
      rpn_msb.clear()
      rpn_lsb.clear()
      bend_range.clear()
    elif active_source:
      match = _RPP_EVENT.match(stripped)
      if not match:
        continue
      delta, status_hex, d1_hex, d2_hex = match.groups()
      tick += int(delta)
      status = int(status_hex, 16)
      data1 = int(d1_hex, 16)
      data2 = int(d2_hex, 16)
      kind = status & 0xF0
      channel = status & 0x0F
      if kind == 0xB0:
        if data1 == 101:
          rpn_msb[channel] = data2
        elif data1 == 100:
          rpn_lsb[channel] = data2
        elif (data1 == 6 and rpn_msb.get(channel) == 0 and
              rpn_lsb.get(channel) == 0):
          bend_range[channel] = data2
      elif kind == 0xE0:
        if bend_range.get(channel) != 12:
          raise ValueError(
              f'{rpp_path}: pitch wheel at tick {tick} on {track_name!r} '
              'without an active +/-12-semitone RPN declaration')
        bends.append((track_name or '', tick, (data2 << 7 | data1) - 8192))

  return ppq or 0, bends


def _add_source_project_pitch_bends(
    ns: note_seq.NoteSequence, midi_path: Path, manifest_record: dict
) -> None:
  """Add bend labels from the source RPP that legacy corpus MIDI omitted."""
  source_project = manifest_record.get('source_project')
  if not source_project:
    return
  rpp_path = Path(source_project)
  if not rpp_path.is_file():
    raise FileNotFoundError(f'{manifest_record["id"]}: missing {rpp_path}')
  rpp_ppq, bends = _read_rpp_pitch_bends(rpp_path)
  if not bends:
    return
  midi_ppq, tick_to_seconds = _tick_to_seconds_fn(midi_path)
  if rpp_ppq != midi_ppq:
    raise ValueError(
        f'{manifest_record["id"]}: RPP PPQ {rpp_ppq} != MIDI PPQ {midi_ppq}')

  info_instruments = {info.name: info.instrument for info in ns.instrument_infos}
  source_to_canonical = {
      part.get('reaper_track_name', part['track_name']): part['track_name']
      for part in manifest_record['parts']
  }
  programs = {}
  for note in ns.notes:
    programs.setdefault(note.instrument, note.program)
  for track_name, tick, bend in bends:
    canonical_name = source_to_canonical.get(track_name, track_name)
    instrument = info_instruments.get(canonical_name)
    if instrument is None:
      raise ValueError(
          f'{manifest_record["id"]}: bend track {track_name!r} is absent '
          'from corpus MIDI instrument_infos')
    ns.pitch_bends.add(
        time=tick_to_seconds(tick), bend=bend, instrument=instrument,
        program=programs[instrument], is_drum=False)
  ns.total_time = max(ns.total_time,
                      max(bend.time for bend in ns.pitch_bends))


def _load_records(dataset_dir: Path) -> list[dict]:
  return [json.loads(line) for line in
          (dataset_dir / 'manifest.jsonl').read_text().splitlines()
          if line.strip()]


def _manifest_parts_key(record: dict) -> set[tuple[int | None, bool, bool]]:
  """(program, is_drum, rhythm) triples the manifest declares for a record."""
  key = set()
  for part in record['parts']:
    key.add((None if part['is_drum'] else part['program'],
             bool(part['is_drum']),
             bool(part.get('rhythm'))))
  return key


def _note_sequence_parts_key(
    ns: note_seq.NoteSequence) -> set[tuple[int | None, bool, bool]]:
  """(program, is_drum, rhythm) triples actually present in a NoteSequence."""
  rhythm_by_instrument = note_sequences.instrument_rhythms(ns)
  key = set()
  for note in ns.notes:
    program = None if note.is_drum else note.program
    rhythm = (False if note.is_drum
             else rhythm_by_instrument.get(note.instrument, False))
    key.add((program, bool(note.is_drum), rhythm))
  return key


def midi_to_note_sequence(
    midi_path: Path, example_id: str, audio_path: str,
    manifest_record: dict) -> note_seq.NoteSequence:
  """Reads a corpus MIDI file and cross-checks it against its manifest record."""
  ns = note_seq.midi_file_to_note_sequence(str(midi_path))
  ns.id = example_id
  ns.filename = audio_path
  _add_source_project_pitch_bends(ns, midi_path, manifest_record)

  manifest_key = _manifest_parts_key(manifest_record)
  midi_key = _note_sequence_parts_key(ns)
  if manifest_key != midi_key:
    raise ValueError(
        f'{example_id}: manifest parts {sorted(manifest_key)} do not match '
        f'MIDI parts {sorted(midi_key)}. Re-run `reaper2mt3 build`.')

  note_sequences.validate_note_sequence(ns)
  return ns


def _bytes_feature(value: bytes) -> tf.train.Feature:
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def build_dataset(
    dataset_dir: str | Path, output_dir: str | Path,
    overwrite: bool = False) -> dict[str, Path]:
  """Builds one TFRecord per populated split and returns their paths."""
  dataset_dir = Path(dataset_dir).resolve()
  output_dir = Path(output_dir).resolve()

  records_by_split: dict[str, list[dict]] = collections.defaultdict(list)
  for record in _load_records(dataset_dir):
    records_by_split[record['split']].append(record)

  output_dir.mkdir(parents=True, exist_ok=True)
  outputs = {}
  for split, records in sorted(records_by_split.items()):
    output_path = output_dir / f'{split}.tfrecord'
    if output_path.exists() and not overwrite:
      raise FileExistsError(
          f'{output_path} already exists; use --overwrite to replace it.')
    with tf.io.TFRecordWriter(str(output_path)) as writer:
      for record in records:
        audio_path = dataset_dir / record['audio_path']
        midi_path = dataset_dir / record['midi_path']
        sequence = midi_to_note_sequence(
            midi_path, record['id'], record['audio_path'], record)
        example = tf.train.Example(features=tf.train.Features(feature={
            'audio': _bytes_feature(audio_path.read_bytes()),
            'sequence': _bytes_feature(sequence.SerializeToString()),
            'id': _bytes_feature(record['id'].encode('utf-8')),
        }))
        writer.write(example.SerializeToString())
    outputs[split] = output_path
  return outputs


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True,
                      help='Dataset directory containing manifest.jsonl.')
  parser.add_argument('--output-dir',
                      help='TFRecord output directory (default: DATASET/tfrecord).')
  parser.add_argument('--overwrite', action='store_true',
                      help='Replace existing TFRecords in the output directory.')
  args = parser.parse_args(argv)
  dataset_dir = Path(args.dataset)
  output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / 'tfrecord'
  try:
    outputs = build_dataset(dataset_dir, output_dir, overwrite=args.overwrite)
  except (OSError, ValueError) as error:
    print(f'ERROR: {error}', file=sys.stderr)
    return 1
  for split, output_path in outputs.items():
    print(f'Wrote {split}: {output_path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
