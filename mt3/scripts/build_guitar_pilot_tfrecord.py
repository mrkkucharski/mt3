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
import sys

import note_seq
import tensorflow as tf

from mt3 import note_sequences


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
