"""Build MT3 TFRecords from a validated guitar pilot dataset.

Run from the MT3 repository:

  uv run python mt3/scripts/build_guitar_tfrecord.py --dataset ../data/pilot
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Iterable

import mido
import note_seq
import tensorflow as tf

from mt3 import note_sequences
from mt3.scripts import validate_guitar_dataset


# These are zero-indexed General MIDI program numbers. The mapping is part of
# the training target: MT3 predicts the program, then downstream code maps it
# back to the corresponding fixed guitar lane.
LANE_TO_PROGRAM = {
    'clean-rhythm': 26,       # Electric Guitar (jazz)
    'clean-lead': 27,         # Electric Guitar (clean)
    'distorted-rhythm': 29,   # Electric Guitar (overdriven)
    'distorted-lead': 30,     # Electric Guitar (distortion)
}


def _absolute_messages(track: mido.MidiTrack) -> Iterable[tuple[int, mido.Message]]:
  """Yields a MIDI track's messages with absolute tick positions."""
  tick = 0
  for message in track:
    tick += message.time
    yield tick, message


def _tempo_events(midi_file: mido.MidiFile) -> list[tuple[int, int]]:
  """Returns the effective tempo map as (tick, microseconds_per_beat)."""
  events = [(0, 500000)]
  for track in midi_file.tracks:
    for tick, message in _absolute_messages(track):
      if message.type == 'set_tempo':
        events.append((tick, message.tempo))
  effective_events = []
  # Sort only by tick. Python's stable sort then keeps the default tempo before
  # an explicit tempo event at tick 0, allowing the explicit event to win.
  for tick, tempo in sorted(events, key=lambda event: event[0]):
    if effective_events and effective_events[-1][0] == tick:
      effective_events[-1] = (tick, tempo)
    else:
      effective_events.append((tick, tempo))
  return effective_events


def _ticks_to_seconds(tick: int, tempo_events: list[tuple[int, int]],
                      ticks_per_beat: int) -> float:
  """Converts an absolute MIDI tick using the file's tempo map."""
  seconds = 0.0
  previous_tick, tempo = tempo_events[0]
  for event_tick, event_tempo in tempo_events[1:]:
    if tick <= event_tick:
      return seconds + (tick - previous_tick) * tempo / 1e6 / ticks_per_beat
    seconds += (event_tick - previous_tick) * tempo / 1e6 / ticks_per_beat
    previous_tick, tempo = event_tick, event_tempo
  return seconds + (tick - previous_tick) * tempo / 1e6 / ticks_per_beat


def _lane_note_events(track: mido.MidiTrack, lane_name: str,
                      tempo_events: list[tuple[int, int]],
                      ticks_per_beat: int) -> list[tuple[float, float, int, int]]:
  """Converts one named raw MIDI track into note tuples.

  Reaper may use multiple MIDI channels within a lane, such as one channel per
  guitar string. Pairing by channel and pitch preserves that source structure
  while emitting one MT3 target lane.
  """
  active_notes: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
  completed_notes = []
  for tick, message in _absolute_messages(track):
    if message.type == 'note_on' and message.velocity > 0:
      active_notes[(message.channel, message.note)].append((tick, message.velocity))
    elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
      key = (message.channel, message.note)
      if not active_notes[key]:
        raise ValueError(f'{lane_name}: note-off without matching note-on at tick {tick}.')
      start_tick, velocity = active_notes[key].pop()
      # Reaper can export a note-on and note-off at the same tick. It has no
      # audible or symbolic duration, so omit it rather than rejecting an
      # otherwise valid recording.
      if tick == start_tick:
        continue
      if tick < start_tick:
        raise ValueError(f'{lane_name}: non-positive note duration at tick {tick}.')
      completed_notes.append((
          _ticks_to_seconds(start_tick, tempo_events, ticks_per_beat),
          _ticks_to_seconds(tick, tempo_events, ticks_per_beat),
          message.note,
          velocity))
  unclosed = sum(len(notes) for notes in active_notes.values())
  if unclosed:
    raise ValueError(f'{lane_name}: {unclosed} note-on event(s) have no note-off.')
  return completed_notes


def midi_to_note_sequence(midi_path: str | Path, example_id: str,
                          audio_path: str) -> note_seq.NoteSequence:
  """Converts the four contract lanes in a MIDI file to an MT3 NoteSequence."""
  midi_file = mido.MidiFile(midi_path)
  tempo_events = _tempo_events(midi_file)
  lane_tracks = {}
  for track in midi_file.tracks:
    names = [message.name for message in track if message.type == 'track_name']
    for name in names:
      if name in LANE_TO_PROGRAM:
        lane_tracks[name] = track

  note_sequence = note_seq.NoteSequence(ticks_per_quarter=midi_file.ticks_per_beat)
  note_sequence.id = example_id
  note_sequence.filename = audio_path
  for tick, tempo in tempo_events:
    note_sequence.tempos.add(
        time=_ticks_to_seconds(tick, tempo_events, midi_file.ticks_per_beat),
        qpm=mido.tempo2bpm(tempo))
  for track in midi_file.tracks:
    for tick, message in _absolute_messages(track):
      if message.type == 'time_signature':
        note_sequence.time_signatures.add(
            time=_ticks_to_seconds(tick, tempo_events, midi_file.ticks_per_beat),
            numerator=message.numerator,
            denominator=message.denominator)

  for lane_name, program in LANE_TO_PROGRAM.items():
    if lane_name not in lane_tracks:
      raise ValueError(f'MIDI is missing required lane {lane_name!r}.')
    for start_time, end_time, pitch, velocity in _lane_note_events(
        lane_tracks[lane_name], lane_name, tempo_events, midi_file.ticks_per_beat):
      note_sequence.notes.add(
          start_time=start_time,
          end_time=end_time,
          pitch=pitch,
          velocity=velocity,
          program=program,
          is_drum=False)
      note_sequence.total_time = max(note_sequence.total_time, end_time)
  note_sequences.assign_instruments(note_sequence)
  note_sequences.validate_note_sequence(note_sequence)
  return note_sequence


def _bytes_feature(value: bytes) -> tf.train.Feature:
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _load_records(dataset_dir: Path) -> list[dict]:
  return [json.loads(line) for line in (dataset_dir / 'manifest.jsonl').read_text().splitlines()
          if line.strip()]


def build_dataset(dataset_dir: str | Path, output_dir: str | Path,
                  overwrite: bool = False) -> dict[str, Path]:
  """Builds one TFRecord per populated split and returns their paths."""
  dataset_dir = Path(dataset_dir).resolve()
  output_dir = Path(output_dir).resolve()
  validation = validate_guitar_dataset.validate_dataset(dataset_dir)
  if not validation.ok:
    raise ValueError('Dataset validation failed:\n' + '\n'.join(validation.errors))

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
        sequence = midi_to_note_sequence(midi_path, record['id'], record['audio_path'])
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
