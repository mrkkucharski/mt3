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

"""Phase 0 gate: two tiny rendered excerpts covering the (program, rhythm)
distinction, for the MT3_guitar_transcription_plan.md Phase 0 gate --
"Overfit a tiny randomly initialized MT3 configuration on two rendered
excerpts until onset-plus-offset F1 over (program, rhythm) pairs reaches at
least 0.95."

Importing this module registers the `guitar_pilot_gate` transcription task
as a side effect (referenced by name from
`mt3/gin/guitar_pilot_gate_local.gin`), mirroring how `mt3.tasks` registers
the real datasets.
"""

from __future__ import annotations

import functools
from pathlib import Path

from mt3 import datasets
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import tasks

import note_seq
import numpy as np
import tensorflow as tf

GATE_DIR = Path(__file__).resolve().parents[3] / 'runs' / 'phase0_gate_data'
SAMPLE_RATE = 44100


def _note(notes_list, rhythms_list, pitch, program, start, end, rhythm):
  notes_list.append((pitch, program, start, end))
  rhythms_list.append(rhythm)


def _build_example(example_id: str, notes, rhythms) -> tuple[note_seq.NoteSequence, bytes]:
  """Builds a NoteSequence with (program, rhythm) labels and its synth WAV."""
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  for pitch, program, start, end in notes:
    ns.notes.add(start_time=start, end_time=end, pitch=pitch, velocity=100,
                program=program, is_drum=False)
    ns.total_time = max(ns.total_time, end)
  note_sequences.assign_instruments(ns, note_rhythms=list(rhythms))
  ns.id = example_id

  samples = note_seq.midi_synth.synthesize(ns, sample_rate=SAMPLE_RATE)
  wav_bytes = note_seq.audio_io.samples_to_wav_data(samples, SAMPLE_RATE)
  return ns, wav_bytes


def _gate_examples():
  # Excerpt 1: two guitars sharing a program but differing in rhythm, playing
  # simultaneously -- the case the fused (program-only) scheme could not
  # express -- plus a non-guitar note (always rhythm-absent).
  notes1, rhythms1 = [], []
  _note(notes1, rhythms1, 60, 30, 0.0, 1.0, True)   # distortion-guitar:rhythm
  _note(notes1, rhythms1, 64, 30, 0.0, 1.0, False)  # distortion-guitar
  _note(notes1, rhythms1, 67, 0, 1.0, 2.0, False)   # acoustic-grand-piano

  # Excerpt 2: one guitar program appearing sequentially in both roles, plus
  # an overlapping non-guitar note.
  notes2, rhythms2 = [], []
  _note(notes2, rhythms2, 55, 24, 0.0, 1.0, False)  # acoustic-guitar-nylon
  _note(notes2, rhythms2, 55, 24, 1.0, 2.0, True)   # acoustic-guitar-nylon:rhythm
  _note(notes2, rhythms2, 72, 73, 0.5, 1.5, False)  # flute

  return [('gate_0', notes1, rhythms1), ('gate_1', notes2, rhythms2)]


def _bytes_feature(value: bytes) -> tf.train.Feature:
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def build_gate_dataset(output_dir: Path = GATE_DIR) -> dict[str, Path]:
  """Writes train.tfrecord and test.tfrecord (identical: this is an overfit
  check, not a generalization check) containing the two gate excerpts.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  outputs = {}
  for split in ('train', 'test'):
    path = output_dir / f'{split}.tfrecord'
    with tf.io.TFRecordWriter(str(path)) as writer:
      for example_id, notes, rhythms in _gate_examples():
        ns, wav_bytes = _build_example(example_id, notes, rhythms)
        example = tf.train.Example(features=tf.train.Features(feature={
            'audio': _bytes_feature(wav_bytes),
            'sequence': _bytes_feature(ns.SerializeToString()),
            'id': _bytes_feature(example_id.encode('utf-8')),
        }))
        writer.write(example.SerializeToString())
    outputs[split] = path
  return outputs


GATE_DATASET_CONFIG = datasets.DatasetConfig(
    name='guitar_pilot_gate',
    paths={
        'train': str(GATE_DIR / 'train.tfrecord'),
        'test': str(GATE_DIR / 'test.tfrecord'),
    },
    features={
        'audio': tf.io.FixedLenFeature([], dtype=tf.string),
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
    },
    train_split='train',
    train_eval_split='test',
    infer_eval_splits=[
        datasets.InferEvalSplit(name='test', suffix='eval_test'),
    ])

# include_ties=False (plain onsets+offsets+programs, no tie-section preamble):
# tie/rhythm interaction across segment boundaries is already proven by the
# dedicated round-trip tests (note_sequences_rhythm_test.py). Both excerpts
# fit in a single window anyway, so what this gate isolates is just whether
# the model can learn program+rhythm from audio at all -- the tie-section
# preamble the full NoteEncodingWithTiesSpec target requires is an extra,
# unrelated thing for a randomly initialized tiny model to learn to emit
# before it ever produces a single correct note token.
tasks.add_transcription_task_to_registry(
    dataset_config=GATE_DATASET_CONFIG,
    spectrogram_config=tasks.SPECTROGRAM_CONFIG,
    vocab_config=tasks.VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=False,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=False)


if __name__ == '__main__':
  outputs = build_gate_dataset()
  for split, path in outputs.items():
    print(f'Wrote {split}: {path}')
