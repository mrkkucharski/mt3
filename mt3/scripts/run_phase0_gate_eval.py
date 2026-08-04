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

"""Computes the Phase 0 gate F1 directly, bypassing seqio.Evaluator.

t5x.train's built-in inference evaluator also computes this, but its
TensorBoard/JSON summary path (mt3.summaries) calls fluidsynth for audio
previews, which isn't installed in this environment and makes the whole
metrics computation fail before any score is logged. This script restores
the trained tiny checkpoint directly and scores just the two gate excerpts
against mt3.metrics._program_aware_note_scores(granularity_type='full'),
which is exactly the (program, rhythm)-aware onset+offset F1 the Phase 0
gate requires (>= 0.95).

Run from the MT3 repository (same PYTHONPATH workaround as training):
  PYTHONPATH=mt3/local_cpu_sitecustomize uv run python \
      -m mt3.scripts.run_phase0_gate_eval \
      --checkpoint runs/phase0_gate_v1/checkpoint_2500
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import gin
import jax
import numpy as np
import seqio

from mt3.runtime import tensorflow

tf = tensorflow()

import note_seq
from t5.data import preprocessors as t5_preprocessors
from t5x import adafactor
from t5x import partitioning
from t5x import utils

from mt3 import metrics
from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3.scripts import phase0_gate
from mt3 import spectrograms
from mt3 import vocabularies

INPUT_LENGTH = 256
TARGET_LENGTH = 128
SAMPLE_RATE = 16000

# Must match mt3/gin/guitar_pilot_gate_local.gin's network.T5Config exactly,
# or checkpoint restore fails on a shape mismatch.
TINY_T5_CONFIG = dict(
    dtype='float32',
    emb_dim=32,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    head_dim=8,
    mlp_dim=64,
    mlp_activations=('gelu', 'linear'),
    dropout_rate=0.0,
    logits_via_embedding=False,
)


class GateTranscriber:
  """Minimal Transcriber variant sized for the Phase 0 gate's tiny model."""

  def __init__(self, checkpoint_path: Path):
    self.checkpoint_path = checkpoint_path
    self.batch_size = 1
    self.sequence_length = {'inputs': INPUT_LENGTH, 'targets': TARGET_LENGTH}
    self.partitioner = partitioning.PjitPartitioner(num_partitions=1)
    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.codec = vocabularies.build_codec(
        vocabularies.VocabularyConfig(num_velocity_bins=1))
    self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
    self.output_features = {
        'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        'targets': seqio.Feature(vocabulary=self.vocabulary),
    }
    self.model = models.ContinuousInputsEncoderDecoderModel(
        module=network.Transformer(config=network.T5Config(
            vocab_size=vocabularies.num_embeddings(self.vocabulary),
            **TINY_T5_CONFIG)),
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(self.spectrogram_config))
    self._restore()

  def _restore(self) -> None:
    initializer = utils.TrainStateInitializer(
        optimizer_def=self.model.optimizer_def,
        init_fn=self.model.get_initial_variables,
        input_shapes={
            'encoder_input_tokens': (self.batch_size, INPUT_LENGTH),
            'decoder_input_tokens': (self.batch_size, TARGET_LENGTH),
        },
        partitioner=self.partitioner)
    # t5x.train() defaults to use_orbax=True and resumes through
    # create_checkpoint_manager_and_restore's Orbax branch.
    # TrainStateInitializer.from_checkpoint_or_scratch's direct
    # checkpoints.Checkpointer.restore(path=...) hits the *non*-Orbax legacy
    # branch instead, which expects a flat 'checkpoint' file and cannot read
    # the Orbax-native (OCDBT) layout t5x.train itself writes. Mirror the
    # working (use_orbax=True) path here.
    restore_config = utils.RestoreCheckpointConfig(
        path=str(self.checkpoint_path), mode='latest', dtype='float32')
    train_state_axes = initializer.train_state_axes

    def predict(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})

    self._predict = self.partitioner.partition(
        predict,
        in_axis_resources=(train_state_axes.params,
                           partitioning.PartitionSpec('data',), None),
        out_axis_resources=partitioning.PartitionSpec('data',))
    valid_restore_cfg, restore_paths = (
        utils.get_first_valid_restore_config_and_paths([restore_config]))
    train_state, _ = utils.create_checkpoint_manager_and_restore(
        train_state_initializer=initializer,
        partitioner=self.partitioner,
        restore_checkpoint_cfg=valid_restore_cfg,
        restore_path=restore_paths[0] if restore_paths else None,
        fallback_init_rng=jax.random.PRNGKey(0),
        save_checkpoint_cfg=None,
        model_dir=str(self.checkpoint_path),
        ds_iter=None,
        use_orbax=True)
    self._train_state = train_state or initializer.from_scratch(
        jax.random.PRNGKey(0))

  def _dataset(self, audio: np.ndarray) -> tf.data.Dataset:
    hop = self.spectrogram_config.hop_width
    audio = np.pad(audio, (0, (-len(audio)) % hop), mode='constant')
    frames = spectrograms.split_audio(audio, self.spectrogram_config)
    frame_times = (np.arange(len(audio) // hop)
                  / self.spectrogram_config.frames_per_second)
    dataset = tf.data.Dataset.from_tensors(
        {'inputs': frames, 'input_times': frame_times})
    chain = [
        functools.partial(t5_preprocessors.split_tokens_to_inputs_length,
                          sequence_length=self.sequence_length,
                          output_features=self.output_features,
                          feature_key='inputs',
                          additional_feature_keys=['input_times']),
        preprocessors.add_dummy_targets,
        functools.partial(preprocessors.compute_spectrograms,
                          spectrogram_config=self.spectrogram_config),
    ]
    for preprocessor in chain:
      dataset = preprocessor(dataset)
    return dataset

  def transcribe(self, audio: np.ndarray) -> note_seq.NoteSequence:
    dataset = self._dataset(audio)
    model_dataset = self.model.FEATURE_CONVERTER_CLS(pack=False)(
        dataset, task_feature_lengths=self.sequence_length).batch(self.batch_size)
    predictions = []
    for example, batch in zip(dataset.as_numpy_iterator(),
                              model_dataset.as_numpy_iterator()):
      tokens, _ = self._predict(self._train_state.params, batch, jax.random.PRNGKey(0))
      start_time = example['input_times'][0]
      start_time -= start_time % (1 / self.codec.steps_per_second)
      decoded = self.vocabulary.decode_tf(tokens[0]).numpy()
      if vocabularies.DECODED_EOS_ID in decoded:
        decoded = decoded[:np.argmax(decoded == vocabularies.DECODED_EOS_ID)]
      predictions.append({'est_tokens': np.asarray(decoded, np.int32),
                          'start_time': start_time, 'raw_inputs': []})
    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec,
        encoding_spec=note_sequences.NoteEncodingSpec)
    print(f'  invalid_events={result["est_invalid_events"]} '
          f'dropped_events={result["est_dropped_events"]}')
    return result['est_ns']


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  args = parser.parse_args()

  transcriber = GateTranscriber(Path(args.checkpoint).resolve())

  f1s = []
  for example_id, notes, rhythms in phase0_gate._gate_examples():
    ref_ns, wav_bytes = phase0_gate._build_example(example_id, notes, rhythms)
    audio = note_seq.audio_io.wav_data_to_samples_librosa(
        wav_bytes, sample_rate=SAMPLE_RATE)
    print(f'{example_id}:')
    est_ns = transcriber.transcribe(audio)
    print(f'  ref notes={len(ref_ns.notes)} est notes={len(est_ns.notes)}')
    scores = metrics._program_aware_note_scores(
        ref_ns, est_ns, granularity_type='full')
    f1 = scores['Onset + offset + program F1 (full)']
    print(f'  (program, rhythm) onset+offset F1 = {f1:.3f}')
    f1s.append(f1)

  overall = float(np.mean(f1s))
  print(f'\nMean (program, rhythm) onset+offset F1 over both excerpts: {overall:.3f}')
  print('GATE PASSED' if overall >= 0.95 else 'GATE NOT YET PASSED')
  return 0 if overall >= 0.95 else 1


if __name__ == '__main__':
  raise SystemExit(main())
