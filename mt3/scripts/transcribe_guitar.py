"""Transcribe one WAV file with a fine-tuned MT3 guitar checkpoint.

Run from the MT3 repository:

  uv run python mt3/scripts/transcribe_guitar.py \
      --checkpoint ../runs/guitar_pilot_finetune/checkpoint_100 \
      --input /path/to/song.wav \
      --output /path/to/song_transcription.mid

The output always contains the four named guitar lanes. Notes predicted with
programs outside the lane mapping are omitted.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import gin
import jax
import mido
import numpy as np
import seqio
import tensorflow as tf

# TensorFlow 2.20's tf.data meta-optimizer is incompatible with the MT3/SeqIO
# inference pipeline on the project's Apple-Silicon environment. This must be
# configured before importing note_seq, which can initialize TensorFlow.
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

import note_seq
from t5.data import preprocessors as t5_preprocessors
from t5x import adafactor
from t5x import partitioning
from t5x import utils

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies


SAMPLE_RATE = 16000
INPUT_LENGTH = 256
TARGET_LENGTH = 1024

LANES = (
    ('clean-rhythm', 26),
    ('clean-lead', 27),
    ('distorted-rhythm', 29),
    ('distorted-lead', 30),
)


def _trim_eos(tokens: np.ndarray) -> np.ndarray:
  tokens = np.asarray(tokens, np.int32)
  if vocabularies.DECODED_EOS_ID in tokens:
    tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
  return tokens


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo: int) -> int:
  return max(0, round(seconds * ticks_per_beat * 1_000_000 / tempo))


def write_guitar_midi(note_sequence: note_seq.NoteSequence, output_path: Path) -> dict[str, int]:
  """Writes a NoteSequence to four fixed named MIDI lane tracks."""
  midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
  tempo = 500000  # 120 BPM; event seconds are preserved by tick conversion.
  conductor = mido.MidiTrack()
  conductor.append(mido.MetaMessage('track_name', name='guitar-transcription', time=0))
  conductor.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
  midi_file.tracks.append(conductor)

  counts = {}
  for channel, (lane_name, program) in enumerate(LANES):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=lane_name, time=0))
    track.append(mido.Message('program_change', channel=channel, program=program, time=0))
    lane_notes = [note for note in note_sequence.notes
                  if not note.is_drum and note.program == program
                  and note.end_time > note.start_time]
    events = []
    for note in lane_notes:
      start = _seconds_to_ticks(note.start_time, midi_file.ticks_per_beat, tempo)
      end = _seconds_to_ticks(note.end_time, midi_file.ticks_per_beat, tempo)
      end = max(end, start + 1)
      # Note-offs precede note-ons at the same tick.
      events.append((start, 1, 'on', note.pitch, note.velocity))
      events.append((end, 0, 'off', note.pitch, 0))
    previous_tick = 0
    for tick, _, kind, pitch, velocity in sorted(events):
      message_type = 'note_on' if kind == 'on' else 'note_off'
      track.append(mido.Message(
          message_type,
          channel=channel,
          note=int(pitch),
          velocity=max(0, min(127, int(velocity))),
          time=tick - previous_tick))
      previous_tick = tick
    track.append(mido.MetaMessage('end_of_track', time=0))
    midi_file.tracks.append(track)
    counts[lane_name] = len(lane_notes)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  midi_file.save(output_path)
  return counts


class GuitarInferenceModel:
  """Minimal local MT3 inference wrapper for the guitar fine-tuning model."""

  def __init__(self, checkpoint_path: Path):
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
    self._parse_model_gin()
    self.model = self._build_model()
    self._restore(checkpoint_path)

  def _parse_model_gin(self) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bindings = [
        'from __gin__ import dynamic_registration',
        'from mt3 import vocabularies',
        'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
        'vocabularies.VocabularyConfig.num_velocity_bins=1',
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          [str(repo_root / 'mt3/gin/model.gin')], bindings,
          finalize_config=False)

  def _build_model(self):
    model_config = gin.get_configurable(network.T5Config)()
    return models.ContinuousInputsEncoderDecoderModel(
        module=network.Transformer(config=model_config),
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(self.spectrogram_config))

  def _restore(self, checkpoint_path: Path) -> None:
    # Legacy T5X checkpoint shard metadata is resolved by TensorStore and
    # requires an absolute base path on current macOS TensorStore builds.
    checkpoint_path = checkpoint_path.resolve()
    initializer = utils.TrainStateInitializer(
        optimizer_def=self.model.optimizer_def,
        init_fn=self.model.get_initial_variables,
        input_shapes={
            'encoder_input_tokens': (self.batch_size, INPUT_LENGTH),
            'decoder_input_tokens': (self.batch_size, TARGET_LENGTH),
        },
        partitioner=self.partitioner)
    restore_config = utils.RestoreCheckpointConfig(
        path=str(checkpoint_path), mode='specific', dtype='float32')
    train_state_axes = initializer.train_state_axes

    def predict(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})

    self._predict = self.partitioner.partition(
        predict,
        in_axis_resources=(train_state_axes.params,
                           partitioning.PartitionSpec('data',), None),
        out_axis_resources=partitioning.PartitionSpec('data',))
    self._train_state = initializer.from_checkpoint_or_scratch(
        [restore_config], init_rng=jax.random.PRNGKey(0))

  def _dataset(self, audio: np.ndarray) -> tf.data.Dataset:
    hop = self.spectrogram_config.hop_width
    padding = (-len(audio)) % hop
    audio = np.pad(audio, (0, padding), mode='constant')
    frames = spectrograms.split_audio(audio, self.spectrogram_config)
    frame_times = np.arange(len(audio) // hop) / self.spectrogram_config.frames_per_second
    dataset = tf.data.Dataset.from_tensors({'inputs': frames, 'input_times': frame_times})
    chain = [
        functools.partial(
            t5_preprocessors.split_tokens_to_inputs_length,
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
    for example, batch in zip(dataset.as_numpy_iterator(), model_dataset.as_numpy_iterator()):
      tokens, _ = self._predict(self._train_state.params, batch, jax.random.PRNGKey(0))
      start_time = example['input_times'][0]
      start_time -= start_time % (1 / self.codec.steps_per_second)
      predictions.append({
          'est_tokens': _trim_eos(self.vocabulary.decode_tf(tokens[0]).numpy()),
          'start_time': start_time,
          'raw_inputs': [],
      })
    return metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec,
        encoding_spec=note_sequences.NoteEncodingWithTiesSpec)['est_ns']


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True, type=Path)
  parser.add_argument('--input', required=True, type=Path, help='Input WAV file.')
  parser.add_argument('--output', required=True, type=Path, help='Output four-lane MIDI file.')
  args = parser.parse_args(argv)
  if not args.input.is_file():
    parser.error(f'input WAV does not exist: {args.input}')
  audio = note_seq.audio_io.wav_data_to_samples_librosa(
      args.input.read_bytes(), sample_rate=SAMPLE_RATE)
  model = GuitarInferenceModel(args.checkpoint)
  sequence = model.transcribe(audio)
  counts = write_guitar_midi(sequence, args.output)
  retained = sum(counts.values())
  discarded = sum(1 for note in sequence.notes
                  if not note.is_drum and note.program not in dict(LANES))
  print(f'Wrote {args.output}: {retained} guitar-lane notes.')
  print('Lane counts: ' + ', '.join(f'{name}={count}' for name, count in counts.items()))
  if discarded:
    print(f'Discarded {discarded} non-guitar-program prediction(s).')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
