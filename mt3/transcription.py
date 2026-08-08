"""Public API for multi-instrument MT3 transcription."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import gin
import jax
import numpy as np
import seqio

from mt3.runtime import tensorflow

tf = tensorflow()

# ``note_seq`` may initialize TensorFlow, so it must follow runtime setup.
import note_seq
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
# Encoder window in spectrogram frames. spectrograms.py fixes a 16 kHz sample
# rate and a hop width of 128, so there are 125 frames per second: the 256
# default is a ~2.048 s window, and 512 is the ~4.096 s window that
# gin/context_4s.gin trains. These must match the checkpoint being loaded --
# a checkpoint adapted to 512 frames will still RESTORE at 256 (the T5X
# relative position bias is sequence-length independent), it will just be
# evaluated outside the context it was tuned for, which reads as a quality
# regression rather than an error.
INPUT_LENGTH = 256
TARGET_LENGTH = 1024


@dataclass(frozen=True)
class TranscriptionResult:
  """Summary of a completed multi-instrument transcription."""

  output_path: Path
  note_count: int
  programs: tuple[int, ...]
  drum_note_count: int
  # Decode diagnostics from metrics_utils.event_predictions_to_ns, summed
  # across all encoder windows. est_dropped_events grows with overlap by
  # design under head-crop (MT3_HEADCROP_OVERLAP_PLAN.md Phase 2): every
  # discarded lookahead tail counts as "dropped", so this is only a
  # meaningful error signal when compared within one hop/window setting.
  est_invalid_events: int = 0
  est_dropped_events: int = 0
  # Resolved encoder window geometry, so an ablation run's own configuration
  # travels with its output rather than living only in the invocation.
  window_frames: int = 0
  hop_frames: int = 0
  lookahead_seconds: float = 0.0
  cost_multiplier: float = 1.0

  def as_dict(self) -> dict[str, object]:
    return {
        'output': str(self.output_path),
        'note_count': self.note_count,
        'programs': list(self.programs),
        'drum_note_count': self.drum_note_count,
        'est_invalid_events': self.est_invalid_events,
        'est_dropped_events': self.est_dropped_events,
        'window_frames': self.window_frames,
        'hop_frames': self.hop_frames,
        'lookahead_seconds': self.lookahead_seconds,
        'cost_multiplier': self.cost_multiplier,
    }


def write_multitrack_midi(note_sequence: note_seq.NoteSequence,
                          output_path: Path) -> int:
  """Writes all predicted MIDI programs and drum events to a standard MIDI file."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  note_seq.sequence_proto_to_midi_file(note_sequence, str(output_path))
  return len(note_sequence.notes)


class Transcriber:
  """A reusable MT3 multi-instrument inference model.

  Construct one instance per checkpoint and call :meth:`transcribe_file` for
  every WAV file.  Keeping the instance alive avoids restoring the checkpoint
  for each request.

  ``input_length`` is the encoder window in spectrogram frames and must match
  the window the checkpoint was trained with (256 for the ~2 s baseline, 512
  for a checkpoint adapted with gin/context_4s.gin).  It is a plain constructor
  argument rather than a gin binding because inference here builds its model
  from gin/model.gin only, which carries architecture but no task lengths.

  ``lookahead_frames`` (MT3_HEADCROP_OVERLAP_PLAN.md) requests that many
  spectrogram frames of right-context lookahead per window, by overlapping
  consecutive windows and discarding the lookahead portion of each window's
  decoded output (see metrics_utils.decode_and_combine_predictions). It is
  expressed in frames rather than seconds because it must divide evenly into
  the checkpoint's own frame grid; convert via
  spectrograms.SpectrogramConfig().frames_per_second (125 at the defaults) to
  go from seconds. 0 (the default) reproduces the original non-overlapping
  baseline exactly.
  """

  def __init__(self,
               checkpoint_path: str | Path,
               input_length: int = INPUT_LENGTH,
               target_length: int = TARGET_LENGTH,
               lookahead_frames: int = 0):
    self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not self.checkpoint_path.exists():
      raise FileNotFoundError(f'MT3 checkpoint does not exist: {self.checkpoint_path}')
    if not 0 <= lookahead_frames < input_length:
      raise ValueError(
          f'lookahead_frames must be in [0, {input_length}); got '
          f'{lookahead_frames}')
    self.batch_size = 1
    self.sequence_length = {'inputs': input_length, 'targets': target_length}
    self.lookahead_frames = lookahead_frames
    self.hop_frames = input_length - lookahead_frames
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
    self._restore()

  def _parse_model_gin(self) -> None:
    model_gin = Path(__file__).resolve().parent / 'gin/model.gin'
    bindings = [
        'from __gin__ import dynamic_registration',
        'from mt3 import vocabularies',
        'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
        'vocabularies.VocabularyConfig.num_velocity_bins=1',
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          [str(model_gin)], bindings, finalize_config=False)

  def _build_model(self):
    model_config = gin.get_configurable(network.T5Config)()
    return models.ContinuousInputsEncoderDecoderModel(
        module=network.Transformer(config=model_config),
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(self.spectrogram_config))

  def _restore(self) -> None:
    initializer = utils.TrainStateInitializer(
        optimizer_def=self.model.optimizer_def,
        init_fn=self.model.get_initial_variables,
        input_shapes={
            'encoder_input_tokens': (self.batch_size, INPUT_LENGTH),
            'decoder_input_tokens': (self.batch_size, TARGET_LENGTH),
        },
        partitioner=self.partitioner)
    restore_config = utils.RestoreCheckpointConfig(
        path=str(self.checkpoint_path), mode='specific', dtype='float32')
    train_state_axes = initializer.train_state_axes

    def predict(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})

    self._predict = self.partitioner.partition(
        predict,
        in_axis_resources=(train_state_axes.params,
                           partitioning.PartitionSpec('data',), None),
        out_axis_resources=partitioning.PartitionSpec('data',))
    # NOT initializer.from_checkpoint_or_scratch: it calls
    # checkpoints.Checkpointer.restore(path=...) directly, the *non*-Orbax
    # legacy branch, which expects a flat 'checkpoint' file and cannot read
    # the Orbax-native (OCDBT) layout t5x.train itself writes -- so this
    # loaded the official released checkpoint fine but broke on any
    # self-trained checkpoint from this fork's own training pipeline.
    # t5x.train() defaults to use_orbax=True and resumes through
    # create_checkpoint_manager_and_restore's Orbax branch, which reads
    # both layouts correctly; mirror that here (same fix already applied
    # to mt3/scripts/run_phase0_gate_eval.py's GateTranscriber).
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
    frame_times = np.arange(len(audio) // hop) / self.spectrogram_config.frames_per_second
    dataset = tf.data.Dataset.from_tensors({'inputs': frames, 'input_times': frame_times})
    chain = [
        functools.partial(preprocessors.split_tokens_strided,
                          window_tokens=self.sequence_length['inputs'],
                          hop_tokens=self.hop_frames,
                          feature_key='inputs',
                          additional_feature_keys=['input_times']),
        preprocessors.add_dummy_targets,
        functools.partial(preprocessors.compute_spectrograms,
                          spectrogram_config=self.spectrogram_config),
    ]
    for preprocessor in chain:
      dataset = preprocessor(dataset)
    return dataset

  def _transcribe_with_diagnostics(self, audio: np.ndarray) -> dict:
    """Runs inference and returns the full event_predictions_to_ns result.

    Includes 'est_ns' plus decode diagnostics ('est_invalid_events',
    'est_dropped_events'). Split out from transcribe() so transcribe_file()
    can populate TranscriptionResult's diagnostic fields without changing
    transcribe()'s public return type -- evaluate_checkpoint.py and other
    scripts call transcribe() expecting a bare NoteSequence.
    """
    dataset = self._dataset(audio)
    model_dataset = self.model.FEATURE_CONVERTER_CLS(pack=False)(
        dataset, task_feature_lengths=self.sequence_length).batch(self.batch_size)
    predictions = []
    for example, batch in zip(dataset.as_numpy_iterator(), model_dataset.as_numpy_iterator()):
      tokens, _ = self._predict(self._train_state.params, batch, jax.random.PRNGKey(0))
      start_time = example['input_times'][0]
      start_time -= start_time % (1 / self.codec.steps_per_second)
      decoded = self.vocabulary.decode_tf(tokens[0]).numpy()
      if vocabularies.DECODED_EOS_ID in decoded:
        decoded = decoded[:np.argmax(decoded == vocabularies.DECODED_EOS_ID)]
      predictions.append({'est_tokens': np.asarray(decoded, np.int32),
                          'start_time': start_time, 'raw_inputs': []})
    return metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec,
        encoding_spec=note_sequences.NoteEncodingWithTiesSpec)

  def transcribe(self, audio: np.ndarray) -> note_seq.NoteSequence:
    return self._transcribe_with_diagnostics(audio)['est_ns']

  def transcribe_file(self, input_path: str | Path,
                      output_path: str | Path) -> TranscriptionResult:
    """Transcribes one WAV file and writes a standard multi-track MIDI file."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
      raise FileNotFoundError(f'input WAV does not exist: {input_path}')
    audio = note_seq.audio_io.wav_data_to_samples_librosa(
        input_path.read_bytes(), sample_rate=SAMPLE_RATE)
    result = self._transcribe_with_diagnostics(audio)
    sequence = result['est_ns']
    note_count = write_multitrack_midi(sequence, output_path)
    window_frames = self.sequence_length['inputs']
    return TranscriptionResult(
        output_path=output_path,
        note_count=note_count,
        programs=tuple(sorted({note.program for note in sequence.notes if not note.is_drum})),
        drum_note_count=sum(note.is_drum for note in sequence.notes),
        est_invalid_events=int(result['est_invalid_events']),
        est_dropped_events=int(result['est_dropped_events']),
        window_frames=window_frames,
        hop_frames=self.hop_frames,
        lookahead_seconds=(
            self.lookahead_frames / self.spectrogram_config.frames_per_second),
        cost_multiplier=window_frames / self.hop_frames)
