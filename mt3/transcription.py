"""Public API for multi-instrument MT3 transcription."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
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
class WindowGeometry:
  """Describes how an encoder window splits into [lookback | keep | lookahead].

  ``keep`` (== ``hop_frames``) is the region whose decoded output is
  actually kept; ``lookback``/``lookahead`` are overlap-only context
  discarded after decoding (see MT3_HEADCROP_OVERLAP_PLAN.md for the
  lookahead half of this; lookback is its mirror on the other side of the
  window).

  ``lookback_frames=0, lookahead_frames=0`` reproduces the original
  non-overlapping baseline exactly: ``keep_frames == window_frames`` and
  ``cost_multiplier == 1.0``.

  Frames are the primitive; seconds are derived via ``.seconds()`` using
  ``spectrograms.SpectrogramConfig.frames_per_second`` (125 at the fixed
  16 kHz / 128-sample-hop defaults). Converting a requested seconds value
  into frames is the caller's job (e.g. the CLI), since only the caller
  knows whether truncation to the frame grid is acceptable.
  """
  window_frames: int
  lookback_frames: int = 0
  lookahead_frames: int = 0

  def __post_init__(self):
    if self.lookback_frames < 0:
      raise ValueError(
          f'lookback_frames must be >= 0; got {self.lookback_frames}')
    if self.lookahead_frames < 0:
      raise ValueError(
          f'lookahead_frames must be >= 0; got {self.lookahead_frames}')
    if self.lookback_frames + self.lookahead_frames >= self.window_frames:
      raise ValueError(
          f'window_frames={self.window_frames} leaves no kept region: '
          f'lookback_frames={self.lookback_frames} + '
          f'lookahead_frames={self.lookahead_frames} >= '
          f'{self.window_frames}')

  @property
  def keep_frames(self) -> int:
    return self.window_frames - self.lookback_frames - self.lookahead_frames

  @property
  def hop_frames(self) -> int:
    return self.keep_frames

  @property
  def cost_multiplier(self) -> float:
    return self.window_frames / self.keep_frames

  def seconds(self, frames: int, spectrogram_config) -> float:
    return frames / spectrogram_config.frames_per_second


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
  # est_suppressed_events is lookback's analogue: every event discarded
  # from a window's suppressed (lookback) prefix counts here, so it is
  # similarly only meaningful within one geometry, and is 0 whenever
  # lookback_seconds is 0.
  est_invalid_events: int = 0
  est_dropped_events: int = 0
  est_suppressed_events: int = 0
  # Resolved encoder window geometry, so an ablation run's own configuration
  # travels with its output rather than living only in the invocation.
  window_frames: int = 0
  # keep_frames is the unambiguous name (window_frames - lookback_frames -
  # lookahead_frames); hop_frames is kept alongside it for continuity with
  # callers that predate lookback, where the two were always equal.
  keep_frames: int = 0
  hop_frames: int = 0
  lookback_seconds: float = 0.0
  lookahead_seconds: float = 0.0
  cost_multiplier: float = 1.0
  # GM program every note was forced onto, or None if program tokens were
  # decoded normally (see Transcriber's force_program).
  force_program: int | None = None
  # whether rhythm tokens were ignored, collapsing the rhythm/lead split
  # (see Transcriber's no_rhythm). Always true when force_program is set,
  # which implies it, and when rhythm_vocab is False.
  no_rhythm: bool = False
  # whether the checkpoint's vocabulary carries the rhythm/lead flag at all
  # (see Transcriber's rhythm_vocab). False means the model was trained with
  # gin/no_rhythm.gin and cannot emit rhythm tokens in the first place --
  # a different thing from decoding a rhythm-aware model with no_rhythm.
  rhythm_vocab: bool = True

  def as_dict(self) -> dict[str, object]:
    return {
        'output': str(self.output_path),
        'note_count': self.note_count,
        'programs': list(self.programs),
        'drum_note_count': self.drum_note_count,
        'est_invalid_events': self.est_invalid_events,
        'est_dropped_events': self.est_dropped_events,
        'est_suppressed_events': self.est_suppressed_events,
        'window_frames': self.window_frames,
        'keep_frames': self.keep_frames,
        'hop_frames': self.hop_frames,
        'lookback_seconds': self.lookback_seconds,
        'lookahead_seconds': self.lookahead_seconds,
        'cost_multiplier': self.cost_multiplier,
        'force_program': self.force_program,
        'no_rhythm': self.no_rhythm,
        'rhythm_vocab': self.rhythm_vocab,
    }


def write_multitrack_midi(note_sequence: note_seq.NoteSequence,
                          output_path: Path) -> int:
  """Writes all predicted MIDI programs and drum events to a standard MIDI file."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  note_seq.sequence_proto_to_midi_file(note_sequence, str(output_path))
  return len(note_sequence.notes)


def _windowed_input_dataset(
    audio: np.ndarray,
    spectrogram_config,
    geometry: 'WindowGeometry',
) -> tf.data.Dataset:
  """Frames `audio` and splits it into overlapping encoder windows per `geometry`.

  Window k covers real audio starting at frame `k * geometry.keep_frames -
  geometry.lookback_frames` (relative to the start of `audio`); to give
  window 0 a full `lookback_frames` of left context without reusing any
  real audio twice, `audio` is left-padded with that many frames of silence
  before framing. `input_times` stays in ORIGINAL-audio time throughout --
  the pad occupies frames `[-lookback_frames, 0)` -- rather than shifting
  the whole timeline, so no downstream consumer needs to know the pad
  happened.

  Left-padding grows the total frame count, which can produce trailing
  windows whose *kept* region starts at or past the end of the real audio
  (pure silence, contributing nothing but wasted compute and a risk of
  hallucinated events). Those are dropped here.

  Args:
    audio: mono audio samples.
    spectrogram_config: a spectrograms.SpectrogramConfig.
    geometry: the WindowGeometry describing the encoder window.

  Returns:
    A dataset of examples with 'inputs' (raw audio frames) and
    'input_times' (original-audio-time timestamp of each frame),
    windowed per `geometry`.
  """
  hop = spectrogram_config.hop_width
  # Right-pad to a whole number of frames first, exactly as before this
  # windowing became overlap-aware -- this only rounds the tail up to the
  # frame grid, it is not the lookback/lookahead overlap padding.
  audio = np.pad(audio, (0, (-len(audio)) % hop), mode='constant')
  num_orig_frames = len(audio) // hop

  lookback = geometry.lookback_frames
  if lookback:
    audio = np.pad(audio, (lookback * hop, 0), mode='constant')

  frames = spectrograms.split_audio(audio, spectrogram_config)
  frame_times = (
      (np.arange(len(audio) // hop) - lookback)
      / spectrogram_config.frames_per_second)
  dataset = tf.data.Dataset.from_tensors(
      {'inputs': frames, 'input_times': frame_times})
  dataset = preprocessors.split_tokens_strided(
      dataset,
      window_tokens=geometry.window_frames,
      hop_tokens=geometry.keep_frames,
      feature_key='inputs',
      additional_feature_keys=['input_times'])

  if lookback:
    lookback_seconds = lookback / spectrogram_config.frames_per_second
    orig_duration_seconds = num_orig_frames / spectrogram_config.frames_per_second
    dataset = dataset.filter(
        lambda x: x['input_times'][0] + lookback_seconds < orig_duration_seconds)

  return dataset


def _quantize(t: float, codec) -> float:
  """Floors `t` onto the codec's own step grid (multiples of 1/steps_per_second).

  Computed via floor(t * steps_per_second) / steps_per_second, not the
  more obvious `t - t % step`: floating-point modulo of `t` against a step
  like 0.01 (not exactly representable in binary) routinely lands a hair
  below zero for a `t` that is decimally an exact multiple -- e.g.
  `5.0 % 0.01` is ~0.00999999999999990, not 0.0 -- which then floors a
  whole extra step, e.g. 5.0 down to 4.99. A tiny epsilon before flooring
  absorbs that noise while being far too small to affect any genuine
  non-boundary `t` (codec steps are >= 0.01s, 7+ orders of magnitude
  larger than the epsilon).
  """
  steps = math.floor(t * codec.steps_per_second + 1e-9)
  return steps / codec.steps_per_second


def _min_decode_time(start_time: float, geometry: WindowGeometry, codec,
                     spectrogram_config) -> float | None:
  """A window's kept-region start, quantized onto the same codec step grid as `start_time`.

  `start_time` must already be quantized (see `_quantize`) -- both are
  quantized against the same grid so a boundary event can't fall through
  the crack between this window's own max_decode_time (the next window's
  min_decode_time) and this value.

  Returns None when `geometry.lookback_frames` is 0, rather than
  `start_time` itself: passing decode_events a non-None `min_time` (even
  one that can never actually suppress anything, since a segment's own
  cur_time can never fall below its own start_time) switches its
  max_time boundary test from the legacy inclusive `cur_time > max_time`
  to the half-open `cur_time >= max_time` -- silently dropping an event
  that lands exactly on a window boundary, which is common rather than
  rare once both sides are quantized onto the same codec step grid. None
  preserves the exact legacy comparison, which is the whole point of the
  lookback_frames=0 compatibility anchor.
  """
  if not geometry.lookback_frames:
    return None
  lookback_seconds = geometry.seconds(geometry.lookback_frames, spectrogram_config)
  return _quantize(start_time + lookback_seconds, codec)


def _should_cap_last_segment_tail(geometry: WindowGeometry) -> bool:
  """Whether the last segment's tail should be capped at the real audio duration.

  Only when overlap is actually in use: capping is new behavior relative
  to the pre-overlap baseline (previously the last segment's
  max_decode_time was always unbounded), and the lookback_frames=0,
  lookahead_frames=0 compatibility anchor must reproduce that baseline
  exactly, not this improvement.
  """
  return bool(geometry.lookback_frames or geometry.lookahead_frames)


def _cap_last_segment_tail(predictions: list[dict], audio_duration_seconds: float) -> None:
  """Caps the chronologically-last prediction's decode window at the real end of the audio.

  Mutates `predictions` in place. Without this, the last segment's
  max_decode_time stays unbounded and a lookahead tail can decode events
  from the padding-only silence past the end of the file. No-op on an
  empty list.

  The cap is nudged a hair past audio_duration_seconds rather than sitting
  exactly on it. If this segment also has its own min_decode_time (true
  whenever lookback_frames > 0), run_length_encoding.decode_events treats
  max_time as the exclusive end of a half-open [min_time, max_time)
  interval -- correct when max_time is the *next* segment's own
  min_decode_time (that segment's min_time picks up anything at the exact
  boundary instead), but there is no next segment here to do that. Without
  the nudge, a real note ending at exactly the last frame of the file --
  entirely plausible, not an edge case a real recording avoids -- would be
  silently dropped instead of kept. The nudge is far smaller than a single
  codec step, so it can't admit a genuinely hallucinated event from the
  padding tail, which would need at least one full step past
  audio_duration_seconds to be encoded at all.
  """
  if not predictions:
    return
  last = max(predictions, key=lambda pred: pred['start_time'])
  last['max_decode_time'] = audio_duration_seconds + 1e-6


class Transcriber:
  """A reusable MT3 multi-instrument inference model.

  Construct one instance per checkpoint and call :meth:`transcribe_file` for
  every WAV file.  Keeping the instance alive avoids restoring the checkpoint
  for each request.

  ``rhythm_vocab`` likewise must match the checkpoint. It defaults to False,
  because rhythm-free checkpoints are the project default; pass True for a
  checkpoint deliberately trained with the optional rhythm/lead flag. Because
  that range is last, every other event keeps its token ids and the padded
  vocabulary size is unchanged, so the mismatch is silent rather than a shape
  error -- a rhythm-aware checkpoint decoded with ``rhythm_vocab=False``
  would turn each of its rhythm tokens into an invalid event. Use
  ``no_rhythm`` (not this) to suppress the rhythm/lead split of a
  rhythm-aware checkpoint.

  ``input_length`` is the encoder window in spectrogram frames and must match
  the window the checkpoint was trained with (256 for the ~2 s baseline, 512
  for a checkpoint adapted with gin/context_4s.gin).  It is a plain constructor
  argument rather than a gin binding because inference here builds its model
  from gin/model.gin only, which carries architecture but no task lengths.

  Each encoder window splits into three parts, ``[lookback | keep |
  lookahead]`` (see WindowGeometry): ``keep`` is the region whose decoded
  output actually ends up in the transcription; ``lookback_frames`` and
  ``lookahead_frames`` are extra context the encoder sees but whose own
  decoded output is discarded, by overlapping consecutive windows
  (MT3_HEADCROP_OVERLAP_PLAN.md documents the lookahead half of this in
  detail; lookback is its mirror on the other side of the window --
  ``_windowed_input_dataset`` left-pads the audio so window 0 gets real
  silence rather than reused content, and
  metrics_utils.decode_and_combine_predictions crops each window's decoded
  output to ``[this window's kept-region start, the next window's
  kept-region start)``). Both are expressed in frames rather than seconds
  because they must divide evenly into the checkpoint's own frame grid;
  convert via spectrograms.SpectrogramConfig().frames_per_second (125 at
  the defaults) to go from seconds. 0 for both (the default) reproduces
  the original non-overlapping baseline exactly -- this is the
  compatibility anchor the whole feature leans on, and is covered by its
  own regression tests.
  """

  def __init__(self,
               checkpoint_path: str | Path,
               input_length: int = INPUT_LENGTH,
               target_length: int = TARGET_LENGTH,
               lookahead_frames: int = 0,
               lookback_frames: int = 0,
               force_program: int | None = None,
               no_rhythm: bool = False,
               rhythm_vocab: bool = False):
    self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not self.checkpoint_path.exists():
      raise FileNotFoundError(f'MT3 checkpoint does not exist: {self.checkpoint_path}')
    self.geometry = WindowGeometry(
        window_frames=input_length,
        lookback_frames=lookback_frames,
        lookahead_frames=lookahead_frames)
    if force_program is not None and not (
        note_seq.MIN_MIDI_PROGRAM <= force_program <= note_seq.MAX_MIDI_PROGRAM):
      raise ValueError(
          f'force_program must be in [{note_seq.MIN_MIDI_PROGRAM}, '
          f'{note_seq.MAX_MIDI_PROGRAM}]; got {force_program}')
    self.force_program = force_program
    # force_program implies it (NoteDecodingState.__post_init__ does the same
    # for the decoding state), so the recorded/reported value matches what
    # decoding actually did rather than what was passed in. A checkpoint whose
    # vocabulary has no rhythm range implies it too, trivially: there are no
    # rhythm tokens for the model to emit.
    self.rhythm_vocab = rhythm_vocab
    self.no_rhythm = no_rhythm or force_program is not None or not rhythm_vocab
    self.batch_size = 1
    self.sequence_length = {'inputs': input_length, 'targets': target_length}
    self.lookahead_frames = lookahead_frames
    self.hop_frames = self.geometry.hop_frames
    self.partitioner = partitioning.PjitPartitioner(num_partitions=1)
    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.codec = vocabularies.build_codec(
        vocabularies.VocabularyConfig(num_velocity_bins=1,
                                      include_rhythm=rhythm_vocab))
    self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
    self.output_features = {
        'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        'targets': seqio.Feature(vocabulary=self.vocabulary),
    }
    self._encoding_spec = self._build_encoding_spec()
    self._parse_model_gin()
    self.model = self._build_model()
    self._restore()

  def _build_encoding_spec(self) -> note_sequences.NoteEncodingSpecType:
    """The decoding spec to use, honouring force_program/no_rhythm if set.

    NoteEncodingWithTiesSpec's init_decoding_state_fn is ordinarily just the
    NoteDecodingState class; when either pin is requested, it's replaced with
    a partial that applies the pin to every fresh decoding state (including
    the per-window lookback shadow state NoteDecodingState creates
    internally), so every decoded note lands under the one program and/or
    the non-rhythm role regardless of what tokens the model emits.
    """
    if self.force_program is None and not self.no_rhythm:
      return note_sequences.NoteEncodingWithTiesSpec
    return replace(
        note_sequences.NoteEncodingWithTiesSpec,
        init_decoding_state_fn=functools.partial(
            note_sequences.NoteDecodingState,
            force_program=self.force_program,
            no_rhythm=self.no_rhythm))

  def _parse_model_gin(self) -> None:
    model_gin = Path(__file__).resolve().parent / 'gin/model.gin'
    bindings = [
        'from __gin__ import dynamic_registration',
        'from mt3 import vocabularies',
        'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
        'vocabularies.VocabularyConfig.num_velocity_bins=1',
        # Must agree with self.codec: model.gin builds its own codec from
        # these bindings to size the output vocabulary.
        f'vocabularies.VocabularyConfig.include_rhythm={self.rhythm_vocab}',
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
    dataset = _windowed_input_dataset(audio, self.spectrogram_config, self.geometry)
    chain = [
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
      start_time = _quantize(example['input_times'][0], self.codec)
      decoded = self.vocabulary.decode_tf(tokens[0]).numpy()
      if vocabularies.DECODED_EOS_ID in decoded:
        decoded = decoded[:np.argmax(decoded == vocabularies.DECODED_EOS_ID)]
      predictions.append({
          'est_tokens': np.asarray(decoded, np.int32),
          'start_time': start_time,
          'min_decode_time': _min_decode_time(
              start_time, self.geometry, self.codec, self.spectrogram_config),
          'raw_inputs': [],
      })
    if _should_cap_last_segment_tail(self.geometry):
      # _windowed_input_dataset() left-pads for lookback but never
      # right-pads beyond a whole frame, so this cap is exact, not an
      # approximation.
      _cap_last_segment_tail(
          predictions, len(audio) / self.spectrogram_config.sample_rate)
    return metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec,
        encoding_spec=self._encoding_spec)

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
        est_suppressed_events=int(result['est_suppressed_events']),
        window_frames=window_frames,
        keep_frames=self.geometry.keep_frames,
        hop_frames=self.hop_frames,
        lookback_seconds=self.geometry.seconds(
            self.geometry.lookback_frames, self.spectrogram_config),
        lookahead_seconds=self.geometry.seconds(
            self.geometry.lookahead_frames, self.spectrogram_config),
        cost_multiplier=window_frames / self.hop_frames,
        force_program=self.force_program,
        no_rhythm=self.no_rhythm,
        rhythm_vocab=self.rhythm_vocab)
