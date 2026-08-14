"""Command-line interface for the reusable MT3 inference package."""

from __future__ import annotations

import argparse
import json


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Transcribe a WAV file with multi-instrument MT3.')
  parser.add_argument('--checkpoint', required=True, help='Path to an MT3 checkpoint directory.')
  parser.add_argument('--input', required=True, help='Input WAV file.')
  parser.add_argument('--output', required=True, help='Output MIDI file.')
  parser.add_argument('--json', action='store_true', help='Write the result object as JSON to stdout.')
  parser.add_argument(
      '--input-length', type=int, default=None,
      help='Encoder window in spectrogram frames (125 frames = 1 s). Defaults '
           'to the 256-frame (~2 s) baseline; pass 512 for a checkpoint '
           'adapted to the ~4 s window. Must match how the checkpoint was '
           'trained.')
  lookahead = parser.add_mutually_exclusive_group()
  lookahead.add_argument(
      '--lookahead-frames', type=int, default=None,
      help='MT3_HEADCROP_OVERLAP_PLAN.md: request this many spectrogram '
           'frames of right-context lookahead per window by overlapping '
           'consecutive windows (0, the default, reproduces the baseline). '
           'Combined with --lookback-frames/--lookback-seconds, cost scales '
           'as input-length / (input-length - lookback-frames - '
           'lookahead-frames).')
  lookahead.add_argument(
      '--lookahead-seconds', type=float, default=None,
      help='Same as --lookahead-frames, in seconds. Converted at the fixed '
           '125 frames/s the spectrogram front end uses (16 kHz sample rate, '
           '128-sample hop -- see spectrograms.py); truncated to whole '
           'frames.')
  lookback = parser.add_mutually_exclusive_group()
  lookback.add_argument(
      '--lookback-frames', type=int, default=None,
      help='Same idea as --lookahead-frames, mirrored onto the left of the '
           'window: request this many spectrogram frames of left-context '
           'by overlapping consecutive windows and discarding their '
           'suppressed prefix (0, the default, reproduces the baseline). '
           'lookback-frames + lookahead-frames must leave a positive kept '
           'region within --input-length.')
  lookback.add_argument(
      '--lookback-seconds', type=float, default=None,
      help='Same as --lookback-frames, in seconds. Converted at the fixed '
           '125 frames/s the spectrogram front end uses; truncated to '
           'whole frames.')
  return parser


# 125 frames/s: spectrograms.SpectrogramConfig().frames_per_second at its
# default sample rate (16 kHz) and hop width (128 samples).
_FRAMES_PER_SECOND = 125


def _resolve_transcriber_kwargs(args: argparse.Namespace) -> dict:
  """Converts parsed CLI args into Transcriber() kwargs.

  Pure and import-free (no `mt3.transcription`), so it's testable without
  paying for the heavy ML imports `main()` otherwise delays until after
  argument parsing.
  """
  kwargs = {} if args.input_length is None else {'input_length': args.input_length}
  if args.lookahead_frames is not None:
    kwargs['lookahead_frames'] = args.lookahead_frames
  elif args.lookahead_seconds is not None:
    kwargs['lookahead_frames'] = int(args.lookahead_seconds * _FRAMES_PER_SECOND)
  if args.lookback_frames is not None:
    kwargs['lookback_frames'] = args.lookback_frames
  elif args.lookback_seconds is not None:
    kwargs['lookback_frames'] = int(args.lookback_seconds * _FRAMES_PER_SECOND)
  return kwargs


def main(argv: list[str] | None = None) -> int:
  parser = _parser()
  args = parser.parse_args(argv)
  # Delay the heavy ML imports until argument parsing has completed.
  from mt3 import transcription

  kwargs = _resolve_transcriber_kwargs(args)

  # Validate the resolved geometry ourselves, before Transcriber spends time
  # loading a checkpoint: a bad combination should read as a normal argparse
  # usage error, not a stack trace from deep inside model construction.
  input_length = kwargs.get('input_length', transcription.INPUT_LENGTH)
  try:
    geometry = transcription.WindowGeometry(
        window_frames=input_length,
        lookback_frames=kwargs.get('lookback_frames', 0),
        lookahead_frames=kwargs.get('lookahead_frames', 0))
  except ValueError as e:
    parser.error(str(e))

  result = transcription.Transcriber(
      args.checkpoint, **kwargs).transcribe_file(args.input, args.output)
  if args.json:
    print(json.dumps(result.as_dict(), sort_keys=True))
  else:
    print(f'Wrote {result.output_path}: {result.note_count} predicted notes.')
    print(f'Programs: {list(result.programs)}; drum notes: {result.drum_note_count}.')
    # From `result`, the geometry Transcriber actually ran with -- not the
    # `geometry` object above, which exists only for pre-flight validation
    # before a checkpoint is loaded. The two happen to agree today (both
    # derive from the same kwargs via the same WindowGeometry constructor),
    # but reading the executed result is the one that can't drift from what
    # actually ran.
    lookback_frames = round(result.lookback_seconds * _FRAMES_PER_SECOND)
    window_seconds = result.window_frames / _FRAMES_PER_SECOND
    print(f'window {result.window_frames}f ({window_seconds:.2f}s) = '
          f'{lookback_frames}f lookback + {result.keep_frames}f keep + '
          f'{round(result.lookahead_seconds * _FRAMES_PER_SECOND)}f lookahead, '
          f'cost {result.cost_multiplier:.2f}x')
  return 0
