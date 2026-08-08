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
           'Must be less than --input-length; cost scales as '
           'input-length / (input-length - lookahead-frames).')
  lookahead.add_argument(
      '--lookahead-seconds', type=float, default=None,
      help='Same as --lookahead-frames, in seconds. Converted at the fixed '
           '125 frames/s the spectrogram front end uses (16 kHz sample rate, '
           '128-sample hop -- see spectrograms.py); truncated to whole '
           'frames.')
  return parser


def main(argv: list[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  # Delay the heavy ML imports until argument parsing has completed.
  from mt3.transcription import Transcriber

  kwargs = {} if args.input_length is None else {'input_length': args.input_length}
  if args.lookahead_frames is not None:
    kwargs['lookahead_frames'] = args.lookahead_frames
  elif args.lookahead_seconds is not None:
    # Matches spectrograms.SpectrogramConfig().frames_per_second at its
    # default sample rate and hop width; see the --lookahead-seconds help.
    kwargs['lookahead_frames'] = int(args.lookahead_seconds * 125)
  result = Transcriber(args.checkpoint, **kwargs).transcribe_file(args.input, args.output)
  if args.json:
    print(json.dumps(result.as_dict(), sort_keys=True))
  else:
    print(f'Wrote {result.output_path}: {result.note_count} predicted notes.')
    print(f'Programs: {list(result.programs)}; drum notes: {result.drum_note_count}.')
  return 0
