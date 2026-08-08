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
  return parser


def main(argv: list[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  # Delay the heavy ML imports until argument parsing has completed.
  from mt3.transcription import Transcriber

  kwargs = {} if args.input_length is None else {'input_length': args.input_length}
  result = Transcriber(args.checkpoint, **kwargs).transcribe_file(args.input, args.output)
  if args.json:
    print(json.dumps(result.as_dict(), sort_keys=True))
  else:
    print(f'Wrote {result.output_path}: {result.note_count} predicted notes.')
    print(f'Programs: {list(result.programs)}; drum notes: {result.drum_note_count}.')
  return 0
