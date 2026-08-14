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

"""Manual smoke test for the lookback feature against a real checkpoint.

The unit and end-to-end tests (transcription_test.py, lookback_e2e_test.py)
prove the segmentation/decode plumbing is correct using synthetic inputs and
a perfect simulated model -- they cannot catch output that is well-formed
but musically wrong, which needs a real model and, ultimately, listening to
the result. This script is the documented one-liner for that: it runs one
real WAV file through all four geometries named in the original feature
request and reports note counts and decode diagnostics side by side, so a
regression or an implausible geometry is visible at a glance before
spending time listening.

This script is not run in CI (it needs a real checkpoint and WAV file) --
run it manually after any change that touches the lookback decode path, and
actually listen to at least the best- and worst-looking outputs; per
CLAUDE.md, F1 and by-ear quality can genuinely disagree in this project.

Usage:
  uv run python -m mt3.scripts.lookback_smoke_test \
      --checkpoint /path/to/checkpoint \
      --input /path/to/audio.wav \
      --output-dir /tmp/lookback_smoke

Add --input-length 512 for a checkpoint adapted to the ~4s window
(gin/context_4s.gin); the four geometries below scale with it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _geometries(input_length: int) -> dict[str, tuple[int, int]]:
  """(lookback_frames, lookahead_frames) for the four geometries from the original feature request.

  Scaled to `input_length` at the fixed 125 frames/s (so e.g. "1s/2s/1s"
  becomes exactly a quarter of input_length on each overlap side for the
  256-frame/~2s default, and matches the literal 125/250/125-frame split
  at the 512-frame/~4s window).
  """
  quarter = input_length // 4
  return {
      '0s/window/0s (baseline, no overlap)': (0, 0),
      '1x/2x/1x (symmetric overlap)': (quarter, quarter),
      '1.5x/1x/1.5x (overlap > keep)': (quarter + quarter // 2, quarter + quarter // 2),
      '1x/2.5x/0.5x (asymmetric)': (quarter, quarter // 2),
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True, help='Path to an MT3 checkpoint directory.')
  parser.add_argument('--input', required=True, help='Input WAV file.')
  parser.add_argument('--output-dir', required=True, help='Directory to write one MIDI file per geometry.')
  parser.add_argument('--input-length', type=int, default=None,
                      help='Encoder window in spectrogram frames. Defaults to the '
                           '256-frame (~2s) baseline; pass 512 for a checkpoint '
                           'adapted to the ~4s window. Must match how the '
                           'checkpoint was trained.')
  args = parser.parse_args()

  # Delay the heavy ML imports until argument parsing has completed.
  from mt3.transcription import INPUT_LENGTH, Transcriber

  input_length = args.input_length or INPUT_LENGTH
  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  rows = []
  for name, (lookback_frames, lookahead_frames) in _geometries(input_length).items():
    output_path = output_dir / (name.split(' ')[0].replace('/', '_') + '.mid')
    start = time.time()
    transcriber = Transcriber(
        args.checkpoint, input_length=input_length,
        lookback_frames=lookback_frames, lookahead_frames=lookahead_frames)
    result = transcriber.transcribe_file(args.input, output_path)
    elapsed = time.time() - start
    rows.append((name, result, elapsed))

  header = (f'{"geometry":<32} {"notes":>6} {"invalid":>8} {"dropped":>8} '
           f'{"suppressed":>10} {"cost":>6} {"seconds":>8}')
  print(header)
  print('-' * len(header))
  for name, result, elapsed in rows:
    print(f'{name:<32} {result.note_count:>6} {result.est_invalid_events:>8} '
          f'{result.est_dropped_events:>8} {result.est_suppressed_events:>10} '
          f'{result.cost_multiplier:>5.2f}x {elapsed:>7.1f}s')
  print()
  print(f'MIDI files written to {output_dir}')
  print('Listen to at least the best- and worst-looking rows above before '
        'drawing any conclusion from note counts alone.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
