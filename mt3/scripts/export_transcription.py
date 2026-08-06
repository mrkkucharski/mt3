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

"""Transcribe a WAV file and export it as a reviewable REAPER project.

One-command Phase 1 "export" entry point
(`MT3_guitar_transcription_plan.md`, "Local training and deliverables").
Fuses the two steps every fine-tune leg has run by hand in sequence so far
(see `PROJECT_LOG.md`, e.g. "First fine-tune on the new corpus"):
`mt3-transcribe` (raw multitrack MIDI from a checkpoint), then
`midi2reaper build` (a REAPER project with real instrument chains, for
listening). The two tools live in separate uv-managed environments -- this
fork's own TF/JAX-heavy one and midi2reaper's lightweight one -- and share no
code, so the second step runs as a subprocess against midi2reaper's own venv
binary rather than an in-process import.

Run from the MT3 repository, with `midi2reaper` checked out as a sibling
directory (see its own README):

  uv run python -m mt3.scripts.export_transcription \
      --checkpoint runs/guitar_pilot_v1/checkpoint_1002000 \
      --input ../reaper/instrumental.wav \
      --out-dir ../runs/instrumental_transcriptions
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def find_midi2reaper_bin(start: Path) -> Path:
  """midi2reaper's own venv binary, located relative to the shared fork
  root -- the same way `modal_train.py`'s `_find_fork_root` locates this
  fork's own root -- rather than a hardcoded absolute path that would only
  work on one machine."""
  for candidate in (start, *start.parents):
    sibling = candidate / "midi2reaper" / ".venv" / "bin" / "midi2reaper"
    if sibling.is_file():
      return sibling
  raise FileNotFoundError(
      f"no midi2reaper/.venv/bin/midi2reaper found above {start} -- set up "
      "midi2reaper (see its own README) as a sibling checkout")


def export_transcription(
    checkpoint: Path, input_wav: Path, out_dir: Path, *, force: bool = True,
) -> Path:
  """Transcribes `input_wav` with `checkpoint` and builds a REAPER project
  from the result. Returns the written .RPP path."""
  # Delayed: pulls in JAX/T5X/TensorFlow, which argument parsing (and
  # --help) should not pay for.
  from mt3.transcription import Transcriber

  midi2reaper = find_midi2reaper_bin(Path(__file__).resolve())

  out_dir.mkdir(parents=True, exist_ok=True)
  raw_midi = out_dir / f"{input_wav.stem}.mid"
  result = Transcriber(checkpoint).transcribe_file(input_wav, raw_midi)
  print(f"Transcribed {result.note_count} notes ({len(result.programs)} programs, "
        f"{result.drum_note_count} drum notes) to {raw_midi}")

  argv = [str(midi2reaper), "build", str(raw_midi), "-o", str(out_dir)]
  if force:
    argv.append("-f")
  subprocess.run(argv, check=True)

  # `midi2reaper build` exits 0 even when every input is rejected (e.g. no
  # guitar part) or left untouched by an existing project without --force --
  # its own report is the only place that shows up, not the exit code. This
  # command's whole point is producing that .RPP, so confirm it exists
  # rather than trusting a clean exit code silently.
  rpp_path = out_dir / f"{raw_midi.stem}.RPP"
  if not rpp_path.is_file():
    raise FileNotFoundError(
        f"midi2reaper build exited cleanly but did not write {rpp_path} -- "
        "see its own output above for a REJECT/EXISTS reason")
  return rpp_path


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path,
                      help="Path to an MT3 checkpoint directory.")
  parser.add_argument("--input", required=True, type=Path, help="Input WAV file.")
  parser.add_argument("--out-dir", required=True, type=Path,
                      help="Directory for the transcribed MIDI and REAPER project.")
  parser.add_argument("--no-force", action="store_true",
                      help="don't overwrite an existing REAPER project of the same name")
  args = parser.parse_args(argv)

  try:
    rpp_path = export_transcription(
        args.checkpoint.expanduser().resolve(), args.input.expanduser().resolve(),
        args.out_dir.expanduser().resolve(), force=not args.no_force)
  except (FileNotFoundError, subprocess.CalledProcessError) as error:
    print(f"ERROR: {error}", file=sys.stderr)
    return 1

  print(f"Wrote {rpp_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
