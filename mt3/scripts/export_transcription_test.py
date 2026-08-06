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

"""Tests for export_transcription."""

from pathlib import Path
import subprocess

import note_seq
import pytest

from mt3.scripts import export_transcription


class _FakeTranscriber:
  """Stands in for `mt3.transcription.Transcriber`: no checkpoint restore,
  no JAX inference -- just writes a fixed one-note MIDI file."""

  def __init__(self, checkpoint_path):
    self.checkpoint_path = checkpoint_path

  def transcribe_file(self, input_path, output_path):
    from mt3.transcription import TranscriptionResult, write_multitrack_midi

    sequence = note_seq.NoteSequence()
    note = sequence.notes.add()
    note.start_time, note.end_time = 0.0, 0.5
    note.pitch, note.velocity, note.program = 40, 100, 30
    note_count = write_multitrack_midi(sequence, Path(output_path))
    return TranscriptionResult(
        output_path=Path(output_path), note_count=note_count, programs=(30,), drum_note_count=0)


def test_find_midi2reaper_bin_walks_up_to_a_sibling_checkout(tmp_path):
  fork_root = tmp_path / "transcription"
  mt3_scripts = fork_root / "mt3" / "mt3" / "scripts"
  mt3_scripts.mkdir(parents=True)
  bin_path = fork_root / "midi2reaper" / ".venv" / "bin" / "midi2reaper"
  bin_path.parent.mkdir(parents=True)
  bin_path.write_text("#!/bin/sh\n")

  found = export_transcription.find_midi2reaper_bin(mt3_scripts / "export_transcription.py")

  assert found == bin_path


def test_find_midi2reaper_bin_raises_when_no_sibling_exists(tmp_path):
  with pytest.raises(FileNotFoundError, match="midi2reaper"):
    export_transcription.find_midi2reaper_bin(tmp_path / "somewhere" / "deep")


def _fake_midi2reaper_build(calls: list[list[str]]):
  """Mimics `midi2reaper build <midi> -o <out_dir>`'s own naming convention
  (`{stem}.RPP` in `out_dir`) as a side effect, the same way the real binary
  would when run as a subprocess."""

  def _run(argv, check=False):
    calls.append(argv)
    midi_path = Path(argv[2])
    out_dir = Path(argv[argv.index("-o") + 1])
    (out_dir / f"{midi_path.stem}.RPP").write_text("fake rpp")
    return subprocess.CompletedProcess(argv, 0)

  return _run


def test_export_transcription_writes_midi_then_builds_a_reaper_project(tmp_path, monkeypatch):
  import mt3.transcription as transcription_module
  monkeypatch.setattr(transcription_module, "Transcriber", _FakeTranscriber)
  fake_bin = Path("/fake/midi2reaper/.venv/bin/midi2reaper")
  monkeypatch.setattr(export_transcription, "find_midi2reaper_bin", lambda start: fake_bin)

  calls: list[list[str]] = []
  monkeypatch.setattr(export_transcription.subprocess, "run", _fake_midi2reaper_build(calls))

  input_wav = tmp_path / "song.wav"
  input_wav.write_bytes(b"fake audio")
  out_dir = tmp_path / "out"

  rpp_path = export_transcription.export_transcription(
      tmp_path / "checkpoint", input_wav, out_dir)

  assert rpp_path == out_dir / "song.RPP"
  assert rpp_path.is_file()
  assert (out_dir / "song.mid").is_file()
  [argv] = calls
  assert argv[0] == str(fake_bin)
  assert argv[1] == "build"
  assert argv[2] == str(out_dir / "song.mid")
  assert "-f" in argv  # force is the default


def test_export_transcription_omits_force_flag_when_disabled(tmp_path, monkeypatch):
  import mt3.transcription as transcription_module
  monkeypatch.setattr(transcription_module, "Transcriber", _FakeTranscriber)
  monkeypatch.setattr(export_transcription, "find_midi2reaper_bin",
                      lambda start: Path("/fake/midi2reaper"))

  calls: list[list[str]] = []
  monkeypatch.setattr(export_transcription.subprocess, "run", _fake_midi2reaper_build(calls))

  input_wav = tmp_path / "song.wav"
  input_wav.write_bytes(b"fake audio")

  export_transcription.export_transcription(
      tmp_path / "checkpoint", input_wav, tmp_path / "out", force=False)

  [argv] = calls
  assert "-f" not in argv


def test_export_transcription_raises_when_midi2reaper_rejects_the_input(tmp_path, monkeypatch):
  """`midi2reaper build` exits 0 even on a REJECT/EXISTS outcome -- this must
  not be mistaken for success just because the subprocess didn't raise."""
  import mt3.transcription as transcription_module
  monkeypatch.setattr(transcription_module, "Transcriber", _FakeTranscriber)
  monkeypatch.setattr(export_transcription, "find_midi2reaper_bin",
                      lambda start: Path("/fake/midi2reaper"))

  def _run_without_writing_rpp(argv, check=False):
    return subprocess.CompletedProcess(argv, 0)

  monkeypatch.setattr(export_transcription.subprocess, "run", _run_without_writing_rpp)

  input_wav = tmp_path / "song.wav"
  input_wav.write_bytes(b"fake audio")

  with pytest.raises(FileNotFoundError, match="did not write"):
    export_transcription.export_transcription(tmp_path / "checkpoint", input_wav, tmp_path / "out")


def test_main_reports_the_written_project(tmp_path, monkeypatch, capsys):
  import mt3.transcription as transcription_module
  monkeypatch.setattr(transcription_module, "Transcriber", _FakeTranscriber)
  monkeypatch.setattr(export_transcription, "find_midi2reaper_bin",
                      lambda start: Path("/fake/midi2reaper"))
  calls: list[list[str]] = []
  monkeypatch.setattr(export_transcription.subprocess, "run", _fake_midi2reaper_build(calls))

  input_wav = tmp_path / "song.wav"
  input_wav.write_bytes(b"fake audio")
  checkpoint_dir = tmp_path / "checkpoint"
  checkpoint_dir.mkdir()

  exit_code = export_transcription.main([
      "--checkpoint", str(checkpoint_dir),
      "--input", str(input_wav),
      "--out-dir", str(tmp_path / "out"),
  ])

  assert exit_code == 0
  assert "Wrote" in capsys.readouterr().out


def test_main_returns_nonzero_on_midi2reaper_failure(tmp_path, monkeypatch, capsys):
  import mt3.transcription as transcription_module
  monkeypatch.setattr(transcription_module, "Transcriber", _FakeTranscriber)
  monkeypatch.setattr(export_transcription, "find_midi2reaper_bin",
                      lambda start: Path("/fake/midi2reaper"))

  def _raise(argv, check=False):
    raise subprocess.CalledProcessError(1, argv)

  monkeypatch.setattr(export_transcription.subprocess, "run", _raise)

  input_wav = tmp_path / "song.wav"
  input_wav.write_bytes(b"fake audio")
  checkpoint_dir = tmp_path / "checkpoint"
  checkpoint_dir.mkdir()

  exit_code = export_transcription.main([
      "--checkpoint", str(checkpoint_dir),
      "--input", str(input_wav),
      "--out-dir", str(tmp_path / "out"),
  ])

  assert exit_code == 1
  assert "ERROR" in capsys.readouterr().err
