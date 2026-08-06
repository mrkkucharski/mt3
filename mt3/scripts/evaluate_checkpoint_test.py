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

"""Tests for evaluate_checkpoint."""

import copy
import json
from pathlib import Path
import wave

import mido
import note_seq
import pytest

from mt3.scripts import evaluate_checkpoint


def _write_wav(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with wave.open(str(path), "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(44100)
    wav_file.writeframes(b"\x01\x00" * 33075)


def _write_midi(path: Path) -> None:
  """One rhythm guitar part, matching `_write_fixture`'s manifest record."""
  path.parent.mkdir(parents=True, exist_ok=True)
  midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
  conductor = mido.MidiTrack()
  conductor.append(mido.MetaMessage("track_name", name="conductor", time=0))
  conductor.append(mido.MetaMessage("set_tempo", tempo=468750, time=0))
  midi_file.tracks.append(conductor)

  rhythm_track = mido.MidiTrack()
  rhythm_track.append(mido.MetaMessage("track_name", name="distortion-guitar:rhythm", time=0))
  rhythm_track.append(mido.Message("program_change", channel=0, program=30, time=0))
  rhythm_track.append(mido.Message("note_on", channel=0, note=40, velocity=100, time=0))
  rhythm_track.append(mido.Message("note_off", channel=0, note=40, velocity=0, time=480))
  midi_file.tracks.append(rhythm_track)

  midi_file.save(str(path))


def _write_fixture(dataset_dir: Path, split: str = "test") -> dict:
  _write_wav(dataset_dir / f"audio/{split}/ex_0001.wav")
  _write_midi(dataset_dir / f"midi/{split}/ex_0001.mid")

  record = {
      "id": "ex_0001",
      "split": split,
      "audio_path": f"audio/{split}/ex_0001.wav",
      "midi_path": f"midi/{split}/ex_0001.mid",
      "sample_rate_hz": 44100,
      "channels": 1,
      "source_midi_id": "source_0001",
      "parts": [
          {
              "track_name": "distortion-guitar:rhythm",
              "program": 30,
              "is_drum": False,
              "rhythm": True,
              "note_count": 1,
          },
      ],
      "renderer": "test renderer",
      "normalization": "peak:-1dBFS",
  }
  (dataset_dir / "manifest.jsonl").write_text(json.dumps(record) + "\n")
  return record


def test_load_split_records_filters_by_split(tmp_path):
  train_record = {"id": "ex_0001", "split": "train"}
  test_record = {"id": "ex_0002", "split": "test"}
  (tmp_path / "manifest.jsonl").write_text(
      json.dumps(train_record) + "\n" + json.dumps(test_record) + "\n")

  assert evaluate_checkpoint._load_split_records(tmp_path, "test") == [test_record]
  assert evaluate_checkpoint._load_split_records(tmp_path, "train") == [train_record]


class _FakeTranscriber:
  """Stands in for `mt3.transcription.Transcriber`: no checkpoint restore,
  no JAX inference -- just returns a NoteSequence handed to it up front."""

  instances: list["_FakeTranscriber"] = []

  def __init__(self, checkpoint_path, predicted: note_seq.NoteSequence):
    self.checkpoint_path = checkpoint_path
    self._predicted = predicted
    _FakeTranscriber.instances.append(self)

  def transcribe(self, audio) -> note_seq.NoteSequence:
    return self._predicted


def test_evaluate_scores_a_perfect_prediction_as_f1_one(tmp_path, monkeypatch):
  dataset_dir = tmp_path / "dataset"
  record = _write_fixture(dataset_dir)
  ref_ns = evaluate_checkpoint.midi_to_note_sequence(
      dataset_dir / record["midi_path"], record["id"], record["audio_path"], record)

  import mt3.transcription as transcription_module
  monkeypatch.setattr(
      transcription_module, "Transcriber",
      lambda checkpoint_path: _FakeTranscriber(checkpoint_path, copy.deepcopy(ref_ns)))

  result = evaluate_checkpoint.evaluate(tmp_path / "checkpoint", dataset_dir, "test")

  assert result["example_count"] == 1
  assert result["mean_f1"] == pytest.approx(1.0)
  assert result["examples"][0]["ref_notes"] == 1
  assert result["examples"][0]["est_notes"] == 1


def test_evaluate_scores_a_missed_prediction_below_one(tmp_path, monkeypatch):
  dataset_dir = tmp_path / "dataset"
  record = _write_fixture(dataset_dir)
  empty_ns = note_seq.NoteSequence()

  import mt3.transcription as transcription_module
  monkeypatch.setattr(
      transcription_module, "Transcriber",
      lambda checkpoint_path: _FakeTranscriber(checkpoint_path, empty_ns))

  result = evaluate_checkpoint.evaluate(tmp_path / "checkpoint", dataset_dir, "test")

  assert result["mean_f1"] < 1.0
  assert result["examples"][0]["est_notes"] == 0


def test_evaluate_raises_for_a_split_with_no_examples(tmp_path):
  dataset_dir = tmp_path / "dataset"
  _write_fixture(dataset_dir, split="train")

  with pytest.raises(ValueError, match="test"):
    evaluate_checkpoint.evaluate(tmp_path / "checkpoint", dataset_dir, "test")


def test_main_writes_a_report(tmp_path, monkeypatch, capsys):
  dataset_dir = tmp_path / "dataset"
  record = _write_fixture(dataset_dir)
  ref_ns = evaluate_checkpoint.midi_to_note_sequence(
      dataset_dir / record["midi_path"], record["id"], record["audio_path"], record)

  import mt3.transcription as transcription_module
  monkeypatch.setattr(
      transcription_module, "Transcriber",
      lambda checkpoint_path: _FakeTranscriber(checkpoint_path, copy.deepcopy(ref_ns)))

  report_path = tmp_path / "report.json"
  checkpoint_dir = tmp_path / "checkpoint"
  checkpoint_dir.mkdir()
  exit_code = evaluate_checkpoint.main([
      "--checkpoint", str(checkpoint_dir),
      "--dataset", str(dataset_dir),
      "--report", str(report_path),
  ])

  assert exit_code == 0
  written = json.loads(report_path.read_text())
  assert written["mean_f1"] == pytest.approx(1.0)
  assert "mean (program, rhythm) onset+offset F1" in capsys.readouterr().out
