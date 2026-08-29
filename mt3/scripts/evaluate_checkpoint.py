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

"""Score a real MT3 checkpoint against a DATA_CONTRACT.md dataset's held-out split.

One-command Phase 1 "evaluate" entry point
(`MT3_guitar_transcription_plan.md`, "Local training and deliverables").
`mt3/scripts/run_phase0_gate_eval.py` already does this for Phase 0's tiny
32-dim gate model against two synthetic excerpts; this is the same
(program, rhythm)-aware note scoring (`mt3.metrics._program_aware_note_scores`,
`granularity_type='full'`) applied to the real T5.1.1 Small architecture
(`mt3.transcription.Transcriber`, the same class `mt3-transcribe` uses) and a
real corpus's held-out split, so a checkpoint like `checkpoint_1002000` can be
scored in one command instead of the manual transcribe-then-eyeball-note-counts
process every fine-tune leg has used so far (see `PROJECT_LOG.md`, e.g.
"First fine-tune on the new corpus").

Run from the MT3 repository:

  uv run python -m mt3.scripts.evaluate_checkpoint \
      --checkpoint runs/guitar_pilot_v1/checkpoint_1002000 \
      --dataset ../data/pilot \
      --report ../runs/eval_reports/checkpoint_1002000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import note_seq

from mt3.scripts.build_guitar_pilot_tfrecord import midi_to_note_sequence

F1_KEY = "Onset + offset + program F1 (full)"
PRECISION_KEY = "Onset + offset + program precision (full)"
RECALL_KEY = "Onset + offset + program recall (full)"


def _load_split_records(dataset_dir: Path, split: str) -> list[dict]:
  records = [
      json.loads(line)
      for line in (dataset_dir / "manifest.jsonl").read_text().splitlines()
      if line.strip()
  ]
  return [record for record in records if record["split"] == split]


def evaluate(checkpoint_path: Path, dataset_dir: Path, split: str,
             rhythm_vocab: bool = False) -> dict:
  """Transcribes every `split` example and scores it against its corpus MIDI.

  Rhythm-free evaluation is the default. Pass `rhythm_vocab=True` only for a
  checkpoint trained with the optional rhythm/lead flag; it then scores that
  distinction on both prediction and reference.
  """
  # Delayed: these pull in JAX/T5X/TensorFlow, which argument parsing (and
  # --help) should not pay for.
  from mt3 import metrics
  from mt3.transcription import SAMPLE_RATE, Transcriber

  records = _load_split_records(dataset_dir, split)
  if not records:
    raise ValueError(f"no {split!r} examples in {dataset_dir / 'manifest.jsonl'}")

  transcriber = Transcriber(checkpoint_path, rhythm_vocab=rhythm_vocab)
  per_example = []
  for record in records:
    audio_path = dataset_dir / record["audio_path"]
    midi_path = dataset_dir / record["midi_path"]
    ref_ns = midi_to_note_sequence(midi_path, record["id"], record["audio_path"], record)
    audio = note_seq.audio_io.wav_data_to_samples_librosa(
        audio_path.read_bytes(), sample_rate=SAMPLE_RATE)
    est_ns = transcriber.transcribe(audio)
    scores = metrics._program_aware_note_scores(
        ref_ns, est_ns, granularity_type="full", include_rhythm=rhythm_vocab)

    entry = {
        "id": record["id"],
        "source_midi_id": record["source_midi_id"],
        "ref_notes": len(ref_ns.notes),
        "est_notes": len(est_ns.notes),
        "f1": scores[F1_KEY],
        "precision": scores[PRECISION_KEY],
        "recall": scores[RECALL_KEY],
    }
    per_example.append(entry)
    print(f"{entry['id']} [{entry['source_midi_id'][:40]}]: "
          f"ref={entry['ref_notes']} est={entry['est_notes']} F1={entry['f1']:.3f}")

  mean_f1 = sum(entry["f1"] for entry in per_example) / len(per_example)
  return {
      "checkpoint": str(checkpoint_path),
      "dataset": str(dataset_dir),
      "split": split,
      "example_count": len(per_example),
      "mean_f1": mean_f1,
      "examples": per_example,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path,
                      help="Path to an MT3 checkpoint directory.")
  parser.add_argument("--dataset", required=True, type=Path,
                      help="Dataset directory containing manifest.jsonl.")
  parser.add_argument("--split", default="test", help="Manifest split to evaluate.")
  parser.add_argument("--report", type=Path, help="Write the full JSON report here.")
  parser.add_argument("--include-rhythm-vocab", action="store_true",
                      help="The checkpoint was trained with the optional "
                           "rhythm/lead flag; score by (program, rhythm).")
  args = parser.parse_args(argv)

  result = evaluate(
      args.checkpoint.expanduser().resolve(), args.dataset.expanduser().resolve(),
      args.split, rhythm_vocab=args.include_rhythm_vocab)
  scored_by = "(program, rhythm)" if args.include_rhythm_vocab else "program"
  print(f"\n{result['example_count']} {args.split} example(s), "
        f"mean {scored_by} onset+offset F1 = {result['mean_f1']:.3f}")

  if args.report:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {args.report}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
