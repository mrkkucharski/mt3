"""Run one local guitar-pilot example through MT3 tokenization.

Run from the MT3 repository:

  uv run python mt3/scripts/smoke_guitar_task.py
"""

from __future__ import annotations

import argparse

from mt3 import tasks  # Registers the task as an import side effect.
from mt3 import vocabularies
import seqio


DEFAULT_TASK = 'guitar_pilot_notes_ties_vb1_train'


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--task', default=DEFAULT_TASK)
  parser.add_argument('--split', default='train', choices=('train', 'eval'),
                      help='MT3 dataset split: train, or eval (the held-out test TFRecord).')
  parser.add_argument('--input-length', type=int, default=256)
  parser.add_argument('--target-length', type=int, default=1024)
  args = parser.parse_args(argv)

  task = seqio.TaskRegistry.get(args.task)
  dataset = task.get_dataset(
      sequence_length={'inputs': args.input_length, 'targets': args.target_length},
      split=args.split,
      use_cached=False,
      shuffle=False)
  example = next(dataset.as_numpy_iterator())
  codec = vocabularies.build_codec(tasks.VOCAB_CONFIG_NOVELOCITY)
  raw_tokens = task.output_features['targets'].vocabulary.decode(example['targets'])
  programs = sorted({
      codec.decode_event_index(int(token)).value
      for token in raw_tokens
      if int(token) >= 0 and codec.decode_event_index(int(token)).type == 'program'
  })
  print(f'task: {args.task}')
  print(f'inputs shape: {example["inputs"].shape}')
  print(f'raw audio samples: {example["raw_inputs"].shape}')
  print(f'target tokens: {example["targets"].shape}')
  print(f'program IDs in sampled chunk: {programs}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
