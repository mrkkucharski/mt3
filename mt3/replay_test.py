# Copyright 2026 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the compact original-data replay registrations."""

from mt3 import datasets
from mt3 import tasks  # pylint: disable=unused-import

import seqio
import tensorflow as tf


class ReplayRegistrationTest(tf.test.TestCase):

  def test_compact_slakh_schema_does_not_require_removed_stems(self):
    for config in (datasets.REPLAY_SLAKH_CONFIG,
                   datasets.REPLAY_CERBERUS4_CONFIG):
      self.assertNotIn('stems', config.features)
      self.assertNotIn('stems_shape', config.features)
      self.assertContainsSubset(
          {'mix', 'note_sequences', 'audio_sample_rate', 'track_id'},
          config.features)

  def test_replay_rates_are_exactly_twenty_percent(self):
    self.assertAlmostEqual(sum(tasks.REPLAY_TASK_RATES.values()), 0.2)
    mixture = seqio.MixtureRegistry.get(
        'guitar_pilot_replay_notes_ties_vb1nr_train')
    rates = dict(mixture._task_to_rate)
    self.assertAlmostEqual(
        rates['guitar_pilot_notes_ties_vb1nr_train'], 0.8)
    self.assertAlmostEqual(sum(rates.values()), 1.0)

  def test_inference_eval_is_guitar_only(self):
    mixture = seqio.MixtureRegistry.get(
        'guitar_pilot_replay_notes_ties_vb1nr_eval')
    self.assertEqual(
        [task.name for task in mixture.tasks],
        ['guitar_pilot_notes_ties_vb1nr_eval_test'])


if __name__ == '__main__':
  tf.test.main()
