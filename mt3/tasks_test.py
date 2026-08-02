"""Tests for the local guitar-pilot MT3 task registration."""

import seqio

from mt3 import datasets
from mt3 import tasks  # pylint: disable=unused-import


def test_guitar_pilot_task_is_registered_with_full_lane_specs():
  task = seqio.TaskRegistry.get('guitar_pilot_notes_ties_vb1_train')
  assert task.name == 'guitar_pilot_notes_ties_vb1_train'
  assert [track.name for track in datasets.GUITAR_PILOT_CONFIG.track_specs] == [
      'clean-rhythm', 'clean-lead', 'distorted-rhythm', 'distorted-lead']
