"""Fail-fast actions for MT3 training runs."""

from __future__ import annotations

import gin
import numpy as np
from t5x import trainer


@gin.configurable
class FailOnNonFiniteAction(trainer.BaseAction):
  """Raises when a scalar training metric becomes NaN or infinite.

  Unlike T5X's ``TerminateOnNanAction``, raising makes the training subprocess
  exit nonzero instead of requesting a graceful stop that may save the current
  (already poisoned) train state.
  """

  def __init__(self, task: str = 'train', metric: str = 'loss'):
    self._task = task
    self._metric = metric

  def run(self, train_state, metrics_by_task) -> bool:
    if self._task not in metrics_by_task:
      raise KeyError(
          f'FailOnNonFiniteAction task {self._task!r} is absent; '
          f'available tasks: {tuple(metrics_by_task)}')
    task_metrics = metrics_by_task[self._task]
    if self._metric not in task_metrics:
      raise KeyError(
          f'FailOnNonFiniteAction metric {self._metric!r} is absent from '
          f'task {self._task!r}; available metrics: {tuple(task_metrics)}')

    metric = task_metrics[self._metric]
    if not hasattr(metric, 'value'):
      raise TypeError(
          f'FailOnNonFiniteAction expected a scalar metric value for '
          f'{self._task}/{self._metric}, got {type(metric).__name__}')

    value = np.asarray(metric.value)
    if not np.all(np.isfinite(value)):
      step = int(np.asarray(train_state.step))
      raise FloatingPointError(
          f'Non-finite training metric at step {step}: '
          f'{self._task}/{self._metric}={value}')
    return False
