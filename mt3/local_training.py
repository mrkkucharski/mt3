"""Local training helpers for the guitar-pilot workflow."""

from __future__ import annotations

import gin
from t5x import utils


@gin.configurable
def restore_checkpoint_config(
    path: str,
    mode: str = 'specific',
    dtype: str | None = 'float32',
) -> utils.RestoreCheckpointConfig:
  """Builds a T5X restore config for Gin files on the current T5X release."""
  return utils.RestoreCheckpointConfig(path=path, mode=mode, dtype=dtype)
