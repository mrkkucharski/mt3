"""Runtime configuration shared by the standalone inference entry points."""

from __future__ import annotations


def tensorflow():
  """Imports and configures TensorFlow for the supported MT3 inference path.

  TensorFlow 2.20 on Apple Silicon can fail in MT3's SeqIO input pipeline when
  its tf.data meta-optimizer is enabled.  Keeping this detail here means API
  and CLI consumers never need to set a ``PYTHONPATH`` sitecustomize hook.
  """
  import tensorflow as tf

  tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
  return tf
