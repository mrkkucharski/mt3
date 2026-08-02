"""Local-only TensorFlow compatibility hook for the MT3 CPU smoke run."""

import tensorflow as tf

# TensorFlow 2.20's tf.data meta-optimizer fails on the MT3/SeqIO pipeline in
# this Apple Silicon environment. The smoke-run command adds this directory to
# PYTHONPATH so this hook is not active for ordinary project commands.
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
