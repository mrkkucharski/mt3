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

"""Tests for cli.py's argument parsing and Transcriber-kwargs resolution.

_parser() and _resolve_transcriber_kwargs() are both import-free (no
mt3.transcription), so most of this file runs without paying for the heavy
ML imports main() otherwise delays until after argument parsing.
MainValidationTest is the one exception -- it drives main() itself, so it
does pay that cost, but only up to the point where a bad geometry exits
before any checkpoint is touched.
"""

import contextlib
import io

import tensorflow as tf
from mt3 import cli

_BASE_ARGS = ['--checkpoint', '/tmp/ckpt', '--input', 'in.wav', '--output', 'out.mid']


def _parse(extra_args):
  return cli._parser().parse_args(_BASE_ARGS + extra_args)


class ParserTest(tf.test.TestCase):

  def test_lookback_frames_and_seconds_are_mutually_exclusive(self):
    with self.assertRaises(SystemExit):
      _parse(['--lookback-frames', '10', '--lookback-seconds', '0.1'])

  def test_lookahead_frames_and_seconds_are_mutually_exclusive(self):
    with self.assertRaises(SystemExit):
      _parse(['--lookahead-frames', '10', '--lookahead-seconds', '0.1'])

  def test_lookahead_and_lookback_flags_are_independent(self):
    # A lookback flag and a lookahead flag are not in the same mutually
    # exclusive group -- both may be given together.
    args = _parse(['--lookahead-frames', '10', '--lookback-frames', '20'])
    self.assertEqual(args.lookahead_frames, 10)
    self.assertEqual(args.lookback_frames, 20)

  def test_default_flags_are_none(self):
    args = _parse([])
    self.assertIsNone(args.lookback_frames)
    self.assertIsNone(args.lookback_seconds)
    self.assertIsNone(args.lookahead_frames)
    self.assertIsNone(args.lookahead_seconds)
    self.assertIsNone(args.force_program)
    self.assertFalse(args.no_rhythm)
    self.assertFalse(args.include_rhythm_vocab)

  def test_force_program_out_of_range_is_rejected(self):
    with self.assertRaises(SystemExit):
      _parse(['--force-program', '128'])
    with self.assertRaises(SystemExit):
      _parse(['--force-program', '-1'])

  def test_force_program_in_range_is_accepted(self):
    self.assertEqual(_parse(['--force-program', '12']).force_program, 12)

  def test_no_rhythm_is_a_flag(self):
    self.assertTrue(_parse(['--no-rhythm']).no_rhythm)

  def test_include_rhythm_vocab_is_an_opt_in_flag(self):
    args = _parse(['--include-rhythm-vocab'])
    self.assertTrue(args.include_rhythm_vocab)
    self.assertFalse(args.no_rhythm)


class ResolveTranscriberKwargsTest(tf.test.TestCase):

  def test_no_flags_yields_empty_kwargs(self):
    self.assertEqual(cli._resolve_transcriber_kwargs(_parse([])), {})

  def test_force_program_passed_through(self):
    kwargs = cli._resolve_transcriber_kwargs(
        _parse(['--force-program', '12']))
    self.assertEqual(kwargs, {'force_program': 12})

  def test_no_rhythm_passed_through(self):
    kwargs = cli._resolve_transcriber_kwargs(_parse(['--no-rhythm']))
    self.assertEqual(kwargs, {'no_rhythm': True})

  def test_rhythm_free_vocab_is_the_default(self):
    self.assertEqual(cli._resolve_transcriber_kwargs(_parse([])), {})

  def test_include_rhythm_vocab_passed_through(self):
    kwargs = cli._resolve_transcriber_kwargs(_parse(['--include-rhythm-vocab']))
    self.assertEqual(kwargs, {'rhythm_vocab': True})

  def test_no_rhythm_and_force_program_combine(self):
    kwargs = cli._resolve_transcriber_kwargs(
        _parse(['--no-rhythm', '--force-program', '12']))
    self.assertEqual(kwargs, {'force_program': 12, 'no_rhythm': True})

  def test_lookback_frames_passed_through(self):
    kwargs = cli._resolve_transcriber_kwargs(
        _parse(['--lookback-frames', '64']))
    self.assertEqual(kwargs, {'lookback_frames': 64})

  def test_lookback_seconds_truncated_to_whole_frames(self):
    # 0.5s * 125 frames/s = 62.5 -> truncated (not rounded) to 62.
    kwargs = cli._resolve_transcriber_kwargs(
        _parse(['--lookback-seconds', '0.5']))
    self.assertEqual(kwargs, {'lookback_frames': 62})

  def test_lookahead_seconds_truncated_to_whole_frames(self):
    kwargs = cli._resolve_transcriber_kwargs(
        _parse(['--lookahead-seconds', '0.5']))
    self.assertEqual(kwargs, {'lookahead_frames': 62})

  def test_lookahead_and_lookback_seconds_combine_with_input_length(self):
    kwargs = cli._resolve_transcriber_kwargs(_parse([
        '--lookahead-seconds', '1.0', '--lookback-seconds', '1.0',
        '--input-length', '512',
    ]))
    self.assertEqual(kwargs, {
        'input_length': 512,
        'lookahead_frames': 125,
        'lookback_frames': 125,
    })


class MainValidationTest(tf.test.TestCase):
  """Tests that a bad geometry is rejected before any checkpoint is touched."""

  def test_over_budget_geometry_exits_before_loading_a_checkpoint(self):
    stderr = io.StringIO()
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
      cli.main([
          '--checkpoint', '/nonexistent/path/for/mt3/tests',
          '--input', 'in.wav', '--output', 'out.mid',
          '--input-length', '256',
          '--lookback-frames', '200', '--lookahead-frames', '200',
      ])
    # Names all three numbers, per WindowGeometry's own error message --
    # a clean argparse usage error, not a stack trace from Transcriber.
    self.assertIn('window_frames=256', stderr.getvalue())
    self.assertIn('lookback_frames=200', stderr.getvalue())
    self.assertIn('lookahead_frames=200', stderr.getvalue())


if __name__ == '__main__':
  tf.test.main()
