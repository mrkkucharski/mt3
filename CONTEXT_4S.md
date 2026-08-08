# Extending the acoustic window from ~2 s to ~4 s

Preparation branch (`4s-context`). Nothing here has been trained or evaluated
yet; this branch only makes the 4 s window *runnable* and documents what was
verified against the code while setting it up.

## What this branch changes

| File | Change |
| --- | --- |
| `mt3/gin/context_4s.gin` | New. Thin overlay rebinding `TASK_FEATURE_LENGTHS` to `{'inputs': 512, 'targets': 2048}`. |
| `mt3/scripts/measure_target_lengths.py` | New. Measures the target-token distribution at a given window so `targets` can be sized before a run, not after a crash. |
| `mt3/transcription.py` | `Transcriber` takes `input_length` / `target_length`; defaults unchanged at 256/1024. |
| `mt3/cli.py` | `mt3-transcribe --input-length 512`. Defaults to the old behaviour when omitted. |

The 2 s path is unchanged. The overlay is a separate file rather than an edit
to `guitar_pilot_finetune_modal.gin` specifically so a 2 s-vs-4 s ablation
differs in one causal factor only.

## Verified while preparing this (not assumptions)

**The window is expressed only in frames, never seconds.** `spectrograms.py`
pins `DEFAULT_SAMPLE_RATE = 16000` and `DEFAULT_HOP_WIDTH = 128`, so
`frames_per_second = 125`. 256 frames = 2.048 s; 512 frames = 4.096 s.

**One value drives both chunking paths.** In `mt3/tasks.py`, the training task
chunks via `select_random_chunk` and the eval task via
`split_tokens_to_inputs_length`; both read `sequence_length['inputs']`, so the
gin rebinding reaches both with no further edits.

**`checkpoint_0` restores directly — no positional surgery.** t5x's `T5Config`
has no sequence-length-dependent field, and `network.py` builds position
information with `layers.RelativePositionBiases(num_buckets=32,
max_distance=128)` invoked as `relative_embedding(inputs.shape[-2],
inputs.shape[-2], ...)`. The length is read from the live input; the learned
table is shaped `[num_buckets, num_heads]` and is length-independent. Every
checkpoint tensor keeps its shape.

> This contradicts the whitepaper's Initiative 4, which flags "positional
> migration" as the main compatibility risk and prescribes extending a position
> table by interpolation/extrapolation. That warning assumes **absolute**
> position embeddings. This codebase uses relative bias, so the migration step
> it describes does not exist here.

**But there is a real behavioural caveat.** `max_distance=128` (~1.02 s) means
every frame pair beyond that distance already shares the single coarsest
bucket. At 512 frames a much larger share of each window sits in that saturated
bucket than the pretrained model ever had to resolve. The weights load cleanly;
the model has not learned to *use* the extra context. Expect adaptation steps,
and expect a 4 s model evaluated at step 0 to look no better than the 2 s one.

## Measured: `targets` must go to 2048

A 4 s window carries roughly twice the note events of a 2 s window, and
`targets` does not scale automatically. This was measured rather than
estimated, with `scripts/measure_target_lengths.py` on
`guitar_pilot_notes_ties_vb1_train`:

| window | examples | median | p95 | p99 | max | vs `targets=1024` |
| --- | --- | --- | --- | --- | --- | --- |
| 256 (~2.048 s) | 1040 (full epoch) | 112 | 323 | 435 | 608 | fits |
| 256 (~2.048 s) | 500 | 118 | 295 | 501 | 911 | fits, thin |
| 512 (~4.096 s) | 1040 (full epoch) | 187 | 571 | 811 | 1249 | fits |
| 512 (~4.096 s) | 500 | 185 | 518 | 893 | **1533** | **overflows, 2/500 = 0.4%** |

Leaving `targets` at 1024 does not degrade gracefully. `mt3/tasks.py` ends the
training pipeline with `handle_too_long(skip=skip_too_long)`, and
`skip_too_long` defaults to `False` — the `assert_not_too_long` branch, a
`tf.debugging.assert_less_equal` that **raises** `Value for "targets" field
exceeds maximum length`. A 0.4% per-window overflow rate is a near-certain
crash over a real run, at an unpredictable step, on a paid GPU.

**2048 is a judgement, not a proven bound.** It clears the highest length seen
anywhere (1533) by ~33%. But `select_random_chunk(uniform_random_start=True)`
draws a different chunk from each cached segment every epoch, so two full-epoch
passes returned maxima of 1249 and 1533 — no single pass bounds the tail. The
dataset yields 1040 windows per epoch regardless of window size (`split_tokens`
segments at `MAX_NUM_CACHED_FRAMES=2000` before chunking), so `--max_examples`
above ~1040 just re-samples rather than covering more ground. Re-run the script
if the corpus changes.

The rejected alternative is `skip_too_long=True`, which filters instead of
raising. It trades a crash for silent bias: the dropped windows are exactly the
densest passages, so the model would be quietly trained away from busy music.

### This costs real compute

`seqio` pads every feature to its `task_feature_length` — `models.py`'s
converter docstring: *"Each feature in the `task_feature_lengths` is
trimmed/padded"* — because XLA needs static shapes. The decoder therefore
computes at the **full** target length every step regardless of content, and
with a median of 187 almost all of that work is padding.

So `1024 → 2048` quadruples decoder self-attention cost, on top of the ~4x
encoder attention from the window change itself. Both are unconditional and
they compound. This is the main reason a 4 s run is not simply "2x the 2 s
run".

### Is the current 2 s run at risk?

Not on this evidence — no overflow in either pass. But the worst pass reached
911 against a 1023 ceiling (89% utilisation), so the margin is thinner than it
looks. Worth knowing if the corpus ever grows denser.

## Smoke test results

**Restore and forward pass at 512 work — verified by running them, not by
reading the code.** `Transcriber(checkpoint_0, input_length=512,
target_length=2048)` restores with no shape error and transcribes end to end
to MIDI. The seqio task pipeline also materialises a full 1040-window epoch at
512. The compatibility claim above is therefore empirical.

**A local training step could not be tested — pre-existing environment fault,
unrelated to this branch.** `python -m t5x.train` on this Mac dies in
`prepare_train_iter` with:

```
InternalError: Can't find an output tensor for the output node:
identity_RetVal [Op:MakeIterator]
```

The identical command **without** `context_4s.gin` (i.e. at the 256/1024
baseline) fails in exactly the same place, so this is the Apple-Silicon
TensorFlow input-pipeline problem the project already knows about, not the
window change. Note the data pipeline itself is fine — `task.get_dataset()`
iterates happily at 512; it is t5x's `clu` train-iterator wrapper that breaks.
The first real training step at 512 must therefore happen on Modal.

**Inference throughput is roughly flat, which is not the win it looks like.**
Warm timings (model constructed and JIT-compiled first), 30 s of audio on an
M4 CPU:

| window | wall clock | realtime factor | notes predicted |
| --- | --- | --- | --- |
| 256 (2 s) | 21.5 s | 1.40x | 1097 |
| 512 (4 s) | 20.3 s | 1.48x | 460 |

Per-window cost rises, but there are half as many windows, so whole-file
throughput roughly cancels out. **However, the 4 s column is faster partly
because it predicts 460 notes instead of 1097** — the un-adapted checkpoint
under-predicts badly at a window it was never trained for, and the
autoregressive decoder simply runs fewer steps. This is the behavioural caveat
from the previous section showing up as a number. Re-measure once the model is
actually adapted; expect the gap to close as note counts recover.

Training cost does *not* get this reprieve: targets are padded to the full 2048
every step regardless of content, so the 4x decoder-attention increase there is
unconditional.

## Relationship to `main`

Branched from `63b4de6`. At branch time `main` had **uncommitted** work
belonging to the in-flight `guitar_pilot_finetune_64ex_it2` run. The
`modal_train.py` half of it — pinning `--gin.train.eval_period` to
`save_period`, because `train.gin` hardcodes `eval_period=5000` and t5x
requires the checkpoint/eval/GC periods to be multiples of each other — was a
genuine bug fix independent of the window change and a 4 s run would hit it
too, so it was committed to `main` on its own (`d85b36e`) and this branch
rebased onto it. The branch is now `main` plus exactly one commit.

The `MODEL_DIR` edit pointing at `..._64ex_it2` is run-specific and remains
deliberately uncommitted, as do the untracked `modal_eval.py` and
`guitar_pilot_eval_modal.gin`.

This branch lives in a `git worktree` at `../mt3-4s-context`, so the `main`
checkout and the running job were never touched. Remove it with
`git worktree remove ../mt3-4s-context` when finished.

## Merging is safe; reverting is not a git operation

Every change to a pre-existing file is additive with a behaviour-preserving
default (`--input-length` defaults to `None`, `input_length` defaults to 256),
and `context_4s.gin` / `measure_target_lengths.py` are inert unless explicitly
loaded or run. Merging this branch is therefore a no-op for the 2 s path.

"Going back to 2 s" means dropping `--gin_file=mt3/gin/context_4s.gin` from the
training invocation and `--input-length` from inference. Existing 2 s
checkpoints live in their own `MODEL_DIR`s and are never written to. No
`git revert` and no retraining is involved — which is the reason the window
lives in an overlay rather than an in-place edit to
`guitar_pilot_finetune_modal.gin`.

## Suggested order

1. ~~Run `measure_target_lengths.py` at 256 and 512~~ — done; `targets` set to
   2048 on the evidence above.
2. ~~Local smoke run~~ — restore and forward pass at 512 confirmed. A local
   training step is not possible on this machine (see above).
3. **Next:** a 1-step Modal preflight with `context_4s.gin` appended to the
   gin chain, to confirm a real optimizer step at 512/2048 and to get the true
   peak-memory number on the A10G.
4. Then the real run, holding data and depth fixed. Give it its own
   `MODEL_DIR` so the 2 s checkpoints stay independently usable.
5. Evaluate at 512, and also evaluate the 2 s checkpoint at 512, to separate
   "the model learned to use context" from "the window changed". Re-run the
   inference timing once note counts recover.
