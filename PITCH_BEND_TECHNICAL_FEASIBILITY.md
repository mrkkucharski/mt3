# MT3 pitch-bend generation: technical feasibility and retraining consequences

**Investigation date:** 2026-08-31  
**Status:** feasibility investigation, not an implementation decision

## Question

Can this MT3 fork be extended so that the decoder generates pitch-bend events,
and can the existing pretrained/fine-tuned checkpoints be reused without
training a new model from scratch?

## Short conclusion

Yes. The existing event codec, autoregressive decoder, `NoteSequence` carrier,
and MIDI toolchain can all be extended to carry pitch bends. The lowest-risk
architecture is an optional pitch-bend event range appended after all existing
event ranges.

The current checkpoint tensors are already padded to 1,536 vocabulary rows.
The rhythm-aware codec uses 1,493 vocabulary entries and the rhythm-free codec
uses 1,491, including special and T5 extra IDs. A bend-aware layout must always
reserve the two rhythm positions, including when rhythm is disabled. That
leaves up to **43 appended bend event classes** without changing the decoder
embedding or output-projection shapes. Existing event IDs can remain unchanged,
and bend IDs can remain identical when rhythm is switched on or off. This makes
warm-starting from the current checkpoint technically feasible.

Shape compatibility does not mean that bends can be learned cheaply or
automatically. The current live corpus contains no pitch-bend labels at all,
and bend activity is sparse even in the source RPPs. A bend-aware run requires
new corpus serialization, new event encoding and decoding, bend-aware sampling,
and a deliberate initialization/fine-tuning experiment for the newly activated
vocabulary rows.

## Confirmed current state

### The current model has no bend event

`mt3/vocabularies.py::build_codec` currently declares:

- shift;
- pitch;
- velocity;
- tie;
- program;
- drum;
- optional rhythm.

`mt3/note_sequences.py` converts only note-related data into those events. Its
decoding state likewise produces notes only.

The `PitchBendError` in `mt3/preprocessors.py` applies to the Slakh-style track
merge path. The guitar-pilot path uses `tokenize_transcription_example`, which
does not reject bends: it silently ignores `NoteSequence.pitch_bends` because
it extracts only note onsets and offsets.

### The current training data contains no bend targets

The 193 MIDI files under `../data/pilot/midi` were inspected with `mido`:

- pitch-wheel events: **0**;
- RPN-related CC6/CC100/CC101 events: **0**.

Consequently, the rebuilt TFRecords and the current 193-example no-rhythm run
cannot learn to emit bends. That run is learning the present asymmetric task:
bent audio is supervised as one sustained discrete-pitch note.

The loss cannot discover a new event class that is absent from both the codec
and every target sequence.

### Bend source material exists before corpus MIDI generation

The live `../reaper/generated/it7_set_48` projects contain:

- 48 live RPPs;
- 36 RPPs with `0xE0` events;
- 28 of those projects in the frozen train split and 8 in test;
- 47,275 raw pitch-wheel messages;
- 729 RPN 0 declarations, all followed by CC6 value 12;
- 45,358 non-centred pitch-wheel messages.

The current `SLIDE_CONVERSION_REPORT.md` reports only 327 conversions and still
names `it7_set_50`. The live RPPs contain 729 RPN-per-gesture markers. The
report is therefore stale relative to the live set and must not be used as the
authoritative bend-label count.

All bend-bearing manifest source projects are generated arrangements; none is
marked `live_recording`. A pilot can establish whether the architecture learns
the converter's linear slides, but it will not by itself establish
generalization to human bends or vibrato.

## Codec and checkpoint feasibility

### Existing sizes

For the current `num_velocity_bins=1` configurations:

| Codec | Codec classes | SeqIO vocabulary size | Padded model size |
|---|---:|---:|---:|
| Rhythm-free | 1,388 | 1,491 | 1,536 |
| Rhythm-aware | 1,390 | 1,493 | 1,536 |

The official checkpoint was inspected directly:

- `target.decoder.token_embedder.embedding`: `[1536, 512]`;
- `target.decoder.logits_dense.kernel`: `[512, 1536]`.

Appending no more than 43 bend classes leaves those shapes unchanged for both
codec variants. If the bend range is appended last, all existing event IDs
also remain unchanged. The T5 extra-ID positions move upward inside the same
padded matrix, but those IDs are not transcription targets.

### Mandatory fixed rhythm slot

The current implementation physically omits the two rhythm classes when
`include_rhythm=False`. That is safe today only because rhythm is the last event
range and nothing follows it. Pitch bends would invalidate that assumption.

Bend tokens must not reuse those two positions in the rhythm-free codec. If
they did, the same checkpoint row would mean `rhythm=0/1` with rhythm enabled
and two bend values with rhythm disabled. Bend token IDs would also shift by
two when the option changed, making checkpoint configuration mistakes silent
and destructive.

The bend-aware codec therefore needs one fixed physical layout:

1. all existing shift/pitch/velocity/tie/program/drum ranges;
2. two rhythm positions;
3. the pitch-bend range;
4. T5 extra IDs and padding.

When rhythm is enabled, the two positions decode as `rhythm=0` and
`rhythm=1`. When rhythm is disabled, they must remain allocated but decode as
reserved/invalid events and must never appear in training targets. One
implementation is an internal `reserved_rhythm` event range used only to keep
the codec offsets stable; `codec.has_event_type('rhythm')` must still return
false, preserving the current no-rhythm encoding behavior.

This changes the logical no-rhythm SeqIO vocabulary size from 1,491 to 1,493,
but not the 1,536-row model tensors. Every pre-existing non-rhythm event keeps
its old ID. It also deliberately reduces the apparent 45-row no-rhythm slack
to the same **43 usable bend rows** available in rhythm-aware mode.

### What checkpoint compatibility does and does not guarantee

It guarantees that the checkpoint can be restored without resizing the two
vocabulary-dependent tensors.

It does not guarantee useful initial parameters for the new classes. The rows
that would become bend tokens were previously unused as transcription targets
and participated only as non-target logits. A preliminary norm inspection did
not show corrupt or zero rows, but norm statistics cannot tell whether they
are a good starting point.

A small ablation is required:

1. reuse the existing rows unchanged;
2. reinitialize only the new token-embedding rows and output columns;
3. optionally initialize them from the mean of existing event rows or from
   nearby pitch-token rows.

This is localized checkpoint surgery, not training from scratch. Encoder,
decoder blocks, attention, spectrogram front end, and every old event row can
be retained.

### Future expansion beyond 1,536 rows

Vocabulary semantics can remain backward compatible even after the current
1,536-row geometry is exhausted, but an old checkpoint will no longer restore
strictly into the larger tensors without a migration step.

The compatibility invariant is append-only allocation:

- never renumber an existing event;
- never insert a new range before an existing range;
- keep the two rhythm positions fixed even when rhythm is disabled;
- append later event ranges after the established bend range;
- version and test the event-ID layout independently of which optional targets
  are enabled.

With those rules, every old token sequence has exactly the same meaning under
the enlarged model. A new model can train on old TFRecords and decode old
predictions. The reverse is necessarily limited: an old model does not know
new IDs and cannot reproduce their semantics.

`vocabularies.num_embeddings` rounds the model vocabulary to a multiple of
128. Crossing the present boundary would normally change the geometry from
1,536 to 1,664 rows. Only two checkpoint parameters depend directly on this
dimension in the current model:

- decoder token embeddings: `[1536, 512]` -> `[1664, 512]`;
- decoder output projection: `[512, 1536]` -> `[512, 1664]`.

A checkpoint upgrader can make this a warm start rather than a retrain:

1. create the larger tensors;
2. copy all 1,536 old embedding rows exactly;
3. copy all 1,536 old output columns exactly;
4. initialize only the appended rows/columns;
5. copy every vocabulary-independent parameter unchanged;
6. either migrate the vocabulary-shaped optimizer state or, more simply for a
   new fine-tuning phase, reset the optimizer state while retaining model
   parameters.

The last choice matters because T5X checkpoints contain training/optimizer
state in addition to model parameters. Parameter padding alone is sufficient
for inference and parameter-only fine-tuning, but not for a byte-for-byte
continuation of the previous optimizer trajectory.

New enlarged checkpoints will not load into the old 1,536-row architecture
without truncation. A compatibility export could truncate the appended rows
for a notes-only old runtime, but it would necessarily discard the new event
capabilities.

An alternative is to move to a deliberately larger fixed geometry, such as
1,664 or 2,048, during the first vocabulary migration and reserve the unused
tail for future event types. This pays the migration cost once. It adds only
embedding/output parameters for the reserved rows, but also enlarges the
decoder's output softmax; inactive IDs should ideally be masked during training
and decoding rather than treated as valid events.

### Candidate codebook sizes

Raw 14-bit MIDI pitch wheel must not become a 16,384-class event range. It is
both unnecessary and incompatible with the padded checkpoint.

Here, "25/43 tokens" means **25/43 possible vocabulary classes**, not that
every bend emits that many target tokens. A target emits one bend event only
when its quantized value changes. For example, a -4-semitone slide represented
by 31 raw wheel samples becomes five whole-semitone target events:
`-4, -3, -2, -1, 0`. The 43-class proposal is only a finer menu of possible
values; it does not force all 43 values into a gesture.

#### Can a bend token be parameterized?

At the Python codec level it already is: `Event('pitch_bend', -4)` has an event
type and a value. The current T5 decoder cannot emit that pair as one native
structured object, however. At each autoregressive step its softmax selects one
integer vocabulary ID, so the codec normally flattens `(type, value)` into one
ID. Under the 25-class direct representation, the sequence
`bend(-4), bend(-3), bend(-2), bend(-1), bend(0)` is five emitted token
occurrences selected from the same reusable 25-class range.

There are two ways to avoid allocating one vocabulary class per value:

1. **Prefix plus parameter token.** Add one `bend` prefix and reuse an existing
   128-value range as its parameter. For example, a designated pitch-token
   subset could encode offsets after the prefix:

   ```text
   bend, value(-4), bend, value(-3), ..., bend, value(0)
   ```

   The decoder state interprets the token immediately after `bend` as a bend
   value rather than as its ordinary event type. This needs only one new
   vocabulary class, but each bend point costs two emitted tokens, introduces
   a context-sensitive grammar, and needs rules for a missing/invalid parameter.
   It also reuses embeddings trained for a different meaning.

2. **A genuinely structured decoder output.** Change the model so one step
   predicts an event-type head plus a separate bend-value classification or
   regression head. This is a true parameterized event, but it changes the
   network, loss, decoder-input embedding, beam search and T5X integration. The
   transformer body can still be warm-started, but it is no longer a small
   codec extension.

A sign/magnitude or binary factorization is another prefix scheme, but needs
two or more parameter tokens per value and an even stricter grammar.

For the feasibility pilot, 25 direct classes are cheaper than they sound: they
occupy only 25 of 1,536 model rows, leave 18 rows after the fixed rhythm slot,
and encode each bend point in one step instead of two. The prefix scheme is
technically feasible if preserving vocabulary rows is more important than
target length and implementation simplicity.

Two technically viable direct codebooks are:

1. **25 whole-semitone values, -12 through +12.** This preserves every endpoint
   generated by the current slide converter and leaves 18 common padded rows
   available for later event types after the rhythm slot has been reserved. It
   is coarse for continuous expressive pitch.
2. **43-value hybrid codebook.** Include every integer from -12 through +12
   and half-semitone positions from -8.5 through +8.5. This uses all common
   shape-compatible capacity. It preserves integer endpoints and provides
   50-cent resolution across the range containing almost all current events.

In the live RPPs, the absolute bend-value quantiles are approximately 2.67,
6.89, 7.94, 9.00 and 12.00 semitones at p50, p90, p95, p99 and maximum.

The model should encode **semantic semitone offsets**, not raw wheel values.
Raw values are meaningful only together with the pitch-bend sensitivity. The
current source uses a fixed +/-12-semitone RPN declaration, so a fixed semantic
range is sufficient for the first experiment. Supporting arbitrary RPN ranges
inside the model is not required to establish feasibility.

### Examples from the live RPPs

REAPER stores a pitch-wheel event as `E <delta-ticks> e0 <LSB> <MSB>` for MIDI
channel 0. Its 14-bit wheel value is `LSB + 128 * MSB`; `00 40` is centre
(8,192). The converter precedes every gesture with:

```text
E 0 b0 65 00    # RPN MSB = 0
E 0 b0 64 00    # RPN LSB = 0
E 0 b0 06 0c    # bend sensitivity = 12 semitones
```

The following examples were parsed directly from the live `it7_set_48` RPPs,
not from the removed/stale conversion report.

#### 1. Monophonic slide-in: -4 semitones to centre

`Pink Floyd-Another Brick In The Wall, Part 2-06-28-2026-REV.RPP`,
`overdriven-guitar`, note 57, approximately 165.577 s:

```text
E   0 e0 56 2a    # value 5462, approximately -4.00 semitones
E 132 e0 34 2b    # value 5556, approximately -3.86
E 133 e0 12 2c    # value 5650, approximately -3.72
...               # approximately 5 ms between samples
E 132 e0 44 3e    # value 8004, approximately -0.28
E 133 e0 22 3f    # value 8098, approximately -0.14
E 132 e0 00 40    # centre; the glide has arrived
E 3839 e0 00 40   # centre again after note-off
```

The source contains 31 wheel messages. Consecutive-value collapse gives:

- 25-class whole-semitone codebook: **5 emitted bend events**,
  `[-4, -3, -2, -1, 0]`;
- 43-class hybrid codebook: **9 emitted bend events**.

#### 2. Anchored monophonic slide-out: hold, then 0 to -9, then reset

`Gary Moore-Still Got The Blues-07-25-2026-REV.RPP`, distortion guitar,
note 71, approximately 12.434 s:

```text
E     0 e0 00 40    # centre at note-on
E 16800 e0 00 40    # still centred after a 1.234 s hold
E    69 e0 2c 3e    # approximately -0.31 semitones
E    69 e0 58 3c    # approximately -0.62
...                 # descending samples every approximately 5 ms
E    70 e0 28 13    # approximately -8.38
E    69 e0 55 11    # approximately -8.69
E    69 e0 00 40    # instantaneous reset after note-off
```

The source contains 31 wheel messages. With literal quantized state events:

- whole-semitone codebook: **11 emitted changes**,
  `[0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 0]`;
- hybrid codebook: **19 emitted changes**.

The repeated centre before the ramp is redundant under literal MIDI
sample-and-hold semantics. It is not redundant under a sparse
piecewise-linear representation: there it marks the end of the flat hold and
must be retained or expressed with an explicit curve-mode token.

#### 3. Parallel three-note slide-out: one shared bend stream

The same Gary Moore project, notes 45, 52 and 57, approximately 205.156 s:

```text
E    0 e0 00 40    # all three notes begin centred
E 2688 e0 00 40    # hold for approximately 190 ms
E   71 e0 26 3f    # approximately -0.13 semitones
E   70 e0 4c 3e    # approximately -0.26
...                # one channel-wide curve for the chord
E   71 e0 5f 26    # approximately -4.74
E   70 e0 05 26    # approximately -4.87
E   71 e0 00 40    # reset after note-off
```

The source contains 40 wheel messages. It becomes 7 whole-semitone changes
`[0, -1, -2, -3, -4, -5, 0]`, or 12 hybrid-codebook changes. The event count
is not multiplied by three: MIDI pitch wheel is channel-wide and all three
notes share it. This matches the converter's present constraint that all
voices must move by the same interval and that no unrelated note may sound on
the channel while the bend is away from centre.

## Event representation choices

### A. Quantized pitch-wheel state events

Append one `pitch_bend` range and emit a token whenever the quantized bend
value changes. The bend applies to the currently selected `(program, rhythm)`
track, analogously to how a pitch event uses the current program and rhythm
state.

Advantages:

- smallest change to MT3's existing event language;
- directly decodes to `NoteSequence.pitch_bends`;
- generic enough for slides, bends and later vibrato data;
- fits the current padded checkpoint.

Costs and open details:

- 25 whole-semitone values produce staircase curves if exported literally;
- the 43-value codebook is more precise but consumes all current padded slack;
- bend state must be carried and reconciled across segment boundaries;
- redundant-value removal must be scoped per instrument, not globally.

The existing generic `remove_redundant_state_changes_fn` keeps only one state
per event type. It cannot safely deduplicate pitch bends because two programs
can simultaneously have the same or different bend states. Bend compression
must happen before flattening, keyed by `(program, rhythm)`.

### B. Sparse piecewise-linear control points

The converted gestures are piecewise linear, so another option is to encode
only curve control points and densify them during MIDI export. This would need
far fewer target events and could reproduce smooth curves.

The semantics cannot simply be "interpolate between all different bend
values." Anchored slides deliberately repeat the same value at `note_start`
and `glide_start`; that duplicate marks a flat hold followed by a ramp. A
deduplicator would remove it and incorrectly bend throughout the held part.
Slide-outs also reach a non-centred endpoint and reset to centre at the same
time, which requires ordered same-time events and a distinction between a
linear arrival and an instantaneous reset.

A robust sparse language would therefore need either:

- separate `bend_set` and `bend_linear_to` modes plus a shared value codebook;
  or
- explicit gesture tokens.

For example, two mode classes plus 25 value classes still fit in 27 appended
classes. This is compact and checkpoint-compatible, but it introduces a
custom curve grammar and postprocessor rather than ordinary MIDI state-event
semantics.

### C. Separate continuous/framewise bend head

A second decoder head could estimate a continuous bend curve on the acoustic
frame grid while the autoregressive decoder continues to emit notes.

This avoids vocabulary quantization, but it is not a small extension:

- bend curves must be assigned to programs/tracks or individual active notes;
- a new loss, masking rule and output head are required;
- the head must be stitched across inference windows;
- the existing T5X checkpoint does not contain its parameters;
- the available bend-positive data is small.

This option can reuse the shared encoder and note decoder, but it requires more
new training than appended tokens. It is not the preferred feasibility pilot.

## Required encoding and decoding state

The current conversion uses one channel-wide bend shared by all voices in a
gesture and rejects cases where unrelated simultaneous notes would be
detuned. The first MT3 implementation can use the same track-wide semantics.
Per-string or independently bent polyphony would require MPE-like channel or
string identity and is outside this feasibility result.

A direct-token implementation needs:

- an encoding value carrying time, bend value, program, instrument and rhythm;
- `NoteEncodingState.current_bends[(program, rhythm)]`;
- a deterministic ordering for simultaneous note-off, bend/RPN, reset and
  note-on events;
- bend state in segment-start state events when a note crosses a boundary;
- decoding state that distinguishes confirmed bend state from newly emitted
  bend events;
- reconciliation across overlapping windows, parallel to the active-note
  reconciliation already used by this fork;
- final centring of any track left bent when decoding ends;
- instrument assignment for decoded `NoteSequence.pitch_bends`, whose proto
  has `time`, `bend`, `instrument`, `program` and `is_drum` but no rhythm field.

Rhythm can follow the existing convention: instrument identity and the
`:rhythm` suffix carry the distinction. `assign_instruments` or a companion
function must assign both notes and bends to the same reconstructed
instrument.

The tie section is the subtle part. Merely prepending a bend token at every
segment would create duplicate curve points. A segment-start bend declaration
must act as state confirmation when that bend is already active, while still
being able to establish a non-centred bend for a segment decoded in isolation.
Missing or disagreeing bend state at a kept-region boundary also needs an
explicit rule: retain the preceding segment, accept the new segment, or centre
at the boundary. This requires targeted boundary tests before training.

## Serialization and export feasibility

### `NoteSequence` is sufficient as an in-memory carrier

`note_seq.midi_file_to_note_sequence` preserves pitch-wheel messages as
`NoteSequence.pitch_bends` and preserves RPN messages as ordinary control
changes. A test MIDI with CC101=0, CC100=0, CC6=12 and two pitch-wheel values
survived MIDI -> `NoteSequence` parsing with the expected values, programs and
instrument IDs.

The proto does not record the semantic bend range on each pitch-bend message.
The first implementation should normalize input bends to semitones using the
RPN state during preprocessing, then treat +/-12 as the model/output contract.

### The standard `note_seq` writer is not ordering-safe for this use

The same round-trip test exposed a concrete problem. Source order at tick zero
was:

1. RPN MSB (CC101);
2. RPN LSB (CC100);
3. data entry (CC6=12);
4. pitch wheel;
5. note-on.

`note_seq.sequence_proto_to_midi_file` rewrote the same-time events as:

1. pitch wheel;
2. CC6;
3. CC100;
4. CC101;
5. note-on.

The values survive, but the converter's intentional RPN-before-bend ordering
does not. A bend-aware `write_multitrack_midi` should therefore use an
ordering-controlled `mido` writer or otherwise enforce RPN -> bend -> note-on
at equal ticks. Relying on the current one-line `note_seq` exporter is unsafe.

`midi2reaper` also currently discards bends: `midiscan.py` retains notes and
programs, while `rpp.py::midi_events` writes note events only. Both paths must
be extended before a predicted bend can reach a reviewable REAPER project.

These are implementation changes, not architectural blockers.

## Target-length impact

Raw 200 Hz wheel streams should not be used directly. Quantizing and removing
redundant values gave the following static estimates across the 48 live RPPs:

| Representation | Total bend value changes | Maximum in any 2.048 s window | Maximum in any 4.096 s window |
|---|---:|---:|---:|
| Whole semitone, 25 values | 6,084 | 40 | 60 |
| Hybrid half-step, 43 values | 11,196 | 76 | 114 |

The current real training pipeline was sampled for 1,000 examples:

| Input window | Median current targets | p99 | Maximum | 1,023-token effective ceiling headroom |
|---|---:|---:|---:|---:|
| 256 frames / 2.048 s | 30 | 261 | 347 | 676 |
| 512 frames / 4.096 s | 52 | 476 | 638 | 385 |

These independent maxima cannot simply be added as a proof, because the
densest note and bend windows may coincide. They do show that a quantized bend
stream is unlikely to require a larger 1,024-token target budget for the
current 2-second run, and probably fits the 4-second configuration as well.
The real bend-aware preprocessing pipeline must be measured exhaustively before
launch; the existing target-length script samples random chunks and is not a
formal upper bound.

Sparse curve/gesture tokens would add far fewer events than either direct
codebook.

### Observed long-window NaN limit (2026-09-06)

The length estimates above were not sufficient to predict optimization
stability. A controlled training probe on the restored 241-example corpus
measured the actual 8.192-second, bend-aware, rhythm-free configuration
(1,024 input frames, fixed +/-12-semitone bends, batch size 1):

| Decoder target length | Result | Evidence |
|---:|---|---|
| 2,048 | Stable | 1,000-step probe remained finite; the restarted production run remained finite through relative step 2,000 |
| 3,072 and above | Unstable in the earlier long-window sweep | Stored metrics reached NaN and zero F1; this establishes a warning threshold, not a proof that every corpus/configuration fails at exactly 3,072 |
| 4,096 | NaN | The original run diverged by the first checkpoint interval; on the current corpus, halving the learning rate and normalizing loss by real target tokens still produced NaN |

The result points to decoder target length as the dominant trigger, rather than
the learning rate or padding contribution alone. The 2,048 result is now the
operational limit for this 8-second pilot: it provides roughly 20% headroom
over the measured maximum target length (1,673 over 1,341 yielded train
windows), while avoiding the demonstrated 4,096 failure mode. This is an
empirical stability limit for the present task and optimizer, not a claim that
the MT3 architecture has a universal 2,048-token limit.

The production run now includes a fail-fast training action that raises an
explicit `FloatingPointError` when the monitored loss becomes non-finite,
rather than allowing T5X to stop gracefully after a poisoned state has been
saved. The failure path was intentionally not injected; finite metrics through
the first 2,000 relative steps confirm that the guard does not interfere with
the normal path. Any future increase beyond 2,048 must be treated as a new
stability experiment, with NaN detection and checkpoint inspection enabled.

## Retraining consequences

### The current run is not bend-pretraining

Because it has no bend targets, the current 193-example no-rhythm run should be
treated as a note-model baseline. It may learn invariance to pitch motion while
holding the underlying note, but it does not learn a latent bend output task.

It can still be a warm-start candidate for a bend-aware run. The comparison
should include both:

- the official `checkpoint_0` -> bend-aware fine-tune;
- the best notes-only guitar checkpoint -> bend-aware fine-tune.

The second may have better guitar acoustics and worse pressure to emit bends,
because it has been explicitly optimized to explain bent audio without bend
tokens. This is empirical, not knowable from checkpoint shape alone.

### Positive supervision is sparse

Using the source RPP timing and manifest durations:

- train audio duration: about 5.26 hours;
- non-centred bend activity: about 2.37 minutes, or **0.752%** of train audio;
- test bend activity: about 1.22 minutes, or **1.323%** of test audio;
- approximately **6.19%** of uniformly selected 2.048-second training windows
  overlap any bend;
- approximately **9.96%** of uniformly selected 4.096-second windows overlap
  any bend.

Uniform random chunking will therefore present mostly negative examples. This
is useful for learning not to bend, but it is weak positive supervision for
new output classes.

A bend-aware task should support a controlled mixture such as:

- ordinary random windows, preserving the original note distribution;
- bend-positive windows sampled around known gestures;
- optionally bend-bearing examples without forcing every sampled window to
  contain a bend.

The positive fraction is a hyperparameter. Oversampling all bend windows too
aggressively risks hallucinated bends and overfitting the 28 generated train
songs. Loss weighting on bend tokens is another possibility, but changing
sequence-level cross-entropy weighting is more invasive than data sampling.

### Expected training scope

The technically justified starting point is continued fine-tuning, not a full
retrain:

1. restore all existing parameters;
2. initialize or reuse only the newly activated bend rows;
3. fine-tune on a separate bend-aware task/corpus version;
4. monitor both old note metrics and new bend metrics;
5. retain no-bend material so the model preserves negative calibration.

A from-scratch or large mixed-corpus pretraining run becomes justified only if
the pilot shows that the small bend corpus cannot overcome the old decoder's
bias, or if the representation expands beyond the 43-row padded capacity.

### Catastrophic-forgetting and metric consequences

Existing note tokens keep their IDs and parameters, but target sequences gain
extra events. Note F1 can regress through ordinary fine-tuning even without a
shape change. Every bend experiment needs a fixed notes-only regression suite
and the previous checkpoint evaluated through the unchanged note codec.

Current primary metrics score discrete notes and programs. They do not expose
bend accuracy. The pianoroll path passes through PrettyMIDI and may shift energy
according to bends, so its behavior must be audited rather than assumed stable.
A bend-aware evaluation should report separately:

- existing onset/offset/program F1;
- bend-activity precision and recall;
- bend value error in cents/semitones during reference bend activity;
- endpoint error and timing error for converted slides;
- false bend duration on references with no bend.

The 8 generated test songs with bends provide a development gate, not strong
evidence of real-performance generalization.

## Audio-label validity remains a prerequisite

The RPP converter correctly writes RPN 0 with range 12 before every gesture,
but this does not prove the instrument obeys it. Ample Sound's official manual
documents a plugin-side **Bender Range** control whose illustrated/default
value is 2, plus a **Poly Bender** option. The documentation found does not say
that incoming RPN 0 changes that setting:

- <https://www.amplesound.net/en/tutorial.asp>
- <https://amplesound.net/en/Main_Panel_Manual-AGLP.pdf>

This strengthens rather than resolves the concern in `../PROJECT_LOG.md`.
Before any bend-aware retraining, render a controlled note through every
bend-bearing plugin family:

- centre and full-scale wheel with no RPN;
- the same sequence with RPN +/-12;
- the same with the plugin UI explicitly set to 12;
- a chord with the relevant poly-bender setting.

Measure the output fundamental. For A4, a true +12-semitone bend should reach
approximately 880 Hz; a +/-2-clamped full wheel reaches approximately 493.9
Hz. If the rendered corpus audio is clipped to +/-2 while labels claim +/-12,
the bend task is invalid regardless of model architecture.

## Feasibility gates before a real training leg

1. **Renderer gate:** prove the intended semitone trajectories are present in
   audio for each plugin family.
2. **Serialization gate:** RPP -> corpus MIDI -> `NoteSequence` must preserve
   bend values, instrument ownership, range normalization and event ordering.
3. **Codec regression gate:** every old token ID must remain unchanged; the
   rhythm-free codec must reserve the two rhythm positions; bend IDs must be
   identical with rhythm on and off; and both vocabularies must remain padded
   to 1,536.
4. **Round-trip gate:** encode/decode notes plus bends, including anchored
   holds, slide-ins, slide-outs, parallel chords and same-time reset events.
5. **Boundary gate:** bends crossing ordinary, lookback and lookahead segment
   boundaries must neither duplicate nor remain stuck.
6. **Length gate:** run the real bend-aware SeqIO pipeline through the target
   length audit for 256- and 512-frame windows.
7. **Restore gate:** restore the chosen existing checkpoint and complete one
   forward/backward/save/resume step with the new task.
8. **Overfit gate:** overfit a tiny balanced set containing at least one bend
   of each supported direction/shape and verify decoded MIDI numerically.
9. **Pilot gate:** compare row initialization choices and sampling ratios while
   holding the train/test material fixed.
10. **Regression gate:** require note F1 and false-bend rate on no-bend audio to
    remain within predefined tolerances.

## Current recommendation

For a technical-feasibility pilot, use one fixed physical vocabulary layout:
the two rhythm positions are always allocated, and an optional pitch-bend
range follows them. The no-rhythm configuration treats the rhythm positions as
reserved/invalid rather than allowing bend tokens to occupy them. Warm-start
the existing model and begin with either the 25-value direct state
representation or the 27-class sparse representation consisting of two curve
modes plus 25 values. Both preserve checkpoint tensor shapes, keep bend IDs
stable across the rhythm switch, and leave more room than the 43-value hybrid
codebook.

Do not launch a bend-aware training leg until the renderer and serialization
gates pass. Once they do, the main uncertainty is data efficiency and decoder
adaptation, not whether MT3 can technically represent or emit the events.
