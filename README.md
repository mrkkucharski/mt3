# MT3: Multi-Task Multitrack Music Transcription

MT3 is a multi-instrument automatic music transcription model that uses the [T5X framework](https://github.com/google-research/t5x).

This is not an officially supported Google product.

## Use MT3 as a provisioned inference dependency

This fork exposes a stable, standalone inference boundary for applications such
as VGT.  Keep it in its own uv project and invoke its CLI; do not add the
TensorFlow/JAX/T5X dependency set to the application's Python environment.

VGT (or another host application) should clone a tagged release into its own
cache, then create the locked environment and download the model explicitly.
After this change is released as `v0.1.0`, the provisioning commands are:

```sh
git clone --branch v0.1.0 https://github.com/mrkkucharski/mt3.git /path/to/mt3
uv sync --project /path/to/mt3 --frozen
uv run --project /path/to/mt3 mt3-download-model \
  --output-dir /path/to/models/mt3 --json
```

The model command is resumable and verifies each downloaded file against the
integrity metadata published by the official MT3 bucket.  It does not run as
part of package installation.

Use the installed CLI for transcription:

```sh
uv run --project /path/to/mt3 mt3-transcribe \
  --checkpoint /path/to/models/mt3/checkpoint_0 \
  --input /path/to/drums.wav \
  --output /path/to/drums.mid \
  --json
```

With `--json`, stdout contains only this stable result object, suitable for a
host program to parse:

```json
{"drum_note_count": 81, "note_count": 284, "output": "/path/to/drums.mid", "programs": [0, 33, 48]}
```

The same environment also exposes a reusable Python API.  Keep one
`Transcriber` instance alive to process several files without repeatedly
restoring the checkpoint:

```python
from mt3.transcription import Transcriber

transcriber = Transcriber("/path/to/models/mt3/checkpoint_0")
result = transcriber.transcribe_file("drums.wav", "drums.mid")
```

The runtime applies its Apple-Silicon TensorFlow compatibility setting itself;
callers do not need to use the local `sitecustomize` `PYTHONPATH` hook.

## Reproducible Mac CPU environment

This fork uses [uv](https://docs.astral.sh/uv/) and Python 3.11 for the
Apple Silicon CPU development environment. Git-based dependencies are resolved
to immutable commits in `uv.lock`.

Install the system audio dependency once (skip this if `ffmpeg` is already on
your `PATH`):

```sh
brew install ffmpeg
```

```sh
uv sync --frozen
uv run python -c "import jax; print(jax.devices())"
uv run python -c "import mt3; print(mt3.__path__)"
uv run pytest -q mt3
```

The local `.venv` is intentionally not committed. Use a separate lock or
container definition for the Linux CUDA training environment; do not install
`jax-metal` into this CPU environment.

### Validate the guitar pilot dataset

From the MT3 repository, validate the dataset before adding it to a training
run:

```sh
uv run python mt3/scripts/validate_guitar_dataset.py --dataset ../data/pilot
```

Add `--strict` to treat incomplete metadata, such as an unrecorded render
preset, as a failure.

### Build MT3 training records from the pilot dataset

After validation, convert the MIDI/WAV pairs into MT3-compatible TFRecords:

```sh
uv run python mt3/scripts/build_guitar_tfrecord.py --dataset ../data/pilot
```

The builder preserves the four guitar lanes with distinct MT3 program IDs:
`clean-rhythm=26`, `clean-lead=27`, `distorted-rhythm=29`, and
`distorted-lead=30`. It writes one TFRecord per populated split to
`../data/pilot/tfrecord/`.

### Smoke-test the MT3 preprocessing task

Confirm that a local TFRecord is converted to spectrogram frames and
lane-labelled event tokens before starting training:

```sh
uv run python mt3/scripts/smoke_guitar_task.py
```

The pilot task is `guitar_pilot_notes_ties_vb1_train`. Keep
`PROGRAM_GRANULARITY = 'full'` in its training configuration so program IDs
continue to distinguish the four lanes.

### Run the local training smoke test

This three-step CPU run checks the task, model, optimizer, checkpoint and log
paths together. It is not a quality or accuracy experiment.

```sh
PYTHONPATH=mt3/local_cpu_sitecustomize uv run python -m t5x.train \
  --gin_file=mt3/gin/model.gin \
  --gin_file=mt3/gin/train.gin \
  --gin_file=mt3/gin/local_tiny.gin \
  --gin_file=mt3/gin/guitar_pilot_local.gin
```

### Transcribe a WAV with a fine-tuned guitar checkpoint

```sh
uv run python mt3/scripts/transcribe_guitar.py \
  --checkpoint ../runs/guitar_pilot_finetune/checkpoint_100 \
  --input /path/to/song.wav \
  --output /path/to/song_transcription.mid
```

The generated MIDI always contains `clean-rhythm`, `clean-lead`,
`distorted-rhythm`, and `distorted-lead` tracks. Lanes without predicted notes
are left empty.

On the local Apple-Silicon environment, the command automatically applies the
same TensorFlow input-pipeline workaround used by the training smoke run.

### Transcribe general music with the original MT3 checkpoint

```sh
uv run mt3-transcribe \
  --checkpoint ../model/mt3/checkpoint_0 \
  --input /path/to/song.wav \
  --output /path/to/transcription.mid
```

This command retains MT3's complete multi-instrument prediction, including all
predicted MIDI programs and drums. It does not apply the project's guitar-lane
mapping.

## Transcribe your own audio

Use our [colab notebook](https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb) to
transcribe audio files of your choosing.  You can use a pretrained checkpoint from
either a) the piano transcription model described in [our ISMIR 2021 paper](https://archives.ismir.net/ismir2021/paper/000030.pdf)
or b) the multi-instrument transcription model described in
[our ICLR 2022 paper](https://openreview.net/pdf?id=iMSjopcOn0p).


## Train a model

For now, we do not (easily) support training.  If you like, you can try to
follow the [T5X training instructions](https://github.com/google-research/t5x#training)
and use one of the tasks defined in [tasks.py](mt3/tasks.py).
