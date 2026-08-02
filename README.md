# MT3: Multi-Task Multitrack Music Transcription

MT3 is a multi-instrument automatic music transcription model that uses the [T5X framework](https://github.com/google-research/t5x).

This is not an officially supported Google product.

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
