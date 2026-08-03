"""Compatibility wrapper for :mod:`mt3.cli`.

New integrations should invoke the installed ``mt3-transcribe`` command or
use :class:`mt3.transcription.Transcriber` directly.
"""

from mt3.cli import main
from mt3.transcription import write_multitrack_midi


if __name__ == '__main__':
  raise SystemExit(main())
