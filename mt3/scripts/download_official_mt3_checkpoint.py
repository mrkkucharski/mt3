"""Compatibility wrapper for the installed ``mt3-download-model`` command."""

from mt3.model_download import download_model
from mt3.model_download import main


# Preserve the old importable helper name for local callers.
download = download_model


if __name__ == '__main__':
  raise SystemExit(main())
