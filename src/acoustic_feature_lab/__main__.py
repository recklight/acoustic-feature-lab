"""Run the command-line interface with ``python -m acoustic_feature_lab``.

Equivalent to the ``acoustic-feature-lab`` console script, and usable straight from a
checkout without installing (``PYTHONPATH=src``).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess test
    main()
