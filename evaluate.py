"""Entry point: see src/aidetect/evaluate.py (python evaluate.py --help)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from aidetect.evaluate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
