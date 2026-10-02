"""Start the local web application (see src/aidetect/app.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from aidetect.app import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
