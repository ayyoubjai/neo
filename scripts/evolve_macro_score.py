import os
import sys


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from evolve_score import main


if __name__ == "__main__":
    raise SystemExit(main())
