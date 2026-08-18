"""Put eval/ on sys.path so tests import its modules the same way the eval
scripts do (``from query_parser import parse``)."""

import os
import sys

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
