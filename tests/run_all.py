"""Run every test module. `python tests/run_all.py`"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chunker
import test_dedup
import test_diet
import test_extract
import test_gate
import test_groundtruth
import test_judge
import test_nutrition
import test_present
import test_query
import test_rerank
import test_web

if __name__ == "__main__":
    modules = (
        test_diet, test_nutrition, test_dedup, test_chunker,
        test_gate, test_judge, test_extract, test_groundtruth,
        test_query, test_rerank, test_web, test_present,
    )
    failed = sum(module.run() for module in modules)
    print("ALL PASSED" if not failed else f"{failed} MODULE(S) FAILED")
    raise SystemExit(1 if failed else 0)
