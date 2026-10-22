"""Ask MedRAC-QA from the command line.

    python scripts/ask.py "What are the symptoms of anemia?"
    python scripts/ask.py --mode llm_only "What causes asthma?"
    python scripts/ask.py --json "..."          # full result as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrac_qa.service import build_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--mode", default="full", choices=["full", "rag", "llm_only"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pipeline = build_pipeline()
    for q in args.questions:
        r = pipeline.answer(q, args.mode)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
            continue
        print(f"\nQ: {q}\n[{r.route}]  {r.timings_ms.get('total')} ms  evidence={r.evidence_score}")
        if r.cache:
            print(f"cache: {r.cache['reason']}")
        print(r.answer)
        for c in r.citations:
            print(f"  [{c['n']}] {c['focus']} ({c['source']}) {c['url']}")
        if r.validation and not r.validation["passed"]:
            print(f"validation: {r.validation['reason']}")


if __name__ == "__main__":
    main()
