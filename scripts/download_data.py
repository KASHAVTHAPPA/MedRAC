"""Download the datasets and the embedding model.

    python scripts/download_data.py            # everything missing
    python scripts/download_data.py --model    # only the MiniLM ONNX model

Downloads to data/raw/ (git-ignored) and models/minilm/:
  * MedQuAD (CC BY 4.0)                 knowledge base + cache workload
  * TREC 2017 LiveQA-Med test questions  retrieval evaluation
  * PubMedQA labelled set (MIT)          phase-2 evaluation
  * all-MiniLM-L6-v2, ONNX quint8 (Apache 2.0)
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODEL = ROOT / "models" / "minilm"

MEDQUAD_ZIP = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
LIVEQA_XML = (
    "https://raw.githubusercontent.com/abachaa/LiveQA_MedicalTask_TREC2017/master/TestDataset/"
    "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
)
PUBMEDQA = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
HF = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
MODEL_FILES = {"model_quint8_avx2.onnx": HF + "onnx/model_quint8_avx2.onnx", "tokenizer.json": HF + "tokenizer.json"}


def fetch(url: str) -> bytes:
    print("downloading", url)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def save(url: str, path: Path) -> None:
    if path.exists():
        print("exists     ", path.relative_to(ROOT))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fetch(url))


def medquad() -> None:
    target = RAW / "MedQuAD"
    if target.exists():
        print("exists      data/raw/MedQuAD")
        return
    with zipfile.ZipFile(io.BytesIO(fetch(MEDQUAD_ZIP))) as z:
        z.extractall(RAW)
    shutil.move(str(RAW / "MedQuAD-master"), str(target))
    # The judged LiveQA answers ship inside MedQuAD as a zip.
    inner = target / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip"
    with zipfile.ZipFile(inner) as z:
        z.extractall(RAW / "liveqa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="only download the embedding model")
    args = parser.parse_args()

    for name, url in MODEL_FILES.items():
        save(url, MODEL / name)
    if args.model:
        return
    medquad()
    save(LIVEQA_XML, RAW / "liveqa" / LIVEQA_XML.rsplit("/", 1)[1])
    save(PUBMEDQA, RAW / "ori_pqal.json")
    print("done")


if __name__ == "__main__":
    main()
