"""PDF 청킹 후보를 비교해 프로젝트에 맞는 설정을 추천한다."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


CANDIDATES = ((300, 40), (400, 50), (500, 80), (700, 100))
DEFAULT_PDF_DIR = Path(__file__).resolve().parents[1] / "datasets"


def load_pages(pdf_dir: Path) -> list[str]:
    pages = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pages.extend(
            document.page_content.strip()
            for document in PyMuPDFLoader(str(pdf_path)).load()
            if document.page_content.strip()
        )
    return pages


def evaluate(pages: list[str], chunk_size: int, chunk_overlap: int) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [chunk for page in pages for chunk in splitter.split_text(page)]
    lengths = [len(chunk) for chunk in chunks]
    short_ratio = sum(length < chunk_size * 0.25 for length in lengths) / max(len(chunks), 1)
    sentence_end_ratio = sum(bool(re.search(r"[.!?。]|다\.?$", chunk.rstrip())) for chunk in chunks) / max(len(chunks), 1)
    overlap_cost = chunk_overlap / chunk_size

    # 지나치게 짧은 조각과 중복 저장은 감점하고, 문장 경계 보존은 가점한다.
    score = sentence_end_ratio * 0.55 + (1 - short_ratio) * 0.30 + (1 - overlap_cost) * 0.15
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": len(chunks),
        "average_length": round(sum(lengths) / max(len(lengths), 1), 1),
        "short_ratio": round(short_ratio, 3),
        "sentence_end_ratio": round(sentence_end_ratio, 3),
        "score": round(score, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    args = parser.parse_args()

    pages = load_pages(args.pdf_dir)
    if not pages:
        raise SystemExit(f"PDF를 찾지 못했습니다: {args.pdf_dir}")

    results = [evaluate(pages, size, overlap) for size, overlap in CANDIDATES]
    results.sort(key=lambda result: result["score"], reverse=True)

    print(f"평가 문서: {len(pages)}페이지")
    print("size overlap chunks avg_len short_ratio sentence_end score")
    for result in results:
        print(
            f"{result['chunk_size']:>4} {result['chunk_overlap']:>7} "
            f"{result['chunks']:>6} {result['average_length']:>7} "
            f"{result['short_ratio']:>11} {result['sentence_end_ratio']:>12} "
            f"{result['score']:>5}"
        )

    best = results[0]
    print(
        "\n추천 설정: "
        f"chunk_size={best['chunk_size']}, "
        f"chunk_overlap={best['chunk_overlap']}"
    )


if __name__ == "__main__":
    main()
