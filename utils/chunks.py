import json
import re
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "process data"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNK_SIZES = {
    "textbook": 550,
    "financial_policy": 300,
    "staff_handbook": 400,
    "default": 400,
}


def find_repeated_lines(pages: list[dict[str, Any]]) -> set[str]:
    line_counts: Counter[str] = Counter()
    for page in pages:
        page_lines = {
            line.strip().casefold()
            for line in page["text"].splitlines()
            if line.strip() and len(line.strip()) <= 100
        }
        line_counts.update(page_lines)

    threshold = max(3, ceil(len(pages) * 0.25))
    return {line for line, count in line_counts.items() if count >= threshold}


def normalize_page_text(text: str, repeated_lines: set[str]) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = line.strip()
        if not line or line.casefold() in repeated_lines:
            continue
        if re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,8})", line, re.IGNORECASE):
            continue
        if re.fullmatch(r"page\s*\|\s*\d+", line, re.IGNORECASE):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def get_document_type(source: str) -> str:
    source_name = source.lower()
    if "artificial intelligence" in source_name:
        return "textbook"
    if "financial" in source_name:
        return "financial_policy"
    if "staff" in source_name and "handbook" in source_name:
        return "staff_handbook"
    return "default"


def build_document_text(
    pages: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int, int]]]:
    text_parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    position = 0
    repeated_lines = find_repeated_lines(pages)

    for page in pages:
        page_text = normalize_page_text(page["text"], repeated_lines)
        if not page_text:
            continue

        if text_parts:
            text_parts.append("\n\n")
            position += 2

        start = position
        text_parts.append(page_text)
        position += len(page_text)
        page_spans.append((start, position, int(page["page"])))

    return "".join(text_parts), page_spans


def get_page_range(
    start: int,
    end: int,
    page_spans: list[tuple[int, int, int]],
) -> tuple[int, int]:
    pages = [
        page
        for page_start, page_end, page in page_spans
        if start < page_end and end > page_start
    ]
    if not pages:
        raise ValueError(f"Unable to map chunk character range {start}:{end} to a page")
    return min(pages), max(pages)


def chunk_file(input_path: Path) -> Path:
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"Expected a non-empty page list in {input_path}")

    source = pages[0]["source"]
    document_type = get_document_type(source)
    full_text, page_spans = build_document_text(pages)
    if not full_text:
        raise ValueError(f"No extracted text found in {input_path}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZES[document_type],
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    split_documents = splitter.create_documents([full_text])

    chunks = []
    for index, document in enumerate(split_documents, start=1):
        start = document.metadata.get("start_index")
        if start is None or start < 0:
            raise ValueError(f"LangChain did not provide a valid chunk offset for {source}")

        page_start, page_end = get_page_range(
            start,
            start + len(document.page_content),
            page_spans,
        )
        chunks.append(
            {
                "chunk_id": f"{input_path.stem}-{index:04d}",
                "text": document.page_content,
                "metadata": {
                    "source": source,
                    "document_type": document_type,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": index,
                },
            }
        )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHUNKS_DIR / f"{input_path.stem}_chunks.json"
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created {len(chunks)} chunks: {input_path.name} -> {output_path.name}")
    return output_path


def main() -> None:
    input_files = sorted(PROCESSED_DATA_DIR.glob("*.json"))
    if not input_files:
        raise FileNotFoundError(f"No processed JSON files found in {PROCESSED_DATA_DIR}")

    for input_path in input_files:
        chunk_file(input_path)


if __name__ == "__main__":
    main()
