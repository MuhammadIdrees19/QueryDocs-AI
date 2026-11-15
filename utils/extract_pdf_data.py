import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "process data"


def main() -> None:
    pdf_paths = sorted(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {RAW_DATA_DIR}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        documents = PyPDFLoader(str(pdf_path)).load()
        pages = [
            {
                "source": pdf_path.name,
                "page": document.metadata["page"] + 1,
                "text": document.page_content,
            }
            for document in documents
        ]
        output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}.json"
        output_path.write_text(
            json.dumps(pages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Extracted {len(pages)} pages: {pdf_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
