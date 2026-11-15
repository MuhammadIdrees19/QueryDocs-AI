import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance


logger = logging.getLogger(__name__)


class QdrantVectorStore(LangChainQdrantVectorStore):
    """Skip the collection lookup LangChain repeats before every search."""

    @classmethod
    def _validate_collection_for_dense(
        cls,
        client: QdrantClient,
        collection_name: str,
        vector_name: str,
        distance: Distance,
        dense_embeddings: Any,
    ) -> None:
        return


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
COLLECTIONS = {
    "textbook": "AI _ML",
    "financial_policy": "Finance",
    "staff_handbook": "HR",
}
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


def get_compute_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model_kwargs(*, local_files_only: bool = False) -> dict[str, Any]:
    device = get_compute_device()
    model_kwargs: dict[str, Any] = {"device": device}
    if local_files_only:
        model_kwargs["local_files_only"] = True
    if device == "cuda":
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return model_kwargs


def build_embeddings() -> HuggingFaceEmbeddings:
    device = get_compute_device()
    logger.info("Loading %s embeddings on %s.", EMBEDDING_MODEL, device)
    try:
        return HuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            model_kwargs=build_model_kwargs(local_files_only=True),
        )
    except OSError:
        logger.info("Downloading %s embeddings.", EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            model_kwargs=build_model_kwargs(),
        )


def load_chunk_documents(
    chunk_files: list[Path],
) -> dict[str, list[tuple[str, Document]]]:
    documents_by_collection: dict[str, list[tuple[str, Document]]] = {
        collection: [] for collection in COLLECTIONS.values()
    }

    for path in chunk_files:
        chunks = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(chunks, list):
            raise ValueError(f"Expected a list of chunks in {path}")

        for chunk in chunks:
            metadata = chunk["metadata"]
            collection_name = COLLECTIONS.get(metadata["document_type"])
            if collection_name is None:
                raise ValueError(
                    f"Unsupported document type {metadata['document_type']!r} "
                    f"in {path.name}"
                )

            metadata = {**metadata, "chunk_id": chunk["chunk_id"]}
            document = Document(page_content=chunk["text"], metadata=metadata)
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{collection_name}:{metadata['source']}:{chunk['chunk_id']}",
                )
            )
            documents_by_collection[collection_name].append((point_id, document))

    return documents_by_collection


def validate_collection(
    client: QdrantClient,
    collection_name: str,
) -> str:
    try:
        info = client.get_collection(collection_name)
    except UnexpectedResponse as error:
        raise ValueError(
            f"Qdrant collection {collection_name!r} does not exist. "
            "Create it with 1024-dimensional cosine vectors first."
        ) from error
    vectors = info.config.params.vectors
    vector_name = ""
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            raise ValueError(
                f"Collection {collection_name!r} must have exactly one vector configuration."
            )
        vector_name, vector_config = next(iter(vectors.items()))
    else:
        vector_config = vectors

    if vector_config.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Collection {collection_name!r} has vector size {vector_config.size}; "
            f"{EMBEDDING_MODEL} requires {EMBEDDING_DIMENSION}."
        )
    if vector_config.distance != Distance.COSINE:
        raise ValueError(
            f"Collection {collection_name!r} must use cosine distance."
        )
    return vector_name


def require_qdrant_settings() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    required_variables = ("QDRANT_URL", "QDRANT_API_KEY")
    missing_variables = [
        variable for variable in required_variables if not os.getenv(variable)
    ]
    if missing_variables:
        raise RuntimeError(
            "Set these variables in the project .env file: "
            + ", ".join(missing_variables)
        )


def build_qdrant_client() -> QdrantClient:
    require_qdrant_settings()
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=30,
    )


def build_vector_stores(
    client: QdrantClient | None = None,
) -> dict[str, QdrantVectorStore]:
    qdrant_client = client or build_qdrant_client()
    vector_names = {
        collection_name: validate_collection(qdrant_client, collection_name)
        for collection_name in COLLECTIONS.values()
    }

    embeddings = build_embeddings()
    dimension = embeddings._client.get_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{EMBEDDING_MODEL} returned {dimension} dimensions; "
            f"Qdrant collections require {EMBEDDING_DIMENSION}."
        )

    return {
        collection_name: QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
            vector_name=vector_names[collection_name],
            validate_collection_config=False,
        )
        for collection_name in COLLECTIONS.values()
    }


def existing_document_matches(point: Any, document: Document) -> bool:
    payload = point.payload or {}
    return (
        payload.get("page_content") == document.page_content
        and payload.get("metadata") == document.metadata
    )


def index_documents(
    client: QdrantClient,
    stores: dict[str, QdrantVectorStore],
    documents_by_collection: dict[str, list[tuple[str, Document]]],
) -> None:
    for collection_name, entries in documents_by_collection.items():
        if not entries:
            continue

        point_ids = [point_id for point_id, _ in entries]
        existing_points = client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        existing_by_id = {str(point.id): point for point in existing_points}
        documents_to_upsert = []
        ids_to_upsert = []

        for point_id, document in entries:
            existing = existing_by_id.get(point_id)
            if existing is None or not existing_document_matches(existing, document):
                ids_to_upsert.append(point_id)
                documents_to_upsert.append(document)

        if documents_to_upsert:
            stores[collection_name].add_documents(
                documents=documents_to_upsert,
                ids=ids_to_upsert,
            )
            print(
                f"Stored {len(documents_to_upsert)} new or changed chunks "
                f"in Qdrant collection {collection_name}."
            )
        else:
            print(f"Qdrant collection {collection_name} is already up to date.")


def main() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk JSON files found in {CHUNKS_DIR}. Run utils/chunks.py first."
        )

    documents_by_collection = load_chunk_documents(chunk_files)
    client = build_qdrant_client()
    stores = build_vector_stores(client)
    index_documents(client, stores, documents_by_collection)


if __name__ == "__main__":
    main()
