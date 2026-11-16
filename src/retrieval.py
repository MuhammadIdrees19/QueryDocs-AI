import json
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Literal, override

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field

from src.embedding import build_model_kwargs, build_vector_stores, get_compute_device
from src.generation import build_chat_model, format_page_label


logger = logging.getLogger(__name__)
RERANKER_MODEL = "BAAI/bge-reranker-base"
RESULTS_PER_COLLECTION = 10
MAX_RESULTS = 6
CollectionName = Literal["AI _ML", "Finance", "HR"]
ALL_COLLECTIONS: tuple[CollectionName, ...] = ("AI _ML", "Finance", "HR")
COLLECTION_ALIASES = {
    "ai _ml": "AI _ML",
    "ai_ml": "AI _ML",
    "ai/ml": "AI _ML",
    "ai ml": "AI _ML",
    "aiml": "AI _ML",
    "machine learning": "AI _ML",
    "finance": "Finance",
    "hr": "HR",
}


class RouteDecision(BaseModel):
    collections: list[CollectionName] = Field(min_length=1)


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Route the user's question to one or more knowledge collections. "
            "Choose AI _ML for artificial intelligence or machine learning, "
            "Finance for financial policies, and HR for staff handbook topics. "
            "Choose multiple collections when the question spans topics. "
            "If you cannot confidently identify the relevant collection, choose "
            "all three. Return only the structured collection selection; do not "
            "answer the question.",
        ),
        ("human", "{question}"),
    ]
)


def build_router() -> Runnable:
    llm = build_chat_model(for_routing=True)
    return ROUTER_PROMPT | llm.with_structured_output(
        RouteDecision,
        method="function_calling",
    )


def _keyword_collections(question: str) -> list[str]:
    text = question.casefold()
    selected: list[str] = []

    def mentions(*terms: str) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)

    if mentions("machine learning", "deep learning", "artificial intelligence", "neural", "ai", "ml"):
        selected.append("AI _ML")
    if mentions("finance", "financial", "budget", "invoice", "expense", "payroll"):
        selected.append("Finance")
    if mentions("hr", "human resources", "staff", "handbook", "employee", "leave"):
        selected.append("HR")
    return selected


def _normalize_collection(name: object) -> str | None:
    key = " ".join(str(name).strip().casefold().split())
    mapped = COLLECTION_ALIASES.get(key, str(name).strip())
    if mapped in ALL_COLLECTIONS:
        return mapped
    return None


def _decision_from_payload(payload: object) -> RouteDecision | None:
    if isinstance(payload, RouteDecision):
        return payload
    if isinstance(payload, BaseMessage):
        tool_calls = getattr(payload, "tool_calls", None) or []
        for call in tool_calls:
            parsed = _decision_from_payload(call.get("args"))
            if parsed is not None:
                return parsed
        return _decision_from_payload(payload.content)
    if isinstance(payload, list):
        for item in payload:
            parsed = _decision_from_payload(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(payload, str):
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return _decision_from_payload(json.loads(payload[start : end + 1]))
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        if "parsed" in payload:
            parsed = _decision_from_payload(payload.get("parsed"))
            if parsed is not None:
                return parsed
        names = payload.get("collections", payload.get("collection"))
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list):
            return None
        collections = []
        for name in names:
            mapped = _normalize_collection(name)
            if mapped and mapped not in collections:
                collections.append(mapped)
        if not collections:
            return None
        return RouteDecision(collections=collections)
    return None


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """Keep the cross-encoder score LangChain's compressor otherwise drops."""

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        del callbacks
        if not documents:
            return []

        scores = self.model.score(
            [(query, document.page_content) for document in documents]
        )
        if len(scores) != len(documents):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {len(documents)} documents."
            )

        ranked_results = sorted(
            zip(documents, scores, strict=True),
            key=lambda result: result[1],
            reverse=True,
        )
        selected = []
        for document, score in ranked_results[: self.top_n]:
            metadata = {
                **document.metadata,
                "reranker_score": float(score),
                "page_label": format_page_label(
                    document.metadata["page_start"],
                    document.metadata["page_end"],
                ),
            }
            selected.append(document.model_copy(update={"metadata": metadata}))
        return selected


def build_reranker() -> HuggingFaceCrossEncoder:
    device = get_compute_device()
    logger.info("Loading %s reranker on %s.", RERANKER_MODEL, device)
    try:
        return HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            model_kwargs=build_model_kwargs(local_files_only=True),
        )
    except OSError:
        logger.info("Downloading %s reranker.", RERANKER_MODEL)
        return HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            model_kwargs=build_model_kwargs(),
        )


def route_query(question: str, router: Runnable) -> list[str]:
    question = question.strip()
    if not question:
        raise ValueError("The question must not be empty.")

    keyword_match = _keyword_collections(question)
    if keyword_match:
        return keyword_match

    try:
        decision = _decision_from_payload(router.invoke({"question": question}))
    except Exception:
        logger.exception("Question routing failed. Searching all collections.")
        return list(ALL_COLLECTIONS)
    if decision is None:
        logger.warning("Router did not return collections. Searching all collections.")
        return list(ALL_COLLECTIONS)
    return list(dict.fromkeys(decision.collections))


def retrieve_documents(
    question: str,
    collection_names: list[str],
    vector_stores: dict[str, QdrantVectorStore],
    reranker: HuggingFaceCrossEncoder,
) -> list[tuple[Document, float]]:
    for collection_name in collection_names:
        if collection_name not in vector_stores:
            raise ValueError(f"Unknown Qdrant collection: {collection_name}")
    if not collection_names:
        return []

    search_started = perf_counter()
    query_vector = vector_stores[collection_names[0]].embeddings.embed_query(question)

    def search_collection(collection_name: str) -> list[Document]:
        return vector_stores[collection_name].similarity_search_by_vector(
            query_vector,
            k=RESULTS_PER_COLLECTION,
        )

    if len(collection_names) == 1:
        candidates = search_collection(collection_names[0])
    else:
        with ThreadPoolExecutor(max_workers=len(collection_names)) as pool:
            batches = pool.map(search_collection, collection_names)
        candidates = [document for batch in batches for document in batch]
    logger.info(
        "Vector search and query embedding took %.2f seconds for %d candidate(s).",
        perf_counter() - search_started,
        len(candidates),
    )

    if not candidates:
        return []

    rerank_started = perf_counter()
    reranked = ScoredCrossEncoderReranker(
        model=reranker,
        top_n=MAX_RESULTS,
    ).compress_documents(candidates, question)
    logger.info(
        "Reranking %d candidate(s) took %.2f seconds.",
        len(candidates),
        perf_counter() - rerank_started,
    )
    return [
        (document, float(document.metadata["reranker_score"]))
        for document in reranked
    ]


def route_and_retrieve(
    question: str,
    router: Runnable,
    vector_stores: dict[str, QdrantVectorStore],
    reranker: HuggingFaceCrossEncoder,
) -> tuple[list[str], list[tuple[Document, float]]]:
    route_started = perf_counter()
    collection_names = route_query(question, router)
    logger.info(
        "Question routing took %.2f seconds.",
        perf_counter() - route_started,
    )
    documents = retrieve_documents(
        question,
        collection_names,
        vector_stores,
        reranker,
    )
    return collection_names, documents


def main() -> None:
    router = build_router()
    vector_stores = build_vector_stores()
    reranker = build_reranker()
    print("Router, retrieval, and reranking are ready. Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("Please enter a question.")
            continue

        collections, results = route_and_retrieve(
            question,
            router,
            vector_stores,
            reranker,
        )
        print(f"Routed to: {', '.join(collections)}")
        if not results:
            print("No matching chunks were found.")
            continue

        for index, (document, score) in enumerate(results, start=1):
            metadata = document.metadata
            print(
                f"\n[{index}] {metadata['source']} | PDF page(s) {metadata['page_label']} "
                f"| reranker score {score:.4f}"
            )
            print(document.page_content)


if __name__ == "__main__":
    main()
