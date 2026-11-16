import os

import streamlit as st
from groq import APIStatusError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage

SECRET_KEYS = ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


def apply_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into the environment the rest of the app reads."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in SECRET_KEYS:
        value = secrets.get(key) if hasattr(secrets, "get") else None
        if value:
            os.environ.setdefault(key, str(value))


st.set_page_config(
    page_title="QueryDocs AI",
    page_icon="Q",
    layout="centered",
)
apply_streamlit_secrets()

from src.embedding import build_vector_stores
from src.generation import build_answer_chain, generate_answer
from src.retrieval import build_reranker, build_router, route_and_retrieve


def collection_label(name: str) -> str:
    if name == "AI _ML":
        return "AI/ML"
    return name


def source_rows(ranked_documents: list) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for document, _score in ranked_documents:
        metadata = document.metadata
        key = (metadata["source"], metadata["page_start"], metadata["page_end"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "source": metadata["source"],
                "page_label": metadata["page_label"],
            }
        )
    return rows


@st.cache_resource(show_spinner="Loading models and connecting to Qdrant Cloud…")
def load_pipeline() -> dict:
    return {
        "router": build_router(),
        "vector_stores": build_vector_stores(),
        "reranker": build_reranker(),
        "answer_chain": build_answer_chain(),
    }


def chat_history(messages: list[dict]) -> list:
    history = []
    for message in messages[-8:]:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))
    return history


def answer_question(question: str, messages: list[dict]) -> dict:
    pipeline = load_pipeline()
    collections, ranked_documents = route_and_retrieve(
        question,
        pipeline["router"],
        pipeline["vector_stores"],
        pipeline["reranker"],
    )
    documents = [document for document, _score in ranked_documents]
    answer = generate_answer(
        question,
        documents,
        chat_history=chat_history(messages),
        chain=pipeline["answer_chain"],
    )
    return {
        "role": "assistant",
        "content": answer,
        "collections": [collection_label(name) for name in collections],
        "sources": source_rows(ranked_documents),
    }


def render_message(message: dict) -> None:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            st.caption(
                "Sources: "
                + " · ".join(
                    f"{item['source']} — page {item['page_label']}" for item in sources
                )
            )
        collections = message.get("collections") or []
        if collections:
            st.caption("Searched: " + ", ".join(collections))


st.title("QueryDocs AI")
st.caption("Ask a question about your documents. Answers include the source page.")

with st.sidebar:
    st.header("QueryDocs AI")
    st.write("Questions are routed to AI/ML, Finance, or HR, then answered from those documents.")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    load_pipeline()
except RuntimeError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "The search models could not be loaded. Check the Qdrant Cloud settings and try again."
    )
    st.caption(str(error))
    st.stop()

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask anything about your documents...")
if not question:
    st.stop()

question = question.strip()
if not question:
    st.stop()
if len(question) > 4000:
    st.warning("Ask a question of 4000 characters or fewer.")
    st.stop()

st.session_state.messages.append({"role": "user", "content": question})
render_message(st.session_state.messages[-1])

try:
    with st.spinner("Searching the documents…"):
        reply = answer_question(question, st.session_state.messages[:-1])
except RateLimitError:
    reply = {
        "role": "assistant",
        "content": "Groq rejected the request due to its rate limit. Wait briefly and try again.",
    }
except APIStatusError as error:
    if error.status_code == 401:
        detail = "Groq rejected the API key. Check GROQ_API_KEY."
    elif error.status_code == 403:
        detail = "Groq blocked this request. Check the key and account limits."
    else:
        detail = (
            f"Groq request failed with HTTP {error.status_code}. "
            "Check model availability and account limits."
        )
    reply = {"role": "assistant", "content": detail}
except Exception as error:
    reply = {
        "role": "assistant",
        "content": f"The question could not be answered. {error}",
    }

st.session_state.messages.append(reply)
render_message(reply)
