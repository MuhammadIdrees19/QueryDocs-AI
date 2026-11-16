import os
from uuid import uuid4

import streamlit as st
from groq import APIStatusError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage

from src.accounts import authenticate, save_chats, signup
from src.citations import matching_sources

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


def chat_title(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 42:
        return compact
    return compact[:41] + "…"


def source_rows(ranked_documents: list) -> list[dict]:
    rows = []
    seen = set()
    for document, score in ranked_documents:
        metadata = document.metadata
        key = (metadata["source"], metadata["page_start"], metadata["page_end"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "source": metadata["source"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "page_label": metadata["page_label"],
                "score": float(score),
            }
        )
    return rows


def ensure_session() -> None:
    if "user" not in st.session_state:
        st.session_state.user = None
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "auth_error" not in st.session_state:
        st.session_state.auth_error = ""


def sync_active_chat() -> None:
    active_id = st.session_state.active_chat_id
    if not active_id:
        return
    for chat in st.session_state.chats:
        if chat["id"] != active_id:
            continue
        chat["messages"] = st.session_state.messages
        first_question = next(
            (message["content"] for message in chat["messages"] if message["role"] == "user"),
            "",
        )
        if first_question:
            chat["title"] = chat_title(first_question)
        return


def persist_chats() -> None:
    user = st.session_state.user
    if user is None:
        return
    sync_active_chat()
    save_chats(user["email"], st.session_state.chats)


def start_new_chat() -> None:
    sync_active_chat()
    if st.session_state.user is not None and st.session_state.messages:
        save_chats(st.session_state.user["email"], st.session_state.chats)
    st.session_state.active_chat_id = None
    st.session_state.messages = []


def open_chat(chat_id: str) -> None:
    if chat_id == st.session_state.active_chat_id:
        return
    sync_active_chat()
    if st.session_state.user is not None:
        save_chats(st.session_state.user["email"], st.session_state.chats)
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            st.session_state.active_chat_id = chat_id
            st.session_state.messages = list(chat.get("messages") or [])
            return


def begin_chat(question: str) -> None:
    if st.session_state.active_chat_id:
        return
    chat_id = str(uuid4())
    st.session_state.chats.insert(
        0,
        {"id": chat_id, "title": chat_title(question), "messages": []},
    )
    st.session_state.active_chat_id = chat_id


def sign_out() -> None:
    st.session_state.user = None
    st.session_state.chats = []
    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.auth_error = ""


@st.cache_resource(show_spinner="Loading models and connecting to Qdrant Cloud…")
def load_pipeline() -> dict:
    return {
        "router": build_router(),
        "vector_stores": build_vector_stores(),
        "reranker": build_reranker(),
        "answer_chain": build_answer_chain(),
    }


def model_history(messages: list[dict]) -> list:
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
        chat_history=model_history(messages),
        chain=pipeline["answer_chain"],
    )
    return {
        "role": "assistant",
        "content": answer,
        "collections": [collection_label(name) for name in collections],
        "sources": matching_sources(answer, source_rows(ranked_documents)),
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


def render_account() -> None:
    mode = st.radio("Account", ["Sign in", "Sign up"], horizontal=True)
    if mode == "Sign in":
        with st.form("sign-in"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                user, error = authenticate(email, password)
            except Exception as error:
                st.session_state.auth_error = f"Sign in failed. {error}"
                return
            if error:
                st.session_state.auth_error = error
                return
            st.session_state.user = {"email": user["email"], "name": user["name"]}
            st.session_state.chats = list(user["chats"])
            st.session_state.active_chat_id = None
            st.session_state.messages = []
            st.session_state.auth_error = ""
            if st.session_state.chats:
                first = st.session_state.chats[0]
                st.session_state.active_chat_id = first["id"]
                st.session_state.messages = list(first.get("messages") or [])
            st.rerun()
        return

    with st.form("sign-up"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign up", use_container_width=True)
    if submitted:
        try:
            error = signup(name, email, password)
        except Exception as error:
            st.session_state.auth_error = f"Sign up failed. {error}"
            return
        if error:
            st.session_state.auth_error = error
            return
        try:
            user, sign_in_error = authenticate(email, password)
        except Exception as error:
            st.session_state.auth_error = f"Account created, but sign in failed. {error}"
            return
        if sign_in_error or user is None:
            st.session_state.auth_error = sign_in_error or "Account created. Sign in to continue."
            return
        st.session_state.user = {"email": user["email"], "name": user["name"]}
        st.session_state.chats = []
        st.session_state.active_chat_id = None
        st.session_state.messages = []
        st.session_state.auth_error = ""
        st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("QueryDocs AI")
        if st.session_state.user is None:
            render_account()
            if st.session_state.auth_error:
                st.error(st.session_state.auth_error)
            return

        st.caption(st.session_state.user["name"])
        if st.button("New chat", use_container_width=True):
            start_new_chat()
            st.rerun()
        st.divider()
        if st.session_state.chats:
            for chat in st.session_state.chats:
                selected = chat["id"] == st.session_state.active_chat_id
                if st.button(
                    chat.get("title") or "New chat",
                    key=f"chat-{chat['id']}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    open_chat(chat["id"])
                    st.rerun()
        else:
            st.caption("Your chats will appear here.")
        st.divider()
        if st.button("Sign out", use_container_width=True):
            sign_out()
            st.rerun()


ensure_session()
render_sidebar()

st.title("QueryDocs AI")
st.caption("Ask a question about your documents. Answers include the source page.")

if st.session_state.user is None:
    st.info("Sign in or create an account to start a chat.")
    st.stop()

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

begin_chat(question)
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
try:
    persist_chats()
except Exception as error:
    st.caption(f"The chat was not saved. {error}")
