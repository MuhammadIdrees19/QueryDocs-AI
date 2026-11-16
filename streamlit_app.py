import os
from uuid import uuid4

import streamlit as st
from groq import APIStatusError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage

from src.accounts import authenticate, login_with_google, reset_password, save_chats, signup
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
    initial_sidebar_state="collapsed",
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
    defaults = {
        "user": None,
        "chats": [],
        "active_chat_id": None,
        "messages": [],
        "auth_error": "",
        "auth_notice": "",
        "auth_view": "login",
        "settings_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def complete_login(user: dict) -> None:
    st.session_state.user = {"email": user["email"], "name": user["name"]}
    st.session_state.chats = list(user.get("chats") or [])
    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.auth_error = ""
    st.session_state.auth_notice = ""
    st.session_state.auth_view = "login"
    st.session_state.settings_open = False
    if st.session_state.chats:
        first = st.session_state.chats[0]
        st.session_state.active_chat_id = first["id"]
        st.session_state.messages = list(first.get("messages") or [])


def sign_out() -> None:
    logout = getattr(st, "logout", None)
    if logout is not None and bool(getattr(getattr(st, "user", None), "is_logged_in", False)):
        logout()
    st.session_state.user = None
    st.session_state.chats = []
    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.auth_error = ""
    st.session_state.auth_notice = ""
    st.session_state.auth_view = "login"
    st.session_state.settings_open = False


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


def inject_styles(*, hide_sidebar: bool, auth_layout: bool = False) -> None:
    hide_rule = (
        """
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        """
        if hide_sidebar
        else ""
    )
    auth_rule = ""
    if auth_layout:
        auth_rule = """
        .stApp:has(.auth-page) header[data-testid="stHeader"] { display: none; }
        .stApp:has(.auth-page) .block-container {
            max-width: 100%;
            padding: 0;
        }
        .stApp:has(.auth-page) [data-testid="stHorizontalBlock"] {
            gap: 0;
            align-items: stretch;
            min-height: 100vh;
        }
        .stApp:has(.auth-page) [data-testid="column"]:first-child {
            background:
                radial-gradient(circle at 18% 22%, rgba(255,255,255,.16), transparent 28%),
                radial-gradient(circle at 88% 78%, rgba(255,255,255,.12), transparent 26%),
                linear-gradient(180deg, #4d7dff 0%, #2f62f6 100%);
            color: white;
            min-height: 100vh;
        }
        .stApp:has(.auth-page) [data-testid="column"]:last-child {
            background: #ffffff;
            min-height: 100vh;
            display: flex;
            align-items: center;
        }
        .stApp:has(.auth-page) [data-testid="column"]:last-child [data-testid="stVerticalBlockBorderWrapper"] {
            width: min(100%, 440px);
            margin: 0 auto;
            padding: 48px 28px;
        }
        .brand-panel {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 28px 36px 24px;
            color: white;
        }
        .brand-lockup { display: flex; align-items: center; gap: 12px; font-weight: 700; }
        .brand-mark {
            width: 36px; height: 36px; border-radius: 10px;
            display: grid; place-items: center;
            background: white; color: #2f62f6; font-weight: 800;
        }
        .brand-copy h1 {
            color: white; font-size: 42px; line-height: 1.15;
            margin: 0 0 16px; max-width: 460px;
        }
        .brand-copy p { color: rgba(255,255,255,.88); font-size: 16px; max-width: 420px; margin: 0; }
        .brand-foot { color: rgba(255,255,255,.75); font-size: 13px; }
        .auth-kicker { color: #5c677a; margin: 0 0 18px; }
        .auth-divider { color: #8b95a7; text-align: center; margin: 8px 0 4px; font-size: 14px; }
        """
    st.markdown(
        f"""
        <style>
        {hide_rule}
        {auth_rule}
        section[data-testid="stSidebar"] > div:first-child {{
            height: 100%;
            display: flex;
            flex-direction: column;
        }}
        div[data-testid="stSidebarContent"] {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        .sidebar-spacer {{
            flex: 1 1 auto;
            min-height: 2rem;
        }}
        .sidebar-bottom {{
            margin-top: auto;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(49, 51, 63, 0.16);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def google_auth_configured() -> bool:
    try:
        auth = st.secrets["auth"]
    except Exception:
        return False
    return all(str(auth.get(key, "")).strip() for key in ("client_id", "client_secret", "redirect_uri", "cookie_secret"))


def start_google_login() -> None:
    if not google_auth_configured():
        st.session_state.auth_error = (
            "Google sign-in needs client_id, client_secret, redirect_uri, and cookie_secret "
            "under [auth] in the Streamlit secrets."
        )
        st.rerun()
    login = getattr(st, "login", None)
    if login is None:
        st.session_state.auth_error = "Update Streamlit to use Google sign-in."
        st.rerun()
    login()


def finish_google_login() -> None:
    user_info = getattr(st, "user", None)
    if not bool(getattr(user_info, "is_logged_in", False)):
        return
    email = str(getattr(user_info, "email", "") or "")
    name = str(getattr(user_info, "name", "") or email)
    try:
        account, error = login_with_google(email, name)
    except Exception as error:
        st.session_state.auth_error = f"Google sign-in failed. {error}"
        return
    if error or account is None:
        st.session_state.auth_error = error or "Google sign-in failed."
        return
    complete_login(account)
    st.rerun()


def render_google_button() -> None:
    if st.button("Continue with Google", use_container_width=True):
        start_google_login()


def render_login_page() -> None:
    inject_styles(hide_sidebar=True, auth_layout=True)
    st.markdown('<div class="auth-page"></div>', unsafe_allow_html=True)
    finish_google_login()
    brand, form = st.columns([1.05, 1], gap="small")
    with brand:
        st.markdown(
            """
            <div class="brand-panel">
              <div class="brand-lockup"><span class="brand-mark">Q</span><span>QueryDocs AI</span></div>
              <div class="brand-copy">
                <h1>Get answers straight from your documents.</h1>
                <p>Ask questions about ML, Finance and HR and get clear, cited answers from your own files.</p>
              </div>
              <div class="brand-foot">© 2026 QueryDocs AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with form:
        if st.session_state.auth_notice:
            st.success(st.session_state.auth_notice)
        if st.session_state.auth_error:
            st.error(st.session_state.auth_error)

        view = st.session_state.auth_view
        if view == "login":
            st.header("Log in")
            st.markdown('<p class="auth-kicker">Ask questions from your documents.</p>', unsafe_allow_html=True)
            with st.form("login-form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Password")
                submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")
            if submitted:
                try:
                    user, error = authenticate(email, password)
                except Exception as error:
                    st.session_state.auth_error = f"Sign in failed. {error}"
                    st.rerun()
                if error:
                    st.session_state.auth_error = error
                    st.rerun()
                complete_login(user)
                st.rerun()

            if st.button("Forgot password?"):
                st.session_state.auth_view = "forgot"
                st.session_state.auth_error = ""
                st.session_state.auth_notice = ""
                st.rerun()
            st.markdown('<p class="auth-divider">New to QueryDocs AI?</p>', unsafe_allow_html=True)
            if st.button("Create new account", use_container_width=True):
                st.session_state.auth_view = "signup"
                st.session_state.auth_error = ""
                st.session_state.auth_notice = ""
                st.rerun()
            return

        if view == "signup":
            st.header("Create new account")
            st.markdown('<p class="auth-kicker">Ask questions from your documents.</p>', unsafe_allow_html=True)
            render_google_button()
            with st.form("signup-form"):
                name = st.text_input("Full name", placeholder="Your name")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="At least 6 characters")
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Repeat your password",
                )
                submitted = st.form_submit_button(
                    "Create account",
                    use_container_width=True,
                    type="primary",
                )
            if submitted:
                if password != confirm_password:
                    st.session_state.auth_error = "Passwords do not match."
                    st.rerun()
                try:
                    error = signup(name, email, password)
                except Exception as error:
                    st.session_state.auth_error = f"Sign up failed. {error}"
                    st.rerun()
                if error:
                    st.session_state.auth_error = error
                    st.rerun()
                try:
                    user, sign_in_error = authenticate(email, password)
                except Exception as error:
                    st.session_state.auth_error = f"Account created, but sign in failed. {error}"
                    st.session_state.auth_view = "login"
                    st.rerun()
                if sign_in_error or user is None:
                    st.session_state.auth_error = (
                        sign_in_error or "Account created. Log in to continue."
                    )
                    st.session_state.auth_view = "login"
                    st.rerun()
                complete_login(user)
                st.rerun()
            if st.button("Back to log in", use_container_width=True):
                st.session_state.auth_view = "login"
                st.session_state.auth_error = ""
                st.rerun()
            return

        st.subheader("Reset password")
        with st.form("forgot-form"):
            email = st.text_input("Email", placeholder="Email address")
            new_password = st.text_input(
                "New password",
                type="password",
                placeholder="New password",
            )
            submitted = st.form_submit_button(
                "Update password",
                use_container_width=True,
                type="primary",
            )
        if submitted:
            try:
                error = reset_password(email, new_password)
            except Exception as error:
                st.session_state.auth_error = f"Password reset failed. {error}"
                st.rerun()
            if error:
                st.session_state.auth_error = error
                st.rerun()
            st.session_state.auth_error = ""
            st.session_state.auth_notice = "Password updated. Log in with the new password."
            st.session_state.auth_view = "login"
            st.rerun()
        if st.button("Back to log in", use_container_width=True):
            st.session_state.auth_view = "login"
            st.session_state.auth_error = ""
            st.rerun()


def render_sidebar() -> None:
    inject_styles(hide_sidebar=False)
    with st.sidebar:
        st.header("QueryDocs AI")
        st.caption("Ask about ML, Finance, and HR")
        if st.button("New chat", use_container_width=True):
            start_new_chat()
            st.rerun()
        st.divider()
        st.caption("Chat history")
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

        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)
        if st.button("Settings", use_container_width=True):
            st.session_state.settings_open = not st.session_state.settings_open
            st.rerun()
        if st.session_state.settings_open:
            st.caption(st.session_state.user["email"])
            if st.button("Sign out", use_container_width=True):
                sign_out()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


ensure_session()

if st.session_state.user is None:
    render_login_page()
    st.stop()

render_sidebar()

st.title("QueryDocs AI")
st.caption("Ask a question about your documents. Answers include the source page.")

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
