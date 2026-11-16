import hashlib
import os
import secrets
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.embedding import PROJECT_ROOT, build_qdrant_client


USERS_COLLECTION = "AppUsers"
VECTOR_SIZE = 4


def _user_id(email: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"querydocs-user:{email}"))


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _password_hash(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return digest.hex()


def _client():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    missing = [
        name
        for name in ("QDRANT_URL", "QDRANT_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Set these variables in the project .env file or in the Streamlit app secrets: "
            + ", ".join(missing)
        )
    client = build_qdrant_client()
    exists = getattr(client, "collection_exists", None)
    if exists is not None:
        missing = not exists(USERS_COLLECTION)
    else:
        try:
            client.get_collection(USERS_COLLECTION)
            missing = False
        except UnexpectedResponse:
            missing = True
    if missing:
        client.create_collection(
            collection_name=USERS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    return client


def _read_user(email: str) -> dict | None:
    client = _client()
    points = client.retrieve(
        collection_name=USERS_COLLECTION,
        ids=[_user_id(email)],
        with_payload=True,
    )
    if not points or not points[0].payload:
        return None
    return dict(points[0].payload)


def _write_user(user: dict) -> None:
    client = _client()
    client.upsert(
        collection_name=USERS_COLLECTION,
        points=[
            PointStruct(
                id=_user_id(user["email"]),
                vector=[0.0, 0.0, 0.0, 0.0],
                payload=user,
            )
        ],
    )


def signup(name: str, email: str, password: str) -> str | None:
    clean_name = " ".join(name.split())
    clean_email = _normalize_email(email)
    if not clean_name:
        return "Enter your name."
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        return "Enter a valid email."
    if len(password) < 6:
        return "Use a password of at least 6 characters."
    if _read_user(clean_email) is not None:
        return "An account with this email already exists. Sign in instead."

    salt = secrets.token_hex(16)
    _write_user(
        {
            "email": clean_email,
            "name": clean_name,
            "salt": salt,
            "password_hash": _password_hash(password, salt),
            "chats": [],
        }
    )
    return None


def authenticate(email: str, password: str) -> tuple[dict | None, str | None]:
    clean_email = _normalize_email(email)
    user = _read_user(clean_email)
    if user is None:
        return None, "Email or password is incorrect."
    salt = str(user.get("salt", ""))
    expected = str(user.get("password_hash", ""))
    if not salt or not secrets.compare_digest(_password_hash(password, salt), expected):
        return None, "Email or password is incorrect."
    chats = user.get("chats") or []
    return {"email": clean_email, "name": user.get("name") or clean_email, "chats": chats}, None


def save_chats(email: str, chats: list[dict]) -> None:
    user = _read_user(_normalize_email(email))
    if user is None:
        raise RuntimeError("Account not found.")
    user["chats"] = chats
    _write_user(user)
