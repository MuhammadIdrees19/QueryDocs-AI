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


def _public_user(user: dict, email: str) -> dict:
    return {
        "email": email,
        "name": user.get("name") or email,
        "chats": user.get("chats") or [],
    }


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
            "provider": "password",
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
    password_matches = bool(salt) and secrets.compare_digest(_password_hash(password, salt), expected)
    if not password_matches:
        if user.get("provider") == "google":
            return None, "This email uses Google sign-in."
        return None, "Email or password is incorrect."
    return _public_user(user, clean_email), None


def login_with_google(email: str, name: str) -> tuple[dict | None, str | None]:
    clean_email = _normalize_email(email)
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        return None, "Google did not return an email address."
    clean_name = " ".join((name or "").split()) or clean_email
    user = _read_user(clean_email)
    if user is None:
        salt = secrets.token_hex(16)
        _write_user(
            {
                "email": clean_email,
                "name": clean_name,
                "salt": salt,
                "password_hash": _password_hash(secrets.token_hex(32), salt),
                "provider": "google",
                "chats": [],
            }
        )
        user = _read_user(clean_email)
    if user is None:
        return None, "The Google account could not be saved."
    return _public_user(user, clean_email), None


def save_chats(email: str, chats: list[dict]) -> None:
    user = _read_user(_normalize_email(email))
    if user is None:
        raise RuntimeError("Account not found.")
    user["chats"] = chats
    _write_user(user)


def reset_password(email: str, new_password: str) -> str | None:
    clean_email = _normalize_email(email)
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        return "Enter a valid email."
    if len(new_password) < 6:
        return "Use a password of at least 6 characters."
    user = _read_user(clean_email)
    if user is None:
        return "No account was found for that email."
    salt = secrets.token_hex(16)
    user["salt"] = salt
    user["password_hash"] = _password_hash(new_password, salt)
    _write_user(user)
    return None
