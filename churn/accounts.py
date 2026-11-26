"""Accounts: register, login, current user. Passwords are PBKDF2-SHA256 hashed, sessions are signed JWTs.

Signed-in requests send `Authorization: Bearer <token>` and own their data under the owner key
`user:<id>`. Requests without a token fall back to the anonymous guest demo (`X-Client-Id`).
"""

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


ALGO = "HS256"
ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _secret() -> str:
    explicit = os.getenv("JWT_SECRET")
    if explicit:
        return explicit
    # No separate secret configured: derive one from the database URL, which is already a secret in deployment.
    seed = os.getenv("DATABASE_URL", "local-dev-only")
    return hashlib.sha256(("jwt:" + seed).encode()).hexdigest()


_url = os.getenv("DATABASE_URL", "sqlite:///./churn_accounts.db")
if _url.startswith(("postgres://", "postgresql://")):
    _url = "postgresql+psycopg2://" + _url.split("://", 1)[1]
engine = create_engine(_url, pool_pre_ping=True, connect_args={"check_same_thread": False} if _url.startswith("sqlite") else {})
Session = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "churn_users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split("$")
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), ITERATIONS)
    return hmac.compare_digest(dk.hex(), dk_hex)


def create_token(user_id: int, minutes: int = 60 * 24 * 7) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return jwt.encode({"sub": str(user_id), "exp": exp}, _secret(), algorithm=ALGO)


def user_id_from_token(token: str) -> int:
    return int(jwt.decode(token, _secret(), algorithms=[ALGO])["sub"])


def identity(authorization: str, x_client_id: str) -> str:
    """Owner key for a request: a signed-in account, or an anonymous guest id."""
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Invalid Authorization header")
        try:
            return f"user:{user_id_from_token(token)}"
        except (jwt.PyJWTError, ValueError):
            raise HTTPException(401, "Session expired or invalid, please log in again") from None
    if not (8 <= len(x_client_id) <= 64) or not x_client_id.replace("-", "").isalnum():
        raise HTTPException(400, "Missing or invalid X-Client-Id header")
    return x_client_id


class Credentials(BaseModel):
    email: str = Field(max_length=200)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="", max_length=120)


class LoginIn(BaseModel):
    email: str = Field(max_length=200)
    password: str = Field(max_length=128)


def _get_db():
    with Session() as s:
        yield s


router = APIRouter(prefix="/auth", tags=["auth"])


def _out(user: User) -> dict:
    return {
        "access_token": create_token(user.id),
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name},
    }


@router.post("/register", status_code=201)
def register(body: Credentials, db=Depends(_get_db)):
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "Enter a valid email address")
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    user = User(email=email, name=body.name.strip(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return _out(user)


@router.post("/login")
def login(body: LoginIn, db=Depends(_get_db)):
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong email or password")
    return _out(user)


@router.get("/me")
def me(
    authorization: str = Header(default=""),
    x_client_id: str = Header(default=""),
    db=Depends(_get_db),
):
    owner = identity(authorization, x_client_id)
    if not owner.startswith("user:"):
        raise HTTPException(401, "Not signed in")
    user = db.get(User, int(owner.split(":")[1]))
    if not user:
        raise HTTPException(401, "Account not found")
    return {"id": user.id, "email": user.email, "name": user.name}


class Prediction(Base):
    __tablename__ = "churn_predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("churn_users.id", ondelete="CASCADE"), index=True)
    features: Mapped[dict] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    will_churn: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def init():
    Base.metadata.create_all(engine)


def save_prediction(owner: str, features: dict, probability: float, will_churn: bool, version: str):
    if not owner.startswith("user:"):
        return
    with Session() as s:
        s.add(Prediction(user_id=int(owner[5:]), features=features, probability=probability,
                         will_churn=int(will_churn), model_version=version))
        s.commit()


history_router = APIRouter(tags=["history"])


@history_router.get("/history")
def history(authorization: str = Header(default=""), x_client_id: str = Header(default=""), db=Depends(_get_db)):
    owner = identity(authorization, x_client_id)
    if not owner.startswith("user:"):
        raise HTTPException(401, "Log in to see saved predictions")
    rows = db.scalars(select(Prediction).where(Prediction.user_id == int(owner[5:])).order_by(Prediction.id.desc()).limit(50)).all()
    return [{"id": r.id, "features": r.features, "churn_probability": r.probability, "will_churn": bool(r.will_churn),
             "model_version": r.model_version, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]


@history_router.delete("/history", status_code=204)
def clear_history(authorization: str = Header(default=""), x_client_id: str = Header(default=""), db=Depends(_get_db)):
    owner = identity(authorization, x_client_id)
    if not owner.startswith("user:"):
        raise HTTPException(401, "Log in first")
    for r in db.scalars(select(Prediction).where(Prediction.user_id == int(owner[5:]))).all():
        db.delete(r)
    db.commit()
