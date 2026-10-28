"""鉴权与角色控制（无外部依赖：数据库用依赖覆盖替换）"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.core import deps
from src.core.deps import get_current_user, has_role, require_roles
from src.infra.database import get_db
from src.modules.user.model import Role
from src.utils import jwt_utils
from src.utils.jwt_utils import TokenError, encode_jwt, verify_jwt
from src.utils.password_utils import hash_password, verify_password


# ── JWT ──────────────────────────────────────────────────────────────────

def test_jwt_roundtrip():
    payload = verify_jwt(encode_jwt({"sub": "7", "role": "doctor"}))
    assert payload["sub"] == "7" and payload["role"] == "doctor"


def test_jwt_tampered_is_rejected():
    token = encode_jwt({"sub": "7"})
    with pytest.raises(TokenError):
        verify_jwt(token[:-2] + ("aa" if not token.endswith("aa") else "bb"))


def test_jwt_signed_with_other_key_is_rejected():
    forged = jwt.encode({"sub": "1", "role": "admin"}, "other-secret", algorithm="HS256")
    with pytest.raises(TokenError):
        verify_jwt(forged)


def test_jwt_expired_is_rejected():
    now = datetime.now(timezone.utc)
    expired = jwt.encode(
        {"sub": "1", "iat": now - timedelta(hours=2), "exp": now - timedelta(hours=1)},
        jwt_utils._get_secret(),
        algorithm="HS256",
    )
    with pytest.raises(TokenError, match="过期"):
        verify_jwt(expired)


def test_jwt_alg_none_is_rejected():
    unsigned = jwt.encode({"sub": "1"}, key=None, algorithm="none")
    with pytest.raises(TokenError):
        verify_jwt(unsigned)


# ── 密码 ─────────────────────────────────────────────────────────────────

def test_password_hash_and_verify():
    hashed = hash_password("correct-horse")
    assert hashed != "correct-horse"
    assert verify_password("correct-horse", hashed)
    assert not verify_password("wrong-pass", hashed)


# ── 角色判断 ─────────────────────────────────────────────────────────────

def test_admin_passes_every_check():
    assert has_role(Role.ADMIN.value, (Role.OPERATOR.value,))


def test_role_must_be_in_allowed_list():
    assert has_role(Role.DOCTOR.value, (Role.DOCTOR.value,))
    assert not has_role(Role.PATIENT.value, (Role.DOCTOR.value,))
    assert not has_role(Role.DOCTOR.value, (Role.OPERATOR.value,))


# ── 接口层：401 / 403 / 200 ──────────────────────────────────────────────

class _FakeDB:
    def __init__(self, users):
        self._users = users

    async def get(self, _model, user_id):
        return self._users.get(user_id)


def _client(users):
    app = FastAPI()

    async def fake_db():
        yield _FakeDB(users)

    app.dependency_overrides[get_db] = fake_db

    @app.get("/any")
    async def any_user(user=Depends(get_current_user)):
        return {"id": user.id}

    @app.get("/doctor")
    async def doctor_only(user=Depends(require_roles(Role.DOCTOR))):
        return {"role": user.role}

    return TestClient(app)


def _user(uid, role, active=True):
    return SimpleNamespace(id=uid, role=role, is_active=active)


def _bearer(uid, role="patient"):
    return {"Authorization": f"Bearer {encode_jwt({'sub': str(uid), 'role': role})}"}


def test_missing_token_returns_401():
    assert _client({}).get("/any").status_code == 401


def test_garbage_token_returns_401():
    r = _client({}).get("/any", headers={"Authorization": "Bearer not-a-token"})
    assert r.status_code == 401


def test_unknown_user_returns_401():
    assert _client({}).get("/any", headers=_bearer(99)).status_code == 401


def test_disabled_user_returns_401():
    c = _client({1: _user(1, "patient", active=False)})
    assert c.get("/any", headers=_bearer(1)).status_code == 401


def test_valid_token_returns_200():
    c = _client({1: _user(1, "patient")})
    r = c.get("/any", headers=_bearer(1))
    assert r.status_code == 200 and r.json() == {"id": 1}


def test_patient_gets_403_on_doctor_route():
    c = _client({1: _user(1, "patient")})
    assert c.get("/doctor", headers=_bearer(1)).status_code == 403


def test_doctor_and_admin_pass_doctor_route():
    c = _client({1: _user(1, "doctor"), 2: _user(2, "admin")})
    assert c.get("/doctor", headers=_bearer(1)).status_code == 200
    assert c.get("/doctor", headers=_bearer(2)).status_code == 200


def test_role_in_token_is_ignored_in_favor_of_database():
    # token 里声称自己是 admin，但数据库里是 patient：以数据库为准
    c = _client({1: _user(1, "patient")})
    assert c.get("/doctor", headers=_bearer(1, role="admin")).status_code == 403
