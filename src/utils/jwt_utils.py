import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi.security import OAuth2PasswordBearer
from loguru import logger

from src.core.config import get_settings

ALGORITHM = "HS256"

# 从 Authorization: Bearer <token> 头中提取令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

_fallback_secret: str | None = None


def _get_secret() -> str:
    """
    优先使用环境变量 JWT_SECRET_KEY。
    未配置时生成进程内随机密钥：服务可以启动，但重启后旧 token 全部失效。
    """
    global _fallback_secret
    secret = get_settings().JWT_SECRET_KEY
    if secret:
        return secret
    if _fallback_secret is None:
        _fallback_secret = secrets.token_hex(32)
        logger.warning("未配置 JWT_SECRET_KEY，使用临时随机密钥，重启后 token 将失效")
    return _fallback_secret


class TokenError(Exception):
    """token 缺失、过期或不合法"""


def encode_jwt(payload: dict) -> str:
    """登录成功后签发 token。payload 里不要放敏感数据。"""
    now = datetime.now(timezone.utc)
    claims = {
        **payload,
        "iat": now,
        "exp": now + timedelta(minutes=get_settings().JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(claims, key=_get_secret(), algorithm=ALGORITHM)


def verify_jwt(token: str) -> dict:
    """校验 token 并返回 payload。失败时抛出 TokenError。"""
    try:
        return jwt.decode(token, key=_get_secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise TokenError("token 已过期")
    except jwt.InvalidTokenError:
        raise TokenError("非法 token")
