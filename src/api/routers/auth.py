from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_current_user
from src.infra.database import get_db
from src.modules.user.model import Role, User
from src.utils.jwt_utils import encode_jwt
from src.utils.password_utils import hash_password, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=72)  # bcrypt 只取前 72 字节


class UserInfo(BaseModel):
    id: int
    username: str
    role: str
    department_id: int | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _to_info(user: User) -> UserInfo:
    return UserInfo(
        id=user.id, username=user.username, role=user.role, department_id=user.department_id
    )


@router.post("/register", response_model=UserInfo, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    自助注册，只会创建 patient 角色。
    医生、运营、管理员账号只能由管理员通过 scripts/create_user.py 创建，
    注册接口不接受 role 字段，避免用户自行提权。
    """
    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        role=Role.PATIENT.value,
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已存在")
    return _to_info(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """表单登录（兼容 Swagger 页面右上角的 Authorize 按钮）。"""
    result = await db.execute(select(User).where(User.username == form.username))
    user = result.scalar_one_or_none()
    # 用户不存在与密码错误返回同样的信息，避免被用来枚举用户名
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号已被禁用")
    return TokenResponse(access_token=encode_jwt({"sub": str(user.id), "role": user.role}))


@router.get("/me", response_model=UserInfo)
async def me(user: User = Depends(get_current_user)):
    return _to_info(user)
