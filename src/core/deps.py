from fastapi import Depends, HTTPException, status
from fastapi.params import Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.database import get_db
from src.modules.user.model import Role, User
from src.utils.jwt_utils import TokenError, oauth2_scheme, verify_jwt


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


# 前端每次请求携带：Authorization: Bearer <token>
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """从 JWT 解析当前登录用户。角色以数据库为准，避免 token 签发后角色变更仍然有效。"""
    try:
        user_id = int(verify_jwt(token)["sub"])
    except (TokenError, KeyError, ValueError) as e:
        raise _unauthorized(str(e) if isinstance(e, TokenError) else "非法 token")

    user = await db.get(User, user_id)
    if user is None:
        raise _unauthorized("用户不存在")
    if not user.is_active:
        raise _unauthorized("账号已被禁用")
    return user


def has_role(user_role: str, allowed: tuple[str, ...]) -> bool:
    """admin 拥有全部权限，其余角色必须在允许列表内。"""
    return user_role == Role.ADMIN or user_role in allowed


def require_roles(*roles: Role):
    """
    路由级角色校验依赖：
        @router.post("/upload", dependencies=[Depends(require_roles(Role.DOCTOR))])
    或需要用户对象时：
        user: User = Depends(require_roles(Role.OPERATOR))
    """
    allowed = tuple(r.value for r in roles)

    async def checker(user: User = Depends(get_current_user)) -> User:
        if not has_role(user.role, allowed):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "当前角色无权访问该资源")
        return user

    return checker


# api/v1/users?page=1&page_size=10&keyword=张三
class PageParams:
    """通用分页参数，通过 Depends 注入到接口中"""
    def __init__(
        self,
        page: int = Query(1, ge=1, description="页码，从1开始"),
        page_size: int = Query(10, ge=1, le=100, description="每页条数"),
        keyword: str | None = Query(None, description="搜索关键词"),
    ):
        self.page = page
        self.page_size = page_size
        self.keyword = keyword

    @property
    def offset(self) -> int:
        """计算 SQL OFFSET"""
        return (self.page - 1) * self.page_size
