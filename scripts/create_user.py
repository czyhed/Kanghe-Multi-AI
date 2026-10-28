"""
创建指定角色的账号（医生 / 运营 / 管理员）。
自助注册接口只能创建 patient，其他角色只能通过本脚本由管理员创建。

用法:
    python scripts/create_user.py <username> <role> [--department-id N]
密码通过环境变量 NEW_USER_PASSWORD 传入，未设置时交互输入，避免出现在命令行历史里。
"""
import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infra.database import AsyncSessionLocal
from src.modules.user.model import Role, User
from src.utils.password_utils import hash_password


async def main() -> None:
    parser = argparse.ArgumentParser(description="创建用户")
    parser.add_argument("username")
    parser.add_argument("role", choices=[r.value for r in Role])
    parser.add_argument("--department-id", type=int, default=None)
    args = parser.parse_args()

    password = os.environ.get("NEW_USER_PASSWORD") or getpass.getpass("密码（至少 8 位）: ")
    if len(password) < 8:
        sys.exit("密码至少 8 位")

    async with AsyncSessionLocal() as session:
        session.add(
            User(
                username=args.username,
                password_hash=hash_password(password),
                role=args.role,
                department_id=args.department_id,
            )
        )
        await session.commit()
    print(f"已创建 {args.role} 账号: {args.username}")


if __name__ == "__main__":
    asyncio.run(main())
