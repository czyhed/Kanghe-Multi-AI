from enum import StrEnum

from sqlalchemy import Boolean, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base_model import BaseModel


class Role(StrEnum):
    PATIENT = "patient"    # 患者：问诊、知识问答、药物咨询
    DOCTOR = "doctor"      # 医生：患者权限 + 管理知识库文档
    OPERATOR = "operator"  # 运营：查询运营统计数据
    ADMIN = "admin"        # 管理员：全部权限


class User(BaseModel):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="登录名")
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False, comment="bcrypt 哈希")
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default=Role.PATIENT.value, comment="角色：patient/doctor/operator/admin"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="账号是否启用")
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id", ondelete="SET NULL"), comment="所属科室（医生）"
    )
    patient_id: Mapped[int | None] = mapped_column(
        ForeignKey("patients.id", ondelete="SET NULL"), comment="关联的患者档案（患者）"
    )

    __table_args__ = (Index("ix_users_role", "role"),)
