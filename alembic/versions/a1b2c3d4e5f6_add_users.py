"""add users table for authentication

Revision ID: a1b2c3d4e5f6
Revises: f6b326bbdd4e
Create Date: 2026-10-28 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'f6b326bbdd4e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('username', sa.String(length=64), nullable=False, comment='登录名'),
    sa.Column('password_hash', sa.String(length=128), nullable=False, comment='bcrypt 哈希'),
    sa.Column('role', sa.String(length=20), nullable=False, comment='角色：patient/doctor/operator/admin'),
    sa.Column('is_active', sa.Boolean(), nullable=False, comment='账号是否启用'),
    sa.Column('department_id', sa.BigInteger(), nullable=True, comment='所属科室（医生）'),
    sa.Column('patient_id', sa.BigInteger(), nullable=True, comment='关联的患者档案（患者）'),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='创建时间'),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='更新时间'),
    sa.ForeignKeyConstraint(['department_id'], ['departments.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    op.create_index('ix_users_role', 'users', ['role'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_users_role', table_name='users')
    op.drop_table('users')
