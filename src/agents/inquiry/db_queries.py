# step8:PostgreSQL 患者上下文 + 问诊记录

# src/agents/inquiry/db_queries.py
from __future__ import annotations
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from src.modules.medical.model import Patient, Consultation
from src.agents.inquiry.state import PatientContext

"""
从业务视角看，这个文件就是整条问诊 Agent 的**「医院病历库网关」**——它是唯一与 PostgreSQL（HIS/病历库）打交道的地方。业务上只干两件事：①问诊前"查病历"、②问诊后"存病历"。
"""

# 问诊前"翻病历"
async def load_patient_context(
    patient_id: int | None,                # 患者ID(即 patients.id)；None=未登录，直接返回空上下文
    db: AsyncSession | None = None,        # 数据库会话；None 时本函数内部自动创建
    recent_consultations_limit: int = 5,   # 取「最近 N 条」问诊记录，默认 5 条
) -> PatientContext:                       # 返回患者上下文对象(含基本信息+既往史+过敏史)
    """
    从 PostgreSQL 加载患者上下文：基本信息 + 近期问诊记录。
    patient_id 为 None 时返回空上下文。
    db 为 None 时自动创建 session。
    """
    if patient_id is None:                 # 未登录 => 无需查库
        return PatientContext()            # 返回一个空上下文（全字段默认值）
    from src.infra.database import AsyncSessionLocal  # 延迟导入，避免启动时循环依赖
    _own_session = db is None              # 记录 session 是否「本函数自建」(True=自建)
    if _own_session:
        db = AsyncSessionLocal()           # 自建会话；最后 finally 要负责关闭它
    try:
        # 查患者基本信息
        patient_result = await db.execute(            # 异步执行查询语句
            select(Patient).where(Patient.id == patient_id)  # 按主键 id 查患者，where=过滤条件
        )
        patient = patient_result.scalar_one_or_none() # 取唯一一行；查不到返回 None
        if not patient:
            logger.warning(f"患者 ID {patient_id} 不存在")
            return PatientContext()        # 查无此人 => 返回空上下文
        # 解析既往病史和过敏史（数据库中用逗号分隔存储）
        medical_history = (                 # 【列表推导】按逗号切分→去空格→去掉空串
            [h.strip() for h in patient.medical_history.split(",") if h.strip()]
            if patient.medical_history else []   # 该字段为 None/空时，退化为空列表
        )
        allergy_history = (
            [a.strip() for a in patient.allergy_history.split(",") if a.strip()]
            if patient.allergy_history else []
        )
        # 查近期问诊记录（取最近 N 条）
        consultations_result = await db.execute(
            select(Consultation)                        # 查问诊表
            .where(Consultation.patient_id == patient_id)  # 只查该患者
            .order_by(desc(Consultation.created_at))     # 按创建时间降序(最新在前)
            .limit(recent_consultations_limit)           # 截取前 N 条
        )
        consultations = consultations_result.scalars().all()  # 取出所有行的列表

        # 把历史诊断结论加入 medical_history（用于置信度加权）
        for c in consultations:            # 遍历近期每一条问诊
            if c.diagnosis and c.diagnosis not in medical_history:  # 有诊断且不重复
                medical_history.append(c.diagnosis)  # 汇入既往史，供后面置信度加权使用

        return PatientContext(             # 组装并返回患者上下文对象
            patient_id=patient_id,         # 患者ID
            age=patient.age,               # 年龄
            gender=patient.gender,         # 性别
            allergy_history=allergy_history,        # 过敏史
            medical_history=medical_history,        # 既往史(已包含历史诊断结论)
        )
    except Exception as e:
        logger.error(f"加载患者上下文失败: {e}")
        return PatientContext()            # 出错兜底：也返回空上下文，不让流程中断
    finally:
        if _own_session:
            await db.close()               # 只关闭「本函数自建」的会话，外部传入的不关



# 问诊后"存病历"
async def save_consultation_record(
    patient_id: int | None,                # 患者ID(patients.id)；未登录用户为 None
    session_id: str,                       # 会话ID，用于把同一次问诊的多轮记录关联起来
    chief_complaint: str,                  # 主诉(常用确认症状前5个拼接，见 graph node_save_record)
    diagnosis: str,                        # 诊断结论文案
    department_name: str,                  # 科室名(用于查出科室ID)
    urgency_level: str,                    # 紧急程度(如 normal/urgent)
    db: AsyncSession | None = None,        # 数据库会话；None 时自动创建(如从 Supervisor 工具链调用)
) -> int | None:                           # 返回新记录的主键ID；失败返回 None
    """
    将本次问诊结果保存到 PostgreSQL consultations 表。
    返回新记录的 ID，失败时返回 None。
    db 为 None 时（如从 Supervisor 工具链调用）自动创建 session。
    """
    from src.infra.database import AsyncSessionLocal  # 延迟导入
    # db 为 None 时自己创建 session，用完后关闭
    _own_session = db is None              # 是否自建会话
    if _own_session:
        db = AsyncSessionLocal()           # 自建会话
    try:
        # 查科室 ID（通过科室名查）
        from src.modules.medical.model import Department  # 延迟导入科室模型，避免循环依赖
        dept_result = await db.execute(
            select(Department).where(Department.name == department_name)  # 按科室名查询
        )
        dept = dept_result.scalar_one_or_none()  # 取唯一科室；查不到为 None
        dept_id = dept.id if dept else None      # 取科室ID；查不到=>外键置 NULL
        consultation = Consultation(       # 组装问诊记录对象(尚未入库)
            patient_id=patient_id,  # 未登录用户为 None，存 NULL
            department_id=dept_id,           # 外键：科室ID
            chief_complaint=chief_complaint, # 主诉
            diagnosis=diagnosis,             # 诊断结论
            urgency_level=urgency_level,     # 紧急程度
            session_id=session_id,           # 会话ID
        )

        db.add(consultation)               # 把对象加入会话，标记为「待插入」
        await db.commit()                  # 提交事务，写入数据库永久生效
        await db.refresh(consultation)     # 刷新对象，取回数据库生成的字段(如自增主键 id)
        logger.info(f"问诊记录已保存，ID={consultation.id}")
        return consultation.id             # 返回新记录主键 ID
    except Exception as e:
        await db.rollback()                # 出错回滚，撤销未提交的写入
        logger.error(f"保存问诊记录失败: {e}")
        return None                        # 失败返回 None，上层据此降级处理
    finally:
        if _own_session:
            await db.close()               # 只关闭自建会话
