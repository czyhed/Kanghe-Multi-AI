
"""
step1:状态机定义（InquiryState + InquiryPhase）
step2:移交数据包（InquiryHandoffPayload）

"""

from enum import Enum                          # Python 标准库
from typing import Annotated                    # Python 标准库
from pydantic import BaseModel, Field           # Pydantic（数据校验）
from langgraph.graph.message import add_messages # ★ LangGraph
# add_messages（reducer）:消息"只追加不覆盖"的规则
from langchain_core.messages import BaseMessage  # LangChain Core

# from langgraph.graph import START,END

# ── 阶段枚举：驱动 LangGraph 路由逻辑 ──────────────────────────────────
class InquiryPhase(str, Enum):
    CLARIFY            = "CLARIFY"            # 用户描述模糊，引导澄清
    GRAPH_QUERY        = "GRAPH_QUERY"        # 有明确症状，查询 Neo4j
    SYMPTOM_CONFIRM    = "SYMPTOM_CONFIRM"    # 追问候选症状
    COMPLICATION_CHECK = "COMPLICATION_CHECK" # 追问并发症症状
    CONCLUDE           = "CONCLUDE"           # 输出诊断结论
    HANDOFF            = "HANDOFF"            # 移交 Inquiry Worker Agent
    END                = "__end__"                # 流程结束

# ── 候选疾病 ────────────────────────────────────────────────────────────
# 嫌疑人
class CandidateDisease(BaseModel):
    name: str
    confidence: float = 0.0           # 最终置信度（加权后）
    base_confidence: float = 0.0      # 基础置信度 = 命中症状数 / 总症状数
    matched_symptoms: list[str] = Field(default_factory=list) # 匹配症状
    all_symptoms: list[str] = Field(default_factory=list) # 全部症状
    department: str = "" #科室
    checks: list[str] = Field(default_factory=list) # （检查项）
    complications: list[str] = Field(default_factory=list)  # 并发症ACOMPANY_WITH 关系

# ── 患者上下文（从外部系统加载） ─────────────────────────────────────────
class PatientContext(BaseModel):
    patient_id: int | None = None
    age: int | None = None
    gender: str | None = None                 # "男" / "女"
    allergy_history: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)     # 既往病史
    long_term_memories: list[str] = Field(default_factory=list)  # Milvus 长期记忆摘要

# ── 移交数据包（传给 Inquiry Worker Agent） ──────────────────────────────
class InquiryHandoffPayload(BaseModel):
    patient_id: int | None = None
    patient_context: dict = Field(default_factory=dict)
    confirmed_symptoms: list[str] = Field(default_factory=list)
    denied_symptoms: list[str] = Field(default_factory=list)
    unmatched_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str = ""
    primary_confidence: float = 0.0
    suspected_diseases: list[str] = Field(default_factory=list)
    department: str = ""
    recommended_checks: list[str] = Field(default_factory=list)
    total_rounds: int = 0
    session_id: str = ""

# ── 主状态对象：在 LangGraph 节点间流转的"黑板" ──────────────────────────
#  因为：节点是"独立函数"，彼此没法共享变量，所以需要维护一块黑板
class InquiryState(BaseModel):
    # 对话消息历史（add_messages 注解让 LangGraph 自动追加而非覆盖）
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    # 流程控制
    phase: InquiryPhase = InquiryPhase.CLARIFY
    round: int = 0                    # 当前轮次（1~10）
    session_id: str = ""
    # 症状追踪
    ## 来自移交数据包：
    confirmed_symptoms: list[str] = Field(default_factory=list) # 确认症状
    denied_symptoms: list[str] = Field(default_factory=list)  # 否认症状
    unmatched_symptoms: list[str] = Field(default_factory=list)  # 图谱外症状

    asked_symptoms: list[str] = Field(default_factory=list)      # 已问过，不重复
    # 诊断推理
    candidate_diseases: list[CandidateDisease] = Field(default_factory=list)
    # 患者上下文
    patient_context: PatientContext = Field(default_factory=PatientContext)
    # 结论
    handoff_payload: InquiryHandoffPayload | None = None
    force_conclude: bool = False # 强制收敛开关     # True = 达到 10 轮强制结束 置ture,跳过置信度判断直接出结论。
    # 节点间临时传递数据
    pending_ask_symptoms: list[str] = Field(default_factory=list)  # 本轮准备追问的症状 # 记录"本轮正在问的症状"