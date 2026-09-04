# 智慧问诊 Agent 学习笔记（上）

> 基于学习文档 `4.智慧问诊Agent.md` + 项目实际代码

---

> ## 📁 代码现状说明
>
> 本文档对应的工作区代码**已全部实现**，包括 `src/agents/inquiry/`、`src/agents/workers/inquiry_agent.py`、`src/agents/tools/worker_tools.py`、`src/api/routers/chat.py`、`scripts/init_symptom_index.py` 等。文档中的路径均可直接对照源码阅读。

---

## 一、总览：智慧问诊 Agent 在项目中的位置

### 1.1 改造前（无问诊流程）

原 `src/agents/supervisor_agent.py` 是通用 ReAct Agent：

```
用户消息 → FastAPI → Supervisor Agent（ReAct）
                         ├── save_memory / search_memory（Milvus 长期记忆）
                         └── 无专门问诊流程
```

两个痛点：
1. 无专门问诊流程——"我头疼"直接用 LLM 回答，无结构化诊断
2. 多轮无状态——ReAct 每轮独立推理，无法保持"已确认症状""候选疾病排序"

### 1.2 改造后（目标架构）

```
用户消息 → FastAPI 路由层
               │
               ├── Redis EXISTS inquiry_active?
               │     ├── YES → 直接调用 InquiryGraph（绕过 Supervisor）
               │     └── NO  → 走 Supervisor Agent
               │                  ├── 识别问诊意图 → call_inquiry_agent 工具
               │                  │     └── 启动 InquiryGraph（首轮）
               │                  └── ...
```

### 1.3 核心设计决策：为什么用 StateGraph 不用 ReAct Agent

| 维度 | ReAct Agent（原有） | StateGraph（改造后） |
|------|---------------------|---------------------|
| 流程控制 | LLM 自由推理，不可预测 | 确定性路由，每步可控 |
| 状态管理 | 仅消息列表 | 结构化状态（症状列表、候选疾病、置信度） |
| 多轮对话 | 每轮独立推理 | 贯穿多轮的状态机 |
| 急症处理 | 无法保证优先 | 固定在第一轮检查 |
| 收敛判断 | LLM 自行决定 | 置信度算法 + 阈值 |

---

## 二、编写步骤总览（11 步）

| 步骤 | 文件                                                             | 职责 |
|------|----------------------------------------------------------------|------|
| Step 1 | `src/agents/inquiry/state.py` ✅️                               | 状态机定义（InquiryState + InquiryPhase） |
| Step 2 | `src/agents/inquiry/state.py` ✅️                               | 移交数据包（InquiryHandoffPayload） |
| Step 3 | `src/agents/inquiry/symptom_normalizer.py` ✅️                  | 症状三层标准化 |
| Step 4 | `scripts/init_symptom_index.py`  ✅️                            | Milvus 症状向量索引初始化 |
| Step 5 | `src/agents/inquiry/neo4j_queries.py`  ✅️                      | Neo4j Cypher 查询封装 |
| Step 6 | `src/agents/inquiry/confidence.py`                             | 置信度计算 + 收敛判断 |
| Step 7 | `src/agents/inquiry/prompts.py`                                | 所有 LLM 提示词模板 |
| Step 8 | `src/agents/inquiry/db_queries.py`                             | PostgreSQL 患者上下文 + 问诊记录 |
| Step 9 | `src/agents/inquiry/graph.py`                                  | InquiryGraph 主图构建（核心） |
| Step 10 | `src/agents/workers/inquiry_agent.py`                          | Inquiry Worker Agent（挂号移交） |
| Step 11 | `src/agents/tools/worker_tools.py` + `src/api/routers/chat.py` | 工具改造 + FastAPI 路由分流 |


---

## 三、Step 1-2：状态机定义（state.py）

### 3.1 为什么需要状态机

ReAct Agent 只有 `messages: list` 一个状态字段。StateGraph 允许自定义状态容器（`TypedDict` 或 `BaseModel`），每个节点可以读/写其中的字段。本项目用的是 `BaseModel`（Pydantic），见 3.3。

### 3.2 InquiryPhase 枚举（真实代码）

```python
class InquiryPhase(str, Enum):
    CLARIFY            = "CLARIFY"            # 用户描述模糊，引导澄清
    GRAPH_QUERY        = "GRAPH_QUERY"        # 有明确症状，查询 Neo4j
    SYMPTOM_CONFIRM    = "SYMPTOM_CONFIRM"    # 追问候选症状
    COMPLICATION_CHECK = "COMPLICATION_CHECK" # 追问并发症症状
    CONCLUDE           = "CONCLUDE"           # 输出诊断结论
    HANDOFF            = "HANDOFF"            # 移交 Inquiry Worker Agent
    END                = "__end__"            # 流程结束
```

- `InquiryPhase` 【自定义变量】：问诊流程的 7 个阶段
- `Enum` 【内置方法】：Python 标准枚举，继承 `str` 让值可 JSON 序列化
- 注意：真实代码的阶段值是**大写**字符串，且 `END = "__end__"`（与 LangGraph 的 `END` 常量对齐）

### 3.3 InquiryState 完整字段（真实代码）

> 重要差异：真实代码用 `BaseModel`（Pydantic），**不是** `TypedDict`。节点返回 `dict`，LangGraph 按字段 merge 到 Pydantic 模型上。

```python
class InquiryState(BaseModel):
    # 对话消息历史（add_messages 注解让 LangGraph 自动追加而非覆盖）
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    # 流程控制
    phase: InquiryPhase = InquiryPhase.CLARIFY
    round: int = 0                    # 当前轮次
    session_id: str = ""
    # 症状追踪
    confirmed_symptoms: list[str] = Field(default_factory=list)  # 确认症状
    denied_symptoms: list[str] = Field(default_factory=list)     # 否认症状
    unmatched_symptoms: list[str] = Field(default_factory=list)  # 图谱外症状
    asked_symptoms: list[str] = Field(default_factory=list)      # 已问过，不重复
    # 诊断推理
    candidate_diseases: list[CandidateDisease] = Field(default_factory=list)
    # 患者上下文
    patient_context: PatientContext = Field(default_factory=PatientContext)
    # 结论
    handoff_payload: InquiryHandoffPayload | None = None
    force_conclude: bool = False      # 强制收敛开关
    # 节点间临时传递数据
    pending_ask_symptoms: list[str] = Field(default_factory=list)  # 本轮准备追问的症状
```

关键标注：
- `BaseModel` 【内置方法】：Pydantic 基类，提供数据校验 + 字段默认值
- `Annotated[list[BaseMessage], add_messages]` 【内置方法】：`add_messages` 是 LangGraph 的 reducer，消息"只追加不覆盖"
- `Field(default_factory=list)` 【内置方法】：列表字段用工厂函数，避免可变默认值共享问题
- `PatientContext`、`CandidateDisease`、`InquiryHandoffPayload` 【自定义变量】：Pydantic BaseModel

### 3.4 数据流

```
用户消息（HumanMessage）
    ↓ append 到 state.messages（add_messages reducer 自动执行）
节点函数读取 state.confirmed_symptoms 等字段
    ↓ 处理后返回 dict，LangGraph 自动 merge 到 state
路由函数读取 state.phase 决定下一步
    ↓
下一个节点继续读写 state
```

### 3.5 InquiryHandoffPayload（移交数据包，真实代码）

```python
class InquiryHandoffPayload(BaseModel):
    patient_id: int | None = None
    patient_context: dict = Field(default_factory=dict)
    confirmed_symptoms: list[str] = Field(default_factory=list)  # 确认症状
    denied_symptoms: list[str] = Field(default_factory=list)     # 否认症状
    unmatched_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str = ""                  # 主诊断
    primary_confidence: float = 0.0
    suspected_diseases: list[str] = Field(default_factory=list)  # 疑似疾病
    department: str = ""                       # 建议科室
    recommended_checks: list[str] = Field(default_factory=list)
    total_rounds: int = 0
    session_id: str = ""
```

- 在 `node_conclude`（节点⑧）中构建，传递给 Inquiry Worker Agent 执行挂号
- `BaseModel` 【内置方法】：Pydantic 基类，提供数据验证 + JSON 序列化
- `model_dump_json(indent=2, ensure_ascii=False)` 【内置方法】：序列化为格式化 JSON，注入 Worker Agent Prompt

---

## 四、Step 3：症状三层标准化（symptom_normalizer.py）

### 4.1 解决什么问题

用户说"脑袋嗡嗡的"，系统需要知道这是"头痛"。三层标准化逐层兜底：

```
用户原话 → ① LLM 结构化提取 → ② Neo4j 精确匹配 → ③ Milvus 语义兜底
```

### 4.2 总入口 `normalize_symptoms`（真实代码）

```python
async def normalize_symptoms(
    user_input: str,
    llm: BaseChatModel,
    neo4j_driver: AsyncDriver,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> dict:
    normalized = await extract_and_normalize_symptoms(user_input, llm)
    if not normalized:
        return {"matched": [], "mapped": {}, "unmatched": [], "all_standard": []}
    matched, unmatched_after_exact = await match_symptoms_in_neo4j(normalized, neo4j_driver)
    mapped, still_unmatched = await semantic_match_symptoms(
        unmatched_after_exact, embedding_model, milvus_client
    )
    all_standard = matched + list(mapped.values())
    return {
        "matched": matched,          # 精确命中 Neo4j 的标准症状
        "mapped": mapped,            # {用户原词: 图谱标准词} 语义兜底映射
        "unmatched": still_unmatched,  # 真正的图谱外症状
        "all_standard": all_standard,  # matched + mapped.values()，供 Neo4j 查询
    }
```

数据流：`extract_and_normalize_symptoms`（LLM 提取）→ `match_symptoms_in_neo4j`（Neo4j 精确）→ `semantic_match_symptoms`（Milvus 兜底）→ 汇总四件套。

### 4.3 关键代码解读（真实代码）

**第一层：LLM 结构化提取**

```python
class SymptomsOutput(BaseModel):                      # 【自定义变量】输出 Schema
    symptoms: list[str] = Field(
        description="从用户描述中提取并标准化后的医学症状术语列表，无症状时为空列表"
    )

async def extract_and_normalize_symptoms(user_input: str, llm: BaseChatModel) -> list[str]:
    structured_llm = llm.with_structured_output(SymptomsOutput)  # 【内置方法】
    prompt = SYMPTOM_EXTRACT_PROMPT.format(user_input=user_input)
    try:
        result: SymptomsOutput = await structured_llm.ainvoke(
            [SystemMessage(content=prompt)]
        )
        symptoms = [s.strip() for s in result.symptoms if s.strip()]
        return symptoms
    except Exception as e:
        # 结构化输出失败时静默降级，返回空列表，由后续层处理
        logger.warning(f"LLM 结构化输出失败，降级为空列表: {e}")
        return []
```

- `with_structured_output(SymptomsOutput)` 【内置方法】：把 Pydantic Schema 转成 LLM function calling 约束，直接返回对象而非 JSON 字符串
- `ainvoke` 【内置方法】：LangChain Runnable 异步调用
- `Field(description=...)` 【内置方法】：Pydantic 字段描述，LLM 据此理解字段含义
- 降级策略：失败返回 `[]`，不中断主流程

**第二层：Neo4j 精确匹配**

```python
async def match_symptoms_in_neo4j(
    symptoms: list[str], neo4j_driver: AsyncDriver,
) -> tuple[list[str], list[str]]:
    """返回 (命中列表, 未命中列表)。"""
    if not symptoms:
        return [], []
    async with neo4j_driver.session() as session:   # 【内置方法】Neo4j 异步会话
        result = await session.run(
            "MATCH (s:Symptom) WHERE s.name IN $names RETURN s.name AS name",
            names=symptoms,                          # 参数化绑定，防注入
        )
        records = await result.data()
        matched_set = {r["name"] for r in records}
    matched = [s for s in symptoms if s in matched_set]
    unmatched = [s for s in symptoms if s not in matched_set]
    return matched, unmatched
```

- `driver.session()` 【内置方法】：Neo4j AsyncDriver 创建数据库会话
- `session.run(cypher, names=...)` 【内置方法】：执行 Cypher，参数化绑定防注入

**第三层：Milvus 语义兜底**

```python
SIMILARITY_THRESHOLD = 0.85  # 低于此值视为真正的图谱外症状

async def semantic_match_symptoms(
    unmatched_symptoms: list[str],
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> tuple[dict[str, str], list[str]]:
    if not unmatched_symptoms:
        return {}, []
    # 批量向量化，减少 API 调用次数
    query_embeddings = await embedding_model.aembed_documents(unmatched_symptoms)  # 【内置方法】
    mapped: dict[str, str] = {}
    still_unmatched: list[str] = []
    for symptom, query_vec in zip(unmatched_symptoms, query_embeddings):
        try:
            results = milvus_client.search(
                collection_name="symptom_index",
                data=[query_vec], limit=1, output_fields=["name"],
            )  # 【内置方法】Milvus 向量相似度搜索
            if results and results[0]:
                top_hit = results[0][0]
                score = top_hit["distance"]  # COSINE 相似度，越高越相似
                if score >= SIMILARITY_THRESHOLD:
                    std_name = top_hit["entity"]["name"]
                    mapped[symptom] = std_name
                else:
                    still_unmatched.append(symptom)
            else:
                still_unmatched.append(symptom)
        except Exception as e:
            logger.warning(f"Milvus 语义匹配失败 '{symptom}': {e}")
            still_unmatched.append(symptom)
    return mapped, still_unmatched
```

- `aembed_documents` 【内置方法】：批量文本转向量（复数形式，一次请求处理多个）
- `milvus_client.search` 【内置方法】：在 `symptom_index` 集合中搜最相似向量，返回 `distance`（COSINE）
- 阈值 `0.85`：相似度达标才算"同一个症状"，否则归入"图谱外症状"

### 4.4 与项目现有代码的关系

- 第二层 `driver` 来自 `src/infra/neo4j_client.py` 的 `get_neo4j_driver()` → 返回 `AsyncDriver` 单例
- 第三层 `embedding_model` 是 `DashScopeEmbeddings`，与 `src/agents/supervisor_agent.py` 中 `_get_embedding_model()` 返回同一个模型
- 第三层 Milvus `collection` 是新建的 `symptom_index` 集合，与现有 `agent_long_term_memory` 集合独立

---

## 五、Step 4-5：Milvus 索引初始化 + Neo4j 查询封装

### 5.1 init_symptom_index.py 数据流

```
PostgreSQL symptoms 表（src/modules/medical/model.py Symptom 模型）
    ↓ 读取全部症状名
DashScopeEmbeddings.embed_query(症状名) → 1024 维向量
    ↓
Milvus collection "symptom_index"
    字段：id / name / embedding(1024维) / created_at
    索引：IVF_FLAT + COSINE
```

- 与 `scripts/init_neo4j.py`、`scripts/init_postgres.py` 同目录，是第三个初始化脚本
- Milvus 连接通过 `src/infra/milvus_client.py` 的 `get_milvus_client_alias()` 获取

### 5.2 neo4j_queries.py 核心查询（真实代码）

```cypher
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE s.name IN $confirmed_symptoms
WITH d, collect(s.name) AS matched_symptoms, count(s) AS matched_count
MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
WITH d, matched_symptoms, matched_count, count(all_s) AS total_symptoms
ORDER BY toFloat(matched_count) / total_symptoms DESC
LIMIT $top_k
RETURN
    d.name AS disease,
    matched_symptoms,
    matched_count,
    total_symptoms,
    toFloat(matched_count) / total_symptoms AS base_confidence
```

三个核心函数：

```python
async def query_candidate_diseases(
    confirmed_symptoms: list[str],
    neo4j_driver: AsyncDriver,
    top_k: int = 10,
) -> list[CandidateDisease]:
    """根据已确认症状查候选疾病，按基础置信度（命中数/总症状数）降序取 Top K。"""
    if not confirmed_symptoms:
        return []
    cypher = """...（上面这段 Cypher）..."""
    async with neo4j_driver.session() as session:
        result = await session.run(
            cypher, confirmed_symptoms=confirmed_symptoms, top_k=top_k,
        )
        records = await result.data()
    candidates = []
    for r in records:
        candidates.append(CandidateDisease(
            name=r["disease"],
            base_confidence=round(r["base_confidence"], 4),
            confidence=round(r["base_confidence"], 4),  # 初始值，后续加权
            matched_symptoms=r["matched_symptoms"],
            all_symptoms=[],   # 由 enrich_candidate_details 补充
            department="",
            checks=[],
            complications=[],
        ))
    return candidates

async def enrich_candidate_details(
    candidates: list[CandidateDisease], neo4j_driver: AsyncDriver,
) -> list[CandidateDisease]:
    """补充候选疾病的全部症状（HAS_SYMPTOM）、科室（BELONGS_TO）、检查（NEED_CHECK）、并发症（ACOMPANY_WITH）。"""

async def get_pending_symptoms(
    candidates: list[CandidateDisease],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    asked_symptoms: list[str],
) -> list[tuple[str, int]]:
    """提取尚未确认/否认/追问过的症状，按出现次数升序（区分度高的优先追问）。"""
```

- `CandidateDisease` 【自定义变量】：字段为 `name`、`confidence`、`base_confidence`、`matched_symptoms`、`all_symptoms`、`department`、`checks`、`complications`（注意：是 `confidence` 不是 `confidence_score`，是 `checks` 不是 `recommended_exams`）

---

## 六、Step 6：置信度计算引擎（confidence.py）

### 6.1 上下文加权规则（真实代码）

```python
def apply_context_weights(
    candidates: list[CandidateDisease],
    patient_context: PatientContext,
    denied_symptoms: list[str],
) -> list[CandidateDisease]:
    # 计算每个症状出现在几个候选疾病中（用于判断"核心症状"）
    symptom_disease_count: dict[str, int] = {}
    for c in candidates:
        for s in c.all_symptoms:
            symptom_disease_count[s] = symptom_disease_count.get(s, 0) + 1
    for c in candidates:
        score = c.base_confidence
        # +0.15 既往病史命中
        for history_item in patient_context.medical_history:
            if c.name in history_item:
                score += 0.15
                break
        # +0.10 长期记忆命中
        for memory in patient_context.long_term_memories:
            if c.name in memory:
                score += 0.10
                break
        # -0.20 核心症状被否认（核心症状 = 只属于该疾病且被否认）
        for s in c.all_symptoms:
            if s in denied_symptoms and symptom_disease_count.get(s, 0) == 1:
                score -= 0.20
                break
        c.confidence = round(max(0.0, min(1.0, score)), 4)
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates
```

权重规则：+0.15 既往病史、+0.10 长期记忆、-0.20 核心症状被否认（`symptom_disease_count == 1` 才是"核心症状"）。最后 `clamp` 到 `[0, 1]`，按最终置信度降序排列。

### 6.2 收敛判断（真实代码）

```python
def check_convergence(
    candidates: list[CandidateDisease],
    current_round: int,
    max_rounds: int = 10,
) -> tuple[bool, bool]:
    """返回 (should_conclude, force_conclude)。"""
    if current_round >= max_rounds:
        return True, True          # 轮次上限，强制结束
    if not candidates:
        return False, False
    top1 = candidates[0].confidence
    if top1 >= 0.70:               # 条件1：Top1 ≥ 70%
        return True, False
    if len(candidates) >= 2:
        top2 = candidates[1].confidence
        if top1 - top2 >= 0.30:    # 条件2：Top1 - Top2 ≥ 30%
            return True, False
    return False, False
```

- 返回 `(should_conclude, force_conclude)` 两个布尔值：`force_conclude` 表示"因轮次上限被迫结束"，结论里需标注"信息不足"

### 6.3 数据流（真实代码）

```
node_query_neo4j 调用 query_candidate_diseases → list[CandidateDisease]
    ↓
enrich_candidate_details(candidates, driver)  # 补全部症状/科室/检查/并发症
    ↓
apply_context_weights(candidates, patient_context, denied_symptoms)
    ↓ patient_context.medical_history → +0.15
    ↓ patient_context.long_term_memories → +0.10
    ↓ state.denied_symptoms → -0.20（核心症状）
    ↓
按 confidence 降序排序 → 更新 state.candidate_diseases
    ↓
check_convergence(candidates, round) → (should_conclude, force_conclude)
    ↓ should_conclude=True: phase=CONCLUDE → node_conclude
    ↓ 否则: phase=SYMPTOM_CONFIRM → node_ask_symptoms
```

---

## 七、Step 7-8：提示词模板 + PostgreSQL 查询封装

### 7.1 五个提示词（prompts.py 真实代码）

| 常量 | 用途 | 输出格式 |
|------|------|----------|
| `CLARIFY_PROMPT` | 引导患者澄清模糊描述 | 纯文本 |
| `ASK_SYMPTOMS_PROMPT` | 口语化追问症状（一次最多 3 个） | 纯文本 |
| `PARSE_ANSWER_PROMPT` | 解析用户对追问的回答 | JSON |
| `CONCLUSION_PROMPT` | 生成诊断结论文案 | 纯文本 |
| `EMERGENCY_CHECK_PROMPT` | 识别急症关键词 | JSON |

> 注意：真实代码中**没有** `SYMPTOM_EXTRACTION_PROMPT`，第一层症状提取的提示词在 `symptom_normalizer.py` 的 `SYMPTOM_EXTRACT_PROMPT` 里。
> 提示词的实际用法：`graph.py` 里用 `deps.llm.ainvoke([SystemMessage(content=prompt)])` 拿原始文本后手动 `json.loads` 解析，**不是**配合 `with_structured_output`（只有 `symptom_normalizer.py` 第一层用了 `with_structured_output`）。

### 7.2 db_queries.py 两个核心函数（真实代码）

```python
async def load_patient_context(
    patient_id: int | None,
    db: AsyncSession | None = None,
    recent_consultations_limit: int = 5,
) -> PatientContext:
    """从 PostgreSQL 加载患者上下文：基本信息 + 近期问诊记录。"""
    if patient_id is None:
        return PatientContext()
    from src.infra.database import AsyncSessionLocal
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()
    try:
        patient_result = await db.execute(select(Patient).where(Patient.id == patient_id))
        patient = patient_result.scalar_one_or_none()
        if not patient:
            return PatientContext()
        medical_history = (
            [h.strip() for h in patient.medical_history.split(",") if h.strip()]
            if patient.medical_history else []
        )
        allergy_history = (
            [a.strip() for a in patient.allergy_history.split(",") if a.strip()]
            if patient.allergy_history else []
        )
        consultations_result = await db.execute(
            select(Consultation)
            .where(Consultation.patient_id == patient_id)
            .order_by(desc(Consultation.created_at))
            .limit(recent_consultations_limit)
        )
        consultations = consultations_result.scalars().all()
        for c in consultations:
            if c.diagnosis and c.diagnosis not in medical_history:
                medical_history.append(c.diagnosis)
        return PatientContext(
            patient_id=patient_id, age=patient.age, gender=patient.gender,
            allergy_history=allergy_history, medical_history=medical_history,
        )
    except Exception as e:
        logger.error(f"加载患者上下文失败: {e}")
        return PatientContext()
    finally:
        if _own_session:
            await db.close()

async def save_consultation_record(
    patient_id: int | None,
    session_id: str,
    chief_complaint: str,
    diagnosis: str,
    department_name: str,
    urgency_level: str,
    db: AsyncSession | None = None,
) -> int | None:
    """将本次问诊结果保存到 PostgreSQL consultations 表，返回新记录 ID。"""
    # 内部先按科室名查 Department.id，再构造 Consultation 写入
```

与项目现有代码的关系：
- `Patient` 模型（`src/modules/medical/model.py`）有 `allergy_history`、`medical_history`
- `Consultation` 模型有 `patient_id`、`department_id`、`chief_complaint`、`diagnosis`、`urgency_level`、`session_id`
- `db` 参数是 `AsyncSession`，来自 `src/infra/database.py` 的 `get_db()`（`db=None` 时函数内部自建 `AsyncSessionLocal` 会话）
