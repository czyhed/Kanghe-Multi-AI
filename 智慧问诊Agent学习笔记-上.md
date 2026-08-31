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

| 步骤 | 文件 | 职责 |
|------|------|------|
| Step 1 | `src/agents/inquiry/state.py` | 状态机定义（InquiryState + InquiryPhase） |
| Step 2 | `src/agents/inquiry/state.py` | 移交数据包（InquiryHandoffPayload） |
| Step 3 | `src/agents/inquiry/symptom_normalizer.py` | 症状三层标准化 |
| Step 4 | `scripts/init_symptom_index.py` | Milvus 症状向量索引初始化 |
| Step 5 | `src/agents/inquiry/neo4j_queries.py` | Neo4j Cypher 查询封装 |
| Step 6 | `src/agents/inquiry/confidence.py` | 置信度计算 + 收敛判断 |
| Step 7 | `src/agents/inquiry/prompts.py` | 所有 LLM 提示词模板 |
| Step 8 | `src/agents/inquiry/db_queries.py` | PostgreSQL 患者上下文 + 问诊记录 |
| Step 9 | `src/agents/inquiry/graph.py` | InquiryGraph 主图构建（核心） |
| Step 10 | `src/agents/workers/inquiry_agent.py` | Inquiry Worker Agent（挂号移交） |
| Step 11 | `src/agents/tools/worker_tools.py` + `src/api/routers/chat.py` | 工具改造 + FastAPI 路由分流 |


---

## 三、Step 1-2：状态机定义（state.py）

### 3.1 为什么需要状态机

ReAct Agent 只有 `messages: list` 一个状态字段。StateGraph 允许自定义 `TypedDict` 作为状态容器，每个节点可以读/写其中的字段。

### 3.2 InquiryPhase 枚举

```python
class InquiryPhase(str, Enum):
    CLARIFY = "clarify"
    GRAPH_QUERY = "graph_query"
    SYMPTOM_CONFIRM = "symptom_confirm"
    COMPLICATION_CHECK = "complication_check"
    CONCLUDE = "conclude"
    HANDOFF = "handoff"
    END = "end"
```

- `InquiryPhase` 【自定义变量】：问诊流程的 7 个阶段
- `Enum` 【内置方法】：Python 标准枚举，继承 `str` 让值可 JSON 序列化
- 每个节点执行后更新 `state.phase`，路由函数根据 `phase` 决定下一步

### 3.3 InquiryState 完整字段

```python
class InquiryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 【内置方法】自动追加
    session_id: str
    round: int
    phase: InquiryPhase
    patient_context: PatientContext
    confirmed_symptoms: list[str]
    denied_symptoms: list[str]
    unmatched_symptoms: list[str]
    candidate_diseases: list[CandidateDisease]
    pending_ask_symptoms: list[str]
    handoff_payload: InquiryHandoffPayload | None
    force_conclude: bool
```

关键标注：
- `TypedDict` 【内置方法】：Python typing 模块，定义字典的精确结构，LangGraph 用它做状态类型声明
- `Annotated[list[BaseMessage], add_messages]` 【内置方法】：`add_messages` 是 LangGraph 的 reducer 函数，告诉 LangGraph "新消息追加到列表末尾，不覆盖"
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

### 3.5 InquiryHandoffPayload（移交数据包）

```python
class InquiryHandoffPayload(BaseModel):
    primary_disease: str          # 主诊断
    suspected_diseases: list[str]  # 疑似疾病
    department: str                # 建议科室
    confirmed_symptoms: list[str]  # 确认症状
    denied_symptoms: list[str]     # 否认症状
    recommended_exams: list[str]   # 建议检查
    confidence_score: float        # 置信度
    inquiry_rounds: int            # 问诊轮次
    session_id: str
```

- 在 `node_conclude`（节点8）中构建，传递给 Inquiry Worker Agent 执行挂号
- `BaseModel` 【内置方法】：Pydantic 基类，提供数据验证 + JSON 序列化
- `model_dump_json(indent=2, ensure_ascii=False)` 【内置方法】：序列化为格式化 JSON，注入 Worker Agent Prompt

---

## 四、Step 3：症状三层标准化（symptom_normalizer.py）

### 4.1 解决什么问题

用户说"脑袋嗡嗡的"，系统需要知道这是"头痛"。三层标准化逐层兜底：

```
用户原话 → ① LLM 结构化提取 → ② Neo4j 精确匹配 → ③ Milvus 语义兜底
```

### 4.2 三个函数的数据流

```
normalize_symptoms(raw_input, deps)
    │
    ├── ① llm_extract_symptoms(text, llm)
    │     └── LLM with_structured_output → ["头痛", "发烧", "肚子疼"]
    │
    ├── ② neo4j_match(symptoms, driver)
    │     └── Cypher 精确匹配 → 能匹配的保留，不能的传给第三层
    │
    └── ③ milvus_fallback(unmatched, client, embedding_model)
          └── 向量相似度搜索 → 阈值 0.85 以上替换为标准症状
```

### 4.3 关键代码解读

**第一层：LLM 结构化提取**

```python
class SymptomExtraction(BaseModel):       # 【自定义变量】输出 Schema
    symptoms: list[str] = Field(description="从文本中提取的标准化症状")

structured_llm = llm.with_structured_output(SymptomExtraction)  # 【内置方法】
result = await structured_llm.ainvoke(prompt)  # 【内置方法】异步调用
raw_symptoms = result.symptoms
```

- `with_structured_output(SymptomExtraction)` 【内置方法】：把 Pydantic Schema 转成 LLM function calling 约束
- `ainvoke` 【内置方法】：LangChain Runnable 异步调用
- `Field(description=...)` 【内置方法】：Pydantic 字段描述，LLM 据此理解字段含义

**第二层：Neo4j 精确匹配**

```python
async with driver.session() as session:   # 【内置方法】Neo4j 异步会话
    result = await session.run(cypher, symptoms=raw_symptoms)  # 【内置方法】
    matched = [r["s.name"] for r in await result.data()]
```

- `driver.session()` 【内置方法】：Neo4j AsyncDriver 创建数据库会话
- `session.run(cypher, symptoms=...)` 【内置方法】：执行 Cypher，参数化绑定防注入

**第三层：Milvus 语义兜底**

```python
query_vec = embedding_model.embed_query(symptom)  # 【内置方法】文本→向量
results = collection.search(
    data=[query_vec], anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=1,
)  # 【内置方法】Milvus 向量相似度搜索
if results[0] and results[0][0].score >= 0.85:
    return results[0][0].entity.get("name")
```

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

### 5.2 neo4j_queries.py 核心查询

```cypher
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE s.name IN $symptom_list
WITH d, collect(s.name) AS matched_symptoms, count(s) AS match_count
RETURN d.name AS disease_name, d.department AS department,
       matched_symptoms, match_count, d.common_drug AS common_drug
ORDER BY match_count DESC LIMIT 10
```

核心函数：
```python
async def query_candidate_diseases(driver, symptoms: list[str]) -> list[CandidateDisease]:
    # 1. 执行 Cypher 查询
    # 2. 补充每个候选疾病的全部症状、科室、检查、并发症
    # 3. 返回 CandidateDisease 列表（还未计算置信度）
```

- `CandidateDisease` 【自定义变量】：包含 name、matched_symptoms、all_symptoms、department、confidence_score

---

## 六、Step 6：置信度计算引擎（confidence.py）

### 6.1 三层加权规则

```
基础分 = match_count / total_symptoms_of_disease
    ↓
+0.15 如果疾病在既往病史中（PostgreSQL medical_history）
+0.10 如果疾病在长期记忆中（Milvus search_memory 结果）
-0.20 如果疾病的核心症状被用户否认
    ↓
最终置信度 = clamp(基础分 ± 权重, 0, 1)
```

### 6.2 收敛判断

```python
def should_conclude(candidates, round) -> bool:
    if round >= 10:           return True        # 轮次上限
    if not candidates:         return True       # 无候选疾病
    if candidates[0].confidence_score >= 0.70:   return True  # Top1 ≥ 70%
    if len(candidates) > 1 and \
       candidates[0].confidence_score - candidates[1].confidence_score >= 0.30:
        return True                                # Top1-Top2 ≥ 30%
    return False
```

### 6.3 数据流

```
node_query_neo4j 调用 query_candidate_diseases → list[CandidateDisease]
    ↓
apply_context_weights(candidates, patient_context)
    ↓ patient_context.medical_history → +0.15
    ↓ patient_context.long_term_memories → +0.10
    ↓ state.denied_symptoms → -0.20
    ↓
按 confidence_score 降序排序 → 更新 state.candidate_diseases
    ↓
should_conclude() → True: 走 node_conclude / False: 走 node_ask_symptoms
```

---

## 七、Step 7-8：提示词模板 + PostgreSQL 查询封装

### 7.1 六个提示词（prompts.py）

| 提示词 | 用途 | 输出格式 |
|--------|------|----------|
| EMERGENCY_CHECK_PROMPT | 识别急症关键词 | JSON |
| SYMPTOM_EXTRACTION_PROMPT | 从用户文本提取标准症状 | JSON |
| CLARIFY_PROMPT | 生成澄清引导话术 | 纯文本 |
| ASK_SYMPTOM_PROMPT | 口语化追问症状 | 纯文本 |
| PARSE_ANSWER_PROMPT | 解析用户对追问的回答 | JSON |
| CONCLUDE_PROMPT | 生成诊断结论文案 | 纯文本 |

- 需要 JSON 输出的配合 `with_structured_output` 使用
- 纯文本的直接 `llm.ainvoke(prompt).content`
- `ChatPromptTemplate` 【内置方法】：LangChain 提示词模板，支持变量注入

### 7.2 db_queries.py 两个核心函数

```python
async def get_patient_context(db, patient_id) -> PatientContext:
    # 1. 查 patients 表 → 过敏史、既往病史
    # 2. 查 consultations 表 → 最近 3 次问诊诊断
    # 3. 合并为 PatientContext 返回

async def save_consultation(db, state: InquiryState) -> int:
    # 把诊断结论写入 consultations 表，返回 consultation_id
```

与项目现有代码的关系：
- `Patient` 模型（`src/modules/medical/model.py`）有 `allergy_history`、`medical_history`
- `Consultation` 模型有 `patient_id`、`department_id`、`diagnosis`、`session_id`
- `db` 参数是 `AsyncSession`，来自 `src/infra/database.py` 的 `get_db()`
