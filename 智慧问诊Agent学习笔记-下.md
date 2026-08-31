# 智慧问诊 Agent 学习笔记（下）

> 基于学习文档 `4.智慧问诊Agent.md` + 项目实际代码
>
> 上篇覆盖 Step 1-8（状态机 → 症状标准化 → 置信度 → 提示词 → PostgreSQL 查询）
>
> 下篇覆盖 Step 9-11（InquiryGraph 主图 → Worker Agent → 工具+路由分流）+ 架构总结

---

## 八、Step 9：InquiryGraph 主图构建（graph.py）

### 8.1 为什么这是最核心的文件

`graph.py` 是整个问诊 Agent 的「大脑」。前面 8 步都在准备零件（状态、标准化、置信度、提示词、数据库查询），Step 9 把这些零件组装成一台机器。

核心思路：用 LangGraph 的 `StateGraph` 构建一个确定性有向图，每个节点是一个 async 函数，边由路由函数决定。

### 8.2 图的拓扑结构

```
                          ┌─────────────────────────────┐
                    ┌─────│      START                  │
                    │     └─────────────┬───────────────┘
                    │                   ↓
                    │           ┌───────────────────┐
                    │           │ node_emergency     │  ① 急症检查（首轮必走）
                    │           └─────────┬─────────┘
                    │                     ↓
                    │           ┌───────────────────┐
                    │           │ node_clarify       │  ② 澄清/提取症状
                    │           └─────────┬─────────┘
                    │                     ↓
                    │           ┌───────────────────┐
                    │           │ node_normalize     │  ③ 三层标准化
                    │           └─────────┬─────────┘
                    │                     ↓
                    │           ┌───────────────────┐
                    │           │ node_query_neo4j   │  ④ 知识图谱查询
                    │           └─────────┬─────────┘
                    │                     ↓
                    │           ┌───────────────────┐
                    │           │ node_confidence    │  ⑤ 置信度计算 + 收敛判断
                    │           └─────────┬─────────┘
                    │                     │
              ┌──────┘                     ├──────────────┐
              │                            ↓              ↓
              │          should_conclude?  YES          NO
              │                            │              │
              │                   ┌────────↓──┐  ┌───────↓────────┐
              │                   │node_conclude│  │node_ask_symptom │  ⑥ 追问
              │                   └────┬────────┘  └───────┬────────┘
              │                        ↓                   ↓
              │               ┌────────────────┐  ┌──────────────────┐
              │               │ node_handoff    │  │ node_parse_answer │  ⑦ 解析回答
              │               └────┬───────────┘  └───────┬──────────┘
              │                    ↓                       │
              │              ┌─────↓──────┐                │
              │              │   END      │    ←──────────┘ 回到 node_confidence
              │              └────────────┘
              │
              └── 后续轮次（Redis inquiry_active = true）
                          直接从 node_clarify 进入，跳过 START
```

### 8.3 节点函数统一签名

所有节点函数遵循同一个签名模式：

```python
async def node_xxx(state: InquiryState) -> dict:
    """
    作用：执行节点逻辑，返回需要更新的状态字段
    意思：LangGraph 会把返回的 dict merge 到当前 state 中
    一句话记忆点：节点函数只管「读 state → 处理 → 返回 dict」，LangGraph 负责合并
    """
    # 读 state
    messages = state["messages"]
    confirmed = state["confirmed_symptoms"]

    # 处理（调 LLM / 查数据库 / 计算置信度）
    ...

    # 返回需要更新的字段（不必返回全部，只返回变化的）
    return {
        "messages": [new_message],        # add_messages reducer 自动追加
        "phase": InquiryPhase.GRAPH_QUERY,
        "confirmed_symptoms": updated_list,
    }
```

关键标注：
- `InquiryState` 【自定义变量】：Step 1 定义的 TypedDict，所有节点共享
- 返回 `dict` 而非完整 `InquiryState`：LangGraph 只 merge 返回的字段
- `messages` 字段用 `add_messages` reducer，返回的新消息自动追加到列表末尾

### 8.4 七个核心节点逐个拆解

#### ① node_emergency — 急症检查

```python
async def node_emergency(state: InquiryState) -> dict:
    # 1. 用 EMERGENCY_CHECK_PROMPT + with_structured_output 检查急症关键词
    structured_llm = llm.with_structured_output(EmergencyCheck)  # 【内置方法】
    result = await structured_llm.ainvoke(prompt)                # 【内置方法】

    # 2. 如果是急症 → 直接跳到 node_conclude，设置 force_conclude=True
    if result.is_emergency:
        return {
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": True,
            "messages": [AIMessage(content=URGENT_MESSAGE)],
        }

    # 3. 不是急症 → 正常进入澄清阶段
    return {"phase": InquiryPhase.CLARIFY}
```

- `EmergencyCheck` 【自定义变量】：Pydantic Schema，包含 `is_emergency: bool` + `reason: str`
- 只在第一轮执行，后续轮次跳过此节点
- `force_conclude` 【自定义变量】：强制收敛标志，跳过置信度判断

#### ② node_clarify — 澄清 + 提取症状

```python
async def node_clarify(state: InquiryState) -> dict:
    user_msg = state["messages"][-1].content  # 取最新用户消息

    # 用 CLARIFY_PROMPT 生成引导话术
    reply = await llm.ainvoke(CLARIFY_PROMPT.format(user_message=user_msg))

    return {
        "messages": [reply],                          # AI 澄清话术追加到消息历史
        "phase": InquiryPhase.GRAPH_QUERY,
        "round": state["round"] + 1,                  # 轮次 +1
    }
```

- `ainvoke` 【内置方法】：LangChain Runnable 异步调用
- `CLARIFY_PROMPT.format(...)` 【内置方法】：字符串模板变量替换
- 返回的 `reply` 是 `AIMessage`，通过 `add_messages` reducer 追加

#### ③ node_normalize — 三层标准化

```python
async def node_normalize(state: InquiryState) -> dict:
    # 调用 Step 3 的 normalize_symptoms
    result = await normalize_symptoms(
        raw_input=state["messages"][-1].content,
        deps=DependencyContainer(llm, neo4j_driver, milvus_client, embedding_model),
    )

    # result 返回三个列表
    return {
        "confirmed_symptoms": state["confirmed_symptoms"] + result.matched,
        "unmatched_symptoms": result.unmatched,
        "phase": InquiryPhase.SYMPTOM_CONFIRM,
    }
```

- `normalize_symptoms` 【自定义变量】：Step 3 的三层标准化入口函数
- `DependencyContainer` 【自定义变量】：依赖注入容器，统一管理 LLM / Neo4j / Milvus 连接
- `result.matched` 【自定义变量】：成功标准化的症状列表
- `result.unmatched` 【自定义变量】：三层都未能匹配的症状，保留供后续追问

#### ④ node_query_neo4j — 知识图谱查询

```python
async def node_query_neo4j(state: InquiryState) -> dict:
    # 调用 Step 5 的 query_candidate_diseases
    candidates = await query_candidate_diseases(
        driver=neo4j_driver,
        symptoms=state["confirmed_symptoms"],
    )
    return {
        "candidate_diseases": candidates,
        "phase": InquiryPhase.COMPLICATION_CHECK,
    }
```

- `query_candidate_diseases` 【自定义变量】：Step 5 封装的 Cypher 查询函数
- 此节点只查询不计算置信度，置信度在下一个节点
- `candidates` 此时是无序的 `list[CandidateDisease]`

#### ⑤ node_confidence — 置信度计算 + 收敛判断

```python
async def node_confidence(state: InquiryState) -> dict:
    # 1. 调用 Step 6 的 apply_context_weights
    candidates = apply_context_weights(
        candidates=state["candidate_diseases"],
        patient_context=state["patient_context"],
        denied_symptoms=state["denied_symptoms"],
    )

    # 2. 按置信度降序排序
    candidates.sort(key=lambda c: c.confidence_score, reverse=True)

    # 3. 判断是否收敛
    do_conclude = should_conclude(candidates, state["round"]) or state.get("force_conclude", False)

    return {
        "candidate_diseases": candidates,
        "phase": InquiryPhase.CONCLUDE if do_conclude else InquiryPhase.SYMPTOM_CONFIRM,
    }
```

- `apply_context_weights` 【自定义变量】：Step 6 的三层加权函数
- `should_conclude` 【自定义变量】：Step 6 的收敛判断函数（轮次上限 / Top1>=70% / Top1-Top2>=30%）
- 这是路由枢纽节点：根据收敛结果决定走向 `node_conclude` 还是 `node_ask_symptom`

#### ⑥ node_ask_symptom — 追问症状

```python
async def node_ask_symptom(state: InquiryState) -> dict:
    # 1. 从候选疾病中找到「已确认症状覆盖不足」的疾病
    # 2. 从该疾病的 all_symptoms 中挑选未确认/未否认的症状
    # 3. 用 ASK_SYMPTOM_PROMPT 生成口语化追问
    top_candidate = state["candidate_diseases"][0]
    unasked = [s for s in top_candidate.all_symptoms
               if s not in state["confirmed_symptoms"]
               and s not in state["denied_symptoms"]]

    if not unasked:
        # 该疾病所有症状都已问过 → 直接收敛
        return {"phase": InquiryPhase.CONCLUDE, "force_conclude": True}

    next_symptom = unasked[0]
    reply = await llm.ainvoke(ASK_SYMPTOM_PROMPT.format(symptom=next_symptom))

    return {
        "messages": [reply],
        "pending_ask_symptoms": [next_symptom],
        "phase": InquiryPhase.COMPLICATION_CHECK,
    }
```

- `pending_ask_symptoms` 【自定义变量】：记录当前正在追问的症状名，等待用户回答后与 `node_parse_answer` 对账
- 追问策略：贪心策略，选 Top1 候选疾病的未问症状逐个追问

#### ⑦ node_parse_answer — 解析用户回答

```python
async def node_parse_answer(state: InquiryState) -> dict:
    user_msg = state["messages"][-1].content
    pending = state["pending_ask_symptoms"]

    # 用 PARSE_ANSWER_PROMPT + with_structured_output 解析用户回答
    structured_llm = llm.with_structured_output(AnswerParse)  # 【内置方法】
    result = await structured_llm.ainvoke(
        PARSE_ANSWER_PROMPT.format(symptom=pending[0], user_answer=user_msg)
    )

    confirmed = list(state["confirmed_symptoms"])
    denied = list(state["denied_symptoms"])

    if result.has_symptom:
        confirmed.append(pending[0])
    else:
        denied.append(pending[0])

    # 回到 node_confidence 重新计算置信度
    return {
        "confirmed_symptoms": confirmed,
        "denied_symptoms": denied,
        "pending_ask_symptoms": [],
        "phase": InquiryPhase.COMPLICATION_CHECK,
    }
```

- `AnswerParse` 【自定义变量】：Pydantic Schema，包含 `has_symptom: bool` + `confidence: float`
- 解析完成后回到 `node_confidence`：新一轮置信度计算 + 收敛判断
- 形成闭环：`confidence -> ask -> parse -> confidence` 直到收敛

#### ⑧ node_conclude — 生成诊断结论

```python
async def node_conclude(state: InquiryState) -> dict:
    candidates = state["candidate_diseases"]

    # 1. 构建 InquiryHandoffPayload
    payload = InquiryHandoffPayload(
        primary_disease=candidates[0].name,
        suspected_diseases=[c.name for c in candidates[1:3]],
        department=candidates[0].department,
        confirmed_symptoms=state["confirmed_symptoms"],
        denied_symptoms=state["denied_symptoms"],
        recommended_exams=candidates[0].recommended_exams,
        confidence_score=candidates[0].confidence_score,
        inquiry_rounds=state["round"],
        session_id=state["session_id"],
    )

    # 2. 用 CONCLUDE_PROMPT 生成诊断结论文案
    reply = await llm.ainvoke(
        CONCLUDE_PROMPT.format(payload=payload.model_dump_json(indent=2, ensure_ascii=False))
    )

    # 3. 持久化问诊记录到 PostgreSQL
    consultation_id = await save_consultation(db, state)

    return {
        "messages": [reply],
        "handoff_payload": payload,
        "phase": InquiryPhase.HANDOFF,
    }
```

- `InquiryHandoffPayload` 【自定义变量】：Step 2 定义的移交数据包
- `model_dump_json(indent=2, ensure_ascii=False)` 【内置方法】：Pydantic 序列化为格式化 JSON，`ensure_ascii=False` 保留中文
- `save_consultation` 【自定义变量】：Step 8 的 PostgreSQL 写入函数
- `handoff_payload` 被存入 state，供下一个节点 `node_handoff` 使用

#### ⑨ node_handoff — 挂号移交

```python
async def node_handoff(state: InquiryState) -> dict:
    payload = state["handoff_payload"]

    # 调用 Inquiry Worker Agent 执行挂号
    worker = await get_inquiry_worker_agent()
    result = await worker.ainvoke(
        {"messages": [HumanMessage(content=payload.model_dump_json())]},
        config={"configurable": {"thread_id": state["session_id"]}},
    )

    return {
        "messages": [result["messages"][-1]],
        "phase": InquiryPhase.END,
    }
```

- `get_inquiry_worker_agent()` 【自定义变量】：返回 Worker Agent 单例（Step 10 实现）
- Worker Agent 接收 `InquiryHandoffPayload` JSON，自主调用挂号工具
- 完成后 `phase = END`，图执行结束

### 8.5 路由函数

路由函数是无状态的纯函数，根据 `state.phase` 返回下一个节点名：

```python
def route(state: InquiryState) -> str:
    phase = state["phase"]

    if phase == InquiryPhase.CLARIFY:
        return "node_clarify"
    elif phase == InquiryPhase.GRAPH_QUERY:
        return "node_normalize"
    elif phase == InquiryPhase.SYMPTOM_CONFIRM:
        if state.get("force_conclude") or should_conclude(state["candidate_diseases"], state["round"]):
            return "node_conclude"
        return "node_ask_symptom"
    elif phase == InquiryPhase.COMPLICATION_CHECK:
        return "node_confidence"
    elif phase == InquiryPhase.CONCLUDE:
        return "node_conclude"
    elif phase == InquiryPhase.HANDOFF:
        return "node_handoff"
    elif phase == InquiryPhase.END:
        return END
    return END
```

- `END` 【内置方法】：LangGraph 内置常量，表示图的终止节点
- 路由函数不执行任何副作用，只做判断

### 8.6 组装图

```python
def build_inquiry_graph() -> CompiledStateGraph:
    """构建并编译问诊状态图"""
    graph = StateGraph(InquiryState)                # 【内置方法】创建状态图

    # 1. 添加节点
    graph.add_node("node_emergency", node_emergency)
    graph.add_node("node_clarify", node_clarify)
    graph.add_node("node_normalize", node_normalize)
    graph.add_node("node_query_neo4j", node_query_neo4j)
    graph.add_node("node_confidence", node_confidence)
    graph.add_node("node_ask_symptom", node_ask_symptom)
    graph.add_node("node_parse_answer", node_parse_answer)
    graph.add_node("node_conclude", node_conclude)
    graph.add_node("node_handoff", node_handoff)

    # 2. 设置入口
    graph.set_entry_point("node_emergency")

    # 3. 添加条件边
    graph.add_conditional_edges("node_emergency", route)
    graph.add_conditional_edges("node_clarify", route)
    graph.add_conditional_edges("node_normalize", route)
    graph.add_conditional_edges("node_query_neo4j", route)
    graph.add_conditional_edges("node_confidence", route)
    graph.add_conditional_edges("node_ask_symptom", route)
    graph.add_conditional_edges("node_parse_answer", route)
    graph.add_conditional_edges("node_conclude", route)
    graph.add_conditional_edges("node_handoff", route)

    # 4. 编译
    return graph.compile(checkpointer=inquiry_checkpointer)
```

关键标注：
- `StateGraph(InquiryState)` 【内置方法】：LangGraph 创建状态图，`InquiryState` 是 TypedDict 状态类型
- `add_node(name, func)` 【内置方法】：注册节点，name 是字符串标识，func 是 async 函数
- `set_entry_point(name)` 【内置方法】：设置图的入口节点
- `add_conditional_edges(node, router)` 【内置方法】：从 `node` 出发，由 `router` 返回值决定下一个节点
- `compile(checkpointer=...)` 【内置方法】：编译图，`checkpointer` 让图支持多轮中断续接
- `CompiledStateGraph` 【内置方法】：编译后的图对象，拥有 `ainvoke` / `astream` 方法

### 8.7 多轮对话的 Checkpoint 机制

```
用户第1轮："我头疼发烧"
    ↓
graph.ainvoke(input, config={"configurable": {"thread_id": "user:session"}})
    ↓
node_emergency -> node_clarify -> node_normalize -> node_query_neo4j
    -> node_confidence -> node_ask_symptom
    ↓
checkpoint 保存到 Redis（state 快照）
    ↓
AI 回复："请问头痛是搏动性的还是胀痛？"
    ↓
图暂停（等待用户回答）

用户第2轮："胀痛，还有恶心"
    ↓
graph.ainvoke(input, config={"configurable": {"thread_id": "user:session"}})
    ↓
从 Redis 恢复 state 快照 -> 追加用户消息 -> 从 node_parse_answer 继续
    ↓
node_parse_answer -> node_confidence -> should_conclude? -> node_conclude
    ↓
最终诊断 + 挂号
```

- `thread_id` 【自定义变量】：会话标识，格式 `user_id:session_id`
- checkpointer 使用 `AsyncRedisSaver`，与 Supervisor Agent 共用 Redis 但独立连接池
- 每轮 `ainvoke` 自动从 checkpoint 恢复 + 保存

### 8.8 与现有 Supervisor Agent 的关系

| 维度 | Supervisor Agent | InquiryGraph |
|------|-----------------|--------------|
| 创建方式 | `create_agent()` | `StateGraph().compile()` |
| 调用方式 | `agent.ainvoke({"messages": [...]})` | `graph.ainvoke(input, config)` |
| 流程控制 | LLM ReAct 自由推理 | 确定性有向图路由 |
| 状态 | 仅 `messages` 列表 | `InquiryState` 含 10+ 结构化字段 |
| checkpointer | `AsyncRedisSaver` | `AsyncRedisSaver`（共用模式） |
| store | `MilvusStore`（长期记忆） | 不直接用 store，通过 `patient_context` 读取 |

---

## 九、Step 10：Inquiry Worker Agent（inquiry_agent.py）

### 9.1 Worker Agent 的角色

Worker Agent 是一个轻量级 ReAct Agent，职责单一：接收 `InquiryHandoffPayload`，执行挂号操作。

```
InquiryGraph node_handoff
    ↓ 传入 InquiryHandoffPayload JSON
Worker Agent（ReAct）
    ├── 读取 payload → 理解诊断结论
    ├── 调用 call_registration 工具 → 预约挂号
    └── 返回挂号结果
```

### 9.2 为什么用 ReAct 而不用 StateGraph

| 原因 | 说明 |
|------|------|
| 流程简单 | 只有「读 payload → 挂号」一步，不需要复杂路由 |
| 需要灵活性 | 挂号可能失败需重试或换科室，LLM 自主决策更合适 |
| 复用现有模式 | 与 Supervisor Agent 同一模式，降低维护成本 |

### 9.3 Worker Agent 代码

```python
# src/agents/workers/inquiry_agent.py

from langchain.agents import create_agent
from agents.tools.worker_tools import call_registration

_inquiry_worker = None

async def get_inquiry_worker_agent():
    """返回 Worker Agent 单例"""
    global _inquiry_worker
    if _inquiry_worker is None:
        _inquiry_worker = create_agent(
            model="deepseek-chat",
            tools=[call_registration],
            system_prompt=(
                "你是天宫医疗的挂号助手。"
                "你会收到一个 JSON 格式的问诊结论，包含诊断、建议科室、置信度等信息。"
                "请根据建议科室为患者预约挂号。"
                "如果目标科室无号源，尝试推荐相近科室。"
            ),
            checkpointer=AsyncRedisSaver(redis_client=get_checkpointer_redis()),
        )
        await _inquiry_worker.checkpointer.asetup()
    return _inquiry_worker
```

关键标注：
- `create_agent` 【内置方法】：LangChain 的 Agent 工厂函数，与 Supervisor Agent 同一个方法
- `call_registration` 【自定义变量】：挂号工具，Step 11 实现
- `system_prompt` 【自定义变量】：指导 LLM 如何处理 payload
- `AsyncRedisSaver` 【内置方法】：与 Supervisor Agent 同一 checkpointer 类型，但 `thread_id` 不同
- 单例模式与 `get_supervisor_agent()` 完全一致

### 9.4 Worker Agent 与 InquiryGraph 的数据流

```
时刻① node_handoff 构建 payload
    │
    ↓  payload.model_dump_json()
时刻② Worker Agent.ainvoke({"messages": [HumanMessage(json)]})
    │
    ↓  LLM 读取 JSON → 决定调用 call_registration
时刻③ call_registration(department="神经内科", patient_id="123")
    │
    ↓  调用医院 HIS 接口
时刻④ 返回挂号结果 → Worker Agent 生成回复
    │
    ↓  result["messages"][-1]
时刻⑤ node_handoff 把回复追加到 InquiryState.messages
    │
    ↓  phase = END
时刻⑥ 图执行结束 → Redis inquiry_active 删除
```

---

## 十、Step 11：工具改造 + FastAPI 路由分流

### 10.1 改造前的调用链

```
用户请求 → FastAPI → chat_endpoint(user_id, session_id, message)
    ↓
get_supervisor_agent() → Supervisor Agent.ainvoke(...)
    ↓
LLM ReAct → 可能调用 save_memory / search_memory → 返回回复
```

当前 `src/api/routers/chat.py` 已实现 `POST /api/v1/chat` 与 `/stream` 路由（含 Redis 问诊分流），供 `src/main.py` 注册使用。

### 10.2 改造后的调用链

```
用户请求 → POST /api/v1/chat
    ↓
chat_route(user_id, session_id, message)
    ↓
Redis EXISTS inquiry:{user_id}:{session_id}    ← 【新增】问诊激活标志
    ├── YES → InquiryGraph.ainvoke(...)         ← 直接走问诊图
    └── NO  → Supervisor Agent.ainvoke(...)     ← 走通用 Agent
                ↓
                Supervisor 识别问诊意图
                ↓ 调用 call_inquiry_agent 工具
                ↓
                Redis SET inquiry:{user_id}:{session_id}  ← 激活问诊
                ↓
                InquiryGraph.ainvoke(...)  ← 启动问诊首轮
```

### 10.3 Redis 路由分流标志

```python
# src/infra/redis_cache.py 新增

async def is_inquiry_active(user_id: str, session_id: str) -> bool:
    """检查问诊是否已激活"""
    key = f"inquiry:{user_id}:{session_id}"
    return await _redis_client.exists(key) > 0

async def set_inquiry_active(user_id: str, session_id: str, ttl: int = 3600):
    """激活问诊模式，TTL 1小时自动过期"""
    key = f"inquiry:{user_id}:{session_id}"
    await _redis_client.set(key, "1", ex=ttl)

async def clear_inquiry_active(user_id: str, session_id: str):
    """问诊结束后清除标志"""
    key = f"inquiry:{user_id}:{session_id}"
    await _redis_client.delete(key)
```

- `_redis_client` 【自定义变量】：`redis_cache.py` 中已有的业务 Redis 客户端（`decode_responses=True`）
- `exists` / `set` / `delete` 【内置方法】：redis-py 标准命令
- `ex=ttl` 【内置方法】：设置过期时间（秒），防止异常中断后永久卡在问诊模式

### 10.4 call_inquiry_agent 工具（Supervisor 的工具）

```python
# src/agents/tools/worker_tools.py 新增

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

@tool
async def call_inquiry_agent(runtime: ToolRuntime) -> str:
    """
    当用户描述身体不适或症状时调用此工具，启动结构化问诊流程。
    问诊会逐步确认症状、查询疾病知识、计算置信度，最终给出诊断建议和挂号。
    """
    # 从 thread_id 提取 user_id（复用 store_tools.py 中的逻辑）
    user_id = get_user_id(runtime)

    # 激活问诊模式
    redis = await get_redis_client()
    session_id = runtime.config["configurable"]["thread_id"].split(":")[1]
    await set_inquiry_active(user_id, session_id)

    # 启动 InquiryGraph 首轮
    graph = build_inquiry_graph()
    result = await graph.ainvoke(
        {"messages": runtime.config.get("messages", [])},
        config={"configurable": {"thread_id": f"{user_id}:{session_id}"}},
    )

    return result["messages"][-1].content
```

关键标注：
- `@tool` 【内置方法】：LangChain 装饰器，把函数注册为 Agent 可调用的工具
- `ToolRuntime` 【内置方法】：LangGraph 注入的运行时上下文，携带 `config` 和 `store`
- `get_user_id(runtime)` 【自定义变量】：复用 `store_tools.py` 中的函数，从 `thread_id` 提取 `user_id`
- `build_inquiry_graph()` 【自定义变量】：Step 9 的图构建函数
- 此工具被添加到 Supervisor Agent 的 `tools` 列表中

### 10.5 call_registration 工具（Worker Agent 的工具）

```python
# src/agents/tools/worker_tools.py 新增

@tool
async def call_registration(department: str, runtime: ToolRuntime) -> str:
    """
    为患者预约指定科室的挂号。

    Args:
        department: 科室名称，如"神经内科"、"心内科"
    """
    user_id = get_user_id(runtime)
    # 调用医院 HIS 系统接口（此处用模拟实现）
    # 实际项目通过 src/infra/his_client.py 调用

    appointment_id = f"APT-{int(time.time())}"
    return (
        f"挂号成功\n"
        f"科室：{department}\n"
        f"预约号：{appointment_id}\n"
        f"患者ID：{user_id}\n"
        f"请提前30分钟到院取号。"
    )
```

- `department` 【自定义变量】：从 `InquiryHandoffPayload.department` 传递而来
- 实际项目中此处应调用医院 HIS 接口，当前为模拟实现

### 10.6 改造后的 chat_endpoint

```python
# src/agents/supervisor_agent.py 改造后

async def chat_endpoint(user_id: str, session_id: str, message: str):
    """统一聊天入口：根据问诊标志分流"""
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

    # ① 检查问诊是否已激活
    if await is_inquiry_active(user_id, session_id):
        # 问诊模式：直接走 InquiryGraph
        graph = await get_inquiry_graph()  # 单例
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )

        # 检查是否问诊已结束
        if result.get("phase") == InquiryPhase.END:
            await clear_inquiry_active(user_id, session_id)

        return result["messages"][-1].content

    # ② 非问诊模式：走 Supervisor Agent
    agent = await get_supervisor_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content
```

数据流时刻表：

```
时刻① 用户发消息 → chat_endpoint
    ↓
时刻② Redis EXISTS inquiry:{user_id}:{session_id}
    ├── YES → 时刻③a 直接走 InquiryGraph（续接）
    │         ↓
    │    时刻④a graph.ainvoke → 从 checkpoint 恢复 state → 继续追问/收敛
    │         ↓
    │    时刻⑤a phase == END? → clear_inquiry_active → 返回结果
    │
    └── NO → 时刻③b 走 Supervisor Agent
              ↓
         时刻④b Supervisor ReAct → 识别症状意图 → 调用 call_inquiry_agent
              ↓
         时刻⑤b call_inquiry_agent → set_inquiry_active → 启动 InquiryGraph 首轮
              ↓
         时刻⑥b 返回首轮问诊回复（如澄清话术）
```

### 10.7 FastAPI 路由注册

```python
# src/api/routers/chat.py 新增

from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["chat"])

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat_route(req: ChatRequest):
    from agents.supervisor_agent import chat_endpoint
    reply = await chat_endpoint(req.user_id, req.session_id, req.message)
    return ChatResponse(reply=reply, session_id=req.session_id)
```

```python
# src/main.py 改造后，在 create_app() 中注册路由

from api.routers.chat import router as chat_router

def create_app() -> FastAPI:
    ...
    app.include_router(chat_router, prefix="/api/v1")
    ...
```

- `APIRouter` 【内置方法】：FastAPI 路由分组器
- `BaseModel` 【内置方法】：Pydantic 请求/响应模型
- `Depends` 【内置方法】：FastAPI 依赖注入（当前简化为直接调用）

---

## 十一、完整架构总览

### 11.1 改造后的系统全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI 应用 (src/main.py)                    │
│                                                                      │
│  POST /api/v1/chat ──→ chat_route ──→ chat_endpoint                  │
│                                          │                           │
│                    ┌─────────────────────┴───────────────────┐       │
│                    │  Redis: inquiry:{user}:{session}         │       │
│                    │    EXISTS?                                │       │
│                    └──────┬──────────────────┬───────────────┘       │
│                           │ YES              │ NO                    │
│                           ↓                  ↓                       │
│              ┌────────────────────┐  ┌──────────────────────┐        │
│              │   InquiryGraph     │  │  Supervisor Agent    │        │
│              │   (StateGraph)     │  │  (ReAct create_agent)│        │
│              └────────┬───────────┘  └──────────┬──────────┘        │
│                       │                         │                   │
│         ┌─────────────┼─────────────┐    tools: │                   │
│         │             │             │    ├── save_memory              │
│    ┌────┴───┐   ┌──────┴────┐  ┌────┴──────┐  ├── search_memory       │
│    │节点1-9 │   │ Worker     │  │call_inquiry│  └── call_inquiry_agent│
│    │状态机  │   │ Agent      │  │_agent 工具  │           │             │
│    └────┬───┘   │(ReAct)    │  └─────────────┘           │             │
│         │       └─────┬─────┘                              │             │
│         │  call_reg    │                                    │             │
│         │  istration   │                    set_inquiry_active           │
│         │             │                                    │             │
│         └─────────────┴────────────────────────────────────┘             │
│                                                                          │
│  ┌──────────────────────── 基础设施层 ─────────────────────────┐         │
│  │                                                            │         │
│  │  PostgreSQL        Redis           Milvus        Neo4j      │         │
│  │  (患者/问诊)       (checkpoint     (长期记忆 +   (疾病知识  │         │
│  │                    +问诊标志)      症状索引)     图谱)     │         │
│  │                                                            │         │
│  │  src/infra/         src/infra/      src/infra/   src/infra/ │         │
│  │  database.py        redis_cache.py  milvus_store neo4j_     │         │
│  │                                      .py        client.py  │         │
│  └────────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 11.2 数据流完整闭环

```
步骤1 用户："我头疼发烧两天了"
    ↓
步骤2 FastAPI POST /api/v1/chat → chat_endpoint
    ↓
步骤3 Redis EXISTS inquiry:{user}:{session} → NO（首次）
    ↓
步骤4 Supervisor Agent.ainvoke → LLM 识别问诊意图 → 调用 call_inquiry_agent
    ↓
步骤5 Redis SET inquiry:{user}:{session} → 激活问诊
    ↓
步骤6 InquiryGraph.ainvoke 首轮
    ├── node_emergency：检查无急症
    ├── node_clarify：生成澄清话术 "请描述头痛的性质和部位"
    ├── node_normalize：LLM 提取 ["头痛", "发烧"]，Neo4j 匹配成功
    ├── node_query_neo4j：查到 3 个候选疾病（偏头痛/感冒/脑膜炎）
    ├── node_confidence：计算置信度 [0.45, 0.40, 0.20]，未收敛
    └── node_ask_symptom：追问 "头痛是搏动性的还是胀痛？"
    ↓
步骤7 checkpoint 保存到 Redis → 返回 AI 回复
    ↓
─── 用户第二轮 ───
    ↓
步骤8 用户："胀痛，还有恶心"
    ↓
步骤9 Redis EXISTS inquiry:{user}:{session} → YES
    ↓
步骤10 InquiryGraph.ainvoke 续轮
    ├── 从 checkpoint 恢复 state
    ├── node_parse_answer：解析"胀痛+恶心" → confirmed += ["恶心"]
    ├── node_confidence：重新计算 [0.75, 0.35, 0.15] → 偏头痛 Top1≥70%
    ├── node_conclude：生成诊断结论 + 构建 HandoffPayload
    └── node_handoff：调用 Worker Agent → call_registration → 挂号成功
    ↓
步骤11 phase = END → Redis DELETE inquiry:{user}:{session}
    ↓
步骤12 返回最终回复："诊断为偏头痛（置信度75%），已为您预约神经内科挂号..."
```

### 11.3 与现有代码的对照总结

| 现有代码文件 | 改造内容 | 新增文件 |
|-------------|---------|---------|
| `src/agents/supervisor_agent.py` | `chat_endpoint` 增加 Redis 分流逻辑；`tools` 增加 `call_inquiry_agent` | — |
| `src/agents/store_tools.py` | 不改动，`get_user_id` 被新工具复用 | — |
| `src/infra/redis_cache.py` | 新增 `is_inquiry_active` / `set_inquiry_active` / `clear_inquiry_active` | — |
| `src/infra/milvus_store.py` | 不改动，InquiryGraph 不直接用 Store | — |
| `src/main.py` | `create_app()` 中注册 chat_router | `src/api/routers/chat.py` |
| — | 全部新增 | `src/agents/inquiry/state.py` |
| — | 全部新增 | `src/agents/inquiry/symptom_normalizer.py` |
| — | 全部新增 | `src/agents/inquiry/neo4j_queries.py` |
| — | 全部新增 | `src/agents/inquiry/confidence.py` |
| — | 全部新增 | `src/agents/inquiry/prompts.py` |
| — | 全部新增 | `src/agents/inquiry/db_queries.py` |
| — | 全部新增 | `src/agents/inquiry/graph.py` |
| — | 全部新增 | `src/agents/workers/inquiry_agent.py` |
| — | 全部新增 | `src/agents/tools/worker_tools.py` |
| — | 全部新增 | `scripts/init_symptom_index.py` |

### 11.4 核心设计决策回顾

| 决策 | 选择 | 理由 |
|------|------|------|
| 问诊流程控制 | StateGraph 而非 ReAct | 医疗场景需确定性路由，不可让 LLM 自由推理 |
| 多轮状态管理 | TypedDict + Checkpoint | 结构化字段（症状列表/候选疾病）比消息列表更精确 |
| 症状标准化 | 三层兜底 | LLM 提取有误差，Neo4j 精确匹配 + Milvus 语义兜底确保覆盖 |
| 收敛判断 | 置信度算法 + 阈值 | 比 LLM 自行判断更可量化、可调参 |
| 分流机制 | Redis 标志位 | O(1) 判断，无需查数据库；TTL 防死锁 |
| Worker Agent | ReAct 而非 StateGraph | 挂号流程简单，需灵活性（可能换科室） |
| 诊断移交 | HandoffPayload JSON | 结构化传递，Worker Agent 可读可解析 |

---

## 十二、关键概念速查表

### 12.1 【内置方法】速查

| 方法 | 来源 | 作用 |
|------|------|------|
| `create_agent(model, tools, system_prompt, ...)` | `langchain.agents` | 创建 ReAct Agent |
| `StateGraph(state_type)` | `langgraph.graph` | 创建状态图 |
| `add_node(name, func)` | `StateGraph` | 注册节点 |
| `add_conditional_edges(node, router)` | `StateGraph` | 添加条件边 |
| `compile(checkpointer=...)` | `StateGraph` | 编译图 |
| `ainvoke(input, config)` | `CompiledStateGraph` | 异步执行图 |
| `with_structured_output(schema)` | `BaseChatModel` | 约束 LLM 输出为 Pydantic Schema |
| `@tool` | `langchain_core.tools` | 把函数注册为工具 |
| `ToolRuntime` | `langgraph.prebuilt` | 工具运行时上下文 |
| `add_messages` | `langgraph.graph` | 消息列表 reducer（追加而非覆盖） |
| `Annotated[T, reducer]` | `typing` | 声明 reducer |
| `AsyncRedisSaver(redis_client)` | `langgraph.checkpoint.redis` | Redis Checkpoint |
| `asetup()` | `AsyncRedisSaver` | 初始化 RediSearch 索引 |
| `model_dump_json(indent, ensure_ascii)` | `pydantic.BaseModel` | 序列化为 JSON |
| `embed_query(text)` | `Embeddings` | 文本转向量 |
| `BaseStore` | `langgraph.store.base` | Store 基类 |
| `aput(namespace, key, value)` | `BaseStore` | 写入记忆 |
| `asearch(namespace, query)` | `BaseStore` | 语义检索记忆 |
| `END` | `langgraph.graph` | 图终止常量 |

### 12.2 【自定义变量】速查

| 变量 | 文件 | 作用 |
|------|------|------|
| `InquiryState` | `state.py` | 问诊状态 TypedDict |
| `InquiryPhase` | `state.py` | 问诊阶段枚举 |
| `InquiryHandoffPayload` | `state.py` | 移交数据包 |
| `PatientContext` | `state.py` | 患者上下文 |
| `CandidateDisease` | `neo4j_queries.py` | 候选疾病模型 |
| `normalize_symptoms()` | `symptom_normalizer.py` | 三层标准化入口 |
| `query_candidate_diseases()` | `neo4j_queries.py` | 知识图谱查询 |
| `apply_context_weights()` | `confidence.py` | 置信度加权 |
| `should_conclude()` | `confidence.py` | 收敛判断 |
| `build_inquiry_graph()` | `graph.py` | 构建问诊图 |
| `get_inquiry_worker_agent()` | `inquiry_agent.py` | Worker Agent 单例 |
| `call_inquiry_agent` | `worker_tools.py` | 启动问诊工具 |
| `call_registration` | `worker_tools.py` | 挂号工具 |
| `is_inquiry_active()` | `redis_cache.py` | 检查问诊标志 |
| `set_inquiry_active()` | `redis_cache.py` | 激活问诊 |
| `clear_inquiry_active()` | `redis_cache.py` | 清除问诊标志 |
