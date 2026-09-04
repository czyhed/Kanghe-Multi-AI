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

### 8.2 图的拓扑结构（真实代码）

真实代码共 **10 个节点**（1 个 dispatcher 分发器 + 9 个业务节点）：

```
                          ┌─────────────────────────────┐
                          │        dispatcher            │  入口分发（根据 round 路由）
                          └─────────────┬───────────────┘
                    ┌──────────┬─────────┴──────────┬──────────┐
              round==0    pending_ask_symptoms 非空   否则
                    │          │                    │
                    ↓          ↓                    ↓
              load_context  parse_answer        extract_symptoms
               ①加载上下文   ⑦解析回答            ③症状标准化
                    ↓          │                    │
              check_emergency   └──→ query_neo4j ←──┘
               ②急症识别             ⑤查候选疾病
                    ↓                （含置信度+收敛）
              extract_symptoms              │
               ③症状标准化        ┌─────────┴─────────┐
                    │         收敛(CONCLUDE)    未收敛(SYMPTOM_CONFIRM)
              有症状→query_neo4j       ↓              ↓
              无→clarify④         conclude       ask_symptoms
                                   ⑧诊断结论       ⑥生成追问
                                      ↓              ↓
                                 save_record     END（等用户回答）
                                   ⑨保存记录         └→ parse_answer
```

节点职责一览（真实代码）：

| 节点 | 函数名 | 职责 | 执行时机 |
|------|--------|------|----------|
| dispatcher | `lambda state: {}` | 无操作分发器，仅用于路由 | 每次 |
| ① | `node_load_context` | 从 PostgreSQL 加载患者上下文 | 仅首轮（round==0） |
| ② | `node_check_emergency` | 急症识别，命中直接 END | 仅首轮 |
| ③ | `node_extract_symptoms` | 症状三层标准化 | 每轮 |
| ④ | `node_clarify` | 澄清模糊描述 | 无明确症状时 |
| ⑤ | `node_query_neo4j` | 查候选疾病 + 详情 + 置信度 + 收敛判断 | 每轮 |
| ⑥ | `node_ask_symptoms` | 生成口语化追问话术 | 未收敛时 |
| ⑦ | `node_parse_answer` | 解析用户对追问的回答 | 有 pending 时 |
| ⑧ | `node_conclude` | 生成诊断结论 + 构建 HandoffPayload | 收敛时 |
| ⑨ | `node_save_record` | 保存问诊记录到 PostgreSQL | 结论后 |

> 关键差异：真实代码把"置信度计算 + 收敛判断"合并进了 `node_query_neo4j`（不是单独的 `node_confidence`），并把"急症检查"独立成 `node_check_emergency`。

### 8.3 节点函数统一签名（真实代码）

所有节点函数遵循同一个签名模式（**多了一个 `deps` 参数**）：

```python
async def node_xxx(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    作用：执行节点逻辑，返回需要更新的状态字段
    意思：LangGraph 会把返回的 dict merge 到当前 state 中
    一句话记忆点：节点函数只管「读 state → 调 deps → 返回 dict」，LangGraph 负责合并
    """
    # 读 state
    confirmed = state.confirmed_symptoms   # 注意：BaseModel 用 . 访问，不是 ["..."]
    # 调依赖
    response = await deps.llm.ainvoke(...)
    # 返回需要更新的字段
    return {"phase": InquiryPhase.GRAPH_QUERY, "confirmed_symptoms": updated_list}
```

关键标注：
- `InquiryState` 【自定义变量】：`state.py` 定义的 Pydantic BaseModel（不是 TypedDict），用 `.字段` 访问
- `InquiryDeps` 【自定义变量】：依赖注入容器（`llm`/`neo4j_driver`/`embedding_model`/`milvus_client`/`db_session`），通过闭包绑定到节点
- 返回 `dict` 而非完整 `InquiryState`：LangGraph 只 merge 返回的字段
- `messages` 字段用 `add_messages` reducer，返回的新消息自动追加到列表末尾

### 8.4 九个核心节点逐个拆解（真实代码）

#### ① node_load_context — 加载患者上下文

```python
async def node_load_context(state: InquiryState, deps: InquiryDeps) -> dict:
    if state.round > 0:
        return {}  # 非首轮，不重复加载
    from src.agents.inquiry.db_queries import load_patient_context
    patient_id = int(state.patient_context.patient_id) if state.patient_context.patient_id else None
    patient_ctx = await load_patient_context(patient_id=patient_id, db=deps.db_session)
    merged_ctx = PatientContext(
        patient_id=state.patient_context.patient_id,  # 保持 str，贯穿整个流程
        age=patient_ctx.age, gender=patient_ctx.gender,
        allergy_history=patient_ctx.allergy_history,
        medical_history=patient_ctx.medical_history,
        long_term_memories=state.patient_context.long_term_memories,  # 由外部注入
    )
    return {"patient_context": merged_ctx}
```

- 仅首轮（`round == 0`）执行；从 PostgreSQL 加载病史，Milvus 长期记忆由外部（`run_inquiry`）注入
- `load_patient_context` 【自定义变量】：`db_queries.py` 的查询函数

#### ② node_check_emergency — 急症识别

```python
async def node_check_emergency(state: InquiryState, deps: InquiryDeps) -> dict:
    if state.round > 0:
        return {}  # 非首轮，跳过
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):          # 【内置方法】倒序找最后一条用户消息
            last_user_msg = msg.content
            break
    if not last_user_msg:
        return {}
    prompt = EMERGENCY_CHECK_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)               # 【内置方法】手动解析 JSON
        if result.get("is_emergency"):
            emergency_reply = "⚠️ 根据您描述的症状，这可能是紧急情况！...请立即前往急诊科就诊"
            return {"phase": InquiryPhase.END, "messages": [AIMessage(content=emergency_reply)]}
    except Exception as e:
        logger.warning(f"急症识别解析失败: {e}")
    return {}
```

- 用 `deps.llm.ainvoke` 拿原始文本后手动 `json.loads`（**不是** `with_structured_output`）
- 急症命中直接 `phase = END`，结束整个流程

#### ③ node_extract_symptoms — 症状三层标准化

```python
async def node_extract_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    result = await normalize_symptoms(             # 【自定义变量】三层标准化入口
        user_input=last_user_msg,
        llm=deps.llm, neo4j_driver=deps.neo4j_driver,
        embedding_model=deps.embedding_model, milvus_client=deps.milvus_client,
    )
    new_confirmed = list(set(state.confirmed_symptoms) | set(result["all_standard"]))
    new_unmatched = list(set(state.unmatched_symptoms) | set(result["unmatched"]))
    if new_confirmed:
        return {"confirmed_symptoms": new_confirmed, "unmatched_symptoms": new_unmatched,
                "phase": InquiryPhase.GRAPH_QUERY}   # 有症状 → 查图谱
    return {"unmatched_symptoms": new_unmatched, "phase": InquiryPhase.CLARIFY}  # 无 → 澄清
```

- `normalize_symptoms` 【自定义变量】：`symptom_normalizer.py` 的三层标准化入口
- 有明确症状 → 下一阶段 `GRAPH_QUERY`；无 → `CLARIFY`

#### ④ node_clarify — 澄清模糊描述

```python
async def node_clarify(state: InquiryState, deps: InquiryDeps) -> dict:
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    prompt = CLARIFY_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    return {"round": state.round + 1, "messages": [AIMessage(content=response.content)]}
```

- 只生成澄清话术，然后 `END` 等待用户下一轮输入（不查图谱）

#### ⑤ node_query_neo4j — 查候选疾病 + 置信度 + 收敛判断（合并节点）

```python
async def node_query_neo4j(state: InquiryState, deps: InquiryDeps) -> dict:
    candidates = await query_candidate_diseases(
        confirmed_symptoms=state.confirmed_symptoms, neo4j_driver=deps.neo4j_driver,
    )
    if candidates:
        candidates = await enrich_candidate_details(candidates, deps.neo4j_driver)
        candidates = apply_context_weights(candidates, state.patient_context, state.denied_symptoms)
    should_conclude, force_conclude = check_convergence(candidates, state.round)
    if should_conclude:
        return {"candidate_diseases": candidates, "phase": InquiryPhase.CONCLUDE,
                "force_conclude": force_conclude}
    elif not candidates:
        return {"candidate_diseases": [], "phase": InquiryPhase.CONCLUDE, "force_conclude": True}
    return {"candidate_diseases": candidates, "phase": InquiryPhase.SYMPTOM_CONFIRM}
```

- 关键差异：真实代码把「查询 + 详情补充 + 置信度加权 + 收敛判断」**全部合并进这一个节点**（没有独立的 `node_confidence`）
- `query_candidate_diseases` / `enrich_candidate_details` / `apply_context_weights` / `check_convergence` 【自定义变量】

#### ⑥ node_ask_symptoms — 生成追问话术

```python
async def node_ask_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    pending = await get_pending_symptoms(
        candidates=state.candidate_diseases, confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms, asked_symptoms=state.asked_symptoms,
    )
    top_symptoms = [s for s, _ in pending[:3]]     # 取区分度最高的前 3 个
    if not top_symptoms:
        return {"phase": InquiryPhase.CONCLUDE}     # 无更多可问，直接结论
    human_symptoms = await humanize_symptoms(top_symptoms, deps.llm)  # 口语化
    prompt = ASK_SYMPTOMS_PROMPT.format(symptoms_to_ask="\n".join(f"- {s}" for s in human_symptoms))
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    return {
        "round": state.round + 1,
        "asked_symptoms": state.asked_symptoms + top_symptoms,  # 记入已问，避免重复
        "pending_ask_symptoms": top_symptoms,                   # 本轮问了哪些，供解析对账
        "messages": [AIMessage(content=response.content)],
    }
```

- 追问策略：`get_pending_symptoms` 按出现次数升序（区分度高优先），取前 3 个
- `humanize_symptoms` 【自定义变量】：标准术语转口语（"发热"→"发烧"）
- `asked_symptoms` 记录已问症状，防止重复追问

#### ⑦ node_parse_answer — 解析用户回答

```python
async def node_parse_answer(state: InquiryState, deps: InquiryDeps) -> dict:
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    if not last_user_msg or not state.pending_ask_symptoms:
        return {}
    prompt = PARSE_ANSWER_PROMPT.format(
        asked_symptoms=", ".join(state.pending_ask_symptoms), user_answer=last_user_msg,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)               # {"confirmed": [...], "denied": [...]}
        new_confirmed = list(set(state.confirmed_symptoms) | set(parsed.get("confirmed", [])))
        new_denied = list(set(state.denied_symptoms) | set(parsed.get("denied", [])))
    except Exception as e:
        logger.warning(f"解析用户回答失败: {e}")
        new_confirmed, new_denied = state.confirmed_symptoms, state.denied_symptoms
    return {"confirmed_symptoms": new_confirmed, "denied_symptoms": new_denied,
            "pending_ask_symptoms": []}
```

- 返回的 `confirmed` / `denied` 分别并入对应列表，解析后 `pending_ask_symptoms` 清空
- 路由到 `query_neo4j` 重新计算（形成闭环 `ask → parse → query_neo4j`）

#### ⑧ node_conclude — 生成诊断结论 + 构建 HandoffPayload

```python
async def node_conclude(state: InquiryState, deps: InquiryDeps) -> dict:
    candidates = state.candidate_diseases
    if not candidates:
        no_result_msg = "根据您描述的症状，我暂时无法在知识库中找到匹配的疾病。..."
        return {"phase": InquiryPhase.END, "messages": [AIMessage(content=no_result_msg)]}
    top1 = candidates[0]
    suspected = [c.name for c in candidates[1:5]]
    prompt = CONCLUSION_PROMPT.format(
        confirmed_symptoms="、".join(state.confirmed_symptoms) or "暂无",
        primary_disease=top1.name, confidence=top1.confidence,
        suspected_diseases="、".join(suspected) if suspected else "无",
        department=top1.department or "综合内科",
        checks="、".join(top1.checks[:5]) if top1.checks else "暂无",
        force_conclude=state.force_conclude,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    handoff = InquiryHandoffPayload(
        patient_id=state.patient_context.patient_id,
        patient_context=state.patient_context.model_dump(),
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
        unmatched_symptoms=state.unmatched_symptoms,
        primary_disease=top1.name, primary_confidence=top1.confidence,
        suspected_diseases=suspected, department=top1.department or "综合内科",
        recommended_checks=top1.checks[:5],
        total_rounds=state.round, session_id=state.session_id,
    )
    return {"phase": InquiryPhase.HANDOFF, "handoff_payload": handoff,
            "messages": [AIMessage(content=response.content)]}
```

- 同时生成「结论文案」和「HandoffPayload 挂号数据包」
- `model_dump()` 【内置方法】：Pydantic 转 dict，存入 `patient_context` 字段

#### ⑨ node_save_record — 保存问诊记录

```python
async def node_save_record(state: InquiryState, deps: InquiryDeps) -> dict:
    if not state.handoff_payload:
        return {"phase": InquiryPhase.END}
    from src.agents.inquiry.db_queries import save_consultation_record
    payload = state.handoff_payload
    chief_complaint = "、".join(state.confirmed_symptoms[:5])
    patient_id = int(payload.patient_id) if payload.patient_id else None
    await save_consultation_record(
        patient_id=patient_id, session_id=payload.session_id,
        chief_complaint=chief_complaint, diagnosis=payload.primary_disease,
        department_name=payload.department, urgency_level="normal", db=deps.db_session,
    )
    return {"phase": InquiryPhase.END}
```

- 结论生成后落库，最后 `phase = END`，流程结束



### 8.5 路由函数（真实代码，共 7 个）

路由函数是无状态的纯函数，根据 `state` 返回下一个节点名（不再只靠 `phase`）：

```python
def route_dispatcher(state: InquiryState) -> str:
    """入口分发：首轮走完整流程，后续轮次根据上一轮状态决定入口。"""
    if state.round == 0:
        return "load_context"
    if state.pending_ask_symptoms:      # 上一轮是追问 → 解析用户回答
        return "parse_answer"
    return "extract_symptoms"           # 上一轮是澄清 → 提取新症状

def route_after_emergency(state: InquiryState) -> str:
    """急症检查后：急症直接结束，否则提取症状。"""
    if state.phase == InquiryPhase.END:
        return END
    return "extract_symptoms"

def route_after_extract(state: InquiryState) -> str:
    """症状提取后：有症状查图谱，没症状澄清。"""
    if state.phase == InquiryPhase.GRAPH_QUERY:
        return "query_neo4j"
    return "clarify"

def route_after_neo4j(state: InquiryState) -> str:
    """Neo4j 查询后：收敛则结论，否则追问。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude"
    return "ask_symptoms"

def route_after_ask(state: InquiryState) -> str:
    """追问后：没有更多症状可问则直接结论，否则等待用户回答。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude"
    return END  # 等待用户回答（下一轮从 parse_answer 继续）

def route_after_parse(state: InquiryState) -> str:
    """解析回答后：重新查 Neo4j 更新候选疾病。"""
    return "query_neo4j"

def route_after_conclude(state: InquiryState) -> str:
    """结论后：保存记录，结束。"""
    if state.phase == InquiryPhase.END:
        return END
    return "save_record"
```

- `END` 【内置方法】：LangGraph 内置常量，表示图的终止节点
- 路由函数不执行任何副作用，只做判断
- 每个节点对应一个 `route_after_xxx`，语义更清晰

### 8.6 组装图（真实代码）

```python
def build_inquiry_graph(deps: InquiryDeps):
    """构建并编译问诊 StateGraph。deps 通过闭包注入到每个节点函数中。"""
    # 用闭包把 deps 绑定到节点函数
    async def _load_context(state):    return await node_load_context(state, deps)
    async def _check_emergency(state): return await node_check_emergency(state, deps)
    async def _extract_symptoms(state):return await node_extract_symptoms(state, deps)
    async def _clarify(state):         return await node_clarify(state, deps)
    async def _query_neo4j(state):     return await node_query_neo4j(state, deps)
    async def _ask_symptoms(state):    return await node_ask_symptoms(state, deps)
    async def _parse_answer(state):    return await node_parse_answer(state, deps)
    async def _conclude(state):        return await node_conclude(state, deps)
    async def _save_record(state):     return await node_save_record(state, deps)

    graph = StateGraph(InquiryState)
    # 注册节点（dispatcher 是无操作分发节点，仅用于路由）
    graph.add_node("dispatcher",       lambda state: {})
    graph.add_node("load_context",     _load_context)
    graph.add_node("check_emergency",  _check_emergency)
    graph.add_node("extract_symptoms", _extract_symptoms)
    graph.add_node("clarify",          _clarify)
    graph.add_node("query_neo4j",      _query_neo4j)
    graph.add_node("ask_symptoms",     _ask_symptoms)
    graph.add_node("parse_answer",     _parse_answer)
    graph.add_node("conclude",         _conclude)
    graph.add_node("save_record",      _save_record)

    graph.set_entry_point("dispatcher")
    # 固定边
    graph.add_edge("load_context", "check_emergency")
    graph.add_edge("clarify", END)          # 澄清后等待用户回复
    graph.add_edge("save_record", END)
    # 条件边（路由）
    graph.add_conditional_edges("dispatcher",       route_dispatcher,
        {"load_context": "load_context", "parse_answer": "parse_answer",
         "extract_symptoms": "extract_symptoms"})
    graph.add_conditional_edges("check_emergency",  route_after_emergency,
        {"extract_symptoms": "extract_symptoms", END: END})
    graph.add_conditional_edges("extract_symptoms", route_after_extract,
        {"query_neo4j": "query_neo4j", "clarify": "clarify"})
    graph.add_conditional_edges("query_neo4j",      route_after_neo4j,
        {"conclude": "conclude", "ask_symptoms": "ask_symptoms"})
    graph.add_conditional_edges("ask_symptoms",     route_after_ask,
        {"conclude": "conclude", END: END})
    graph.add_conditional_edges("parse_answer",     route_after_parse,
        {"query_neo4j": "query_neo4j"})
    graph.add_conditional_edges("conclude",         route_after_conclude,
        {"save_record": "save_record", END: END})
    return graph.compile()
```

关键标注：
- `StateGraph(InquiryState)` 【内置方法】：LangGraph 创建状态图，`InquiryState` 是 Pydantic BaseModel
- `add_node(name, func)` 【内置方法】：注册节点，name 是字符串标识，func 是 async 函数
- `set_entry_point(name)` 【内置方法】：设置图的入口节点
- `add_edge(a, b)` 【内置方法】：固定边，无条件从 a 到 b
- `add_conditional_edges(node, router, mapping)` 【内置方法】：从 `node` 出发，`router` 返回值查 `mapping` 得到下一个节点
- 闭包注入模式：`_xxx = lambda state: node_xxx(state, deps)` 把 `deps` 提前绑定
- **注意**：`graph.compile()` 没有传 `checkpointer`（多轮状态由 Redis 手动管理，见 8.7）

### 8.7 多轮对话的状态持久化机制（真实代码：手动 Redis 管理，非 Checkpointer）

> 关键差异：真实代码**没有使用 LangGraph 的 Checkpointer**（`graph.compile()` 无参），而是由 `run_inquiry` 返回 `new_state`，调用方手动序列化到 Redis，下一轮再手动反序列化传回。

```
用户第1轮："我头疼发烧"
    ↓
chat.py → 无活跃标记 → Supervisor → call_inquiry_agent 工具
    ↓
run_inquiry(existing_state=None)   # 首轮，新建 InquiryState
    ↓
graph.ainvoke(state, config)       # dispatcher → load_context → ... → ask_symptoms
    ↓
返回 (reply, new_state)
    ↓
new_state.model_dump_json() → redis.set("inquiry_state:u:s", json, ex=3600)
redis.set("inquiry_active:u:s", "1", ex=3600)     # 设置活跃标记
    ↓
AI 回复："请问头痛是搏动性的还是胀痛？"（图暂停，等待用户回答）

用户第2轮："胀痛，还有恶心"
    ↓
chat.py → redis.exists("inquiry_active:u:s") == True → 直接走 InquiryGraph
    ↓
redis.get("inquiry_state:u:s") → InquiryState.model_validate_json(raw)  # 恢复状态
    ↓
run_inquiry(existing_state=恢复的状态)   # dispatcher 根据 pending_ask_symptoms → parse_answer
    ↓
parse_answer → query_neo4j → ... → conclude → save_record
    ↓
问诊结束 → redis.delete("inquiry_active:u:s", "inquiry_state:u:s") + 触发挂号移交
```

- `thread_id` 【自定义变量】：会话标识，格式 `user_id:session_id`
- Redis 两个键：`inquiry_active:{thread_id}`（活跃标记）、`inquiry_state:{thread_id}`（状态快照，JSON）
- `model_dump_json()` / `model_validate_json()` 【内置方法】：Pydantic 序列化 / 反序列化，实现状态跨轮持久化
- 状态快照存的是**整个 `InquiryState`**（含 messages、候选疾病、症状列表），下一轮完整恢复

### 8.8 与现有 Supervisor Agent 的关系

| 维度 | Supervisor Agent | InquiryGraph |
|------|-----------------|--------------|
| 创建方式 | `create_agent()` | `StateGraph().compile()` |
| 调用方式 | `agent.ainvoke({"messages": [...]})` | `graph.ainvoke(input, config)` |
| 流程控制 | LLM ReAct 自由推理 | 确定性有向图路由 |
| 状态 | 仅 `messages` 列表 | `InquiryState` 含 10+ 结构化字段 |
| 多轮持久化 | `AsyncRedisSaver`（checkpointer） | 手动 Redis 存 `InquiryState` JSON（`inquiry_state:{thread_id}`） |
| store | `MilvusStore`（长期记忆） | 不直接用 store，通过 `patient_context.long_term_memories` 读取 |

---

## 九、Step 10：Inquiry Worker Agent（inquiry_agent.py）

### 9.1 Worker Agent 的角色

Worker Agent 是一个轻量级 ReAct Agent，职责：接收 `InquiryHandoffPayload`，帮患者确认挂号信息、生成问诊单摘要、完成预约挂号（当前工具待接入）。

```
InquiryGraph node_conclude → 构建 InquiryHandoffPayload（phase=HANDOFF）
    ↓ chat.py 路由层检测到 phase==HANDOFF
    ↓ handle_handoff(payload) 传入 InquiryHandoffPayload
Worker Agent（ReAct）
    ├── 读取 payload → 理解诊断结论
    ├── 确认挂号信息 + 生成问诊单摘要
    └── 返回挂号确认消息
```

> 注意：Worker Agent 的调用**不在 graph 节点里**，而是在 `chat.py` 路由层（或 `call_inquiry_agent` 工具）检测到 `phase == HANDOFF` 时触发 `handle_handoff`。

### 9.2 为什么用 ReAct 而不用 StateGraph

| 原因 | 说明 |
|------|------|
| 流程简单 | 只有「读 payload → 确认挂号」一步，不需要复杂路由 |
| 需要灵活性 | 挂号可能失败需重试或换科室，LLM 自主决策更合适 |
| 复用现有模式 | 与 Supervisor Agent 同一模式，降低维护成本 |

### 9.3 Worker Agent 代码（真实代码）

```python
# src/agents/workers/inquiry_agent.py
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from src.core.config import get_settings
from src.agents.inquiry.state import InquiryHandoffPayload
settings = get_settings()

INQUIRY_WORKER_PROMPT = """你是天宫医疗的挂号助手。
你已经收到了智能问诊的结论，现在需要帮助患者完成挂号预约。
问诊结论：
{handoff_payload}
你的职责：
1. 向患者确认挂号信息（科室、时间偏好）
2. 生成问诊单摘要（供医生参考）
3. 完成预约挂号（调用挂号工具，待接入）
请用温和、专业的语气与患者沟通。"""

def get_llm():
    return ChatDeepSeek(
        model=settings.CHAT_MODEL, api_key=settings.DEEPSEEK_API_KEY, temperature=0.3,
    )

def create_inquiry_worker_agent():
    llm = get_llm()
    # 当前阶段：无工具，纯 LLM 推理
    # 未来可添加：预约挂号工具、问诊单生成工具、排班查询工具
    tools = []
    return create_agent(model=llm, tools=tools)

_inquiry_worker_agent = None

def get_inquiry_worker_agent():
    global _inquiry_worker_agent
    if _inquiry_worker_agent is None:
        _inquiry_worker_agent = create_inquiry_worker_agent()
    return _inquiry_worker_agent

async def handle_handoff(payload: InquiryHandoffPayload) -> str:
    """接收问诊移交数据包，执行挂号流程，返回给用户的挂号确认消息。"""
    agent = get_inquiry_worker_agent()
    prompt = INQUIRY_WORKER_PROMPT.format(
        handoff_payload=payload.model_dump_json(indent=2, ensure_ascii=False)
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    return result["messages"][-1].content
```

关键标注：
- `create_agent` 【内置方法】：LangChain 的 Agent 工厂函数，与 Supervisor Agent 同一个方法
- `tools = []`：当前阶段**没有挂号工具**（与伪代码的 `call_registration` 不同，挂号工具待接入）
- `handle_handoff` 【自定义变量】：真正的对外入口，把 `InquiryHandoffPayload` 格式化成 prompt 喂给 Worker Agent
- **无 checkpointer**：Worker Agent 每次独立执行，不依赖多轮状态（与伪代码的 `AsyncRedisSaver` 不同）
- `model_dump_json(indent=2, ensure_ascii=False)` 【内置方法】：payload 序列化为格式化 JSON（保留中文）

### 9.4 Worker Agent 与 InquiryGraph 的数据流（真实代码）

```
时刻① node_conclude 构建 InquiryHandoffPayload（phase=HANDOFF）
    │
    ↓  graph 返回 new_state.phase == HANDOFF
时刻② chat.py 路由层检测到 HANDOFF → 调 handle_handoff(new_state.handoff_payload)
    │
    ↓  payload.model_dump_json(indent=2, ensure_ascii=False)
时刻③ handle_handoff 把 payload 格式化进 INQUIRY_WORKER_PROMPT
    │
    ↓  agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
时刻④ Worker Agent 读取 JSON → 生成挂号确认消息（当前无工具，纯 LLM 推理）
    │
    ↓  result["messages"][-1].content
时刻⑤ chat.py 拼接：reply + "\n\n---\n" + handoff_reply
    │
    ↓  redis.delete("inquiry_active:u:s", "inquiry_state:u:s")
时刻⑥ 问诊结束，Redis 活跃标记清除
```

---

## 十、Step 11：工具改造 + FastAPI 路由分流

### 10.1 改造前的调用链

```
用户请求 → FastAPI → 统一走 Supervisor Agent
    ↓
get_supervisor_agent() → Supervisor Agent.ainvoke(...)
    ↓
LLM ReAct → 可能调用 save_memory / search_memory → 返回回复
```

### 10.2 改造后的调用链（真实代码）

```
用户请求 → POST /api/v1/chat
    ↓
chat(req, db)                        # src/api/routers/chat.py
    ↓
redis.exists("inquiry_active:{thread_id}")    ← 问诊激活标志
    ├── YES → _run_inquiry_turn(message, thread_id, redis, db)
    │           ↓  redis.get("inquiry_state:...") → model_validate_json 恢复状态
    │           ↓  run_inquiry(existing_state) → InquiryGraph 续轮
    └── NO  → get_supervisor_agent() → Supervisor Agent.ainvoke(...)
                ↓
                Supervisor 识别问诊意图
                ↓ 调用 call_inquiry_agent 工具
                ↓
                run_inquiry(existing_state=None) → 启动问诊首轮
                ↓
                redis.set("inquiry_active:...", "1") + redis.set("inquiry_state:...", json)
```

### 10.3 Redis 路由分流标志（真实代码：`redis_cache.py` + `chat.py` 直接操作）

> 关键差异：真实代码**没有** `is_inquiry_active` / `set_inquiry_active` / `clear_inquiry_active` 三个函数，而是由 `chat.py` 直接用 `redis.exists` / `redis.set` / `redis.delete` 操作，键名是 `inquiry_active:{thread_id}` 和 `inquiry_state:{thread_id}`。

`src/infra/redis_cache.py` 只提供两个 Redis 客户端获取函数（真实代码）：

```python
import redis.asyncio as redis
from src.core.config import get_settings
settings = get_settings()

# 业务连接池（decode_responses=True，返回字符串）
redis_pool = redis.ConnectionPool(host=settings.REDIS_HOST, port=settings.REDIS_PORT,
    db=settings.REDIS_DB, password=settings.REDIS_PASSWORD or None,
    decode_responses=True, encoding="utf-8")
_redis_client = redis.Redis(connection_pool=redis_pool)

async def get_redis_client() -> redis.Redis:
    """FastAPI Depends 注入用，返回模块级客户端（连接池复用）。"""
    return _redis_client

# checkpointer 专用连接池（decode_responses=False，bytes 模式）
_checkpointer_pool = redis.ConnectionPool(host=settings.REDIS_HOST, port=settings.REDIS_PORT,
    db=settings.REDIS_DB, password=settings.REDIS_PASSWORD or None, decode_responses=False)
_checkpointer_client = redis.Redis(connection_pool=_checkpointer_pool)

def get_checkpointer_redis() -> redis.Redis:
    """供 RedisSaver(LangGraph checkpointer) 专用，bytes 模式，与业务客户端隔离。"""
    return _checkpointer_client
```

- 两个连接池隔离：`_redis_client`（业务用，`decode_responses=True`）vs `_checkpointer_client`（checkpointer 用，`decode_responses=False` 因为 RedisSaver 要 bytes）
- 问诊分流标志的实际键名（在 `chat.py` 里）：`inquiry_active:{thread_id}`（活跃标记）、`inquiry_state:{thread_id}`（状态快照）
- `exists` / `set` / `delete` 【内置方法】：redis-py 标准命令
- `ex=3600` 【内置方法】：设置过期时间（秒），防止异常中断后永久卡在问诊模式

### 10.4 call_inquiry_agent 工具（Supervisor 的工具，真实代码）

```python
# src/agents/tools/worker_tools.py
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from dataclasses import dataclass
from src.agents.inquiry.graph import run_inquiry, build_inquiry_deps
from src.agents.inquiry.state import InquiryPhase
from src.agents.workers.inquiry_agent import handle_handoff
from src.infra.redis_cache import get_checkpointer_redis

@dataclass
class UserContext:                       # 【自定义变量】用户上下文
    user_id: str
    session_id: str

@tool
async def call_inquiry_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    启动智慧问诊流程。
    适用场景：患者首次描述症状、询问挂哪个科室时。
    后续多轮对话由系统自动路由，无需再次调用此工具。
    Args:
        message: 患者描述的症状或问诊需求
    """
    session_id = runtime.context.session_id   # 【内置方法】从运行时上下文取值
    user_id = runtime.context.user_id
    deps = build_inquiry_deps()               # 【自定义变量】构建依赖容器
    reply, new_state = await run_inquiry(     # 【自定义变量】执行一轮问诊
        user_message=message,
        thread_id=f"{user_id}:{session_id}",
        deps=deps, existing_state=None,       # 首轮，无历史状态
        user_id=user_id,
    )
    # 首轮即收敛（精确症状/急症等）：触发挂号移交，不设活跃标记
    if new_state.phase == InquiryPhase.END:
        if new_state.handoff_payload:
            handoff_reply = await handle_handoff(new_state.handoff_payload)
            return f"{reply}\n\n---\n{handoff_reply}"
        return reply
    # 问诊未结束：保存状态 + 设置活跃标记，等待后续轮次
    redis = get_checkpointer_redis()
    state_key = f"inquiry_state:{user_id}:{session_id}"
    await redis.set(state_key, new_state.model_dump_json(), ex=3600)
    active_key = f"inquiry_active:{user_id}:{session_id}"
    await redis.set(active_key, "1", ex=3600)
    return reply
```

关键标注：
- `@tool` 【内置方法】：LangChain 装饰器，把函数注册为 Agent 可调用的工具
- `ToolRuntime[UserContext]` 【内置方法】：LangGraph 注入的运行时上下文，泛型指定 context 类型为 `UserContext`
- `runtime.context.session_id` 【内置方法】：通过 `context` 字段注入的运行时上下文取值（不是 `runtime.config`）
- `build_inquiry_deps()` / `run_inquiry()` 【自定义变量】：Step 9 的依赖构建和问诊入口函数
- `handle_handoff()` 【自定义变量】：Worker Agent 的挂号移交函数
- 此工具在 `worker_tools.py` 的 `WORKER_TOOLS` 列表中，被 Supervisor Agent 使用

### 10.5 挂号工具（真实代码：当前未实现）

> 关键差异：真实代码中**没有** `call_registration` 工具。Worker Agent 目前 `tools = []`（纯 LLM 推理），挂号工具是预留项。

```python
# src/agents/workers/inquiry_agent.py（真实代码）

def create_inquiry_worker_agent():
    llm = get_llm()
    # 当前阶段：无工具，纯 LLM 推理
    # 未来可添加：预约挂号工具、问诊单生成工具、排班查询工具
    tools = []
    return create_agent(model=llm, tools=tools)
```

- Worker Agent 的提示词 `INQUIRY_WORKER_PROMPT` 里明确写了"完成预约挂号（调用挂号工具，待接入）"
- 未来接入挂号时，需在 `tools = []` 里添加工具，并通过 `handle_handoff` 传给 Worker Agent

### 10.6 改造后的 chat 接口（真实代码）

```python
# src/api/routers/chat.py
def _make_keys(user_id: str, session_id: str) -> tuple[str, str]:
    """生成 Redis 键名。thread_id 与 Supervisor checkpointer 保持一致。"""
    thread_id = f"{user_id}:{session_id}"
    return f"inquiry_active:{thread_id}", f"inquiry_state:{thread_id}"

async def _run_inquiry_turn(message: str, thread_id: str, redis, db) -> str:
    """执行一轮问诊对话（路由层直接调用，绕过 Supervisor）。
    从 Redis 恢复状态 → 执行 InquiryGraph → 保存新状态 → 返回回复。"""
    active_key = f"inquiry_active:{thread_id}"
    state_key  = f"inquiry_state:{thread_id}"
    # 从 Redis 反序列化恢复上一轮状态
    raw = await redis.get(state_key)
    existing_state = InquiryState.model_validate_json(raw) if raw else None  # 【内置方法】
    deps = build_inquiry_deps(db_session=db)
    reply, new_state = await run_inquiry(
        user_message=message, session_id=thread_id,
        deps=deps, existing_state=existing_state,
    )
    # 问诊结束：清除 Redis 标记，触发挂号移交
    if new_state.phase in (InquiryPhase.HANDOFF, InquiryPhase.END):
        await redis.delete(active_key, state_key)
        if new_state.phase == InquiryPhase.HANDOFF and new_state.handoff_payload:
            handoff_reply = await handle_handoff(new_state.handoff_payload)
            return f"{reply}\n\n---\n{handoff_reply}"
        return reply
    # 问诊继续：更新 Redis 状态，重置 TTL
    await redis.set(state_key,  new_state.model_dump_json(), ex=3600)
    await redis.set(active_key, "1",                          ex=3600)
    return reply

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """非流式对话接口。路由层判断是否有活跃问诊。"""
    redis = get_checkpointer_redis()
    thread_id = f"{req.user_id}:{req.session_id}"
    active_key = f"inquiry_active:{thread_id}"
    # ── 问诊进行中：直接走 InquiryGraph ──
    if await redis.exists(active_key):
        reply = await _run_inquiry_turn(req.message, thread_id, redis, db)
        return ChatResponse(reply=reply, session_id=req.session_id)
    # ── 无活跃问诊：走 Supervisor ──
    agent = await get_supervisor_agent()
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
        context=UserContext(user_id=req.user_id, session_id=req.session_id),
    )
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, session_id=req.session_id)
```

数据流时刻表（真实代码）：

```
时刻① 用户发消息 → POST /api/v1/chat
    ↓
时刻② redis.exists("inquiry_active:{thread_id}")
    ├── YES → 时刻③a _run_inquiry_turn（直接走 InquiryGraph，绕过 Supervisor）
    │         ↓
    │    时刻④a redis.get("inquiry_state:{thread_id}") → model_validate_json 恢复状态
    │         ↓
    │    时刻⑤a run_inquiry(existing_state) → graph 续接 → 返回 (reply, new_state)
    │         ↓
    │    时刻⑥a phase in (HANDOFF, END)? → redis.delete 清除标记 + handle_handoff 挂号
    │
    └── NO → 时刻③b 走 Supervisor Agent（get_supervisor_agent）
              ↓
         时刻④b Supervisor ReAct → 识别症状意图 → 调用 call_inquiry_agent 工具
              ↓
         时刻⑤b call_inquiry_agent → run_inquiry(existing_state=None) → redis.set 活跃标记
              ↓
         时刻⑥b 返回首轮问诊回复（如澄清话术）
```

### 10.7 FastAPI 路由注册（真实代码）

```python
# src/api/routers/chat.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from src.infra.database import get_db

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])   # 前缀含 /chat

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    patient_id: str | None = None     # 患者 ID（问诊时用于加载病史）

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    inquiry_phase: str | None = None  # 当前问诊阶段（供前端判断）

# 非流式接口（见 10.6 的 chat 函数）
@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    ...

# 流式接口（SSE）
@router.post("/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    ...
```

```python
# src/main.py 中注册路由（真实代码）

from src.api.routers.chat import router as chat_router

def create_app() -> FastAPI:
    ...
    app.include_router(chat_router)   # prefix 已写在 router 里，无需重复
    ...
```

- `APIRouter(prefix="/api/v1/chat")` 【内置方法】：FastAPI 路由分组器，前缀写在定义处
- `ChatRequest` 【自定义变量】：含 `patient_id` 可选字段（问诊时用于加载患者病史）
- `Depends(get_db)` 【内置方法】：FastAPI 依赖注入 `AsyncSession`（**不是**简化为直接调用）
- `@router.post("")` 【内置方法】：空路径表示匹配 `/api/v1/chat`（非流式）；`/stream` 是流式接口

---

## 十一、完整架构总览

### 11.1 改造后的系统全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI 应用 (src/main.py)                    │
│                                                                      │
│  POST /api/v1/chat ──→ chat (chat.py)                                │
│                              │                                       │
│              ┌───────────────┴───────────────────┐                   │
│              │  Redis: inquiry_active:{thread_id} │                   │
│              │    EXISTS?                         │                   │
│              └──────┬──────────────────┬─────────┘                   │
│                     │ YES              │ NO                          │
│                     ↓                  ↓                             │
│          ┌────────────────────┐  ┌──────────────────────┐            │
│          │  _run_inquiry_turn │  │  Supervisor Agent    │            │
│          │   (InquiryGraph    │  │  (ReAct create_agent)│            │
│          │    StateGraph)     │  └──────────┬───────────┘            │
│          └────────┬───────────┘             │ tools:                 │
│                   │                         │  ├── save_memory       │
│        dispatcher→load_context→              │  ├── search_memory     │
│        check_emergency→extract_symptoms→     │  └── call_inquiry_agent│
│        query_neo4j→ask_symptoms→             │              │        │
│        parse_answer→conclude→save_record      │              │        │
│                   │                         │              │        │
│           HANDOFF 时 ↓                       │              │        │
│        handle_handoff → Worker Agent (ReAct) │  run_inquiry │        │
│        （挂号，tools=[] 待接入）              │  → redis.set │        │
│                   │                         │  active+state│        │
│                   └─────────────────────────┴──────────────┘        │
│                                                                      │
│  ┌──────────────────────── 基础设施层 ─────────────────────────┐     │
│  │                                                            │     │
│  │  PostgreSQL        Redis           Milvus        Neo4j      │     │
│  │  (患者/问诊)       (问诊状态       (长期记忆 +   (疾病知识  │     │
│  │                    +活跃标志)      症状索引)     图谱)     │     │
│  │                                                            │     │
│  │  src/infra/         src/infra/      src/infra/   src/infra/ │     │
│  │  database.py        redis_cache.py  milvus_store neo4j_     │     │
│  │                                      .py        client.py  │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 数据流完整闭环

```
步骤1 用户："我头疼发烧两天了"
    ↓
步骤2 FastAPI POST /api/v1/chat → chat()
    ↓
步骤3 Redis EXISTS inquiry_active:{thread_id} → NO（首次）
    ↓
步骤4 Supervisor Agent.ainvoke → LLM 识别问诊意图 → 调用 call_inquiry_agent
    ↓
步骤5 run_inquiry(existing_state=None) → redis.set(active+state) 激活问诊
    ↓
步骤6 InquiryGraph 首轮
    ├── dispatcher → load_context：加载患者病史
    ├── check_emergency：检查无急症
    ├── extract_symptoms：LLM 提取 ["头痛","发烧"] → Neo4j 匹配成功
    ├── query_neo4j：查到 3 个候选疾病（偏头痛/感冒/脑膜炎）
    ├── （合并）apply_context_weights：置信度 [0.45, 0.40, 0.20]
    ├── （合并）check_convergence：未收敛
    └── ask_symptoms：追问 "头痛是搏动性的还是胀痛？"
    ↓
步骤7 new_state.model_dump_json() 存 Redis → 返回 AI 回复
    ↓
─── 用户第二轮 ───
    ↓
步骤8 用户："胀痛，还有恶心"
    ↓
步骤9 Redis EXISTS inquiry_active:{thread_id} → YES
    ↓
步骤10 _run_inquiry_turn：model_validate_json 恢复状态 → graph 续轮
    ├── dispatcher（pending_ask_symptoms 非空）→ parse_answer
    ├── parse_answer：解析"胀痛+恶心" → confirmed += ["恶心"]
    ├── query_neo4j：重新计算 [0.75, 0.35, 0.15] → 偏头痛 Top1≥70% 收敛
    ├── conclude：生成诊断结论 + 构建 HandoffPayload
    └── save_record：保存问诊记录
    ↓
步骤11 phase = HANDOFF → redis.delete(active, state) + handle_handoff 挂号
    ↓
步骤12 返回最终回复："诊断为偏头痛（置信度75%），建议就诊神经内科..."
```

### 11.3 与现有代码的对照总结（真实代码）

| 代码文件 | 改造内容 |
|-------------|---------|
| `src/agents/tools/worker_tools.py` | 新增 `UserContext` + `call_inquiry_agent` 工具，加入 `WORKER_TOOLS` 列表 |
| `src/agents/supervisor_agent.py` | `tools` 引入 `WORKER_TOOLS`（含 `call_inquiry_agent`） |
| `src/infra/redis_cache.py` | 不改动（仅提供 `get_redis_client` / `get_checkpointer_redis`） |
| `src/api/routers/chat.py` | **新增**：路由层 Redis 分流 + `_run_inquiry_turn` 问诊续轮 |
| `src/main.py` | `create_app()` 中注册 `chat_router` |
| `src/agents/inquiry/state.py` | **新增**：`InquiryState`(BaseModel) + `InquiryPhase` + `CandidateDisease` + `PatientContext` + `InquiryHandoffPayload` |
| `src/agents/inquiry/symptom_normalizer.py` | **新增**：三层标准化 + 反向口语化 |
| `src/agents/inquiry/neo4j_queries.py` | **新增**：候选疾病查询 + 详情补充 + 待追问症状 |
| `src/agents/inquiry/confidence.py` | **新增**：`apply_context_weights` + `check_convergence` |
| `src/agents/inquiry/prompts.py` | **新增**：5 个提示词 |
| `src/agents/inquiry/db_queries.py` | **新增**：`load_patient_context` + `save_consultation_record` |
| `src/agents/inquiry/graph.py` | **新增**：`build_inquiry_graph` + `build_inquiry_deps` + `run_inquiry` |
| `src/agents/workers/inquiry_agent.py` | **新增**：`create_inquiry_worker_agent` + `handle_handoff` |
| `scripts/init_symptom_index.py` | **新增**：Milvus 症状向量索引初始化 |
| `scripts/init_neo4j.py` | 知识图谱初始化 |

### 11.4 核心设计决策回顾

| 决策 | 选择 | 理由 |
|------|------|------|
| 问诊流程控制 | StateGraph 而非 ReAct | 医疗场景需确定性路由，不可让 LLM 自由推理 |
| 多轮状态管理 | Pydantic BaseModel + 手动 Redis | 结构化字段（症状列表/候选疾病）比消息列表更精确；手动序列化全量状态 |
| 症状标准化 | 三层兜底 | LLM 提取有误差，Neo4j 精确匹配 + Milvus 语义兜底确保覆盖 |
| 收敛判断 | 置信度算法 + 阈值 | 比 LLM 自行判断更可量化、可调参 |
| 分流机制 | Redis 活跃标志位 | O(1) 判断，无需查数据库；TTL 防死锁 |
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
| `AsyncRedisSaver(redis_client)` | `langgraph.checkpoint.redis` | Redis Checkpoint（**仅 Supervisor Agent 用**，InquiryGraph 用手动 Redis） |
| `asetup()` | `AsyncRedisSaver` | 初始化 RediSearch 索引 |
| `model_dump_json(indent, ensure_ascii)` | `pydantic.BaseModel` | 序列化为 JSON |
| `model_validate_json(raw)` | `pydantic.BaseModel` | 从 JSON 反序列化（问诊状态恢复） |
| `aembed_documents(texts)` | `Embeddings` | 批量文本转向量（症状语义匹配用） |
| `BaseStore` | `langgraph.store.base` | Store 基类 |
| `aput(namespace, key, value)` | `BaseStore` | 写入记忆 |
| `asearch(namespace, query)` | `BaseStore` | 语义检索记忆 |
| `END` | `langgraph.graph` | 图终止常量 |

### 12.2 【自定义变量】速查

| 变量 | 文件 | 作用 |
|------|------|------|
| `InquiryState` | `state.py` | 问诊状态（Pydantic BaseModel） |
| `InquiryPhase` | `state.py` | 问诊阶段枚举 |
| `InquiryHandoffPayload` | `state.py` | 移交数据包 |
| `PatientContext` | `state.py` | 患者上下文 |
| `CandidateDisease` | `state.py` | 候选疾病模型 |
| `InquiryDeps` | `graph.py` | 依赖注入容器（llm/neo4j/embedding/milvus/db） |
| `normalize_symptoms()` | `symptom_normalizer.py` | 三层标准化总入口 |
| `extract_and_normalize_symptoms()` | `symptom_normalizer.py` | 第一层 LLM 提取 |
| `match_symptoms_in_neo4j()` | `symptom_normalizer.py` | 第二层 Neo4j 精确匹配 |
| `semantic_match_symptoms()` | `symptom_normalizer.py` | 第三层 Milvus 语义兜底 |
| `humanize_symptoms()` | `symptom_normalizer.py` | 标准术语转口语 |
| `query_candidate_diseases()` | `neo4j_queries.py` | 候选疾病反查 |
| `enrich_candidate_details()` | `neo4j_queries.py` | 补充疾病详情 |
| `get_pending_symptoms()` | `neo4j_queries.py` | 提取待追问症状 |
| `apply_context_weights()` | `confidence.py` | 置信度加权 |
| `check_convergence()` | `confidence.py` | 收敛判断（返回 `(should, force)`） |
| `build_inquiry_graph(deps)` | `graph.py` | 构建问诊图（闭包注入） |
| `build_inquiry_deps()` | `graph.py` | 构建依赖容器 |
| `run_inquiry()` | `graph.py` | 问诊入口（一轮） |
| `create_inquiry_worker_agent()` | `inquiry_agent.py` | 创建 Worker Agent（tools=[]） |
| `handle_handoff()` | `inquiry_agent.py` | 挂号移交入口 |
| `call_inquiry_agent` | `worker_tools.py` | 启动问诊工具 |
| `UserContext` | `worker_tools.py` | 工具运行时上下文（user_id/session_id） |
| `_run_inquiry_turn()` | `chat.py` | 路由层问诊续轮 |
