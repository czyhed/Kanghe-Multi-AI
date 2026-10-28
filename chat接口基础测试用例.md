# chat 接口基础测试用例

> 被测对象：`POST /api/v1/chat`（非流式）与 `POST /api/v1/chat/stream`（SSE 流式）
> 测试重点：**首次提问后，各存储中间件（Redis / PostgreSQL / Milvus / Neo4j / 日志）是否正确写入数据**
> 环境：`http://127.0.0.1:8000`

---

## 一、系统调用链路与「数据入库点」地图

在写用例前，先明确一次 chat 请求会经过哪些存储组件、在什么时机写入什么数据。

```
POST /api/v1/chat
   │
   ├─ HTTP 中间件
   │    ├─ LoggingMiddleware  → 只写日志（方法/路径/耗时/状态码），不入库
   │    └─ CORSMiddleware    → 只做跨域，不入库
   │
   └─ 路由 chat()
        ├─ 读 Redis:  exists("inquiry_active:{user_id}:{session_id}")
        │
        ├─【分支A｜首次提问，无活跃标记】→ 走 Supervisor
        │    ├─ AsyncRedisSaver checkpointer ──► Redis 写入 checkpoint（短期记忆/消息历史）
        │    ├─ save_memory 工具 ──────────────► Milvus 写入长期记忆（LLM 决策触发）
        │    ├─ search_memory 工具 ────────────► Milvus 检索
        │    └─ call_inquiry_agent 工具 ───────► InquiryGraph 首轮（9 节点）
        │         ├─ load_context ────► PostgreSQL 读 patients/consultations（读，不写）
        │         ├─ extract_symptoms─► Neo4j 读 + Milvus 读 symptom_index（读，不写）
        │         ├─ query_neo4j ─────► Neo4j 读候选疾病（读，不写）
        │         └─ 若收敛 save_record ─► PostgreSQL 写 consultations 表
        │
        └─【分支B｜后续轮次，有活跃标记】→ 直接走 InquiryGraph
              ├─ 从 Redis 恢复 inquiry_state 状态
              └─ 收敛后 save_record ─► PostgreSQL 写 consultations
```

### 入库点总表

| 存储组件 | 写入时机 | 写入内容 | Key / 表 | 触发条件 |
|---|---|---|---|---|
| **Redis** | 每次走 Supervisor | checkpoint 会话历史 | `checkpoint:*:{user_id}:{session_id}` 等 | 分支A 必写 |
| **Redis** | 首轮问诊未收敛 | 问诊状态 JSON | `inquiry_state:{uid}:{sid}` | phase ≠ END/HANDOFF |
| **Redis** | 首轮问诊未收敛 | 活跃标记 `"1"` | `inquiry_active:{uid}:{sid}` | phase ≠ END/HANDOFF |
| **PostgreSQL** | 问诊收敛 | 问诊记录 | `consultations` 表 | CONCLUDE → save_record |
| **Milvus** | Supervisor 调 save_memory | 长期记忆 | namespace `(users, uid, memories)` | LLM 决策触发 |
| **Neo4j** | 无写入 | 疾病/症状图谱查询 | — | 只读 |
| **日志** | 每次请求 | 访问日志 | 日志文件/控制台 | LoggingMiddleware |

> 关键结论：**首次提问（未收敛）只写 Redis，不写 PostgreSQL；只有问诊收敛才会写 PostgreSQL consultations；Milvus 长期记忆是否写入取决于 LLM 是否判定用户透露了值得记住的健康信息。**

---

## 二、测试用例

### 用例 1：首次提问 —— 正常问诊链路（核心用例）

- **测试标题**：首次提问（无活跃标记）走 Supervisor + InquiryGraph 首轮，验证 Redis 短期记忆与问诊状态正确入库
- **前置条件**：
  1. 服务已启动，5 项依赖健康（`GET /health/deps` 返回 status=ok）
  2. 清空该会话的测试残留：`DEL inquiry_active:1:222333`、`DEL inquiry_state:1:222333`
  3. 准备请求体（用户描述背景：该用户有高血压心脏病史，但本次未在消息中提及）：

```json
{
  "user_id": "1",
  "session_id": "222333",
  "message": "我最近有点头疼，还有点发烧，肚子不舒服"
}
```

- **测试步骤**：
  1. 向 `POST /api/v1/chat` 发送上述 JSON（Header: `Content-Type: application/json`）
  2. 记录 HTTP 状态码与返回体 `{reply, session_id}`
  3. 查询 Redis 键：`inquiry_active:1:222333`、`inquiry_state:1:222333`
  4. 查询 Redis 中该 thread 的 checkpoint 键（如 `keys 'checkpoint*1:222333*'`）
  5. 查询 PostgreSQL `consultations` 表该 session 的记录
- **预期结果**：
  1. HTTP 返回 `200`，`session_id` 回显为 `"222333"`，`reply` 为非空文本（追问话术或诊断结论）
  2. Redis 存在 `inquiry_active:1:222333 = "1"`，TTL 约 3600 秒
  3. Redis 存在 `inquiry_state:1:222333`，值为合法 JSON，反序列化后 `phase` 字段为 `SYMPTOM_CONFIRM` 或 `CLARIFY`（未收敛），`round >= 1`
  4. Redis 存在该 thread 的 checkpoint 数据（Supervisor 短期记忆已写入）
  5. PostgreSQL `consultations` 表**不新增**记录（本轮未收敛，尚未保存病历）

---

### 用例 2：首次提问 —— 症状三层标准化是否正确（头痛/发烧/腹痛）

- **测试标题**：验证症状提取链路，症状「头疼/发烧/肚子不舒服」被标准化为「头痛/发热/腹痛」并可查询到候选疾病
- **前置条件**：同用例 1，Neo4j 中已存在 `头痛`、`发热`、`腹痛` 等标准症状节点及关联疾病
- **测试步骤**：
  1. 发送用例 1 相同请求
  2. 从 Redis `inquiry_state:1:222333` 取出状态 JSON，查看 `confirmed_symptoms` 字段
  3. 查看服务日志中「节点③症状标准化」的输出
- **预期结果**：
  1. `confirmed_symptoms` 包含 `头痛`、`发热`、`腹痛`（或语义等价的标准化术语）
  2. 日志出现 `标准化症状=['头痛','发热','腹痛']` 或类似内容，`未匹配` 列表为空（或仅含无法识别的口语词）
  3. 若首轮命中候选疾病，`phase` 为 `SYMPTOM_CONFIRM` 且 `candidate_diseases` 非空

---

### 用例 3：首次提问 —— 患者档案读取（load_context 从 PostgreSQL 读病史）

- **测试标题**：验证首轮 load_context 能按 user_id 读到患者既往病史（高血压/心脏病）
- **前置条件**：PostgreSQL `patients` 表存在 `id=1` 的患者，且 `medical_history` 字段含「高血压,心脏病」
- **测试步骤**：
  1. 发送用例 1 相同请求
  2. 查看服务日志「节点①加载患者上下文」输出
- **预期结果**：
  1. 日志显示 `medical_history` 长度 >= 1，内容含「高血压」「心脏病」
  2. 患者上下文被写入 state 并贯穿后续节点（置信度加权可用到既往病史）

---

### 用例 4：首次提问 —— 首轮即收敛则写入 PostgreSQL 病历

- **测试标题**：症状明确、置信度高导致首轮收敛时，`consultations` 表正确入库
- **前置条件**：使用足以让首轮收敛的明确症状描述（如单一典型症状），或预先构造高置信候选疾病数据
- **测试步骤**：
  1. 发送明确症状的 chat 请求（记下 session_id，如 `session_id="converge001"`）
  2. 请求完成后查询 Redis `inquiry_active:{uid}:{sid}`、`inquiry_state:{uid}:{sid}`
  3. 查询 PostgreSQL `consultations` 表 `WHERE session_id='converge001'`
- **预期结果**：
  1. 返回 200，reply 为诊断结论文案（含主诊断、推荐科室、建议检查）
  2. Redis 的 `inquiry_active` / `inquiry_state` 键**被删除**（问诊已结束）
  3. `consultations` 表**新增 1 条**记录，字段校验：
     - `chief_complaint` = 前 5 个确认症状拼接（如「头痛、发热、腹痛」）
     - `diagnosis` = 主诊断疾病名
     - `department_id` 对应推荐科室
     - `urgency_level` = `normal`
     - `session_id` = `converge001`

---

### 用例 5：急症场景 —— 立即结束，不入库病历

- **测试标题**：用户描述胸痛/呼吸困难等急症时，立即返回急诊提示并结束流程
- **前置条件**：无
- **测试步骤**：
  1. 发送 `message="我胸口剧烈疼痛，喘不上气"`（session_id 用全新值，如 `session_id="emer001"`）
  2. 查看返回 reply 与 Redis / PostgreSQL
- **预期结果**：
  1. 返回 200，reply 包含「急诊」「120」等急症警告文案
  2. Redis 不产生 `inquiry_active` / `inquiry_state` 键（急症直接 END，未设置活跃标记）
  3. PostgreSQL `consultations` 表**不新增**记录

---

### 用例 6：多轮追问 —— 活跃标记存在时直接走 InquiryGraph（分支B）

- **测试标题**：第二次提问（active_key 存在）绕过 Supervisor，从 Redis 恢复状态继续问诊
- **前置条件**：用例 1 已执行，`inquiry_active:1:222333` 存在
- **测试步骤**：
  1. 向 `POST /api/v1/chat` 再次发送该会话消息（对上一轮追问的回答，如 `"没有咳嗽，就是头晕"`）
  2. 查看返回 reply 与 Redis `inquiry_state:1:222333` 的最新值
- **预期结果**：
  1. 返回 200，reply 为下一轮追问或诊断结论
  2. 日志显示走的是 `_run_inquiry_turn`（绕过 Supervisor，不重复初始化 agent）
  3. Redis `inquiry_state` 被更新为新状态，`round` 递增、`asked_symptoms` 增长
  4. 若本轮收敛，`inquiry_active` / `inquiry_state` 被删除，`consultations` 表新增记录

---

### 用例 7：长期记忆 Milvus 入库（save_memory 工具触发）

- **测试标题**：用户透露慢性病史/过敏史时，Supervisor 调 save_memory 将长期记忆写入 Milvus
- **前置条件**：Milvus 可用，embedding 模型可用
- **测试步骤**：
  1. 发送含明确个人健康信息的消息，如 `message="我有高血压和心脏病，最近头疼发烧"`（新 session_id，如 `session_id="mem001"`）
  2. 查看服务日志是否有「执行：保存长期记忆」输出
  3. 查询 Milvus store，namespace `("users", "1", "memories")` 下是否有新文档
- **预期结果**：
  1. 日志出现 `🔧执行：保存长期记忆`（LLM 判定需要记住时）
  2. Milvus 中 namespace `users/1/memories` 存在新写入的记忆文档，`value.content` 含「高血压」「心脏病」
  3. 后续同 user_id 新会话问诊时，`search_memory` 能检索到该记忆（日志出现「执行：搜索长期记忆」）

> 说明：save_memory 是否触发由 Supervisor 的 LLM 决策，非 100% 确定，测试时用强提示词提高触发概率，或以日志为准。

---

### 用例 8：长期记忆检索（search_memory 工具 + 上下文加权）

- **测试标题**：已有长期记忆的用户再次问诊时，记忆被检索并注入，用于置信度加权
- **前置条件**：用例 7 已写入「高血压、心脏病」记忆
- **测试步骤**：
  1. 用同一 `user_id="1"` 但新 session_id 发送问诊消息
  2. 查看日志「搜索长期记忆」与「节点①加载患者上下文」输出
- **预期结果**：
  1. 日志出现 `🔧执行：搜索长期记忆`
  2. `long_term_memories` 非空，含相关记忆内容
  3. `apply_context_weights` 对命中疾病的置信度 +0.10（日志「候选疾病 top5」置信度有所反映）

---

### 用例 9：参数校验 —— 缺失必填字段

- **测试标题**：请求体缺少 `user_id` / `session_id` / `message` 时返回 422
- **前置条件**：无
- **测试步骤**：
  1. 分别发送缺少 `message`、缺少 `session_id`、缺少 `user_id` 的请求体
- **预期结果**：
  1. 均返回 HTTP `422`（FastAPI Pydantic 校验失败）
  2. 响应体为 FastAPI 标准错误结构，`detail` 含缺失字段说明
  3. 不触发任何存储写入（无 Redis / PostgreSQL 变更）

---

### 用例 10：空消息 / 纯空格消息

- **测试标题**：message 为空字符串或纯空格时系统行为
- **前置条件**：无
- **测试步骤**：
  1. 发送 `message=""` 与 `message="   "`
- **预期结果**：
  1. `message=""` 通过 Pydantic 校验（非空约束取决于模型定义），返回 200 或按业务返回异常提示，但不应 500
  2. 若进入问诊，症状提取结果为空，`phase` 进入 `CLARIFY`，reply 为澄清引导话术
  3. 全程无异常堆栈泄漏到响应体

---

### 用例 11：流式接口 SSE 返回格式

- **测试标题**：`/stream` 接口返回符合 SSE 规范的分段数据
- **前置条件**：无
- **测试步骤**：
  1. 向 `POST /api/v1/chat/stream` 发送与用例 1 相同的请求体
  2. 以流式方式读取响应，逐行检查
- **预期结果**：
  1. HTTP 返回 200，响应头 `Content-Type: text/event-stream`
  2. 数据帧格式为 `data: {"type":"token","content":"..."}`，可多条
  3. 结束帧为 `data: {"type":"done","session_id":"222333"}`
  4. 异常时推送 `data: {"type":"error","message":"..."}`
  5. 与用例 1 一致，Redis 状态键正确写入

---

### 用例 12：异常兜底 —— 依赖不可用时返回 500 而非崩溃

- **测试标题**：下游依赖（如 Neo4j/Milvus）不可用时，接口返回 500 且服务不中断
- **前置条件**：临时停用 Neo4j 或 Milvus（或修改连接配置指向错误地址）
- **测试步骤**：
  1. 发送正常问诊请求
  2. 观察返回与日志
- **预期结果**：
  1. 返回 HTTP `500`，`detail` 含异常堆栈（当前实现为 `traceback.format_exc()`）
  2. 服务进程不崩溃，后续请求仍可处理（`GET /health` 仍返回 200）

---

### 用例 13：日志中间件验证

- **测试标题**：LoggingMiddleware 正确记录每次请求的方法、路径、耗时、状态码
- **前置条件**：无
- **测试步骤**：
  1. 发送一次 chat 请求
  2. 查看应用日志
- **预期结果**：
  1. 日志出现 `--> POST /api/v1/chat`
  2. 日志出现 `<-- POST /api/v1/chat status=200 xx.xxms`（含耗时与状态码）
  3. 请求异常时状态码记录为对应错误码

---

## 三、执行顺序建议

```
1. 先跑用例 9/10/13（参数校验 + 日志，不依赖业务数据）
2. 再跑用例 1/2/3（首次提问核心链路，验证 Redis 入库 + 症状标准化 + 患者档案读取）
3. 用例 6（紧接用例 1，验证多轮续接）
4. 用例 5（急症，验证不入库）
5. 用例 4（收敛入库 PostgreSQL）
6. 用例 7/8（长期记忆 Milvus 写入与检索）
7. 用例 11（流式接口）
8. 用例 12（异常兜底）
```

## 四、数据入库断言速查（测试后人工/脚本核对用）

```bash
# Redis（需用 checkpointer 同款 bytes 客户端，或 redis-cli）
redis-cli EXISTS "inquiry_active:1:222333"        # 期望 1（未收敛时）
redis-cli TTL    "inquiry_active:1:222333"        # 期望 ≈3600
redis-cli GET    "inquiry_state:1:222333"         # 期望 JSON，含 phase/round/confirmed_symptoms
redis-cli KEYS   "checkpoint*1:222333*"           # 期望非空（短期记忆）

# PostgreSQL
SELECT * FROM consultations WHERE session_id='222333';  # 未收敛时 0 行；收敛后 1 行

# Milvus（长期记忆）
# namespace = ("users", "1", "memories") 下应有新文档

# Neo4j
# 只读查询，不产生写入；可观察慢查询日志确认查询发生
```
