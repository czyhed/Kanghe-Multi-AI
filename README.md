# 康和会诊 Kanghe MDT

面向医疗场景的多 Agent 协作系统。Supervisor 调度问诊、知识库问答、处方审核和 ChatBI 数据分析等 Agent 协同完成任务,并结合 RAG、知识图谱与 NL2SQL 提供可追溯的回答。

A multi-agent system for healthcare, where a supervisor coordinates specialized agents for patient inquiry, knowledge Q&A, prescription review, and ChatBI analytics. It combines RAG, a knowledge graph, and NL2SQL to deliver traceable answers.

> 仅用于学习与技术演示,输出不构成医疗建议。

## 架构

每轮消息先经过**急症预检**(`src/agents/safety.py`,确定性关键词规则,支持否定句,不依赖 LLM),命中则暂停问诊并直接给出就医指引。通过后由 Supervisor 做意图识别、路由和记忆读写:闲聊直接回复,意图不明先询问,医疗问题不直接回答,交给 5 个 Worker 处理。对话入口的设计思路见 [docs/chitchat-flow.png](docs/chitchat-flow.png)(概念设计,只落地了其中的急症预检、闲聊和意图确认)。

| Agent | 状态 | 说明 |
|-------|------|------|
| Inquiry 智慧问诊 | 可用 | LangGraph 多轮问诊:症状归一化 → Neo4j 候选疾病 → 置信度加权 → 收敛后移交挂号 |
| Knowledge 知识问答 | 可用 | 混合检索(向量 + 图谱)、HyDE、查询改写、重排序、幻觉检查 |
| Drug 药物咨询 | 可用 | 药物关系与处方审核,基于图谱与向量检索 |
| Operation 运营数据 | 可用 | NL2SQL 查询统计数据,含 SQL 安全校验与角色过滤 |
| Report 报告解读 | 占位 | 目前只有提示词,没有接入文件解析或视觉模型 |

存储分工:PostgreSQL(结构化数据)、Redis Stack(会话短期记忆与 LangGraph checkpoint)、Milvus(向量与长期记忆)、Neo4j(疾病-症状-药物图谱)、MinIO(知识文档)。

完整设计与流程图见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

需要 Python 3.13、Docker。

```sh
# 1. 配置环境变量,填入 DASHSCOPE_API_KEY、DEEPSEEK_API_KEY 和 JWT_SECRET_KEY
cp .env.example .env

# 2. 启动基础设施 (PostgreSQL / Redis / Milvus / Neo4j / MinIO)
docker compose up -d

# 3. 安装依赖并迁移数据库
pip install -r requirements.txt
alembic upgrade head

# 4. 导入示例数据(可选,按需选择)
python scripts/init_postgres.py
python scripts/init_neo4j.py
python scripts/init_symptom_index.py

# 5. 启动后端
uvicorn src.main:app --port 8000 --reload
```

健康检查:`GET /health`,依赖状态:`GET /health/deps`。接口文档:<http://localhost:8000/docs>。

### 演示页面(Gradio)

```sh
python gradio_rag_test.py     # RAG 测试页 + TruLens 评估,端口 7860
python -m src.nl2sql.app      # ChatBI 数据面板,端口 7861
```

## 鉴权与权限控制

JWT 登录,角色存在 `users` 表。`/health` 和 `/api/v1/auth/*` 以外的接口都要带 `Authorization: Bearer <token>`,缺失或无效返回 401,角色不够返回 403。

| 角色 | 权限 |
|------|------|
| `patient` | 对话(问诊、知识问答、药物咨询)、查看文档、提交反馈 |
| `doctor` | 患者权限 + 上传、删除知识库文档 |
| `operator` | 患者权限 + 通过对话查询运营统计数据 |
| `admin` | 全部权限,含反馈统计 |

设计要点:
- **角色以数据库为准**:token 里的 `role` 不参与授权判断,账号被禁用或改角色后立即生效。
- **注册只能创建 patient**:注册接口不接受 `role` 字段。其他角色由管理员用脚本创建。
- **`user_id` 取自 token**:对话和反馈接口不再接收请求体里的 `user_id`,避免冒用他人身份和会话。
- **Agent 工具层二次校验**:运营数据工具会检查调用者角色,即使 LLM 被提示词注入诱导调用,非运营角色也拿不到数据。

```sh
# 生成密钥写入 .env 的 JWT_SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# 创建医生 / 运营 / 管理员账号(密码交互输入)
python scripts/create_user.py admin01 admin
```

## 主要接口

| 接口 | 权限 | 说明 |
|------|------|------|
| `POST /api/v1/auth/register` | 公开 | 注册患者账号 |
| `POST /api/v1/auth/login` | 公开 | 登录,返回 token |
| `GET /api/v1/auth/me` | 登录 | 当前用户信息 |
| `POST /api/v1/chat` | 登录 | 对话(非流式) |
| `POST /api/v1/chat/stream` | 登录 | 对话(SSE 流式) |
| `POST /api/v1/knowledge/upload` | doctor | 上传知识文档并入库 |
| `GET /api/v1/knowledge/docs` | 登录 | 文档列表 |
| `POST /api/v1/knowledge/feedback` | 登录 | 提交回答反馈 |
| `GET /api/v1/knowledge/feedback/stats` | admin | 反馈统计 |

## 测试

```sh
pytest test/test_confidence.py test/test_sql_security.py test/test_auth.py test/test_safety.py
```

这几个文件不依赖外部服务,GitHub Actions 里也跑它们。

`test_supervisor_agent.py` 依赖真实的 LLM 与 Redis,需要先完成上面的启动步骤。

## 技术栈

Python 3.13 · FastAPI · LangChain / LangGraph · SQLAlchemy 2.0 + Alembic · DeepSeek · DashScope Embedding · TruLens · Gradio

## 已知限制与路线图

- 报告解读 Agent 尚未实现文件解析与视觉模型接入。
- 挂号为模拟:由 LLM 生成预约信息,未对接 HIS,也没有落库。
- 鉴权只做了角色级控制。医生按科室过滤数据、患者只能访问自己的档案,还没有实现;没有刷新 token 和登出(token 无法主动吊销)。
- 登录接口没有限流,没有防暴力破解。
- 没有独立前端工程,目前只有 Gradio 演示页。
- 默认的 `medical123` 等密码只用于本地 docker-compose,部署时必须替换。

## License

Apache-2.0
