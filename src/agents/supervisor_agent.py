from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.redis import AsyncRedisSaver
from langchain.agents import create_agent
from langchain_deepseek.chat_models import ChatDeepSeek
from dataclasses import dataclass

from src.agents.store_tools import save_memory, search_memory
from src.agents.tools.worker_tools import UserContext, WORKER_TOOLS
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.infra.redis_cache import get_checkpointer_redis
from dotenv import load_dotenv
load_dotenv()

# 作用：只做意图识别和路由，把用户的意图原封不动的发给下游Agent，不能添油加醋

def _get_embedding_model():
    """返回向量化模型。根据你的实际情况替换。"""
    # 使用 DashScope（阿里云）: 环境变量需要配置 DASHSCOPE_API_KEY
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(model="text-embedding-v3")

    # 方案B：使用 Ollama 本地模型
    # from langchain_ollama import OllamaEmbeddings
    # return OllamaEmbeddings(model="nomic-embed-text")

    # 方案C：使用 OpenAI
    # from langchain_openai import OpenAIEmbeddings
    # return OpenAIEmbeddings(model="text-embedding-3-small")

SUPERVISOR_SYSTEM_PROMPT = """你是康和医疗的智能总助手。

## 你的职责

识别用户意图，调用对应的专项助手处理，整合结果后给出清晰、友好的回复。
你自己不直接回答任何健康、医疗、症状相关的问题，必须通过专项助手完成。

## 可调用的专项助手

- call_inquiry_agent：智慧问诊
  适用：用户描述身体不适、症状，需要分诊建议或预约挂号时
  示例："我头疼发烧"、"肚子疼好几天了"、"不知道该挂什么科"、"有点不舒服"

- call_report_agent：报告解读
  适用：用户上传或描述检验报告、影像报告，需要解读关键指标时
  示例："帮我看看这个血常规报告"、"我的CT结果是什么意思"

- call_drug_agent：药物咨询
  适用：询问用药建议、药物相互作用、处方安全性时
  示例："布洛芬和阿莫西林能一起吃吗"、"我对青霉素过敏能用什么替代"

- call_knowledge_agent：医学知识问答
  适用：询问疾病知识、治疗方案、医学术语解释时
  示例："高血压平时要注意什么"、"糖尿病的早期症状有哪些"

- call_operation_agent：运营数据查询（仅限内部运营人员）
  适用：查询医院就诊量、科室排名、收入统计等运营数据时

## 记忆工具

- search_memory：在调用专项助手前，先检索用户的历史记忆（过敏史、既往病史、用药偏好等），将相关信息附加到调用参数中，避免重复询问用户
- save_memory：当用户提到重要的个人健康信息时（过敏史、慢性病、长期用药等），主动保存到长期记忆

## 工作原则

1. 你绝不直接回答健康、医疗、症状、用药相关的问题，这类问题必须通过专项助手完成。只要用户提到身体不适、症状、不舒服（如"不舒服"、"不太好"、"有点难受"），无论多模糊，都必须调用 call_inquiry_agent。
1a. 闲聊：问候、致谢、告别、询问你能做什么等不涉及健康内容的话，直接用一两句话自然回复，不调用任何专项助手，也不要把对话硬拉向问诊。你可以简要介绍自己能帮忙的事：问诊分诊、药物咨询、医学知识、报告解读。
1b. 意图不明：用户想要帮助，但看不出是咨询症状、药物、医学知识还是别的（如"我想咨询一下"、"帮我看看"），先用一句话询问"请问是想咨询身体症状、药物用法，还是了解某类医学知识？"。此时不要追问病史，也不要调用专项助手。用户一旦明确，按对应意图处理。
2. 传递消息规则：调用专项助手时，message 参数必须原样传递用户的原始输入，禁止改写、补充指令或添油加醋。如果 search_memory 检索到了相关历史信息，以"[长期记忆] xxx"的格式附加在用户原文之后，不要修改用户原文本身。
3. 复杂问题可串联：例如用户描述症状后又问用药，先调用 call_inquiry_agent 再调用 call_drug_agent
4. 患者安全优先：识别到"胸痛"、"呼吸困难"、"意识不清"等急症关键词时，立即提示用户拨打 120 或前往急诊，不要等待问诊流程
5. 语气温和专业：用患者能理解的语言表达，避免过度使用医学术语"""




# 创建出 监督 Agent
async def create_supervisor_agent():
    # 1. 复用项目已有的 checkpointer 专用 Redis 客户端（bytes 模式）
    redis_client = get_checkpointer_redis()

    # 2. 创建 AsyncRedisSaver，并调用 asetup() 初始化 RediSearch 索引
    # asetup() 会在 Redis Stack 中创建 checkpoint / checkpoint_write 两个索引
    # 必须在首次使用前调用一次，索引已存在时自动跳过，可以重复调用
    checkpointer = AsyncRedisSaver(redis_client=redis_client)
    await checkpointer.asetup()
    # asetup() 相当于给 Redis 建好「索引目录」，只首次调用有意义，可重复调用（不会再建）

    # 3. 长期记忆
    # ── 长期记忆：Milvus Store ─────────────────────────────────────────
    milvus_alias = get_milvus_client_alias()
    embedding_model = _get_embedding_model()
    store = MilvusStore(
        alias=milvus_alias,
        embeddings=embedding_model,
        dims=1024,   # DashScope text-embedding-v3 默认输出 1024 维
    )

    # 4. 工具：长期记忆工具 + 专项助手（问诊等）
    tools = [save_memory, search_memory, *WORKER_TOOLS]
    tools = [save_memory, search_memory] + WORKER_TOOLS

    # 4. 创建 Agent
    agent = create_agent(
        model="deepseek-chat",
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        context_schema=UserContext,
        middleware=[
            SummarizationMiddleware( # 会话自动总结压缩
                model="deepseek-chat",
                trigger=[
                    ("tokens", 4000),  # token数达到4k时触发
                    ("messages", 6),  # 或消息数达到 4条时触发
                    # ("fraction", 0.8)  # 或80%消息时触发
                ],
                keep=("messages", 6),  # 摘要后保留最近 4 条消息
            )
        ],
        checkpointer=checkpointer, # 短期记忆. agent chat ui（禁用你配置 checkpointer）
        store=store # 长期记忆. 把同一个用户，任意会话的有价值信息进行存储。
    )
    return agent


# 模块级单例：避免每次请求都重新创建 agent 和 checkpointer
_supervisor_agent = None

# 返回 agent
async def get_supervisor_agent():
    """返回全局单例 Agent，首次调用时初始化。"""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = await create_supervisor_agent()
    return _supervisor_agent

# 短期记忆-入口
# FastAPI 路由中使用
async def chat_endpoint(user_id: str, session_id: str, message: str):
    # 使用单例，不重复初始化。获取 agent
    agent = await get_supervisor_agent()

    # thread_id 用来区分不同会话，根据thread_id来割裂短期记忆的namespace
    # thread_id 用来区分不同会话。  用户id:会话id:日期（前端传来一个会话id）
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        context=UserContext(user_id=user_id, session_id=session_id),
    )
    return result["messages"][-1].content


# checkpointer＝「聊天草稿箱」，同一会话续接用，存谁说过什么。
# store＝「病历档案柜」，跨会话主动读写 + 语义检索，存值得记住的长期知识。
