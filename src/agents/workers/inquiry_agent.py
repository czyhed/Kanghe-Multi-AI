# step10:Inquiry Worker Agent（挂号移交）

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
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
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
    """
    接收问诊移交数据包，执行挂号流程。
    返回给用户的挂号确认消息。
    """
    agent = get_inquiry_worker_agent()
    prompt = INQUIRY_WORKER_PROMPT.format(
        handoff_payload=payload.model_dump_json(indent=2, ensure_ascii=False)
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    return result["messages"][-1].content