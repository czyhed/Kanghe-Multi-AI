"""
TruSession 连 trulens_eval 独立库；LiteLLM 用 deepseek-v4-flash 当判卷模型；启动 Dashboard
"""



from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings

settings = get_settings()


def get_trulens_session() -> TruSession:
    """
    初始化 TruLens 会话。
    评估结果存入 PostgreSQL，与业务数据隔离。
    """
    session = TruSession(
        database_url=f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
    )
    return session


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider。
    通过 LiteLLM 对接 DeepSeek (OpenAI 兼容 API)。
    """
    return LiteLLM(
        model_engine="openai/deepseek-v4-flash",
        completion_kwargs={
            "api_key": settings.DASHSCOPE_API_KEY,
            "api_base": settings.BASE_URL_CHAT,
        },
    )


def launch_dashboard(session: TruSession | None = None, port: int = 8501):
    """启动 TruLens Dashboard（Streamlit 界面）。"""
    from trulens.dashboard import run_dashboard
    s = session or get_trulens_session()
    run_dashboard(session=s, port=port)