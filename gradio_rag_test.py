# gradio_rag_test.py
"""
Gradio 界面 - RAG 功能测试 + TruLens 在线评估追踪
运行后每次查询自动记录到 TruLens，可在 Dashboard 查看 RAG Triad 评分
"""
import asyncio
import os
import threading
import tempfile

import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
from pymilvus import MilvusClient

from trulens.apps.app import TruApp

from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.infra.database import AsyncSessionLocal
from src.agents.knowledge.doc_ingestion import ingest_file

# 评估模块
from src.agents.knowledge.evaluation.tracked_knowledge import TrackedKnowledge
from src.agents.knowledge.evaluation.trulens_config import (
    get_trulens_session, get_llm_provider, launch_dashboard,
)
from src.agents.knowledge.evaluation.metrics import build_all_metrics

load_dotenv()
settings = get_settings()

# 全局开关：是否启用 TruLens 追踪
ENABLE_TRULENS = os.getenv("ENABLE_TRULENS", "true").lower() == "true"


# ── 依赖工厂 ──────────────────────────────────────────────────────────

def _get_llm():
    return ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )


def _get_embedding_model():
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


def _get_milvus_client():
    get_milvus_client_alias()
    return MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")


# ── TrackedKnowledge + TruApp 单例 ────────────────────────────────────

_tracked_apps: dict[str, tuple[TrackedKnowledge, TruApp]] = {}


def _get_tracked_app(channel: str, db_session=None) -> tuple[TrackedKnowledge, TruApp]:
    """
    获取指定通路的 TrackedKnowledge + TruApp 单例。
    每个 channel 对应一个 app_version，方便 Dashboard 对比。
    """
    if channel in _tracked_apps:
        return _tracked_apps[channel]

    llm = _get_llm()
    embedding_model = _get_embedding_model()
    milvus_client = _get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    tracked = TrackedKnowledge(
        channel=channel,
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        neo4j_driver=neo4j_driver,
        db_session=db_session,
    )

    # 初始化 TruLens
    _ = get_trulens_session()  # 确保 session 初始化（TruApp 会自动用全局 session）
    provider = get_llm_provider()
    metrics = build_all_metrics(provider)

    tru_app = TruApp(
        tracked,
        app_name="knowledge_agent",
        app_version=channel,
        feedbacks=metrics,
    )

    _tracked_apps[channel] = (tracked, tru_app)
    return tracked, tru_app


async def _tracked_query(channel: str, question: str) -> str:
    """
    用 TruApp 包装执行一次查询，自动记录 trace + 评估。
    开启追踪模式时用 with tru_app，否则直接调用 tracked.query。
    """
    async with AsyncSessionLocal() as db:
        tracked, tru_app = _get_tracked_app(channel, db_session=db)

        if ENABLE_TRULENS:
            with tru_app as recording:
                answer = await tracked.query(question)
        else:
            answer = await tracked.query(question)

    return answer


# ── Doc RAG ──────────────────────────────────────────────────────────

async def _doc_rag(question: str, doc_type: str, use_hyde: bool, role: str):
    if not question.strip():
        return "请输入问题", ""

    # 通过 TruApp 追踪执行
    answer = await _tracked_query("doc_rag", question)

    debug_info = (
        f"通路: doc_rag\n"
        f"TruLens 追踪: {'启用' if ENABLE_TRULENS else '禁用'}\n"
        f"评估结果异步计算中，请到 Dashboard 查看"
    )
    return answer, debug_info


def doc_rag_handler(question, doc_type, use_hyde, role):
    return asyncio.run(_doc_rag(question, doc_type, use_hyde, role))


# ── Graph RAG ────────────────────────────────────────────────────────

async def _graph_rag(question: str, role: str):
    if not question.strip():
        return "请输入问题", ""

    answer = await _tracked_query("graph_rag", question)
    debug_info = f"通路: graph_rag\nTruLens 追踪: {'启用' if ENABLE_TRULENS else '禁用'}"
    return answer, debug_info


def graph_rag_handler(question, role):
    return asyncio.run(_graph_rag(question, role))


# ── 融合检索 ─────────────────────────────────────────────────────────

async def _multi_rag(question: str, use_doc: bool, use_graph: bool, role: str):
    if not question.strip():
        return "请输入问题", ""

    answer = await _tracked_query("fusion", question)
    debug_info = f"通路: fusion\nTruLens 追踪: {'启用' if ENABLE_TRULENS else '禁用'}"
    return answer, debug_info


def multi_rag_handler(question, use_doc, use_graph, role):
    return asyncio.run(_multi_rag(question, use_doc, use_graph, role))


# ── 文档上传（不追踪）────────────────────────────────────────────────

async def _upload_doc(file_path: str, doc_type: str, category: str):
    if not file_path:
        return "请上传文件"

    embedding_model = _get_embedding_model()
    milvus_client = _get_milvus_client()
    doc_name = os.path.basename(file_path)

    chunk_count = await ingest_file(
        file_path=file_path,
        doc_name=doc_name,
        doc_type=doc_type,
        category=category,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
    )
    return f"文档 '{doc_name}' 导入成功，共 {chunk_count} 个分块"


def upload_handler(file, doc_type, category):
    if file is None:
        return "请上传文件"
    return asyncio.run(_upload_doc(file.name, doc_type, category))


# ── Dashboard 启动（后台线程）────────────────────────────────────────

_dashboard_started = False


def start_dashboard_handler():
    """在后台线程启动 TruLens Dashboard"""
    global _dashboard_started
    if _dashboard_started:
        return "Dashboard 已在 http://localhost:8501 运行"

    def _run():
        try:
            launch_dashboard(port=8501)
        except Exception as e:
            print(f"Dashboard 启动失败: {e}")

    threading.Thread(target=_run, daemon=True).start()
    _dashboard_started = True
    return "Dashboard 已启动：http://localhost:8501"


# ── Gradio UI ────────────────────────────────────────────────────────

with gr.Blocks(title="康和医疗 - RAG 测试 + TruLens") as demo:
    gr.Markdown("# 康和医疗 - RAG 功能测试 + TruLens 在线评估")
    gr.Markdown(
        f"**TruLens 追踪状态**: {'✅ 启用' if ENABLE_TRULENS else '❌ 禁用'} "
        f"| 设置环境变量 `ENABLE_TRULENS=false` 可关闭"
    )

    with gr.Row():
        dashboard_btn = gr.Button("🚀 启动 TruLens Dashboard", variant="secondary")
        dashboard_status = gr.Textbox(label="Dashboard 状态", interactive=False)
    dashboard_btn.click(start_dashboard_handler, outputs=[dashboard_status])

    with gr.Tab("文档检索 (Doc RAG)"):
        with gr.Row():
            with gr.Column():
                doc_question = gr.Textbox(label="问题", placeholder="例：阿莫西林的禁忌症？", lines=2)
                doc_type_input = gr.Dropdown(
                    choices=["", "guideline", "drug_instruction", "sop", "literature"],
                    value="", label="文档类型过滤（可选）"
                )
                doc_hyde = gr.Checkbox(label="启用 HyDE 增强", value=True)
                doc_role = gr.Dropdown(choices=["patient", "doctor", "pharmacist"], value="patient", label="角色")
                doc_btn = gr.Button("检索", variant="primary")
            with gr.Column():
                doc_answer = gr.Textbox(label="回答", lines=10)
                doc_debug = gr.Textbox(label="追踪信息", lines=4)
        doc_btn.click(doc_rag_handler, [doc_question, doc_type_input, doc_hyde, doc_role], [doc_answer, doc_debug])

    with gr.Tab("知识图谱检索 (Graph RAG)"):
        with gr.Row():
            with gr.Column():
                graph_question = gr.Textbox(label="问题", placeholder="例：糖尿病的常用药？", lines=2)
                graph_role = gr.Dropdown(choices=["patient", "doctor", "pharmacist"], value="patient", label="角色")
                graph_btn = gr.Button("检索", variant="primary")
            with gr.Column():
                graph_answer = gr.Textbox(label="回答", lines=10)
                graph_debug = gr.Textbox(label="追踪信息", lines=4)
        graph_btn.click(graph_rag_handler, [graph_question, graph_role], [graph_answer, graph_debug])

    with gr.Tab("融合检索 (Fusion)"):
        with gr.Row():
            with gr.Column():
                multi_question = gr.Textbox(label="问题", placeholder="例：高血压合并糖尿病的用药方案", lines=2)
                with gr.Row():
                    multi_doc = gr.Checkbox(label="文档通道", value=True)
                    multi_graph = gr.Checkbox(label="图谱通道", value=True)
                multi_role = gr.Dropdown(choices=["patient", "doctor", "pharmacist"], value="patient", label="角色")
                multi_btn = gr.Button("检索", variant="primary")
            with gr.Column():
                multi_answer = gr.Textbox(label="回答", lines=10)
                multi_debug = gr.Textbox(label="追踪信息", lines=4)
        multi_btn.click(multi_rag_handler, [multi_question, multi_doc, multi_graph, multi_role], [multi_answer, multi_debug])

    with gr.Tab("文档上传"):
        with gr.Row():
            with gr.Column():
                upload_file = gr.File(label="上传文档（PDF/Word/TXT/MD）", file_types=[".pdf", ".docx", ".doc", ".txt", ".md"])
                upload_type = gr.Dropdown(
                    choices=["guideline", "drug_instruction", "sop", "literature"],
                    value="guideline", label="文档类型"
                )
                upload_category = gr.Textbox(label="所属分类", value="通用")
                upload_btn = gr.Button("上传并导入", variant="primary")
            with gr.Column():
                upload_result = gr.Textbox(label="导入结果", lines=3)
        upload_btn.click(upload_handler, [upload_file, upload_type, upload_category], [upload_result])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)