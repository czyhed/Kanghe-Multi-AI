# step9:InquiryGraph 主图构建（核心）
# 定义节点
from __future__ import annotations
import json
from typing import Any

from langgraph.types import GraphOutput
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END # ★ LangGraph（图/状态机的核心）
# 状态机/图的全部能力（节点、边、路由、循环）
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import result

from src.agents.inquiry.state import InquiryState
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.core.config import get_settings
from src.agents.inquiry.state import (
    InquiryState, InquiryPhase, InquiryHandoffPayload, PatientContext
)
from src.agents.inquiry.symptom_normalizer import (
    normalize_symptoms, humanize_symptoms
)
from src.agents.inquiry.neo4j_queries import (
    query_candidate_diseases, enrich_candidate_details, get_pending_symptoms
)
from src.agents.inquiry.confidence import apply_context_weights, check_convergence
from src.agents.inquiry.prompts import (
    CLARIFY_PROMPT, ASK_SYMPTOMS_PROMPT, PARSE_ANSWER_PROMPT,
    CONCLUSION_PROMPT, EMERGENCY_CHECK_PROMPT
)
settings = get_settings()

# ── 依赖注入容器（在 graph 编译时注入，避免全局单例） ──────────────────────
class InquiryDeps:
    def __init__(self, llm, neo4j_driver, embedding_model, milvus_client, db_session):
        self.llm = llm                  # 大语言模型：负责所有自然语言理解与生成
        self.neo4j_driver = neo4j_driver        # 图数据库连接（Neo4j 驱动程序）
        self.embedding_model = embedding_model  # 向量模型：把症状文本转成向量
        self.milvus_client = milvus_client      # 向量库客户端：存/查长期记忆与症状向量
        self.db_session = db_session            # 关系型数据库会话（PostgreSQL）

#════════════════════════════════════════════════════════════════════════
# 节点函数（每个节点接收 state，返回 state 的部分更新）
#════════════════════════════════════════════════════════════════════════

"""
节点1：加载患者上下文。
从 PostgreSQL 加载既往病史，从 Milvus 加载长期记忆。
仅在第一轮（round==0）执行，后续轮次跳过。
"""
async def node_load_context(state: InquiryState, deps: InquiryDeps) -> dict:

    if state.round > 0:
        logger.debug("节点①加载患者上下文 非首轮，跳过上下文加载 (round={})", state.round)
        return {}  # 非首轮，不重复加载
    logger.info("节点①加载患者上下文 开始加载患者上下文 | patient_id={} session_id={}",
                state.patient_context.patient_id, state.session_id)

    from src.agents.inquiry.db_queries import load_patient_context
    # user_id 即 patients.id，前端传来的是字符串，转 int 后查患者档案
    patient_id = int(state.patient_context.patient_id) if state.patient_context.patient_id else None
    # 从HIS查询患者信息和就诊记录
    patient_ctx = await load_patient_context(
        patient_id=patient_id,          # 患者ID（转 int 后的外键，查 patients.id）
        db=deps.db_session,             # 数据库会话（PostgreSQL 连接）
    )
    # 待完成？？？ 从Milvus加载长期记忆
    merged_ctx = PatientContext(
        patient_id=state.patient_context.patient_id,  # 保持 str，贯穿整个流程
        age=patient_ctx.age,
        gender=patient_ctx.gender,
        allergy_history=patient_ctx.allergy_history,
        medical_history=patient_ctx.medical_history,
        long_term_memories=state.patient_context.long_term_memories,
    )
    logger.info("节点①加载患者上下文 上下文加载完成 | age={} gender={} medical_history={} allergy={}",
                merged_ctx.age, merged_ctx.gender,
                len(merged_ctx.medical_history), len(merged_ctx.allergy_history))
    return {"patient_context": merged_ctx}  # 把"患者上下文"写回黑板字段，供后续节点读取

"""
节点2：急症识别。
仅在第一轮执行。识别到急症时直接跳转到 CONCLUDE 阶段，
并在 candidate_diseases 中放入一个特殊的"急诊"标记。
"""
async def node_check_emergency(state: InquiryState, deps: InquiryDeps) -> dict:

    if state.round > 0:
        logger.debug("节点②急症识别 非首轮，跳过急症检查")
        return {}
    logger.info("节点②急症识别 开始急症识别检查")
    last_user_msg = ""
    for msg in reversed(state.messages):  # 历史消息： 1-2-3-4-5-6
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    if not last_user_msg:
        return {}  # 没有用户消息（异常兜底），跳过急症判断
    prompt = EMERGENCY_CHECK_PROMPT.format(user_input=last_user_msg)  # 把用户症状填进"是否急症"判断模板
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])  # 调 LLM 判断是否急症
    try:
        content = response.content.strip()   # 去掉首尾空白
        if "```" in content:
            # LLM 爱用代码块包 JSON，剥掉 ```json ... ``` 只留内部纯 JSON
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)                     # 把 LLM 返回的 JSON 字符串解析成字典
        if result.get("is_emergency"):                   # 模型标记"这是急症"→ 走急症分支
            logger.warning("节点②急症识别 ⚠️ 检测到急症！原因: {}", result.get('reason', ''))
            emergency_reply = (
                # 拼一段"立即去急诊"的警告文案，直接展示给用户
                "⚠️ 根据您描述的症状，这可能是紧急情况！\n\n"
                f"原因：{result.get('reason', '存在急症风险')}\n\n"
                "**请立即前往最近医院的急诊科就诊，或拨打 120 急救电话。**\n\n"
                "不要等待，请立即行动！"
            )
            return {
                "phase": InquiryPhase.END,      # 急症 → 流程直接结束
                "messages": [AIMessage(content=emergency_reply)],  # 把"去急诊"警告文案发给用户
            }
        else:
            logger.info("节点②急症识别 未检测到急症，继续正常问诊流程")
    except Exception as e:
        logger.warning(f"急症识别解析失败: {e}")
    return {}
"""
节点3：症状标准化。
从最新的用户消息中提取症状，经三层标准化后合并到 confirmed_symptoms。
"""
async def node_extract_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:

    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    if not last_user_msg:
        logger.warning("节点③症状标准化 未找到用户消息，跳过症状提取")
        return {}
    logger.info("节点③症状标准化 开始提取症状 | 用户输入: {!r}", last_user_msg[:80])
    # 等待标准化三层管线执行
    result = await normalize_symptoms(  # 三层标准化：LLM提取 → 图谱归一 → 向量兜底
        user_input=last_user_msg,        # 用户原话（口语描述）
        llm=deps.llm,                    # 大模型：从口语里抽取症状词
        neo4j_driver=deps.neo4j_driver,  # 图谱：把口语症状映射到标准症状名
        embedding_model=deps.embedding_model,  # 向量模型：算相似度做兜底匹配
        milvus_client=deps.milvus_client,      # 向量库：存/查症状向量
    )
    new_confirmed = list(set(state.confirmed_symptoms) | set(result["all_standard"]))  # 已确认症状 ∪ 本次新提取的症状（去重）
    new_unmatched = list(set(state.unmatched_symptoms) | set(result["unmatched"]))      # 累计"图谱没识别出来"的症状（去重）
    logger.info("节点③症状标准化 提取完成 | 标准化症状={} 未匹配={} 累计确认={}",
                result["all_standard"], result["unmatched"], new_confirmed)
    if new_confirmed:
        # 成功提取到明确症状 → 进入❓查询候选疾病阶段
        return {
            "confirmed_symptoms": new_confirmed, # 写回 state 字段, state.confirmed_symptoms
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.GRAPH_QUERY, # 接下来进入哪个阶段
        }
    else:
        logger.info("节点③症状标准化 无法提取明确症状，进入澄清流程")
        return {
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.CLARIFY,
        }

"""
节点4：澄清模糊描述。
当用户描述没有明确症状时，引导用户进一步描述。
"""
async def node_clarify(state: InquiryState, deps: InquiryDeps) -> dict:

    last_user_msg = ""
    for msg in reversed(state.messages):   # 倒序遍历历史消息，找最近一条用户消息
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content    # 取到用户原话（口语描述）
            break
    logger.info("节点④澄清模糊描述 用户描述模糊，发起澄清引导 | round={}", state.round)
    # 把用户那句"太模糊的描述"套进澄清模板
    prompt = CLARIFY_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])  # 调 LLM 生成一句"再描述详细点"的引导回复
    logger.debug("节点④澄清模糊描述 澄清回复已生成，等待用户下一轮输入")
    return {
        "round": state.round + 1,
        "messages": [AIMessage(content=response.content)],
    }


"""
节点5：查询 Neo4j 候选疾病 = 嫌疑病。
candidates 候选病、嫌疑病
confirmed_symptoms   确认症状
用已确认症状查候选疾病，补充详情，应用上下文权重。
"""
async def node_query_neo4j(state: InquiryState, deps: InquiryDeps) -> dict:

    logger.info("节点⑤查询候选疾病 查询候选疾病 | 确认症状={}", state.confirmed_symptoms)

    # 此处调用neo4j_queries.py的函数
    candidates = await query_candidate_diseases(
        confirmed_symptoms=state.confirmed_symptoms, # ← 从 state.py 取，传给函数
        neo4j_driver=deps.neo4j_driver,  # 图数据库连接（查疾病知识图谱）
    )
    if candidates:
        # 图谱里查到了候选疾病 → 补充详情、计算置信度
        # 此处调用neo4j_queries.py的函数
        candidates = await enrich_candidate_details(candidates, deps.neo4j_driver)  # 补详情：症状列表/推荐科室/检查项
        #  此处加权（传入疾病、病人上下文、否认症状）
        ## apply_context_weights 内部要遍历每个候选病的 all_symptoms
        candidates = apply_context_weights(
            candidates, state.patient_context, state.denied_symptoms
        )  # 用患者背景 + 否认症状对候选疾病置信度重新加权
        logger.info("节点⑤查询候选疾病 候选疾病 top5: {}",
                    [(c.name, round(c.confidence, 3)) for c in candidates[:5]])
    else:
        logger.warning("节点⑤查询候选疾病 未找到候选疾病，将强制结束问诊")
    # 收敛判断（要么超10轮，要么有候选疾病，要么没有更多有用信息）
    should_conclude, force_conclude = check_convergence(candidates, state.round)
    logger.info("节点⑤查询候选疾病 收敛判断 | should_conclude={} force_conclude={} round={}",
                should_conclude, force_conclude, state.round)
    if should_conclude:
        # 达到收敛条件（超轮数 / 已有高置信候选 / 用户没再补充）→ 直接给结论
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": force_conclude,
        }
    elif not candidates:
        # 没有候选疾病（症状太罕见），直接结束
        return {
            "candidate_diseases": [],
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": True,
        }
    else:
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.SYMPTOM_CONFIRM,
        }


"""
节点6：生成追问话术。
从候选疾病中选出区分度最高的 ≤3 个症状，口语化后追问用户。
"""
async def node_ask_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:

    logger.info("节点⑥生成追问话术 生成追问 | round={} 已问症状={}", state.round, state.asked_symptoms)

    # 此处调用neo4j_queries.py的get_pending_symptoms
    pending = await get_pending_symptoms(  # 挑出"还没问过、又有区分度"的症状
        candidates=state.candidate_diseases,      # 当前候选疾病列表
        confirmed_symptoms=state.confirmed_symptoms,  # 已确认症状（予以排除）
        denied_symptoms=state.denied_symptoms,    # 已否认症状（予以排除）
        asked_symptoms=state.asked_symptoms,      # 已提问症状（避免重复问）
    )
    # 取区分度最高的前 3 个（出现次数最少的）
    top_symptoms = [s for s, _ in pending[:3]]
    if not top_symptoms:
        # 没有更多可问的症状，直接收敛
        logger.info("节点⑥生成追问话术 无更多可追问症状，直接进入结论")
        return {"phase": InquiryPhase.CONCLUDE}
    logger.info("节点⑥生成追问话术 本轮追问症状: {}", top_symptoms)

    # 标准术语 → 口语 此处调用symptom_normalizer.py
    human_symptoms = await humanize_symptoms(top_symptoms, deps.llm)
    # 把口语化的候选症状喂进"追问模板"，生成让用户多选的问题
    prompt = ASK_SYMPTOMS_PROMPT.format(
        symptoms_to_ask="\n".join(f"- {s}" for s in human_symptoms)  # 每个症状一行、前加"-"做成清单
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    return {
        "round": state.round + 1,
        "asked_symptoms": state.asked_symptoms + top_symptoms,
        "pending_ask_symptoms": top_symptoms,  # 记录本轮问了哪些，供下一节点解析
        "messages": [AIMessage(content=response.content)],
    }

"""
节点7：解析用户对追问的回答。
判断用户确认/否认了哪些症状，更新 confirmed_symptoms 和 denied_symptoms。
"""
async def node_parse_answer(state: InquiryState, deps: InquiryDeps) -> dict:

    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    if not last_user_msg or not state.pending_ask_symptoms:
        logger.debug("节点⑦解析用户回答 无用户消息或无待解析症状，跳过")
        return {}  # 没回答或本轮没待解析症状 → 跳过（保持原状）
    logger.info("节点⑦解析用户回答 解析用户回答 | 待解析症状={} 用户回答={!r}",
                state.pending_ask_symptoms, last_user_msg[:80])
    prompt = PARSE_ANSWER_PROMPT.format(
        asked_symptoms=", ".join(state.pending_ask_symptoms),  # 本轮到底问了哪几个症状（逗号分隔）
        user_answer=last_user_msg,                              # 用户原始回答文本
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])  # 调 LLM 解析回答 → 得到确认/否认哪些症状
    try:
        content = response.content.strip()   # 去掉首尾空白
        if "```" in content:
            # 剥掉 LLM 的 ```json ... ``` 代码块
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)  # 解析成字典：{confirmed:[...], denied:[...]}
        # 用户回答里新确认/新否认的症状，并集进已有清单（去重）
        new_confirmed = list(
            set(state.confirmed_symptoms) | set(parsed.get("confirmed", []))
        )
        new_denied = list(
            set(state.denied_symptoms) | set(parsed.get("denied", []))
        )
        logger.info("节点⑦解析用户回答 解析结果 | 新增确认={} 新增否认={}",
                    parsed.get("confirmed", []), parsed.get("denied", []))
    except Exception as e:
        logger.warning(f"解析用户回答失败: {e}")
        new_confirmed = state.confirmed_symptoms  # 解析失败 → 沿用原有已确认症状
        new_denied = state.denied_symptoms        # 解析失败 → 沿用原有已否认症状
    return {
        "confirmed_symptoms": new_confirmed,      # 更新后的已确认症状
        "denied_symptoms": new_denied,            # 更新后的已否认症状
        "pending_ask_symptoms": [],               # 清空待解析症状（本轮已处理完）
    }

"""
节点8：生成诊断结论。
整合所有信息，生成用户可读的结论文案，并构建 HandoffPayload。
"""
async def node_conclude(state: InquiryState, deps: InquiryDeps) -> dict:

    candidates = state.candidate_diseases
    if not candidates:
        logger.warning("节点⑧生成诊断结论 无候选疾病，返回无结果提示")
        no_result_msg = (
            "根据您描述的症状，我暂时无法在知识库中找到匹配的疾病。\n\n"
            "这可能是因为症状较为罕见，或者需要更多信息。\n"
            "建议您直接前往医院就诊，由医生进行面诊。"
        )
        return {
            "phase": InquiryPhase.END,
            "messages": [AIMessage(content=no_result_msg)],
        }
    top1 = candidates[0]                              # 置信度最高的那个 → 主诊断
    suspected = [c.name for c in candidates[1:5]]     # 第2~5名 → 作为"疑似疾病"列进结论
    logger.info("节点⑧生成诊断结论 生成诊断结论 | 主诊断={} 置信度={:.3f} 科室={} 疑似={}",
                top1.name, top1.confidence, top1.department, suspected)
    # 把"主诊断+疑似"等信息填进结论文案模板
    prompt = CONCLUSION_PROMPT.format(
        confirmed_symptoms="、".join(state.confirmed_symptoms) or "暂无",  # 已确认症状（顿号连接，空则写"暂无"）
        primary_disease=top1.name,                                          # 主诊断疾病名
        confidence=top1.confidence,                                         # 主诊断置信度（0~1）
        suspected_diseases="、".join(suspected) if suspected else "无",      # 疑似疾病（无则写"无"）
        department=top1.department or "综合内科",                           # 推荐科室（缺省给综合内科）
        checks="、".join(top1.checks[:5]) if top1.checks else "暂无",        # 建议检查项（无则写"暂无"）
        force_conclude=state.force_conclude,                                # 是否强制收敛（超轮/用户无补充）
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])  # 调 LLM 生成一段权威的结论文案
    # 构建移交数据包（挂号数据包）
    handoff = InquiryHandoffPayload(
        patient_id=state.patient_context.patient_id,        # 患者ID（可回查档案）
        patient_context=state.patient_context.model_dump(), # 患者上下文整体转dict，方便序列化存档
        confirmed_symptoms=state.confirmed_symptoms,        # 用户确认的症状清单
        denied_symptoms=state.denied_symptoms,              # 用户明确否认的症状清单
        unmatched_symptoms=state.unmatched_symptoms,        # 图谱没识别出来的症状
        primary_disease=top1.name,                          # 主诊断：置信度最高的疾病名
        primary_confidence=top1.confidence,                 # 主诊断置信度（0~1）
        suspected_diseases=suspected,                       # 疑似疾病列表（第2~5名）
        department=top1.department or "综合内科",           # 推荐挂号科室（缺省给综合内科）
        recommended_checks=top1.checks[:5],                 # 建议做的检查项（取前5个）
        total_rounds=state.round,                           # 累计问了几轮
        session_id=state.session_id,                        # 会话ID（关联Redis/PostgreSQL）
    )
    return {
        "phase": InquiryPhase.HANDOFF,    # 通知下一节点：开始保存就诊记录
        "handoff_payload": handoff,       # 把准备好的"挂号数据包"挂到黑板上
        "messages": [AIMessage(content=response.content)],  # 结论文案展示给用户
    }

"""
节点9：保存问诊记录到 PostgreSQL。
无论用户是否同意挂号都执行。
"""
async def node_save_record(state: InquiryState, deps: InquiryDeps) -> dict:

    if not state.handoff_payload:
        logger.warning("节点⑨保存问诊记录 无 handoff_payload，跳过保存")
        return {"phase": InquiryPhase.END}
    from src.agents.inquiry.db_queries import save_consultation_record
    payload = state.handoff_payload
    chief_complaint = "、".join(state.confirmed_symptoms[:5])  # 取前5个确认症状做"主诉"写入病历
    # user_id 即 patients.id，转 int 后写入外键
    patient_id = int(payload.patient_id) if payload.patient_id else None
    logger.info("节点⑨保存问诊记录 保存问诊记录 | patient_id={} diagnosis={} department={}",
                patient_id, payload.primary_disease, payload.department)
    # 把问诊结果写入 PostgreSQL（挂到病历下）
    await save_consultation_record(
        patient_id=patient_id,                 # 关联的患者ID（patients.id 外键）
        session_id=payload.session_id,         # 本次会话ID
        chief_complaint=chief_complaint,       # 主诉：前5个确认症状
        diagnosis=payload.primary_disease,     # 诊断结论：主诊断疾病名
        department_name=payload.department,    # 推荐就诊科室
        urgency_level="normal",                # 紧急程度（问诊流程默认普通）
        db=deps.db_session,                    # 数据库会话（事务连接）
    )
    logger.info("节点⑨保存问诊记录 问诊记录保存完成，流程结束")
    return {"phase": InquiryPhase.END}




# src/agents/inquiry/graph.py （续）

# ══════════════════════════════════════════════════════════════════════
# 路由函数（决定下一个节点）
#══════════════════════════════════════════════════════════════════════
# 分发员 - 决定本轮从哪个站点入场
# 站点的定义 return 后面都是站点名
def route_dispatcher(state: InquiryState) -> str:
    """入口分发：首轮走完整流程，后续轮次根据上一轮状态决定入口。"""
    if state.round == 0: # round 第几轮
        return "load_context" # ←首轮
    # 上一轮是追问症状（pending_ask_symptoms 非空）→ 解析用户回答

    if state.pending_ask_symptoms: # pending_ask_symptoms:本轮在问的症状
        return "parse_answer" # ← 追问中

    # 上一轮是澄清引导（pending_ask_symptoms 为空）→ 提取新症状
    return "extract_symptoms" # ← 否则


# 急症之后
def route_after_emergency(state: InquiryState) -> str:
    """急症检查后的路由：急症直接结束，否则提取症状。"""
    if state.phase == InquiryPhase.END:
        return END # ← 是急症
    return "extract_symptoms"  # 非急症 → 进入症状提取节点

# 提取症状之后
def route_after_extract(state: InquiryState) -> str:
    """症状提取后的路由：有症状查图谱，没症状澄清。"""
    if state.phase == InquiryPhase.GRAPH_QUERY:
        return "query_neo4j" # ← 有症状
    return "clarify" # 	← 没症状-澄清

# 查完图谱之后
def route_after_neo4j(state: InquiryState) -> str:
    """Neo4j 查询后的路由：收敛则结论，否则追问。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude" # ← 收敛
    return "ask_symptoms" # ← 没收敛 -追问

# route_after_neo4j 收敛的开关
# 这是把 confidence.py 的收敛判断结果和流程走向接起来的地方。

# 追问完之后
def route_after_ask(state: InquiryState) -> str:
    """追问后的路由：没有更多症状可问则直接结论。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude" # 无更多可问 - 给结论
    return END  # 等待用户回答（下一轮消息进来后从 parse_answer 继续）

# 解析完用户回答后
def route_after_parse(state: InquiryState) -> str:
    """解析回答后：重新查 Neo4j 更新候选疾病。"""
    return "query_neo4j"  # 解析完 → 带上新增症状回炉重查候选疾病

# 出完结论后
def route_after_conclude(state: InquiryState) -> str:
    """结论后：保存记录，结束。"""
    if state.phase == InquiryPhase.END:
        return END #
    return "save_record" # 保存记录

#════════════════════════════════════════════════════════════════════════
# 图的组装
#════════════════════════════════════════════════════════════════════════

def build_inquiry_graph(deps: InquiryDeps):
    """
    构建并编译问诊 StateGraph。
    deps 通过闭包注入到每个节点函数中。
    """
    # 用闭包把 （deps - 依赖）绑定到节点函数_load_context
    #  LangGraph 的硬性规定：节点工人函数_load_context只能有一个参数 state
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
    # 注册节点（dispatcher 是无操作的分发节点，仅用于路由），9个业务节点
    # graph.add_node("站名",节点工人函数)
    graph.add_node("dispatcher",       lambda state: {})
    graph.add_node("load_context",     _load_context) # 加载患者上下文（病史/记忆）
    graph.add_node("check_emergency",  _check_emergency) # 急症识别（"胸痛"直接让去急诊）
    graph.add_node("extract_symptoms", _extract_symptoms) # 症状三层标准化
    graph.add_node("clarify",          _clarify) # 澄清模糊描述（"我不舒服"→追问）
    graph.add_node("query_neo4j",      _query_neo4j) # 查 Neo4j 候选疾病
    graph.add_node("ask_symptoms",     _ask_symptoms)# 生成追问话术（"有没有咳嗽？"）
    graph.add_node("parse_answer",     _parse_answer)# 解析用户对追问的回答
    graph.add_node("conclude",         _conclude)# 生成诊断结论 + 构建移交数据包
    graph.add_node("save_record",      _save_record) # 保存问诊记录到 PostgreSQL

    # 入口：dispatcher 根据 round 分发到首轮流程或后续轮次流程
    graph.set_entry_point("dispatcher") # 设定起点站



    # 注册边（固定边）
    graph.add_edge("load_context", "check_emergency")
    graph.add_edge("clarify", END)          # 澄清后等待用户回复
    graph.add_edge("save_record", END) # 存完 → 结束

    # 注册条件边（路由）
    # add_conditional_edges(起点站, 路由函数, {返回的字符串: 要去哪个站})
    # 动态入口
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
    return graph
    # 把图"编译"成可执行对象。之后 run_inquiry 里 graph.ainvoke(state)（graph.py:547）就是让它从 dispatcher 一站一站跑。


# ════════════════════════════════════════════════════════════════════════
# 对外接口：供 call_inquiry_agent 工具调用
#════════════════════════════════════════════════════════════════════════

#  执行一轮的总开关
async def run_inquiry(
    user_message: str,          # 用户本轮说了什么
    thread_id: str,             # 会话id，关联 Redis/PostgreSQL
    deps: InquiryDeps,          # 工具包（上上节讲过）
    existing_state: InquiryState | None = None,  # 上一轮的黑板；首轮=None
    user_id: str | None = None, # 患者id
    long_term_memories: list[str] | None = None, # Supervisor从Milvus查的长期记忆
) -> tuple[str | list[str | Any], InquiryState | list[Any] | GraphOutput[Any] | Any]:  # 返回(回复文本, 新黑板)
    """
    执行一轮问诊对话。
    Args:
        user_message      : 用户本轮输入
        thread_id        : thread_id（用于关联 Redis 和 PostgreSQL 记录）
        deps              : 依赖注入容器
        existing_state    : 上一轮的状态（多轮对话时传入）
        user_id           : 来自 UserContext 的用户 ID，贯穿整个问诊流程
        long_term_memories: Supervisor 从 Milvus 检索到的长期记忆摘要
    Returns:
        (assistant_reply, new_state)
    """
    graph = build_inquiry_graph(deps) # build_inquiry_graph(deps) 现拼一张图（每次调用都重拼，因为要绑 deps）

    if existing_state is None:
        # 首轮：初始化状态
        state = InquiryState(
            session_id=thread_id,
            patient_context=PatientContext(
                patient_id=user_id,
                long_term_memories=long_term_memories or [],
            ),
        )
    else:
        state = existing_state

    # 追加用户消息
    state.messages.append(HumanMessage(content=user_message))#把用户本轮的话贴到黑板的消息区

    # 根据当前阶段决定从哪个节点继续
    # 首轮从 load_context 开始：后续轮次从parse_answer开始（解析上一轮追问的回答）
    if state.round == 0:
        entry = None # 使用 graph 默认入口 load_context
    else:
        entry = "parse_answer"


    logger.info("▶ run_inquiry 开始 | session={} round={} entry={} user_id={}",
                thread_id, state.round,
                "load_context" if state.round == 0 else "parse_answer", user_id)


    # 执行图（dispatcher 节点会根据 round 自动路由到正确的入口）
    config = {"configurable": {"thread_id": thread_id}}

    # ------------------------- 执行引擎 启动按钮 -----------------------
    if entry:
        graph.set_entry_point(entry)
        graph = graph.compile()
        result = await graph.ainvoke(state, config=config, input={"messages" : state.messages}) #  让整张图跑起来
    else:
        graph = graph.compile()
        result = await graph.ainvoke(state, config = config)

    new_state = InquiryState(**result) if isinstance(result, dict) else result

    if state.round == 0:
        entry = None
    else:
        entry = "parse_answer"






    # 取最后一条 AI 消息作为回复
    reply = ""
    for msg in reversed(new_state.messages):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break
    logger.info("◀ run_inquiry 完成 | session={} phase={} round={} reply_len={}",
                thread_id, new_state.phase, new_state.round, len(reply))
    return reply, new_state
    # reply = 分诊这轮生成的回复文本
    # new_state = 跑完这轮后的新黑板/新状态


"""
一个工厂函数：负责去把 5 样外部资源都创建/取出来，塞进一个 InquiryDeps 袋子返回。call_inquiry_agent 和 FastAPI 路由都用它，避免每处自己拼一遍（它有个默认参数 db_session=None，路由层可传自己的数据库会话进来）。
"""
def build_inquiry_deps(db_session=None) -> InquiryDeps:
    """
    构建问诊依赖注入容器的工厂函数。
    供 call_inquiry_agent 工具和 FastAPI 路由共用，避免重复代码。
    """
    # 大语言模型：负责问答、澄清、诊断的所有自然语言理解与生成
    llm = ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )
    embedding_model = DashScopeEmbeddings(  # 向量模型：把症状文本转成向量，用于 Milvus 相似度检索
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    neo4j_driver = get_neo4j_driver()  # 图数据库连接：用于查询疾病知识图谱
    get_milvus_client_alias()  # 确保连接已建立
    milvus_client = MilvusClient(  # 向量数据库客户端：存/查长期记忆与症状向量
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return InquiryDeps(
        llm=llm,
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        db_session=db_session,
    )