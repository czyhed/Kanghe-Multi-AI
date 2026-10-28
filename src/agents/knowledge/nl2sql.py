# src/agents/knowledge/nl2sql.py

from __future__ import annotations
import asyncio
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.agents.knowledge.prompts import NL2SQL_PROMPT, SQL_QA_PROMPT
from src.nl2sql.security import validate_sql as _validate_sql

MAX_SQL_RETRIES = 2
SQL_TIMEOUT_SECONDS = 10

async def _generate_sql(
    question: str, llm: BaseChatModel, error_hint: str = "",
) -> str:
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 SQL 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2SQL_PROMPT.format(question=question) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    sql = response.content.strip()
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql

async def search_sql_raw(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
)-> tuple[list[dict] | str, str]:
    """NL2SQL：LLM 生成 SQL → 安全校验 → 执行（带重试）。

    返回 (data_or_error, final_sql)：
      - data_or_error：查询结果 list[dict]，或错误提示字符串
      - final_sql：最终校验通过并执行的 SQL（失败时为空字符串）
    """
    error_hint = ""
    final_sql = ""
    for attempt in range(MAX_SQL_RETRIES + 1):
        raw_sql = await _generate_sql(question, llm, error_hint)
        logger.info(f"NL2SQL SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated = _validate_sql(raw_sql)
        if not valid:
            logger.warning(f"SQL 安全校验失败: {validated}")
            return f"查询被安全策略拦截：{validated}。请换一种方式提问。", ""

        try:
            query_result = await asyncio.wait_for(
                db.execute(text(validated)),
                timeout=SQL_TIMEOUT_SECONDS,
            )
            rows = query_result.mappings().all()
            data = [dict(row) for row in rows[:100]]
            final_sql = validated
            break
        except asyncio.TimeoutError:
            logger.warning(f"SQL 执行超时 ({SQL_TIMEOUT_SECONDS}s): {validated}")
            return f"查询执行超时（{SQL_TIMEOUT_SECONDS}秒），请简化查询条件后重试。", ""
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_SQL_RETRIES:
                return "数据查询执行失败，请尝试换一种方式提问。", ""
    else:
        return "数据查询执行失败，请尝试换一种方式提问。", ""

    if not data:
        return "未查询到相关数据。",final_sql

    return data , final_sql


async def search_sql(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> str:
    data, final_sql = await search_sql_raw(question, llm, db)
    result_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    # 增强生成回答
    prompt = SQL_QA_PROMPT.format(question=question, sql=final_sql, result=result_str)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content