"""
公共医学数据集初始化脚本。
从 HuggingFace 下载公共医学数据集，清洗后导入 Milvus 知识库。

用法：
    # 导入全部数据集
    python scripts/init_public_datasets.py

    # 只导入指定数据集
    python scripts/init_public_datasets.py --dataset cmirb
    python scripts/init_public_datasets.py --dataset dialogue
    python scripts/init_public_datasets.py --dataset medqa

依赖：
    pip install datasets
"""

import asyncio
import argparse
import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient
from loguru import logger

from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.doc_rag import COLLECTION_NAME

settings = get_settings()
BATCH_SIZE = 50


def _get_deps():
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    ensure_knowledge_collection(milvus_client)
    return milvus_client, embedding_model


async def _insert_texts(
    milvus_client: MilvusClient,
    embedding_model,
    texts: list[str],
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
) -> int:
    """批量向量化并写入 Milvus。"""
    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )

    all_data = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = await embedding_model.aembed_documents(batch)
        for j, (text_content, emb) in enumerate(zip(batch, embeddings)):
            idx = i + j
            all_data.append({
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "category": category,
                "page_number": 0,
                "chunk_index": idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

    if all_data:
        for i in range(0, len(all_data), 1000):
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data[i:i + 1000],
            )
    return len(all_data)


# ═══════════════════════════════════════════════════════════════════════
# 数据集 1：CMIRB/MedicalRetrieval — 10万条医学检索语料
# ═══════════════════════════════════════════════════════════════════════

async def download_cmirb(milvus_client, embedding_model):
    """
    下载 CMIRB/MedicalRetrieval 数据集并导入 Milvus。
    包含 ~101K 条中文医学文本，适合作为文档 RAG 的基础语料。
    """
    from datasets import load_dataset

    print("[INFO] 下载 CMIRB/MedicalRetrieval 数据集...")
    ds = load_dataset("CMIRB/MedicalRetrieval", "corpus", split="corpus")

    texts = []
    for row in ds:
        text = row.get("text", "") or row.get("content", "")
        if text and len(text.strip()) > 20:
            texts.append(text.strip()[:2000])

    print(f"[INFO] CMIRB 共 {len(texts)} 条有效文本，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"cmirb_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"CMIRB医学检索语料_{i // chunk_size + 1}",
            doc_type="literature",
            category="医学文献",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] CMIRB 导入完成，共 {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════════════
# 数据集 2：Chinese-medical-dialogue-data — 6科室医患对话
# ═══════════════════════════════════════════════════════════════════════

async def download_med_dialogue(milvus_client, embedding_model):
    """
    下载中文医患对话数据集并导入 Milvus。
    覆盖：男科、内科、妇产科、肿瘤科、儿科、外科。
    每条记录格式化为"问：xxx 答：xxx"，作为知识文档。
    """
    from datasets import load_dataset

    print("[INFO] 下载 Chinese-medical-dialogue-data 数据集...")
    ds = load_dataset("BillGPT/Chinese-medical-dialogue-data")

    texts = []
    for split_name in ds:
        for row in ds[split_name]:
            q = row.get("ask", "") or row.get("question", "") or ""
            a = row.get("answer", "") or ""
            if q and a and len(q) > 5 and len(a) > 10:
                text = f"问：{q.strip()}\n答：{a.strip()}"
                texts.append(text[:2000])

    print(f"[INFO] 医患对话共 {len(texts)} 条有效记录，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"med_dialogue_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"中文医患对话_{i // chunk_size + 1}",
            doc_type="literature",
            category="医患对话",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] 医患对话导入完成，共 {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════════════
# 数据集 3：MedQA — 执业医师考试题（评测用，存到本地文件）
# ═══════════════════════════════════════════════════════════════════════

async def download_medqa():
    """
    下载 MedQA 中文执业医师考试题，保存到 data/eval/ 目录。
    用于评测知识 Agent 的准确率，不导入 Milvus。
    """
    from datasets import load_dataset
    import json

    print("[INFO] 下载 MedQA 数据集...")
    ds = load_dataset("bigbio/med_qa", "med_qa_zh_4options_bigbio_qa")

    eval_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    records = []
    for split_name in ds:
        for row in ds[split_name]:
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer = row.get("answer", [])
            if question:
                records.append({
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "split": split_name,
                })

    output_path = os.path.join(eval_dir, "medqa_zh.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] MedQA 保存完成：{output_path}，共 {len(records)} 道题")
    return len(records)


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="公共医学数据集初始化")
    parser.add_argument(
        "--dataset",
        choices=["cmirb", "dialogue", "medqa", "all"],
        default="all",
        help="要导入的数据集（默认 all）",
    )
    args = parser.parse_args()

    milvus_client, embedding_model = _get_deps()

    if args.dataset in ("cmirb", "all"):
        await download_cmirb(milvus_client, embedding_model)

    if args.dataset in ("dialogue", "all"):
        await download_med_dialogue(milvus_client, embedding_model)

    if args.dataset in ("medqa", "all"):
        await download_medqa()

    print("\n[DONE] 数据集初始化完成！")


if __name__ == "__main__":
    asyncio.run(main())
