

"""
配置层：
src/rag/
├── config.py          # ① 配置层：4 个 dataclass，一切策略的开关都在这
├── engine.py          # ④ 引擎层：RAGEngine，把下面所有步骤串成一条流水线
├── tools.py           # ⑤ 工具层：3 个 LangChain tool，暴露给上层 Agent 调用
│
├── ingestion/
│   └── chunkers.py    # ② 接入层：3 种文档分块策略
├── retrieval/
│   ├── query_transform.py   # ③ 查询改写
│   ├── hyde.py              # ③ HyDE 假设性文档
│   ├── vector_search.py     # ③ 纯向量检索
│   ├── hybrid_search.py     # ③ 稠密+稀疏混合检索
│   └── reranker.py          # ③ 重排序
├── generation/
│   └── generator.py    # 生成：带上下文的问答 prompt
└── evaluation/         # 评估（RAG Triad）
    ├── trulens_config.py
    ├── feedbacks.py
    └── tracked_rag.py
"""



from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    strategy: str = "fixed"  # "fixed" | "semantic" | "parent_child"
    parent_chunk_size: int = 2048


@dataclass
class RetrievalConfig:
    top_k: int = 20
    rerank_top_k: int = 5
    use_hyde: bool = False
    use_rerank: bool = True
    use_hybrid: bool = False
    similarity_threshold: float = 0.3


@dataclass
class GenerationConfig:
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class RAGConfig:
    collection_name: str = "knowledge_docs"
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)