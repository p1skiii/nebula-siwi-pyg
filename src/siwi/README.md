# SIWI 后端说明

## 概览
- 单一入口：Flask `/api/chat`，默认 llm_only。
- 配置：`.env`（`LLM_PROVIDER`/`GEMINI_API_KEY`/`LLM_MODEL` 等），RAG/Agent/Graph 默认关闭（`RAG_ENABLED=0`、`AGENT_EXPERIMENTAL=0`、`ENABLE_GRAPH_BACKEND=0`）。
- RAG 模块：`src/siwi/rag/` 可在 Python 层独立调用（文档加载、向量化、检索、管线）。
- Agent/Graph：可选实验（`src/siwi/agent/`、`src/siwi/graph_backend/`），不开启不影响主线。

## 快速开始
1) `cp .env.example .env` 并填写真实 LLM Key（默认 `LLM_PROVIDER=gemini`；若需 mock，显式 `LLM_PROVIDER=mock` + `LLM_ALLOW_MOCK=1`）。
2) `uv pip install -r requirements.txt`
3) `UV_CACHE_DIR=.uv_cache PYTHONPATH=src uv run --no-project python -m siwi.api.app`

## 关键配置
- LLM：`LLM_PROVIDER`（默认 gemini）、`GEMINI_API_KEY`/`LLM_API_KEY`、`LLM_MODEL`（默认 `gemini-1.5-flash-002`）、`LLM_ALLOW_MOCK`（显式 1 才可用 mock）。
- 功能开关：`RAG_ENABLED`（默认 0）、`AGENT_EXPERIMENTAL`（默认 0）、`ENABLE_GRAPH_BACKEND`（默认 0）。

## RAG 管线（独立调用示例）
```python
from pathlib import Path
from siwi.rag.doc_loader import DocLoader
from siwi.rag.embedder import TfidfEmbedder
from siwi.rag.retriever import Retriever
from siwi.rag.pipeline import RagPipeline

docs = DocLoader(Path("data/demo_docs")).load()
retriever = Retriever(embedder=TfidfEmbedder(), documents=docs)
pipeline = RagPipeline(retriever=retriever, llm_client=None, top_k=4)
result = pipeline.run("What is SIWI Bot?")
print(result["answer"], result["sources"])
```

## 目录要点
- `api/`：Flask app、路由、配置。
- `rag/`：文档加载、向量化、检索、管线（默认不挂到路由）。
- `agent/`：极简 Agent SDK（可选）。
- `graph_backend/`：NebulaGraph + PyG 实验模块（可选）。
- `legacy/`：旧入口/逻辑归档。
