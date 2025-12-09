# **Nebula-SIWI Bot**

*一个干净的、重构的ChatBot后端，可选的RAG、轻量级代理路由和Gemini风格的界面.*

> 🚀 **本项目是对原始 GNN PoC 的彻底重构**：
> 我将一个复杂且难以运行的 Nebula + PyG 实验仓库，改造成
> **“单入口可复用的 ChatBot 后端 + 可插拔 RAG + Agent + Graph”**。
---

## 🧠 为什么要做这个项目

* 原始仓库高度耦合（Flask、PyG、Nebula、BERT 混在一起），**难以复用、难以理解、无法开箱即用**
* 我希望做一个**真正能跑**、**可演示**、**可逐步扩展**的 ChatBot 后端
* 具备 **RAG / Agent / Graph** 的能力，为未来做 **Agentic Workflow / 多工具协作** 打基础

---

## ✨ 特性亮点

### 🔹 1) 单一入口 `/api/chat`，默认 “LLM-only”

* 统一接口 → 易集成到任意前端
* LLM Provider 可随时切换（Mock / Gemini / OpenAI）

### 🔹 2)  RAG 系统

* 文档自动扫描
* TF-IDF + fallback 策略
* Top-K 段落检索
* 上下文构造 + LLM 回答
* 错误不崩溃，返回 sources 和 meta 信息

### 🔹 3) 极简 Agent SDK（可开关）

* 意图分类 → 工具路由
* 支持 TextRagTool / GraphTool
* trace 记录整个决策链路

### 🔹 4) Graph & PyG 

* 不影响主线运行
* 启用后可进行 1-hop 子图查询
* 为 future GraphRAG 留扩展位

### 🔹 5) 全新前端（Vite + Vue）

* Gemini-style 极简气泡聊天 UI
* 调用 `/api/chat` 即可使用
* 前后端完全解耦

---

## 🧱 架构概述

```
frontend/         # Gemini-style chat UI
   ↓ calls /api/chat
backend/
  siwi/api/       # Flask API, config, deps
  siwi/rag/       # RAG pipeline (loader, embedder, retriever)
  siwi/agent/     # lightweight agent router + tools
  graph_backend/  # optional NebulaGraph + PyG
data/demo_docs/   # built-in RAG documents
```

---

## ⚡ 快速开始

### 1) 配置

```bash
cp .env.example .env
# 默认 LLM_PROVIDER=gemini；RAG/Agent/Graph 均关闭
```

### 2) 安装依赖

```bash
uv pip install -r requirements.txt
```

### 3) 启动后端

```bash
UV_CACHE_DIR=.uv_cache PYTHONPATH=src uv run --no-project python -m siwi.api.app
```

前端：

```bash
cd frontend
npm install
npm run dev
```

访问：`http://localhost:5173`

---

## 💬 API (`POST /api/chat`)

Request:

```json
{
  "message": "这个项目是做什么的？"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [...],
  "meta": {
    "mode": "llm_only | text_rag | graph",
    "llm_provider": "GeminiLLMClient",
    "agent_enabled": false
  },
  "trace": [...]
}
```

---

## 🔍 RAG 设计

* 文档加载（`.txt` / `.md`）自动切分 chunk
* 向量化：TF-IDF → fallback（无 sklearn 时仍可运行）
* 检索：余弦相似度 / 关键词召回
* 统一输出：sources + meta
* 失败不崩溃 → 自动回退到 LLM-only

---

## 🧪 Agent 设计

* intent classifier：`graph / text`
* router → 调用对应工具
* 可插拔 Tools：支持未来扩展 Search / Function Calling
* trace 记录 → 可用于可观测性与运营分析

---

## 🧱 技术亮点

* 对 legacy GNN PoC 进行了 **模块化重构**，建立统一 API 与可维护结构
* 通过环境变量（env）驱动运行模式：LLM-only / RAG / Agent / Graph
* 前端完全重写，实现了 **Gemini-style UI**（体现产品 sense）
* RAG pipeline 完全自定义，可脱离外部服务运行
* Agent 层设计参考 Claude/ChatGPT Tool Router（展示对热点理解）
* 为未来 GraphRAG / 多工具协作预留接口
* tests 覆盖 RAG + API（pytest）

---

## 🛠 未来计划

* [ ] Function Calling 模式
* [ ] Streaming 输出
* [ ] 多工具协作（Sequential Planner）
* [ ] GraphRAG v1（图检索 + 文档检索融合）
* [ ] UI 添加 Source 高亮 / 工具调用可视化
* [ ] API-Key 前端设置面板

---

## 📄 License

Apache-2.0
