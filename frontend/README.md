# SIWI Chat Frontend (Vite + Vue 3)

极简白色系聊天界面，对接后端 `POST /api/chat`。

## 快速开始
1) 安装依赖
```bash
cd frontend
npm install
```
2) 配置环境变量
```bash
cp .env.example .env
# 如需修改 API 地址，设置 VITE_API_BASE_URL（默认 http://localhost:5000；dev 已配置 /api 代理）
```
3) 启动
```bash
# 后端（项目根）
UV_CACHE_DIR=.uv_cache PYTHONPATH=src uv run --no-project python -m siwi.api.app
# 前端（frontend 目录）
npm run dev
# 访问 http://localhost:5173
```

## 特性
- 聊天列表：用户消息右侧蓝色气泡，Bot 左侧浅灰气泡。
- 顶部欢迎语、轻量标题栏；内容区域居中、最大宽度 720px。
- 底部输入框：Enter 发送，Shift+Enter 换行；发送时按钮禁用。
- 展示后端 meta（mode/llm_provider 等）与 sources（标题、snippet、score）。
- 环境变量 `VITE_API_BASE_URL` 控制 API 基址；未设置时 dev 使用 `/api` 代理。

## 结构
```
frontend/
  src/
    api/chatClient.ts   # /api/chat 封装与类型
    components/
      ChatMessageList.vue
      ChatInputBox.vue
      MetaInfoBar.vue
    types.ts
    App.vue
    main.ts
    style.css
  .env.example
  vite.config.ts
  package.json
```
