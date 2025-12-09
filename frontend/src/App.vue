<template>
  <div class="page">
    <header class="header">
      <div class="title">✨ ChatBot </div>
      <div v-if="error" class="error-banner">
        {{ error }}
      </div>
    </header>

    <main class="content">
      <div class="chat-wrapper">
        <ChatMessageList :messages="messages" />
      </div>
    </main>

    <ChatInputBox :loading="loading" @send="onSend" />
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from "vue";
import { sendChatMessage } from "./api/chatClient";
import ChatInputBox from "./components/ChatInputBox.vue";
import ChatMessageList from "./components/ChatMessageList.vue";
import type { UiMessage } from "./types";

const messages = reactive<UiMessage[]>([
  {
    id: "welcome",
    role: "bot",
    text: "你好，我是你的聊天助手，你可以随时开始和我聊天。",
    meta: {
      mode: "llm_only"
    }
  }
]);
const loading = ref(false);
const error = ref("");

const onSend = async (text: string) => {
  error.value = "";
  const userMsg: UiMessage = {
    id: crypto.randomUUID(),
    role: "user",
    text
  };
  messages.push(userMsg);

  loading.value = true;
  try {
    const resp = await sendChatMessage(text);
    const botMsg: UiMessage = {
      id: crypto.randomUUID(),
      role: "bot",
      text: resp.answer,
      meta: resp.meta,
      sources: resp.sources
    };
    messages.push(botMsg);
  } catch (e: any) {
    error.value = e?.message || "请求失败，请稍后重试";
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: #ffffff;
}

.header {
  padding: 14px 20px;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  position: sticky;
  top: 0;
  background: #ffffff;
  z-index: 10;
}

.title {
  font-size: 18px;
  font-weight: 700;
  color: #111827;
}

.error-banner {
  background: #fca5a5;
  color: #7f1d1d;
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 14px;
}

.content {
  flex: 1;
  display: flex;
  justify-content: center;
  padding: 20px 16px;
}

.chat-wrapper {
  width: 100%;
  max-width: 720px;
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  min-height: 60vh;
}
</style>
