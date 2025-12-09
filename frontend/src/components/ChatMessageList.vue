<template>
  <div class="chat-list">
    <div
      v-for="msg in messages"
      :key="msg.id"
      class="message-row"
      :class="msg.role === 'user' ? 'align-right' : 'align-left'"
    >
      <div class="bubble" :class="msg.role">
        <div class="text" v-html="msg.text"></div>
        <MetaInfoBar v-if="msg.role === 'bot' && msg.meta" :meta="msg.meta" />
        <div v-if="msg.role === 'bot' && msg.sources?.length" class="sources">
          <div class="sources-title">Sources</div>
          <ul>
            <li v-for="(s, idx) in msg.sources" :key="idx">
              <div class="source-title">{{ s.title }}</div>
              <div class="source-snippet">{{ truncate(s.snippet, 100) }}</div>
              <div class="source-score">score: {{ s.score.toFixed(3) }}</div>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { UiMessage } from "../types";
import MetaInfoBar from "./MetaInfoBar.vue";

defineProps<{ messages: UiMessage[] }>();

const truncate = (text: string, len: number) => {
  if (text.length <= len) return text;
  return text.slice(0, len) + "…";
};
</script>

<style scoped>
.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message-row {
  display: flex;
  width: 100%;
}

.align-right {
  justify-content: flex-end;
}

.align-left {
  justify-content: flex-start;
}

.bubble {
  max-width: 72%;
  padding: 12px 14px;
  border-radius: 14px;
  line-height: 1.5;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.08);
  word-break: break-word;
  background: #f3f4f6;
  color: #1f2937;
}

.bubble.user {
  background: #2563eb;
  color: #f8fafc;
}

.bubble.bot {
  background: #f3f4f6;
  color: #1f2937;
}

.text {
  white-space: pre-wrap;
  font-size: 15px;
}

.sources {
  margin-top: 10px;
  border-top: 1px solid #e5e7eb;
  padding-top: 8px;
}

.sources-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: #6b7280;
  margin-bottom: 4px;
}

.sources ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sources li {
  padding: 6px 0;
  border-bottom: 1px dashed #e5e7eb;
}

.source-title {
  font-weight: 600;
  color: #111827;
  font-size: 14px;
}

.source-snippet {
  font-size: 13px;
  color: #374151;
}

.source-score {
  font-size: 12px;
  color: #6b7280;
}
</style>
