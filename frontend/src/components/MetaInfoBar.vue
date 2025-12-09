<template>
  <div class="meta">
    <span v-if="meta.mode">mode: {{ meta.mode }}</span>
    <span v-if="meta.llm_provider">llm: {{ meta.llm_provider }}</span>
    <span v-if="meta.agent_enabled !== undefined">agent: {{ meta.agent_enabled ? "on" : "off" }}</span>
    <span v-if="meta.graph_enabled !== undefined">graph: {{ meta.graph_enabled ? "on" : "off" }}</span>
  </div>
</template>

<script setup lang="ts">
import type { ChatMeta } from "../api/chatClient";

defineProps<{ meta: ChatMeta }>();
</script>

<style scoped>
.meta {
  margin-top: 6px;
  font-size: 12px;
  color: #6b7280;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
</style>
