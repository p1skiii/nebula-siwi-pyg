<template>
  <div class="input-bar">
    <textarea
      v-model="draft"
      class="input"
      placeholder="Type a message..."
      :disabled="loading"
      @keydown.enter.exact.prevent="handleEnter"
      @keydown.enter.shift.exact.stop
    ></textarea>
    <button class="send-btn" :disabled="loading || !draft.trim()" @click="emitSend">
      {{ loading ? "Sending..." : "Send" }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";

const emit = defineEmits<{ send: [text: string] }>();
const props = defineProps<{ loading: boolean }>();

const draft = ref("");

const emitSend = () => {
  const text = draft.value.trim();
  if (!text || props.loading) return;
  emit("send", text);
  draft.value = "";
};

const handleEnter = () => {
  emitSend();
};

watch(
  () => props.loading,
  (next) => {
    if (!next) {
      // focus restore could be added if needed
    }
  }
);
</script>

<style scoped>
.input-bar {
  display: flex;
  gap: 10px;
  padding: 12px 16px 24px;
  border-top: 1px solid #e5e7eb;
  background: #ffffff;
}

.input {
  flex: 1;
  min-height: 60px;
  max-height: 140px;
  padding: 12px;
  border-radius: 14px;
  border: 1px solid #d1d5db;
  background: #f9fafb;
  color: #111827;
  resize: vertical;
  font-size: 15px;
}

.input:disabled {
  opacity: 0.6;
}

.send-btn {
  background: #2563eb;
  color: #f8fafc;
  border: none;
  padding: 0 18px;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.1s ease, box-shadow 0.1s ease;
  box-shadow: 0 6px 14px rgba(37, 99, 235, 0.2);
}

.send-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  box-shadow: none;
}

.send-btn:not(:disabled):active {
  transform: translateY(1px);
}
</style>
