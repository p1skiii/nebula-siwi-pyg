export interface ChatSource {
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
}

export interface ChatMeta {
  mode?: string;
  llm_provider?: string;
  agent_enabled?: boolean;
  graph_enabled?: boolean;
  [key: string]: unknown;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  meta?: ChatMeta;
}

export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || "";
  const url = baseUrl ? `${baseUrl}/api/chat` : `/api/chat`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}
