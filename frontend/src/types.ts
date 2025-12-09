import type { ChatMeta, ChatSource } from "./api/chatClient";

export type Role = "user" | "bot";

export interface UiMessage {
  id: string;
  role: Role;
  text: string;
  meta?: ChatMeta;
  sources?: ChatSource[];
}
