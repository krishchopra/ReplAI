import dotenv from "dotenv";

dotenv.config();

export interface MemoryQuote {
  conversation_id: string;
  message_id?: string;
  content?: string;
  author?: string;
  timestamp?: string;
  score?: number;
}

export interface MemoryConversation {
  conversation_id: string;
  score?: number;
  participants?: string[];
  tags?: Array<{ tag_id?: string; name?: string; category?: string }>;
  quotes: MemoryQuote[];
  neighbor_quotes?: Array<
    MemoryQuote & {
      relation?: "previous" | "next";
    }
  >;
}

export interface MemoryBundle {
  short_summary: string;
  conversations: MemoryConversation[];
  quotes: MemoryQuote[];
  metadata: Record<string, unknown>;
}

const DEFAULT_MEMORY_SERVICE_URL = "http://127.0.0.1:8001";

const MEMORY_SERVICE_URL =
  process.env["MEMORY_SERVICE_URL"] ?? DEFAULT_MEMORY_SERVICE_URL;

function buildQueryFromHistory(
  conversationHistory: { role: "user" | "assistant"; content: string }[]
): string | undefined {
  const reversed = [...conversationHistory].reverse();
  const lastUser = reversed.find((msg) => msg.role === "user");
  if (lastUser?.content?.trim()) {
    return lastUser.content.trim();
  }

  const lastFew = conversationHistory.slice(-6);
  if (lastFew.length === 0) {
    return undefined;
  }

  return lastFew
    .map((msg) => `${msg.role}: ${msg.content}`)
    .join("\n")
    .slice(0, 1500);
}

export async function fetchMemoryBundle(
  conversationHistory: { role: "user" | "assistant"; content: string }[],
  participants?: string[]
): Promise<MemoryBundle | null> {
  const query = buildQueryFromHistory(conversationHistory);
  if (!query) {
    return null;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(`${MEMORY_SERVICE_URL}/memory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        participants,
        top_k: 6,
        expand_hops: 2,
        include_neighbors: true,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(
        `⚠️ Memory service responded with ${response.status}: ${response.statusText}`
      );
      return null;
    }

    const payload = (await response.json()) as MemoryBundle;
    if (!payload || typeof payload !== "object") {
      return null;
    }

    return payload;
  } catch (error) {
    if ((error as Error).name === "AbortError") {
      console.warn("⚠️ Memory service request timed out");
    } else {
      console.warn(`⚠️ Memory service error: ${(error as Error).message}`);
    }
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export function formatMemoryForPrompt(bundle: MemoryBundle): string {
  const lines: string[] = [];
  if (bundle.short_summary) {
    lines.push(`Summary: ${bundle.short_summary}`);
  }

  for (const conversation of bundle.conversations.slice(0, 3)) {
    const participants = conversation.participants?.join(", ") ?? "unknown";
    lines.push(`Conversation (${participants}) - score ${conversation.score ?? "?"}:`);
    for (const quote of conversation.quotes.slice(0, 3)) {
      if (quote.content) {
        const author = quote.author ?? "unknown";
        lines.push(`• ${author}: "${quote.content}"`);
      }
    }
  }

  return lines.join("\n");
}

