import { UIMessage } from 'ai';

export const maxDuration = 30;

/**
 * Chat API route handler.
 *
 * This proxies requests to the FastAPI backend which handles:
 * - RAG context injection
 * - Scratchpad context injection
 * - Conversation persistence
 * - Message history
 */
export async function POST(req: Request) {
  const {
    messages,
    useRag = false,
    useScratchpad = false,
    knowledgePoolIds = [],
    conversationId = null,
  }: {
    messages: UIMessage[];
    useRag?: boolean;
    useScratchpad?: boolean;
    knowledgePoolIds?: string[];
    conversationId?: string | null;
  } = await req.json();

  try {
    // Extract authorization token from request headers
    const authHeader = req.headers.get('authorization');
    console.log('=== Chat API Debug ===');
    console.log('Authorization header:', authHeader);

    // Convert UIMessage format to backend format
    const backendMessages = messages.map((msg) => {
      let content = '';
      if (msg.parts && Array.isArray(msg.parts)) {
        content = msg.parts.filter((part: any) => part.type === 'text').map((part: any) => part.text).join('');
      } else if ((msg as any).content) {
        content = typeof (msg as any).content === 'string' ? (msg as any).content : JSON.stringify((msg as any).content);
      }
      return { role: msg.role, content };
    });

    // Build headers for backend request
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    // Forward authorization token if present
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    // Call backend streaming endpoint
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/api/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        messages: backendMessages,
        conversation_id: conversationId,
        use_rag: useRag,
        use_scratchpad: useScratchpad,
        knowledge_pool_ids: knowledgePoolIds,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    // Transform SSE stream to plain text stream
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    const transformStream = new TransformStream({
      async transform(chunk, controller) {
        const text = decoder.decode(chunk, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            try {
              const parsed = JSON.parse(data);

              // Handle different event types from backend
              if (parsed.type === 'content') {
                // Forward just the text content
                controller.enqueue(encoder.encode(parsed.content));
              }
              // Ignore metadata, conversation_id, done, etc.
            } catch (e) {
              // Skip malformed JSON
            }
          }
        }
      },
    });

    // Return plain text stream
    return new Response(response.body.pipeThrough(transformStream), {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
      },
    });
  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({
        error: 'Failed to process chat request',
        details: error instanceof Error ? error.message : 'Unknown error',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
