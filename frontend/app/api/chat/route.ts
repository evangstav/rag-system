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
  }: {
    messages: UIMessage[];
    useRag?: boolean;
    useScratchpad?: boolean;
    knowledgePoolIds?: string[];
  } = await req.json();

  try {
    // Convert UIMessage format to backend format
    const backendMessages = messages.map((msg) => ({
      role: msg.role,
      content: msg.parts?.map((part) => part.type === 'text' ? part.text : '').join('') || '',
    }));

    // Call backend streaming endpoint
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: backendMessages,
        use_rag: useRag,
        use_scratchpad: useScratchpad,
        knowledge_pool_ids: knowledgePoolIds,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}: ${response.statusText}`);
    }

    // Transform backend SSE stream to AI SDK format
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    const transformStream = new TransformStream({
      async transform(chunk, controller) {
        const text = decoder.decode(chunk);
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            try {
              const parsed = JSON.parse(data);

              // Handle different event types from backend
              if (parsed.type === 'content') {
                // Transform to AI SDK format
                const aiSDKChunk = {
                  type: 'text-delta',
                  textDelta: parsed.content,
                };
                controller.enqueue(encoder.encode(`0:${JSON.stringify(aiSDKChunk)}\n`));
              } else if (parsed.type === 'metadata') {
                // Store metadata (could be used to show sources in UI)
                console.log('Context metadata:', parsed.metadata);
              } else if (parsed.type === 'conversation_id') {
                // Store conversation ID (could be used for history)
                console.log('Conversation ID:', parsed.conversation_id);
              } else if (parsed.type === 'done') {
                // End of stream
                controller.enqueue(encoder.encode('0:{"type":"finish","finishReason":"stop"}\n'));
              } else if (parsed.type === 'error') {
                // Error from backend
                console.error('Backend error:', parsed.error);
                controller.enqueue(encoder.encode(`3:${JSON.stringify({error: parsed.error})}\n`));
              }
            } catch (e) {
              // Skip malformed JSON
              console.warn('Failed to parse SSE data:', data);
            }
          }
        }
      },
    });

    // Return the transformed stream
    return new Response(response.body?.pipeThrough(transformStream), {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Vercel-AI-Data-Stream': 'v1',
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
