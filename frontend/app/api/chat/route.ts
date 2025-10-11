import { streamText, convertToModelMessages, UIMessage } from 'ai';
import { openai } from '@ai-sdk/openai';

export const maxDuration = 30;

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

  // Build context for system message
  const contextParts: string[] = [];

  try {
    // 1. Fetch scratchpad context if enabled
    if (useScratchpad) {
      const scratchpadRes = await fetch(
        'http://localhost:8000/api/scratchpad',
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        }
      );

      if (scratchpadRes.ok) {
        const scratchpadData = await scratchpadRes.json();

        // Format scratchpad data
        const parts = [];
        if (scratchpadData.todos?.length > 0) {
          parts.push(
            'Todos:\n' +
              scratchpadData.todos
                .map((t: any) => `- [${t.done ? 'x' : ' '}] ${t.text}`)
                .join('\n')
          );
        }
        if (scratchpadData.notes) {
          parts.push(`Notes:\n${scratchpadData.notes}`);
        }
        if (scratchpadData.journal) {
          parts.push(`Journal:\n${scratchpadData.journal}`);
        }

        if (parts.length > 0) {
          contextParts.push(
            `User's current scratchpad:\n${parts.join('\n\n')}`
          );
        }
      }
    }

    // 2. Fetch RAG context if enabled
    if (useRag && knowledgePoolIds.length > 0) {
      // Get the last user message text
      const lastMessage = messages[messages.length - 1];
      const userMessageText = lastMessage?.parts
        ?.filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(' ') || '';

      const ragRes = await fetch('http://localhost:8000/api/rag/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessageText,
          pool_ids: knowledgePoolIds,
          top_k: 5,
        }),
      });

      if (ragRes.ok) {
        const ragResults = await ragRes.json();

        if (ragResults.documents?.length > 0) {
          const sources = ragResults.documents
            .map(
              (doc: any, i: number) =>
                `Source ${i + 1} (${doc.metadata?.filename || 'Unknown'}):\n${doc.content}`
            )
            .join('\n\n');

          contextParts.push(`Retrieved information:\n${sources}`);
        }
      }
    }

    // 3. Build system message with context
    let systemMessage = 'You are a helpful AI assistant.';
    if (contextParts.length > 0) {
      systemMessage += '\n\n' + contextParts.join('\n\n');
    }

    // 4. Stream response using AI SDK
    const result = streamText({
      model: openai('gpt-4-turbo-preview'),
      system: systemMessage,
      messages: convertToModelMessages(messages),
    });

    return result.toUIMessageStreamResponse();
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
