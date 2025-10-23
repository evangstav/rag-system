import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// GET /api/conversations/[id]/messages - Get all messages for a conversation
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '100';
    const offset = searchParams.get('offset') || '0';

    // Extract authorization token from request headers
    const authHeader = request.headers.get('authorization');

    // Build headers for backend request
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    // Forward authorization token if present
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    const response = await fetch(
      `${API_URL}/api/conversations/${id}/messages?limit=${limit}&offset=${offset}`,
      {
        headers,
      }
    );

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: 'Failed to fetch messages', details: error },
        { status: response.status }
      );
    }

    const messages = await response.json();
    return NextResponse.json(messages);
  } catch (error) {
    console.error('Error fetching messages:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
