import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// GET /api/rag/knowledge-pools - List all knowledge pools
export async function GET(request: NextRequest) {
  try {
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

    const response = await fetch(`${API_URL}/api/rag/knowledge-pools`, {
      headers,
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: 'Failed to fetch knowledge pools', details: error },
        { status: response.status }
      );
    }

    const pools = await response.json();
    return NextResponse.json(pools);
  } catch (error) {
    console.error('Error fetching knowledge pools:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/rag/knowledge-pools - Create a new knowledge pool
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

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

    const response = await fetch(`${API_URL}/api/rag/knowledge-pools`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: 'Failed to create knowledge pool', details: error },
        { status: response.status }
      );
    }

    const pool = await response.json();
    return NextResponse.json(pool);
  } catch (error) {
    console.error('Error creating knowledge pool:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
