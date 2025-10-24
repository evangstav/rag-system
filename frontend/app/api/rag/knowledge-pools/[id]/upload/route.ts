import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// POST /api/rag/knowledge-pools/[id]/upload - Upload a document
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Extract authorization token from request headers
    const authHeader = request.headers.get('authorization');

    // Get the form data from the request
    const formData = await request.formData();

    // Add pool_id to the form data for backend
    formData.append('pool_id', id);

    // Build headers for backend request (no Content-Type for FormData)
    const headers: HeadersInit = {};

    // Forward authorization token if present
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    const response = await fetch(
      `${API_URL}/api/rag/upload`,
      {
        method: 'POST',
        headers,
        body: formData,
      }
    );

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: 'Failed to upload document', details: error },
        { status: response.status }
      );
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error uploading document:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
