'use client';

import { useState } from 'react';
import { useRAGStore } from '@/store/ragStore';
import { useAuthStore } from '@/lib/auth-store';

interface SearchResult {
  document_id: string;
  filename: string;
  content: string;
  score: number;
  metadata?: Record<string, any>;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  num_results: number;
}

export default function SearchInterface() {
  const { knowledgePools, selectedPoolIds } = useRAGStore();
  const accessToken = useAuthStore((state) => state.accessToken);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await fetch('/api/rag/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          query: query.trim(),
          knowledge_pool_ids: selectedPoolIds.length > 0 ? selectedPoolIds : null,
          limit: 10,
        }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-emerald-600 bg-emerald-50';
    if (score >= 0.6) return 'text-blue-600 bg-blue-50';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-slate-600 bg-slate-50';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Search Input */}
      <form onSubmit={handleSearch} className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your documents..."
            className="flex-1 px-4 py-2 text-sm text-slate-800 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            disabled={isSearching}
          />
          <button
            type="submit"
            disabled={isSearching || !query.trim()}
            className="px-4 py-2 bg-gradient-to-br from-violet-500 to-purple-600 text-white text-sm font-medium rounded-lg hover:from-violet-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Scope Info */}
        {selectedPoolIds.length > 0 ? (
          <p className="mt-2 text-xs text-slate-500">
            Searching in {selectedPoolIds.length} selected pool{selectedPoolIds.length > 1 ? 's' : ''}
          </p>
        ) : knowledgePools.length > 0 ? (
          <p className="mt-2 text-xs text-slate-500">
            Searching across all {knowledgePools.length} pool{knowledgePools.length > 1 ? 's' : ''}
          </p>
        ) : (
          <p className="mt-2 text-xs text-amber-600">
            No knowledge pools available. Create one in the Pools tab.
          </p>
        )}
      </form>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Search Results */}
      <div className="flex-1 overflow-y-auto">
        {!hasSearched ? (
          <div className="text-center py-12 text-slate-500">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-slate-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <p className="text-sm font-medium">Search your documents</p>
            <p className="text-xs text-slate-400 mt-1">
              Enter a query to find relevant content
            </p>
          </div>
        ) : isSearching ? (
          <div className="text-center py-12 text-slate-500">
            <div className="inline-block w-8 h-8 border-4 border-violet-500 border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-sm">Searching documents...</p>
          </div>
        ) : results.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-slate-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm font-medium">No results found</p>
            <p className="text-xs text-slate-400 mt-1">
              Try different keywords or upload more documents
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Results Header */}
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-semibold text-slate-700">
                {results.length} result{results.length > 1 ? 's' : ''} found
              </p>
            </div>

            {/* Result Cards */}
            {results.map((result, index) => (
              <div
                key={`${result.document_id}-${index}`}
                className="p-4 rounded-lg bg-white border border-slate-200 hover:border-violet-300 transition-colors"
              >
                {/* Header with filename and score */}
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-xl">📄</span>
                    <h4 className="text-sm font-medium text-slate-800 truncate">
                      {result.filename}
                    </h4>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${getScoreColor(
                      result.score
                    )}`}
                  >
                    {(result.score * 100).toFixed(0)}% match
                  </span>
                </div>

                {/* Content Preview */}
                <p className="text-sm text-slate-600 leading-relaxed line-clamp-4">
                  {result.content}
                </p>

                {/* Metadata */}
                {result.metadata && Object.keys(result.metadata).length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(result.metadata).map(([key, value]) => (
                      <span
                        key={key}
                        className="px-2 py-0.5 text-xs text-slate-600 bg-slate-100 rounded"
                      >
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
