'use client';

import { useEffect } from 'react';
import { useRAGStore } from '@/store/ragStore';

export default function DocumentList() {
  const {
    knowledgePools,
    documents,
    currentPoolId,
    isLoading,
    setCurrentPool,
    loadDocuments,
    deleteDocument,
  } = useRAGStore();

  useEffect(() => {
    if (currentPoolId) {
      loadDocuments(currentPoolId);
    }
  }, [currentPoolId, loadDocuments]);

  const handleDeleteDocument = async (id: string, filename: string) => {
    if (!confirm(`Delete document "${filename}"?`)) {
      return;
    }
    const success = await deleteDocument(id);
    if (success && currentPoolId) {
      // Reload documents after deletion
      loadDocuments(currentPoolId);
    }
  };

  const getStatusBadge = (status: string) => {
    const colors = {
      pending: 'bg-slate-300 text-slate-700',
      processing: 'bg-blue-500 text-white',
      completed: 'bg-emerald-500 text-white',
      failed: 'bg-red-500 text-white',
    };
    return colors[status as keyof typeof colors] || 'bg-slate-300 text-slate-700';
  };

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return '📄';
      case 'doc':
      case 'docx':
        return '📝';
      case 'txt':
      case 'md':
        return '📃';
      case 'json':
      case 'csv':
        return '📊';
      default:
        return '📎';
    }
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes) return 'N/A';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="flex flex-col h-full">
      {/* Pool Selection */}
      <div className="mb-4">
        <label htmlFor="doc-pool-select" className="block text-sm font-semibold text-slate-800 mb-2">
          Knowledge Pool
        </label>
        <select
          id="doc-pool-select"
          value={currentPoolId || ''}
          onChange={(e) => setCurrentPool(e.target.value || null)}
          className="w-full px-3 py-2 text-sm text-slate-800 rounded-lg border border-slate-300 focus:ring-2 focus:ring-violet-500 focus:border-transparent"
        >
          <option value="">Select a pool...</option>
          {knowledgePools.map((pool) => (
            <option key={pool.id} value={pool.id}>
              {pool.name}
            </option>
          ))}
        </select>
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-y-auto">
        {!currentPoolId ? (
          <div className="text-center py-8 text-slate-500 text-sm">
            Select a knowledge pool to view documents
          </div>
        ) : isLoading ? (
          <div className="text-center py-8 text-slate-500 text-sm">
            Loading documents...
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-sm">
            No documents in this pool yet. Upload some files to get started!
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="p-3 rounded-lg bg-white border border-slate-200 hover:border-violet-300 transition-colors"
              >
                <div className="flex items-start gap-3">
                  {/* File Icon */}
                  <span className="text-2xl">{getFileIcon(doc.filename)}</span>

                  {/* Document Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="text-sm font-medium text-slate-800 truncate">
                        {doc.filename}
                      </h4>

                      {/* Delete Button */}
                      <button
                        onClick={() => handleDeleteDocument(doc.id, doc.filename)}
                        className="text-slate-400 hover:text-red-500 transition-colors flex-shrink-0"
                        aria-label={`Delete ${doc.filename}`}
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </button>
                    </div>

                    {/* Status Badge */}
                    <div className="mt-1 flex items-center gap-2 flex-wrap">
                      <span
                        className={`inline-block px-2 py-0.5 text-xs font-medium rounded ${getStatusBadge(
                          doc.status
                        )}`}
                      >
                        {doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}
                      </span>
                      <span className="text-xs text-slate-500">
                        {formatBytes(doc.file_size)}
                      </span>
                      {doc.num_chunks > 0 && (
                        <span className="text-xs text-slate-500">
                          {doc.num_chunks} chunks
                        </span>
                      )}
                      {doc.num_tokens && doc.num_tokens > 0 && (
                        <span className="text-xs text-slate-500">
                          {doc.num_tokens.toLocaleString()} tokens
                        </span>
                      )}
                    </div>

                    {/* Error Message */}
                    {doc.error_message && (
                      <p className="mt-1 text-xs text-red-600">
                        Error: {doc.error_message}
                      </p>
                    )}

                    {/* Metadata */}
                    <div className="mt-1 text-xs text-slate-400">
                      {doc.source_type === 'upload' ? 'Uploaded' : 'URL'} • {formatDate(doc.created_at)}
                      {doc.mime_type && ` • ${doc.mime_type}`}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Summary */}
      {currentPoolId && documents.length > 0 && (
        <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-600">
              Total: {documents.length} document{documents.length !== 1 ? 's' : ''}
            </span>
            <span className="text-slate-600">
              Completed: {documents.filter((d) => d.status === 'completed').length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
