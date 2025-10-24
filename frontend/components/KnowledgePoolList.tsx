'use client';

import { useEffect, useState } from 'react';
import { useRAGStore } from '@/store/ragStore';

export default function KnowledgePoolList() {
  const {
    knowledgePools,
    selectedPoolIds,
    isLoading,
    error,
    loadKnowledgePools,
    createKnowledgePool,
    deleteKnowledgePool,
    togglePoolSelection,
    clearError,
  } = useRAGStore();

  const [newPoolName, setNewPoolName] = useState('');
  const [newPoolDescription, setNewPoolDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);

  useEffect(() => {
    loadKnowledgePools();
  }, [loadKnowledgePools]);

  const handleCreatePool = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPoolName.trim()) return;

    setIsCreating(true);
    const pool = await createKnowledgePool(
      newPoolName,
      newPoolDescription || undefined
    );
    setIsCreating(false);

    if (pool) {
      setNewPoolName('');
      setNewPoolDescription('');
      setShowCreateForm(false);
    }
  };

  const handleDeletePool = async (id: string, name: string) => {
    if (!confirm(`Delete knowledge pool "${name}"? This will remove all documents in this pool.`)) {
      return;
    }
    await deleteKnowledgePool(id);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-800">Knowledge Pools</h3>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 text-white hover:from-violet-600 hover:to-purple-700 transition-all"
        >
          {showCreateForm ? 'Cancel' : '+ New Pool'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200">
          <div className="flex items-start justify-between">
            <p className="text-sm text-red-700">{error}</p>
            <button
              onClick={clearError}
              className="text-red-400 hover:text-red-600 transition-colors"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Create Form */}
      {showCreateForm && (
        <form onSubmit={handleCreatePool} className="mb-4 p-4 rounded-lg bg-violet-50 border border-violet-200">
          <div className="space-y-3">
            <div>
              <label htmlFor="pool-name" className="block text-xs font-medium text-slate-700 mb-1">
                Pool Name *
              </label>
              <input
                id="pool-name"
                type="text"
                value={newPoolName}
                onChange={(e) => setNewPoolName(e.target.value)}
                placeholder="e.g., Research Papers"
                className="w-full px-3 py-2 text-sm text-slate-800 rounded-lg border border-slate-300 focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                required
                disabled={isCreating}
              />
            </div>
            <div>
              <label htmlFor="pool-description" className="block text-xs font-medium text-slate-700 mb-1">
                Description (optional)
              </label>
              <input
                id="pool-description"
                type="text"
                value={newPoolDescription}
                onChange={(e) => setNewPoolDescription(e.target.value)}
                placeholder="e.g., AI and ML research documents"
                className="w-full px-3 py-2 text-sm text-slate-800 rounded-lg border border-slate-300 focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                disabled={isCreating}
              />
            </div>
            <button
              type="submit"
              disabled={isCreating || !newPoolName.trim()}
              className="w-full px-4 py-2 text-sm font-medium rounded-lg bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isCreating ? 'Creating...' : 'Create Pool'}
            </button>
          </div>
        </form>
      )}

      {/* Pools List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {isLoading && knowledgePools.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-sm">
            Loading pools...
          </div>
        ) : knowledgePools.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-sm">
            No knowledge pools yet. Create one to get started!
          </div>
        ) : (
          knowledgePools.map((pool) => {
            const isSelected = selectedPoolIds.includes(pool.id);
            return (
              <div
                key={pool.id}
                className={`p-3 rounded-lg border-2 transition-all cursor-pointer ${
                  isSelected
                    ? 'border-violet-500 bg-violet-50'
                    : 'border-slate-200 bg-white hover:border-violet-300'
                }`}
                onClick={() => togglePoolSelection(pool.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1">
                    {/* Checkbox */}
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => togglePoolSelection(pool.id)}
                      className="mt-1 h-4 w-4 rounded border-slate-300 text-violet-600 focus:ring-violet-500"
                      onClick={(e) => e.stopPropagation()}
                    />

                    {/* Pool Info */}
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-slate-800 truncate">
                        {pool.name}
                      </h4>
                      {pool.description && (
                        <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">
                          {pool.description}
                        </p>
                      )}
                      <p className="text-xs text-slate-400 mt-1">
                        {new Date(pool.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  {/* Delete Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeletePool(pool.id, pool.name);
                    }}
                    className="ml-2 text-slate-400 hover:text-red-500 transition-colors"
                    aria-label={`Delete ${pool.name}`}
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
              </div>
            );
          })
        )}
      </div>

      {/* Selection Summary */}
      {selectedPoolIds.length > 0 && (
        <div className="mt-4 p-3 rounded-lg bg-emerald-50 border border-emerald-200">
          <p className="text-sm text-emerald-700">
            {selectedPoolIds.length} pool{selectedPoolIds.length > 1 ? 's' : ''} selected for RAG
          </p>
        </div>
      )}
    </div>
  );
}
