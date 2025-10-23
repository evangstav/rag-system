'use client';

import { useEffect, useState } from 'react';
import { useConversationStore } from '@/store/conversationStore';

export function ConversationSidebar() {
  const {
    conversations,
    currentConversationId,
    isLoading,
    error,
    setCurrentConversation,
    loadConversations,
    createConversation,
    deleteConversation,
    clearError,
  } = useConversationStore();

  const [isCollapsed, setIsCollapsed] = useState(false);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  // `loadConversations` is a Zustand store action that is re-created on every render.
  // We intentionally omit it from the dependency array to avoid infinite re-renders.
  useEffect(() => {
    loadConversations();
  }, []);

  const handleNewConversation = async () => {
    await createConversation();
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversation(id);
  };

  const handleDeleteConversation = async (id: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent selecting the conversation
    if (confirm('Are you sure you want to delete this conversation?')) {
      await deleteConversation(id);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMs = now.getTime() - date.getTime();
    const diffInDays = Math.floor(diffInMs / (1000 * 60 * 60 * 24));

    if (diffInDays === 0) {
      return 'Today';
    } else if (diffInDays === 1) {
      return 'Yesterday';
    } else if (diffInDays < 7) {
      return `${diffInDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const getTruncatedTitle = (title: string | null) => {
    const displayTitle = title || 'New Conversation';
    return displayTitle.length > 30 ? `${displayTitle.substring(0, 30)}...` : displayTitle;
  };

  if (isCollapsed) {
    return (
      <div className="flex flex-col h-full bg-slate-900 border-r border-slate-700">
        <button
          onClick={() => setIsCollapsed(false)}
          className="p-4 hover:bg-slate-800 transition-colors"
          aria-label="Expand sidebar"
        >
          <svg
            className="w-6 h-6 text-slate-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <h2 className="text-lg font-semibold text-slate-100">Conversations</h2>
        <button
          onClick={() => setIsCollapsed(true)}
          className="p-1 hover:bg-slate-800 rounded transition-colors"
          aria-label="Collapse sidebar"
        >
          <svg
            className="w-5 h-5 text-slate-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
        </button>
      </div>

      {/* New Conversation Button */}
      <div className="p-4 border-b border-slate-700">
        <button
          onClick={handleNewConversation}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-violet-500 to-purple-600 text-white rounded-lg font-medium hover:from-violet-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-violet-500/30"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          New Chat
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-4 mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm text-red-400">{error}</p>
            <button
              onClick={clearError}
              className="text-red-400 hover:text-red-300"
              aria-label="Dismiss error"
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
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading && conversations.length === 0 ? (
          <div className="p-4 text-center text-slate-400">
            <div className="animate-pulse">Loading conversations...</div>
          </div>
        ) : conversations.length === 0 ? (
          <div className="p-4 text-center text-slate-500">
            <p className="text-sm">No conversations yet</p>
            <p className="text-xs mt-2">Click "New Chat" to start</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                onClick={() => handleSelectConversation(conversation.id)}
                className={`group relative flex items-start justify-between gap-2 p-3 rounded-lg cursor-pointer transition-all ${
                  currentConversationId === conversation.id
                    ? 'bg-violet-500/20 border border-violet-500/30'
                    : 'hover:bg-slate-800 border border-transparent'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <h3
                    className={`text-sm font-medium truncate ${
                      currentConversationId === conversation.id
                        ? 'text-violet-200'
                        : 'text-slate-200'
                    }`}
                  >
                    {getTruncatedTitle(conversation.title)}
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    {formatDate(conversation.updated_at)}
                  </p>
                </div>

                {/* Delete Button */}
                <button
                  onClick={(e) => handleDeleteConversation(conversation.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                  aria-label="Delete conversation"
                >
                  <svg
                    className="w-4 h-4 text-red-400 hover:text-red-300"
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
            ))}
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-700">
        <div className="text-xs text-slate-500 text-center">
          {conversations.length} conversation{conversations.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
  );
}
