'use client';

import { useChat } from '@ai-sdk/react';
import { useState, useRef, useEffect } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { Scratchpad } from '@/components/Scratchpad';
import { AuthGuard } from '@/components/AuthGuard';
import { useAuthStore } from '@/lib/auth-store';
import { logout } from '@/lib/api-client';

function ChatContent() {
  const user = useAuthStore((state) => state.user);
  const accessToken = useAuthStore((state) => state.accessToken);
  const [useRag, setUseRag] = useState(false);
  const [useScratchpad, setUseScratchpad] = useState(false);
  const [input, setInput] = useState('');
  const [showScratchpad, setShowScratchpad] = useState(true);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, sendMessage, isLoading } = useChat({
    api: '/api/chat',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex h-screen">
      <PanelGroup direction="horizontal">
        {/* Scratchpad Panel */}
        {showScratchpad && (
          <>
            <Panel defaultSize={25} minSize={20} maxSize={40}>
              <Scratchpad />
            </Panel>
            <PanelResizeHandle className="w-1 bg-slate-200 hover:bg-violet-400 transition-colors" />
          </>
        )}

        {/* Chat Panel */}
        <Panel defaultSize={75} minSize={50}>
          <div className="flex flex-col h-full bg-gradient-to-br from-slate-50 via-white to-slate-50">
            {/* Header */}
            <header className="flex-shrink-0 border-b border-slate-200 bg-white/80 backdrop-blur-xl">
              <div className="max-w-4xl mx-auto px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    {/* Toggle Scratchpad Button */}
                    <button
                      onClick={() => setShowScratchpad(!showScratchpad)}
                      className="p-2 rounded-lg hover:bg-slate-100 transition-colors"
                      title={showScratchpad ? 'Hide scratchpad' : 'Show scratchpad'}
                    >
                      <svg
                        className="w-5 h-5 text-slate-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 6h16M4 12h16M4 18h16"
                        />
                      </svg>
                    </button>

                    <div>
                      <h1 className="text-xl font-semibold text-slate-900">
                        RAG Chat Assistant
                      </h1>
                      <p className="text-sm text-slate-500 mt-0.5">
                        Powered by AI with contextual knowledge
                      </p>
                    </div>
                  </div>

                  {/* Context Pills and User Menu */}
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setUseScratchpad(!useScratchpad)}
                      className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${useScratchpad
                        ? 'bg-violet-100 text-violet-700 ring-2 ring-violet-200'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${useScratchpad ? 'bg-violet-500' : 'bg-slate-400'
                            }`}
                        />
                        Scratchpad
                      </span>
                    </button>

                    <button
                      onClick={() => setUseRag(!useRag)}
                      className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${useRag
                        ? 'bg-emerald-100 text-emerald-700 ring-2 ring-emerald-200'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${useRag ? 'bg-emerald-500' : 'bg-slate-400'
                            }`}
                        />
                        RAG
                      </span>
                    </button>

                    {/* User Menu */}
                    <div className="relative">
                      <button
                        onClick={() => setShowUserMenu(!showUserMenu)}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition-colors"
                      >
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                          <span className="text-white text-xs font-medium">
                            {user?.username?.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <span className="text-sm font-medium text-slate-700">{user?.username}</span>
                      </button>

                      {/* Dropdown Menu */}
                      {showUserMenu && (
                        <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-slate-200 py-1 z-50">
                          <div className="px-4 py-2 border-b border-slate-200">
                            <p className="text-xs text-slate-500">Signed in as</p>
                            <p className="text-sm font-medium text-slate-900 truncate">{user?.email}</p>
                          </div>
                          <button
                            onClick={() => logout()}
                            className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 transition-colors flex items-center gap-2"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                            </svg>
                            Sign Out
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </header>

            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto">
              <div className="max-w-4xl mx-auto px-6 py-8">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center mb-4">
                      <svg
                        className="w-8 h-8 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                        />
                      </svg>
                    </div>
                    <h2 className="text-2xl font-semibold text-slate-800 mb-2">
                      Start a conversation
                    </h2>
                    <p className="text-slate-500 max-w-md">
                      Ask me anything. Enable scratchpad or RAG context for
                      enhanced responses.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {messages.map((m) => (
                      <div
                        key={m.id}
                        className={`flex gap-4 ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                          }`}
                      >
                        {/* Avatar */}
                        <div
                          className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${m.role === 'user'
                            ? 'bg-gradient-to-br from-blue-500 to-indigo-600'
                            : 'bg-gradient-to-br from-violet-500 to-purple-600'
                            }`}
                        >
                          {m.role === 'user' ? (
                            <svg
                              className="w-5 h-5 text-white"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                              />
                            </svg>
                          ) : (
                            <svg
                              className="w-5 h-5 text-white"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                              />
                            </svg>
                          )}
                        </div>

                        {/* Message Bubble */}
                        <div
                          className={`flex-1 max-w-[80%] ${m.role === 'user' ? 'text-right' : 'text-left'
                            }`}
                        >
                          <div
                            className={`inline-block px-5 py-3 rounded-2xl ${m.role === 'user'
                              ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/20'
                              : 'bg-white text-slate-800 shadow-lg shadow-slate-200/50 border border-slate-200'
                              }`}
                          >
                            <div className="whitespace-pre-wrap break-words leading-relaxed">
                              {m.parts?.map((part, partIdx) => {
                                if (part.type === 'text') {
                                  return <span key={partIdx}>{part.text}</span>;
                                }
                                return null;
                              })}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}

                    {isLoading && (
                      <div className="flex gap-4">
                        <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                          <svg
                            className="w-5 h-5 text-white animate-pulse"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                            />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div className="inline-block px-5 py-3 rounded-2xl bg-white shadow-lg shadow-slate-200/50 border border-slate-200">
                            <div className="flex gap-1.5">
                              <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.3s]" />
                              <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.15s]" />
                              <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>
            </div>

            {/* Input Area */}
            <div className="flex-shrink-0 border-t border-slate-200 bg-white/80 backdrop-blur-xl">
              <div className="max-w-4xl mx-auto px-6 py-4">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (input.trim()) {
                      // Pass dynamic values as request-level options
                      sendMessage(
                        { text: input },
                        {
                          body: {
                            useRag,
                            useScratchpad,
                            knowledgePoolIds: [],
                          },
                        }
                      );
                      setInput('');
                    }
                  }}
                  className="relative"
                >
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask me anything..."
                    disabled={isLoading}
                    className="w-full px-5 py-3.5 pr-12 rounded-xl border border-slate-300 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all disabled:opacity-50 disabled:cursor-not-allowed text-slate-800 placeholder:text-slate-400"
                  />
                  <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 text-white hover:from-violet-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-violet-500/20"
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
                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                      />
                    </svg>
                  </button>
                </form>
              </div>
            </div>
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
}

export default function Chat() {
  return (
    <AuthGuard>
      <ChatContent />
    </AuthGuard>
  );
}
