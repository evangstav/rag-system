'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuthStore } from '@/lib/auth-store';
import RAGManager from './RAGManager';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

interface ScratchpadData {
  todos: Todo[];
  notes: string;
  journal: string;
}

interface JournalEntry {
  date: string;
  content: string;
  preview: string;
}

interface Memory {
  id: string;
  content: string;
  importance: number;
  created_at: string;
  updated_at: string;
}

export function Scratchpad() {
  const accessToken = useAuthStore((state) => state.accessToken);
  const [activeTab, setActiveTab] = useState<'scratchpad' | 'knowledge'>('scratchpad');
  const [scratchpadSubTab, setScratchpadSubTab] = useState<'todos' | 'notes' | 'journal' | 'memories'>('todos');
  const [todos, setTodos] = useState<Todo[]>([]);
  const [notes, setNotes] = useState('');
  const [journal, setJournal] = useState('');
  const [newTodoText, setNewTodoText] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [journalHistory, setJournalHistory] = useState<JournalEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [viewingHistoryEntry, setViewingHistoryEntry] = useState<JournalEntry | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [isLoadingMemories, setIsLoadingMemories] = useState(false);
  const [isExtractingFromJournal, setIsExtractingFromJournal] = useState(false);

  // Reset showHistory when leaving the journal sub-tab
  useEffect(() => {
    if (scratchpadSubTab !== 'journal' && showHistory) {
      setShowHistory(false);
    }
  }, [scratchpadSubTab, showHistory]);
  // Reset viewingHistoryEntry when leaving the journal tab
  useEffect(() => {
    if (scratchpadSubTab !== 'journal' && viewingHistoryEntry !== null) {
      setViewingHistoryEntry(null);
    }
  }, [scratchpadSubTab]);
  // Load scratchpad data on mount
  useEffect(() => {
    if (accessToken) {
      loadScratchpad();
    }
  }, [accessToken]);

  // Auto-save when data changes (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (accessToken) {
        saveScratchpad();
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [todos, notes, journal, accessToken]);

  const loadScratchpad = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/scratchpad', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data: ScratchpadData = await response.json();
        setTodos(data.todos || []);
        setNotes(data.notes || '');
        setJournal(data.journal || '');
      }
    } catch (error) {
      console.error('Failed to load scratchpad:', error);
    }
  };

  const saveScratchpad = async () => {
    if (isSaving) return;

    setIsSaving(true);
    try {
      await fetch('http://localhost:8000/api/scratchpad', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ todos, notes, journal }),
      });
    } catch (error) {
      console.error('Failed to save scratchpad:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const addTodo = () => {
    if (!newTodoText.trim()) return;

    const newTodo: Todo = {
      id: crypto.randomUUID(),
      text: newTodoText.trim(),
      completed: false,
    };

    setTodos([...todos, newTodo]);
    setNewTodoText('');
  };

  const toggleTodo = (id: string) => {
    setTodos(
      todos.map((todo) =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  };

  const deleteTodo = (id: string) => {
    setTodos(todos.filter((todo) => todo.id !== id));
  };

  const loadJournalHistory = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/scratchpad/journal/history?limit=10', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setJournalHistory(data.entries || []);
      }
    } catch (error) {
      console.error('Failed to load journal history:', error);
    }
  }, [accessToken]);

  // Load journal history when journal tab is selected
  useEffect(() => {
    if (accessToken && scratchpadSubTab === 'journal') {
      loadJournalHistory();
    }
  }, [accessToken, scratchpadSubTab, loadJournalHistory]);

  // Load memories when memories tab is selected
  useEffect(() => {
    if (accessToken && scratchpadSubTab === 'memories') {
      loadMemories();
    }
  }, [accessToken, scratchpadSubTab]);

  const loadMemories = async () => {
    setIsLoadingMemories(true);
    try {
      const response = await fetch('http://localhost:8000/api/memory/', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setMemories(data || []);
      }
    } catch (error) {
      console.error('Failed to load memories:', error);
    } finally {
      setIsLoadingMemories(false);
    }
  };

  const deleteMemory = async (id: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/memory/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        setMemories(memories.filter(m => m.id !== id));
      }
    } catch (error) {
      console.error('Failed to delete memory:', error);
    }
  };

  const extractFromJournal = async () => {
    setIsExtractingFromJournal(true);
    try {
      const response = await fetch('http://localhost:8000/api/memory/extract/journal?days_back=7', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const newMemories = await response.json();
        if (newMemories && newMemories.length > 0) {
          alert(`Extracted ${newMemories.length} memories from your journal!`);
          loadMemories(); // Reload to get all memories
        } else {
          alert('No new memories found in recent journal entries.');
        }
      }
    } catch (error) {
      console.error('Failed to extract from journal:', error);
      alert('Failed to extract memories from journal.');
    } finally {
      setIsExtractingFromJournal(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-slate-50 to-white border-l border-slate-200">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Workspace</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              {activeTab === 'scratchpad' && (isSaving ? 'Saving...' : 'Auto-saved')}
              {activeTab === 'knowledge' && 'Manage your knowledge base'}
            </p>
          </div>
        </div>
      </div>

      {/* Main Tabs */}
      <div className="flex-shrink-0 px-6 pt-4">
        <div className="flex gap-2 bg-slate-100 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('scratchpad')}
            className={`flex-1 px-4 py-2.5 text-sm font-semibold rounded-lg transition-all duration-200 ${
              activeTab === 'scratchpad'
                ? 'bg-white text-violet-700 shadow-md shadow-violet-100'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Scratchpad
            </div>
          </button>
          <button
            onClick={() => setActiveTab('knowledge')}
            className={`flex-1 px-4 py-2.5 text-sm font-semibold rounded-lg transition-all duration-200 ${
              activeTab === 'knowledge'
                ? 'bg-white text-emerald-700 shadow-md shadow-emerald-100'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              Knowledge
            </div>
          </button>
        </div>
      </div>

      {/* Sub-tabs for Scratchpad */}
      {activeTab === 'scratchpad' && (
        <div className="flex-shrink-0 px-6 pt-3">
          <div className="flex gap-1">
            <button
              onClick={() => setScratchpadSubTab('todos')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                scratchpadSubTab === 'todos'
                  ? 'bg-violet-100 text-violet-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Todos
            </button>
            <button
              onClick={() => setScratchpadSubTab('journal')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                scratchpadSubTab === 'journal'
                  ? 'bg-violet-100 text-violet-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Journal
            </button>
            <button
              onClick={() => setScratchpadSubTab('notes')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                scratchpadSubTab === 'notes'
                  ? 'bg-violet-100 text-violet-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Notes
            </button>
            <button
              onClick={() => setScratchpadSubTab('memories')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                scratchpadSubTab === 'memories'
                  ? 'bg-violet-100 text-violet-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Memories
            </button>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {activeTab === 'scratchpad' && (
          <>
            {scratchpadSubTab === 'todos' && (
              <div className="space-y-4">
                {/* Add Todo */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newTodoText}
                    onChange={(e) => setNewTodoText(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addTodo()}
                    placeholder="Add a todo..."
                    className="flex-1 px-4 py-2.5 text-sm text-slate-800 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
                  />
                  <button
                    onClick={addTodo}
                    className="px-4 py-2.5 bg-gradient-to-r from-violet-500 to-purple-600 text-white text-sm font-medium rounded-lg hover:from-violet-600 hover:to-purple-700 transition-all shadow-md shadow-violet-500/20"
                  >
                    Add
                  </button>
                </div>

                {/* Todo List */}
                <div className="space-y-2">
                  {todos.length === 0 ? (
                    <div className="text-center py-12">
                      <svg className="w-12 h-12 mx-auto text-slate-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      <p className="text-sm text-slate-400">No todos yet</p>
                      <p className="text-xs text-slate-400 mt-1">Add one above to get started!</p>
                    </div>
                  ) : (
                    todos.map((todo) => (
                      <div
                        key={todo.id}
                        className="flex items-center gap-3 p-3 bg-white rounded-lg border border-slate-200 hover:border-violet-300 hover:shadow-sm transition-all group"
                      >
                        <input
                          type="checkbox"
                          checked={todo.completed}
                          onChange={() => toggleTodo(todo.id)}
                          className="w-4 h-4 text-violet-600 rounded focus:ring-2 focus:ring-violet-500 cursor-pointer"
                        />
                        <span
                          className={`flex-1 text-sm ${
                            todo.completed
                              ? 'text-slate-400 line-through'
                              : 'text-slate-700'
                          }`}
                        >
                          {todo.text}
                        </span>
                        <button
                          onClick={() => deleteTodo(todo.id)}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all"
                          aria-label="Delete todo"
                        >
                          <svg
                            className="w-4 h-4 text-slate-400 hover:text-red-500"
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
                    ))
                  )}
                </div>
              </div>
            )}

            {scratchpadSubTab === 'journal' && (
              <div className="space-y-4">
                {/* Header with today's date and history toggle */}
                <div className="flex items-center justify-between pb-3 border-b border-slate-200">
                  <div className="flex items-center gap-2 text-sm text-slate-600">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    {viewingHistoryEntry ? (
                      new Date(viewingHistoryEntry.date).toLocaleDateString('en-US', {
                        weekday: 'long',
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                      })
                    ) : (
                      new Date().toLocaleDateString('en-US', {
                        weekday: 'long',
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                      })
                    )}
                  </div>
                  {journalHistory.length > 0 && (
                    <button
                      onClick={() => setShowHistory(!showHistory)}
                      className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-violet-600 hover:text-violet-700 hover:bg-violet-50 rounded-md transition-all"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {showHistory ? 'Hide History' : 'View History'}
                    </button>
                  )}
                </div>

                {/* History List (Collapsible) */}
                {showHistory && journalHistory.length > 0 && (
                  <div className="space-y-2 p-3 bg-slate-50 rounded-lg border border-slate-200 max-h-64 overflow-y-auto">
                    <h3 className="text-xs font-semibold text-slate-700 mb-2 uppercase tracking-wide">Past Entries</h3>
                    {journalHistory.map((entry, index) => {
                      const entryDate = new Date(entry.date);
                      const isToday = entryDate.toDateString() === new Date().toDateString();
                      return (
                        <button
                          key={entry.date}
                          onClick={() => {
                            if (!isToday) {
                              setViewingHistoryEntry(entry);
                            }
                          }}
                          disabled={isToday}
                          className={`w-full text-left p-3 rounded-md transition-all ${
                            viewingHistoryEntry?.date === entry.date
                              ? 'bg-violet-100 border-violet-300 border'
                              : 'bg-white hover:bg-slate-100 border border-slate-200'
                          } ${isToday ? 'cursor-not-allowed opacity-60' : ''}`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-slate-900">
                              {entryDate.toLocaleDateString('en-US', {
                                month: 'short',
                                day: 'numeric',
                                year: 'numeric',
                              })}
                            </span>
                            {isToday && (
                              <span className="text-xs px-2 py-0.5 bg-violet-100 text-violet-700 rounded-full font-medium">
                                Today
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-slate-600 line-clamp-2">{entry.preview}</p>
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Back to Today Button (when viewing history) */}
                {viewingHistoryEntry && (
                  <button
                    onClick={() => setViewingHistoryEntry(null)}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-violet-600 hover:text-violet-700 bg-violet-50 hover:bg-violet-100 rounded-lg transition-all"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Back to Today
                  </button>
                )}

                {/* Journal Textarea */}
                <textarea
                  value={viewingHistoryEntry ? viewingHistoryEntry.content : journal}
                  onChange={(e) => {
                    if (!viewingHistoryEntry) {
                      setJournal(e.target.value);
                    }
                  }}
                  placeholder={viewingHistoryEntry ? '' : "What's on your mind today?"}
                  readOnly={!!viewingHistoryEntry}
                  className={`w-full h-[calc(100vh-360px)] px-4 py-3 text-sm text-slate-800 bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent resize-none transition-all ${
                    viewingHistoryEntry ? 'cursor-default bg-slate-50' : ''
                  }`}
                />
                {viewingHistoryEntry && (
                  <p className="text-xs text-slate-500 italic">This is a past entry and cannot be edited.</p>
                )}
              </div>
            )}

            {scratchpadSubTab === 'notes' && (
              <div>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Write your notes here..."
                  className="w-full h-[calc(100vh-320px)] px-4 py-3 text-sm text-slate-800 bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent resize-none transition-all"
                />
              </div>
            )}

            {scratchpadSubTab === 'memories' && (
              <div className="space-y-4">
                {/* Header with actions */}
                <div className="flex items-center justify-between pb-3 border-b border-slate-200">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">Your Memories</h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      AI automatically remembers your preferences and context
                    </p>
                  </div>
                  <button
                    onClick={extractFromJournal}
                    disabled={isExtractingFromJournal}
                    className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-violet-600 hover:text-violet-700 bg-violet-50 hover:bg-violet-100 rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    {isExtractingFromJournal ? 'Extracting...' : 'Extract from Journal'}
                  </button>
                </div>

                {/* Memories list */}
                {isLoadingMemories ? (
                  <div className="text-center py-12">
                    <div className="inline-block w-8 h-8 border-3 border-violet-200 border-t-violet-600 rounded-full animate-spin"></div>
                    <p className="text-sm text-slate-500 mt-3">Loading memories...</p>
                  </div>
                ) : memories.length === 0 ? (
                  <div className="text-center py-12">
                    <svg className="w-12 h-12 mx-auto text-slate-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <p className="text-sm text-slate-400">No memories yet</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Chat with the AI or extract from your journal to build memories
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[calc(100vh-380px)] overflow-y-auto">
                    {memories
                      .sort((a, b) => b.importance - a.importance)
                      .map((memory) => (
                        <div
                          key={memory.id}
                          className="p-3 bg-white rounded-lg border border-slate-200 hover:border-violet-300 hover:shadow-sm transition-all group"
                        >
                          <div className="flex items-start gap-3">
                            {/* Importance indicator */}
                            <div className="flex-shrink-0 mt-0.5">
                              {memory.importance >= 0.8 ? (
                                <span className="text-yellow-500" title={`Importance: ${(memory.importance * 100).toFixed(0)}%`}>⭐</span>
                              ) : memory.importance >= 0.6 ? (
                                <span className="text-violet-500" title={`Importance: ${(memory.importance * 100).toFixed(0)}%`}>●</span>
                              ) : (
                                <span className="text-slate-400" title={`Importance: ${(memory.importance * 100).toFixed(0)}%`}>○</span>
                              )}
                            </div>

                            {/* Content */}
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-slate-700 leading-relaxed">{memory.content}</p>
                              <p className="text-xs text-slate-400 mt-1">
                                {new Date(memory.updated_at).toLocaleDateString('en-US', {
                                  month: 'short',
                                  day: 'numeric',
                                  year: 'numeric',
                                })}
                              </p>
                            </div>

                            {/* Delete button */}
                            <button
                              onClick={() => deleteMemory(memory.id)}
                              className="opacity-0 group-hover:opacity-100 flex-shrink-0 p-1 hover:bg-red-50 rounded transition-all"
                              aria-label="Delete memory"
                            >
                              <svg
                                className="w-4 h-4 text-slate-400 hover:text-red-500"
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
                      ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {activeTab === 'knowledge' && (
          <div className="h-[calc(100vh-220px)]">
            <RAGManager />
          </div>
        )}
      </div>
    </div>
  );
}
