'use client';

import { useState, useEffect } from 'react';
import { useAuthStore } from '@/lib/auth-store';

interface Todo {
  id: string;
  text: string;
  done: boolean;
}

interface ScratchpadData {
  todos: Todo[];
  notes: string;
  journal: string;
}

export function Scratchpad() {
  const accessToken = useAuthStore((state) => state.accessToken);
  const [activeTab, setActiveTab] = useState<'todos' | 'notes' | 'journal'>(
    'todos'
  );
  const [todos, setTodos] = useState<Todo[]>([]);
  const [notes, setNotes] = useState('');
  const [journal, setJournal] = useState('');
  const [newTodoText, setNewTodoText] = useState('');
  const [isSaving, setIsSaving] = useState(false);

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
      done: false,
    };

    setTodos([...todos, newTodo]);
    setNewTodoText('');
  };

  const toggleTodo = (id: string) => {
    setTodos(
      todos.map((todo) =>
        todo.id === id ? { ...todo, done: !todo.done } : todo
      )
    );
  };

  const deleteTodo = (id: string) => {
    setTodos(todos.filter((todo) => todo.id !== id));
  };

  return (
    <div className="flex flex-col h-full bg-white border-r border-slate-200">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-900">Scratchpad</h2>
        <p className="text-xs text-slate-500 mt-1">
          {isSaving ? 'Saving...' : 'Auto-saved'}
        </p>
      </div>

      {/* Tabs */}
      <div className="flex-shrink-0 px-6 pt-4">
        <div className="flex gap-1 bg-slate-100 p-1 rounded-lg">
          <button
            onClick={() => setActiveTab('todos')}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'todos'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
              }`}
          >
            Todos
          </button>
          <button
            onClick={() => setActiveTab('notes')}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'notes'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
              }`}
          >
            Notes
          </button>
          <button
            onClick={() => setActiveTab('journal')}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'journal'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
              }`}
          >
            Journal
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {activeTab === 'todos' && (
          <div className="space-y-4">
            {/* Add Todo */}
            <div className="flex gap-2">
              <input
                type="text"
                value={newTodoText}
                onChange={(e) => setNewTodoText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addTodo()}
                placeholder="Add a todo..."
                className="flex-1 px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
              />
              <button
                onClick={addTodo}
                className="px-4 py-2 bg-violet-500 text-white text-sm font-medium rounded-lg hover:bg-violet-600 transition-colors"
              >
                Add
              </button>
            </div>

            {/* Todo List */}
            <div className="space-y-2">
              {todos.length === 0 ? (
                <p className="text-sm text-slate-400 text-center py-8">
                  No todos yet. Add one above!
                </p>
              ) : (
                todos.map((todo) => (
                  <div
                    key={todo.id}
                    className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors group"
                  >
                    <input
                      type="checkbox"
                      checked={todo.done}
                      onChange={() => toggleTodo(todo.id)}
                      className="w-4 h-4 text-violet-500 rounded focus:ring-2 focus:ring-violet-500"
                    />
                    <span
                      className={`flex-1 text-sm ${todo.done
                          ? 'text-slate-400 line-through'
                          : 'text-slate-700'
                        }`}
                    >
                      {todo.text}
                    </span>
                    <button
                      onClick={() => deleteTodo(todo.id)}
                      className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 transition-all"
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
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'notes' && (
          <div>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Write your notes here..."
              className="w-full h-[calc(100vh-300px)] px-4 py-3 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent resize-none"
            />
          </div>
        )}

        {activeTab === 'journal' && (
          <div>
            <div className="mb-3 text-xs text-slate-500">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </div>
            <textarea
              value={journal}
              onChange={(e) => setJournal(e.target.value)}
              placeholder="What's on your mind today?"
              className="w-full h-[calc(100vh-340px)] px-4 py-3 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent resize-none"
            />
          </div>
        )}
      </div>
    </div>
  );
}
