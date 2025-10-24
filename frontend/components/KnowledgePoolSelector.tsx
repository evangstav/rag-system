'use client';

import { useState, useEffect, useRef } from 'react';
import { useRAGStore } from '@/store/ragStore';

export default function KnowledgePoolSelector() {
  const { knowledgePools, selectedPoolIds, togglePoolSelection, loadKnowledgePools } =
    useRAGStore();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadKnowledgePools();
  }, [loadKnowledgePools]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const selectedCount = selectedPoolIds.length;
  const isActive = selectedCount > 0;

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
          isActive
            ? 'bg-emerald-100 text-emerald-700 ring-2 ring-emerald-200'
            : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
        }`}
      >
        <span className="flex items-center gap-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isActive ? 'bg-emerald-500' : 'bg-slate-400'
            }`}
          />
          RAG {selectedCount > 0 && `(${selectedCount})`}
        </span>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-72 bg-white rounded-lg shadow-xl border border-slate-200 z-50">
          <div className="px-4 py-3 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-800">
              Select Knowledge Pools
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Choose which pools to query
            </p>
          </div>

          <div className="max-h-64 overflow-y-auto">
            {knowledgePools.length === 0 ? (
              <div className="px-4 py-6 text-center text-sm text-slate-500">
                No knowledge pools available.
                <br />
                Create one in the RAG tab.
              </div>
            ) : (
              <div className="py-2">
                {knowledgePools.map((pool) => {
                  const isSelected = selectedPoolIds.includes(pool.id);
                  return (
                    <button
                      key={pool.id}
                      onClick={() => togglePoolSelection(pool.id)}
                      className="w-full px-4 py-2 flex items-center gap-3 hover:bg-slate-50 transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => {}}
                        className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                      />
                      <div className="flex-1 text-left min-w-0">
                        <p className="text-sm font-medium text-slate-800 truncate">
                          {pool.name}
                        </p>
                        {pool.description && (
                          <p className="text-xs text-slate-500 truncate">
                            {pool.description}
                          </p>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {selectedCount > 0 && (
            <div className="px-4 py-3 border-t border-slate-200 bg-emerald-50">
              <p className="text-xs text-emerald-700">
                {selectedCount} pool{selectedCount > 1 ? 's' : ''} selected
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
