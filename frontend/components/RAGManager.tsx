'use client';

import { useState } from 'react';
import KnowledgePoolList from './KnowledgePoolList';
import DocumentUpload from './DocumentUpload';
import DocumentList from './DocumentList';

type RAGTab = 'pools' | 'upload' | 'documents';

export default function RAGManager() {
  const [activeTab, setActiveTab] = useState<RAGTab>('pools');

  const tabs: { id: RAGTab; label: string }[] = [
    { id: 'pools', label: 'Pools' },
    { id: 'upload', label: 'Upload' },
    { id: 'documents', label: 'Documents' },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Tab Navigation */}
      <div className="flex gap-2 mb-4 border-b border-slate-200">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-violet-500 text-violet-600'
                : 'border-transparent text-slate-600 hover:text-slate-800 hover:border-slate-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'pools' && <KnowledgePoolList />}
        {activeTab === 'upload' && <DocumentUpload />}
        {activeTab === 'documents' && <DocumentList />}
      </div>
    </div>
  );
}
