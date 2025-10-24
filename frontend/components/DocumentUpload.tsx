'use client';

import { useState, useRef } from 'react';
import { useRAGStore } from '@/store/ragStore';

export default function DocumentUpload() {
  const { knowledgePools, uploads, uploadDocument } = useRAGStore();
  const [selectedPoolId, setSelectedPoolId] = useState<string>('');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    if (!selectedPoolId) {
      alert('Please select a knowledge pool first');
      return;
    }

    // Upload each file
    for (let i = 0; i < files.length; i++) {
      await uploadDocument(selectedPoolId, files[i]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploading':
        return 'bg-blue-500';
      case 'processing':
        return 'bg-yellow-500';
      case 'completed':
        return 'bg-emerald-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-slate-300';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return 'Processing...';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Pool Selection */}
      <div className="mb-4">
        <label htmlFor="pool-select" className="block text-sm font-semibold text-slate-800 mb-2">
          Select Knowledge Pool
        </label>
        <select
          id="pool-select"
          value={selectedPoolId}
          onChange={(e) => setSelectedPoolId(e.target.value)}
          className="w-full px-3 py-2 text-sm text-slate-800 rounded-lg border border-slate-300 focus:ring-2 focus:ring-violet-500 focus:border-transparent"
        >
          <option value="">Choose a pool...</option>
          {knowledgePools.map((pool) => (
            <option key={pool.id} value={pool.id}>
              {pool.name}
            </option>
          ))}
        </select>
        {knowledgePools.length === 0 && (
          <p className="mt-2 text-xs text-slate-500">
            Create a knowledge pool first in the "Pools" tab
          </p>
        )}
      </div>

      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-violet-500 bg-violet-50'
            : 'border-slate-300 bg-white hover:border-violet-400 hover:bg-violet-50'
        } ${!selectedPoolId ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.txt,.md,.csv,.json"
          onChange={(e) => handleFileSelect(e.target.files)}
          className="hidden"
          disabled={!selectedPoolId}
        />

        <div className="space-y-2">
          <div className="text-4xl">📁</div>
          <div>
            <p className="text-sm font-medium text-slate-700">
              {isDragging ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              or click to browse
            </p>
          </div>
          <p className="text-xs text-slate-400">
            Supports: PDF, DOCX, TXT, MD, CSV, JSON
          </p>
        </div>
      </div>

      {/* Upload Progress */}
      {Object.keys(uploads).length > 0 && (
        <div className="mt-4 space-y-2 flex-1 overflow-y-auto">
          <h4 className="text-sm font-semibold text-slate-800">Upload Progress</h4>
          {Object.entries(uploads).map(([uploadId, upload]) => (
            <div
              key={uploadId}
              className="p-3 rounded-lg bg-slate-50 border border-slate-200"
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl">{getFileIcon(upload.filename)}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-800 truncate">
                    {upload.filename}
                  </p>

                  {/* Status Badge */}
                  <div className="mt-1 flex items-center gap-2">
                    <span
                      className={`inline-block px-2 py-0.5 text-xs font-medium text-white rounded ${getStatusColor(
                        upload.status
                      )}`}
                    >
                      {getStatusText(upload.status)}
                    </span>
                    {upload.status === 'uploading' && (
                      <span className="text-xs text-slate-500">
                        {upload.progress}%
                      </span>
                    )}
                  </div>

                  {/* Error Message */}
                  {upload.error && (
                    <p className="mt-1 text-xs text-red-600">{upload.error}</p>
                  )}

                  {/* Progress Bar */}
                  {(upload.status === 'uploading' || upload.status === 'processing') && (
                    <div className="mt-2 w-full bg-slate-200 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all ${
                          upload.status === 'uploading'
                            ? 'bg-blue-500'
                            : 'bg-yellow-500 animate-pulse'
                        }`}
                        style={{
                          width:
                            upload.status === 'uploading'
                              ? `${upload.progress}%`
                              : '100%',
                        }}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Info */}
      <div className="mt-4 p-3 rounded-lg bg-blue-50 border border-blue-200">
        <p className="text-xs text-blue-700">
          💡 Tip: Documents are processed in the background. You can continue using the app while they're being indexed.
        </p>
      </div>
    </div>
  );
}
