import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from 'sonner';
import { useAuthStore } from '@/lib/auth-store';

// Type definitions matching backend schemas
export interface KnowledgePool {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  collection_name: string;
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  knowledge_pool_id: string;
  filename: string;
  source_type: string;
  source_url?: string;
  file_path?: string;
  file_size?: number;
  mime_type?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error_message?: string;
  num_chunks: number;
  num_tokens?: number;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface UploadProgress {
  filename: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  error?: string;
}

interface RAGState {
  // Knowledge Pools
  knowledgePools: KnowledgePool[];
  selectedPoolIds: string[];

  // Documents
  documents: Document[];
  currentPoolId: string | null;

  // Upload state
  uploads: Record<string, UploadProgress>;

  // UI state
  isLoading: boolean;
  error: string | null;

  // Actions - Knowledge Pools
  loadKnowledgePools: () => Promise<void>;
  createKnowledgePool: (name: string, description?: string) => Promise<KnowledgePool | null>;
  deleteKnowledgePool: (id: string) => Promise<boolean>;
  selectPool: (id: string) => void;
  deselectPool: (id: string) => void;
  togglePoolSelection: (id: string) => void;
  clearSelectedPools: () => void;

  // Actions - Documents
  loadDocuments: (poolId: string) => Promise<void>;
  uploadDocument: (poolId: string, file: File) => Promise<void>;
  deleteDocument: (documentId: string) => Promise<boolean>;

  // Actions - UI
  setCurrentPool: (poolId: string | null) => void;
  clearError: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Helper to get auth headers
const getAuthHeaders = (accessToken: string | null): HeadersInit => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };
  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }
  return headers;
};

export const useRAGStore = create<RAGState>()(
  persist(
    (set, get) => ({
      // Initial state
      knowledgePools: [],
      selectedPoolIds: [],
      documents: [],
      currentPoolId: null,
      uploads: {},
      isLoading: false,
      error: null,

      // Knowledge Pool Actions
      loadKnowledgePools: async () => {
        set({ isLoading: true, error: null });
        try {
          const accessToken = useAuthStore.getState().accessToken;
          const response = await fetch('/api/rag/knowledge-pools', {
            headers: getAuthHeaders(accessToken),
          });
          if (!response.ok) {
            throw new Error('Failed to load knowledge pools');
          }
          const pools = await response.json();
          set({ knowledgePools: pools, isLoading: false });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Unknown error',
            isLoading: false
          });
        }
      },

      createKnowledgePool: async (name: string, description?: string) => {
        set({ isLoading: true, error: null });
        try {
          const accessToken = useAuthStore.getState().accessToken;
          const response = await fetch('/api/rag/knowledge-pools', {
            method: 'POST',
            headers: getAuthHeaders(accessToken),
            body: JSON.stringify({ name, description }),
          });

          if (!response.ok) {
            throw new Error('Failed to create knowledge pool');
          }

          const newPool = await response.json();
          set(state => ({
            knowledgePools: [...state.knowledgePools, newPool],
            isLoading: false,
          }));

          toast.success('Knowledge pool created successfully');
          return newPool;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          set({
            error: errorMessage,
            isLoading: false
          });
          toast.error(`Failed to create pool: ${errorMessage}`);
          return null;
        }
      },

      deleteKnowledgePool: async (id: string) => {
        set({ isLoading: true, error: null });
        try {
          const accessToken = useAuthStore.getState().accessToken;
          const response = await fetch(`/api/rag/knowledge-pools/${id}`, {
            method: 'DELETE',
            headers: getAuthHeaders(accessToken),
          });

          if (!response.ok) {
            throw new Error('Failed to delete knowledge pool');
          }

          set(state => ({
            knowledgePools: state.knowledgePools.filter(p => p.id !== id),
            selectedPoolIds: state.selectedPoolIds.filter(pid => pid !== id),
            currentPoolId: state.currentPoolId === id ? null : state.currentPoolId,
            isLoading: false,
          }));

          toast.success('Knowledge pool deleted successfully');
          return true;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          set({
            error: errorMessage,
            isLoading: false
          });
          toast.error(`Failed to delete pool: ${errorMessage}`);
          return false;
        }
      },

      selectPool: (id: string) => {
        set(state => ({
          selectedPoolIds: state.selectedPoolIds.includes(id)
            ? state.selectedPoolIds
            : [...state.selectedPoolIds, id]
        }));
      },

      deselectPool: (id: string) => {
        set(state => ({
          selectedPoolIds: state.selectedPoolIds.filter(pid => pid !== id)
        }));
      },

      togglePoolSelection: (id: string) => {
        set(state => ({
          selectedPoolIds: state.selectedPoolIds.includes(id)
            ? state.selectedPoolIds.filter(pid => pid !== id)
            : [...state.selectedPoolIds, id]
        }));
      },

      clearSelectedPools: () => {
        set({ selectedPoolIds: [] });
      },

      // Document Actions
      loadDocuments: async (poolId: string) => {
        set({ isLoading: true, error: null });
        try {
          const accessToken = useAuthStore.getState().accessToken;
          const response = await fetch(`/api/rag/knowledge-pools/${poolId}/documents`, {
            headers: getAuthHeaders(accessToken),
          });
          if (!response.ok) {
            throw new Error('Failed to load documents');
          }
          const documents = await response.json();
          set({ documents, isLoading: false });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Unknown error',
            isLoading: false
          });
        }
      },

      uploadDocument: async (poolId: string, file: File) => {
        const uploadId = `${poolId}-${file.name}-${Date.now()}`;

        // Initialize upload progress
        set(state => ({
          uploads: {
            ...state.uploads,
            [uploadId]: {
              filename: file.name,
              progress: 0,
              status: 'uploading',
            },
          },
        }));

        try {
          const formData = new FormData();
          formData.append('file', file);

          const token = useAuthStore.getState().accessToken;
          const response = await fetch(`/api/rag/knowledge-pools/${poolId}/upload`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
            },
            body: formData,
          });

          if (!response.ok) {
            throw new Error('Upload failed');
          }

          const result = await response.json();

          // Update upload status
          set(state => ({
            uploads: {
              ...state.uploads,
              [uploadId]: {
                filename: file.name,
                progress: 100,
                status: 'processing',
              },
            },
          }));

          // Poll for processing completion
          const pollInterval = setInterval(async () => {
            const accessToken = useAuthStore.getState().accessToken;
            const docResponse = await fetch(`/api/rag/knowledge-pools/${poolId}/documents`, {
              headers: getAuthHeaders(accessToken),
            });
            if (docResponse.ok) {
              const documents = await docResponse.json();
              const uploadedDoc = documents.find((d: Document) => d.id === result.document_id);

              if (uploadedDoc) {
                if (uploadedDoc.status === 'completed') {
                  set(state => ({
                    uploads: {
                      ...state.uploads,
                      [uploadId]: {
                        filename: file.name,
                        progress: 100,
                        status: 'completed',
                      },
                    },
                    documents: state.currentPoolId === poolId ? documents : state.documents,
                  }));
                  clearInterval(pollInterval);
                  toast.success(`${file.name} uploaded and processed successfully`);

                  // Remove upload status after 3 seconds
                  setTimeout(() => {
                    set(state => {
                      const { [uploadId]: _, ...remainingUploads } = state.uploads;
                      return { uploads: remainingUploads };
                    });
                  }, 3000);
                } else if (uploadedDoc.status === 'failed') {
                  const errorMsg = uploadedDoc.error_message || 'Processing failed';
                  set(state => ({
                    uploads: {
                      ...state.uploads,
                      [uploadId]: {
                        filename: file.name,
                        progress: 100,
                        status: 'failed',
                        error: errorMsg,
                      },
                    },
                    documents: state.currentPoolId === poolId ? documents : state.documents,
                  }));
                  clearInterval(pollInterval);
                  toast.error(`Failed to process ${file.name}: ${errorMsg}`);
                }
              }
            }
          }, 2000); // Poll every 2 seconds

          // Stop polling after 5 minutes
          setTimeout(() => clearInterval(pollInterval), 5 * 60 * 1000);

        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Upload failed';
          set(state => ({
            uploads: {
              ...state.uploads,
              [uploadId]: {
                filename: file.name,
                progress: 0,
                status: 'failed',
                error: errorMessage,
              },
            },
          }));
          toast.error(`Failed to upload ${file.name}: ${errorMessage}`);
        }
      },

      deleteDocument: async (documentId: string) => {
        set({ isLoading: true, error: null });
        try {
          const accessToken = useAuthStore.getState().accessToken;
          const response = await fetch(`/api/rag/documents/${documentId}`, {
            method: 'DELETE',
            headers: getAuthHeaders(accessToken),
          });

          if (!response.ok) {
            throw new Error('Failed to delete document');
          }

          set(state => ({
            documents: state.documents.filter(d => d.id !== documentId),
            isLoading: false,
          }));

          toast.success('Document deleted successfully');
          return true;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          set({
            error: errorMessage,
            isLoading: false
          });
          toast.error(`Failed to delete document: ${errorMessage}`);
          return false;
        }
      },

      // UI Actions
      setCurrentPool: (poolId: string | null) => {
        set({ currentPoolId: poolId });
      },

      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'rag-storage',
      partialize: (state) => ({
        selectedPoolIds: state.selectedPoolIds,
      }),
    }
  )
);
