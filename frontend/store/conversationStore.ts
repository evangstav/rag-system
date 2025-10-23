import { create } from 'zustand';
import { useAuthStore } from '@/lib/auth-store';

export interface Conversation {
  id: string;
  user_id: string;
  title: string | null;
  use_rag: boolean;
  use_scratchpad: boolean;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

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

interface ConversationState {
  // State
  conversations: Conversation[];
  currentConversationId: string | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  setCurrentConversation: (id: string | null) => void;
  loadConversations: () => Promise<void>;
  createConversation: (title?: string) => Promise<string | null>;
  deleteConversation: (id: string) => Promise<void>;
  updateConversation: (id: string, updates: Partial<Conversation>) => Promise<void>;
  clearError: () => void;
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  // Initial state
  conversations: [],
  currentConversationId: null,
  isLoading: false,
  error: null,

  // Set current conversation
  setCurrentConversation: (id) => {
    set({ currentConversationId: id });
  },

  // Load all conversations
  loadConversations: async () => {
    set({ isLoading: true, error: null });
    try {
      const accessToken = useAuthStore.getState().accessToken;
      const response = await fetch('/api/conversations', {
        headers: getAuthHeaders(accessToken),
      });
      if (!response.ok) {
        throw new Error('Failed to load conversations');
      }
      const conversations = await response.json();
      set({ conversations, isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      });
    }
  },

  // Create a new conversation
  createConversation: async (title) => {
    set({ isLoading: true, error: null });
    try {
      const accessToken = useAuthStore.getState().accessToken;
      const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: getAuthHeaders(accessToken),
        body: JSON.stringify({ title }),
      });

      if (!response.ok) {
        throw new Error('Failed to create conversation');
      }

      const newConversation = await response.json();

      // Add to conversations list and set as current
      set((state) => ({
        conversations: [newConversation, ...state.conversations],
        currentConversationId: newConversation.id,
        isLoading: false,
      }));

      return newConversation.id;
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      });
      return null;
    }
  },

  // Delete a conversation
  deleteConversation: async (id) => {
    set({ isLoading: true, error: null });
    try {
      const accessToken = useAuthStore.getState().accessToken;
      const response = await fetch(`/api/conversations/${id}`, {
        method: 'DELETE',
        headers: getAuthHeaders(accessToken),
      });

      if (!response.ok) {
        throw new Error('Failed to delete conversation');
      }

      // Remove from conversations list
      set((state) => ({
        conversations: state.conversations.filter((c) => c.id !== id),
        // If deleted conversation was current, clear current
        currentConversationId:
          state.currentConversationId === id ? null : state.currentConversationId,
        isLoading: false,
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      });
    }
  },

  // Update a conversation (e.g., change title)
  updateConversation: async (id, updates) => {
    set({ isLoading: true, error: null });
    try {
      const accessToken = useAuthStore.getState().accessToken;
      const response = await fetch(`/api/conversations/${id}`, {
        method: 'PATCH',
        headers: getAuthHeaders(accessToken),
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error('Failed to update conversation');
      }

      const updatedConversation = await response.json();

      // Update in conversations list
      set((state) => ({
        conversations: state.conversations.map((c) =>
          c.id === id ? updatedConversation : c
        ),
        isLoading: false,
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      });
    }
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },
}));
