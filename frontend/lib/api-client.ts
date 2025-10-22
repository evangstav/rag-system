/**
 * API client for making authenticated requests to the backend
 */

import { useAuthStore } from './auth-store';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface RequestOptions extends RequestInit {
  requireAuth?: boolean;
}

/**
 * Make an authenticated API request
 */
export async function apiRequest<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { requireAuth = true, ...fetchOptions } = options;

  // Get access token from store
  const accessToken = useAuthStore.getState().accessToken;

  // Build headers
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  // Add authorization header if authenticated
  if (requireAuth && accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  // Make request
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...fetchOptions,
    headers,
  });

  // Handle unauthorized (token expired)
  if (response.status === 401 && requireAuth) {
    // Try to refresh token
    const refreshed = await refreshAccessToken();

    if (refreshed) {
      // Retry original request with new token
      const newAccessToken = useAuthStore.getState().accessToken;
      headers['Authorization'] = `Bearer ${newAccessToken}`;

      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...fetchOptions,
        headers,
      });

      if (!retryResponse.ok) {
        const error = await retryResponse.json().catch(() => ({}));
        throw new Error(error.detail || 'Request failed');
      }

      return retryResponse.json();
    } else {
      // Refresh failed, logout user
      useAuthStore.getState().logout();
      window.location.href = '/login';
      throw new Error('Session expired. Please login again.');
    }
  }

  // Handle other errors
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Refresh the access token using the refresh token
 */
async function refreshAccessToken(): Promise<boolean> {
  try {
    const refreshToken = useAuthStore.getState().refreshToken;

    if (!refreshToken) {
      return false;
    }

    const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      return false;
    }

    const data = await response.json();
    useAuthStore.getState().updateAccessToken(data.access_token);

    return true;
  } catch (error) {
    console.error('Token refresh failed:', error);
    return false;
  }
}

/**
 * Login with email and password
 */
export async function login(email: string, password: string) {
  const response = await apiRequest<{
    access_token: string;
    refresh_token: string;
    user: any;
  }>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
    requireAuth: false,
  });

  // Save tokens and user to store
  useAuthStore.getState().login(
    response.access_token,
    response.refresh_token,
    response.user
  );

  return response;
}

/**
 * Register a new user
 */
export async function register(email: string, username: string, password: string) {
  const response = await apiRequest<{
    access_token: string;
    refresh_token: string;
    user: any;
  }>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, username, password }),
    requireAuth: false,
  });

  // Save tokens and user to store
  useAuthStore.getState().login(
    response.access_token,
    response.refresh_token,
    response.user
  );

  return response;
}

/**
 * Logout the current user
 */
export function logout() {
  useAuthStore.getState().logout();
  window.location.href = '/login';
}
