const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';

function detectDefaultApiBaseUrl(): string {
  if (rawApiBaseUrl) {
    return rawApiBaseUrl.replace(/\/+$/, '');
  }

  if (typeof window === 'undefined') {
    return 'http://localhost:8000';
  }

  const hostname = window.location.hostname;
  const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1';
  return isLocalhost ? 'http://localhost:8000' : '';
}

export const API_BASE_URL = detectDefaultApiBaseUrl();
export const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 45000);
