/// <reference types="vite/client" />
import {API_BASE_URL, REQUEST_TIMEOUT_MS} from '@/config';
import {
  SearchCapabilitiesResponse,
  SimilarMoviesRequest,
  SimilarMoviesResponse,
  WarmupResponse,
} from '@/types';

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: BodyInit | null;
  timeoutMs?: number;
}

function buildUrl(pathname: string): string {
  return API_BASE_URL ? `${API_BASE_URL}${pathname}` : pathname;
}

function describeAbort(timeoutMs: number): string {
  return `The request timed out after ${Math.round(timeoutMs / 1000)} seconds. The backend may still be starting.`;
}

async function readErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const payload = (await response.json()) as {detail?: string};
    if (payload.detail) {
      return payload.detail;
    }
  }
  return `Server error: ${response.status} ${response.statusText}`;
}

async function requestJson<T>(pathname: string, options: RequestOptions = {}): Promise<T> {
  const {timeoutMs = REQUEST_TIMEOUT_MS, signal, ...init} = options;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const abortHandler = () => controller.abort();

  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener('abort', abortHandler, {once: true});
    }
  }

  let response: Response;
  try {
    response = await fetch(buildUrl(pathname), {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(describeAbort(timeoutMs));
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener('abort', abortHandler);
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<T>;
}

export function fetchSearchCapabilities(signal?: AbortSignal): Promise<SearchCapabilitiesResponse> {
  return requestJson<SearchCapabilitiesResponse>('/api/capabilities', {
    method: 'GET',
    signal,
    timeoutMs: 90000,
  });
}

export function warmupBackend(model: SimilarMoviesRequest['model'], signal?: AbortSignal): Promise<WarmupResponse> {
  return requestJson<WarmupResponse>('/api/warmup', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({model}),
    signal,
    timeoutMs: 90000,
  });
}

export function fetchSimilarMovies(
  request: SimilarMoviesRequest,
  signal?: AbortSignal,
): Promise<SimilarMoviesResponse> {
  return requestJson<SimilarMoviesResponse>('/api/similar-movies', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
    signal,
  });
}
