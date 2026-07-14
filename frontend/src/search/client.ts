import type {EngineStatus, MovieResult, SimilarMoviesRequest} from '@/types';
import type {WorkerRequest, WorkerResponse} from './protocol';

type StatusListener = (status: EngineStatus) => void;

interface PendingSearch {
  resolve: (results: MovieResult[]) => void;
  reject: (error: Error) => void;
}

const worker = new Worker(new URL('./search.worker.ts', import.meta.url), {type: 'module'});
const listeners = new Set<StatusListener>();
const pending = new Map<number, PendingSearch>();
let nextRequestId = 1;
let initializationPromise: Promise<void> | null = null;
let resolveInitialization: (() => void) | null = null;
let rejectInitialization: ((error: Error) => void) | null = null;
let currentStatus: EngineStatus = {
  state: 'loading',
  message: 'Starting the private browser search engine…',
  progress: null,
};

worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
  const message = event.data;

  if (message.type === 'status') {
    publish({state: 'loading', message: message.message, progress: message.progress});
    return;
  }

  if (message.type === 'ready') {
    publish({state: 'ready', message: 'Private local search ready', progress: 100});
    resolveInitialization?.();
    resolveInitialization = null;
    rejectInitialization = null;
    return;
  }

  if (message.type === 'results') {
    const request = pending.get(message.requestId);
    if (!request) return;
    pending.delete(message.requestId);
    request.resolve(message.results);
    return;
  }

  const error = new Error(message.message);
  if (message.requestId !== undefined) {
    const request = pending.get(message.requestId);
    if (request) {
      pending.delete(message.requestId);
      request.reject(error);
    }
  } else {
    publish({state: 'error', message: message.message, progress: null});
    rejectInitialization?.(error);
    resolveInitialization = null;
    rejectInitialization = null;
  }
};

worker.onerror = (event) => {
  const error = new Error(event.message || 'The browser search worker crashed.');
  publish({state: 'error', message: error.message, progress: null});
  rejectInitialization?.(error);
  for (const request of pending.values()) request.reject(error);
  pending.clear();
};

function publish(status: EngineStatus): void {
  currentStatus = status;
  for (const listener of listeners) listener(status);
}

export function subscribeToEngineStatus(listener: StatusListener): () => void {
  listeners.add(listener);
  listener(currentStatus);
  return () => listeners.delete(listener);
}

export function initializeSearchEngine(): Promise<void> {
  if (!initializationPromise) {
    initializationPromise = new Promise<void>((resolve, reject) => {
      resolveInitialization = resolve;
      rejectInitialization = reject;
      const message: WorkerRequest = {type: 'initialize'};
      worker.postMessage(message);
    });
  }
  return initializationPromise;
}

export async function searchMovies(request: SimilarMoviesRequest): Promise<MovieResult[]> {
  await initializeSearchEngine();
  const requestId = nextRequestId;
  nextRequestId += 1;

  return new Promise<MovieResult[]>((resolve, reject) => {
    pending.set(requestId, {resolve, reject});
    const message: WorkerRequest = {
      type: 'search',
      requestId,
      text: request.text,
      k: request.k,
    };
    worker.postMessage(message);
  });
}
