import type {MovieResult} from '@/types';

export type WorkerRequest =
  | {type: 'initialize'}
  | {type: 'search'; requestId: number; text: string; k: number};

export type WorkerResponse =
  | {type: 'status'; message: string; progress: number | null}
  | {type: 'ready'}
  | {type: 'results'; requestId: number; results: MovieResult[]}
  | {type: 'error'; requestId?: number; message: string};
