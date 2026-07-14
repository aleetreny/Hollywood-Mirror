export interface MovieResult {
  title: string;
  affinity: number;
}

export interface SimilarMoviesRequest {
  text: string;
  k: number;
}

export type EngineState = 'loading' | 'ready' | 'error';

export interface EngineStatus {
  state: EngineState;
  message: string;
  progress: number | null;
}
