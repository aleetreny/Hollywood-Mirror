export type ModelId = 'mpnet' | 'minilm';

export interface MovieResult {
  title: string;
  affinity: number;
}

export interface SimilarMoviesResponse {
  results: MovieResult[];
}

export interface SimilarMoviesRequest {
  text: string;
  model: ModelId;
  k: number;
}

export interface ModelCapability {
  id: ModelId;
  dimension: number;
  available: boolean;
}

export interface SearchCapabilitiesResponse {
  default_model: ModelId;
  max_k: number;
  models: ModelCapability[];
}

export interface WarmupResponse {
  status: string;
  model: ModelId;
  loaded_now: boolean;
  loaded_models: ModelId[];
  elapsed_ms: number;
}

export const FALLBACK_CAPABILITIES: SearchCapabilitiesResponse = {
  default_model: 'minilm',
  max_k: 50,
  models: [
    {id: 'minilm', dimension: 384, available: false},
    {id: 'mpnet', dimension: 768, available: false},
  ],
};
