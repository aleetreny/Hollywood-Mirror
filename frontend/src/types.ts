export interface MovieResult {
  title: string;
  affinity: number;
}

export interface SimilarMoviesResponse {
  results: MovieResult[];
}

export interface SimilarMoviesRequest {
  text: string;
  model: 'mpnet' | 'minilm';
  k: number;
}
