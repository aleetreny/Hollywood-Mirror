/// <reference types="vite/client" />
import { SimilarMoviesRequest, SimilarMoviesResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchSimilarMovies(request: SimilarMoviesRequest): Promise<SimilarMoviesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/similar-movies`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}
