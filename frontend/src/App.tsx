import React, { useState } from 'react';
import { Layout } from './components/Layout';
import { SearchForm } from './components/SearchForm';
import { ResultsTable } from './components/ResultsTable';
import { fetchSimilarMovies } from './services/api';
import { SimilarMoviesRequest, MovieResult } from './types';

export default function App() {
  const [results, setResults] = useState<MovieResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (request: SimilarMoviesRequest) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetchSimilarMovies(request);
      setResults(response.results);
    } catch (err) {
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred while fetching results.');
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-6 xl:col-span-5">
          <div className="sticky top-24">
            <SearchForm onSearch={handleSearch} isLoading={isLoading} />
          </div>
        </div>
        <div className="lg:col-span-6 xl:col-span-7">
          <ResultsTable results={results} isLoading={isLoading} error={error} />
        </div>
      </div>
    </Layout>
  );
}
