import {startTransition, useEffect, useState} from 'react';

import {Layout} from '@/components/Layout';
import {ResultsTable} from '@/components/ResultsTable';
import {SearchForm} from '@/components/SearchForm';
import {initializeSearchEngine, searchMovies, subscribeToEngineStatus} from '@/search/client';
import type {EngineStatus, MovieResult, SimilarMoviesRequest} from '@/types';

const INITIAL_STATUS: EngineStatus = {
  state: 'loading',
  message: 'Starting the private browser search engine…',
  progress: null,
};

export default function App() {
  const [engineStatus, setEngineStatus] = useState<EngineStatus>(INITIAL_STATUS);
  const [results, setResults] = useState<MovieResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = subscribeToEngineStatus(setEngineStatus);
    const start = () => void initializeSearchEngine().catch(() => undefined);
    const requestIdle = window.requestIdleCallback;

    if (typeof requestIdle === 'function') {
      const id = requestIdle(start, {timeout: 1200});
      return () => {
        window.cancelIdleCallback(id);
        unsubscribe();
      };
    }

    const id = globalThis.setTimeout(start, 0);
    return () => {
      globalThis.clearTimeout(id);
      unsubscribe();
    };
  }, []);

  const handleSearch = async (request: SimilarMoviesRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const nextResults = await searchMovies(request);
      startTransition(() => setResults(nextResults));
    } catch (searchError: unknown) {
      setError(
        searchError instanceof Error
          ? searchError.message
          : 'An unexpected local search error occurred.',
      );
      startTransition(() => setResults(null));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        <div className="lg:col-span-6 xl:col-span-5">
          <SearchForm
            engineStatus={engineStatus}
            isLoading={isLoading}
            onSearch={handleSearch}
          />
        </div>
        <div className="lg:col-span-6 xl:col-span-7">
          <ResultsTable error={error} isLoading={isLoading} results={results} />
        </div>
      </div>
    </Layout>
  );
}
