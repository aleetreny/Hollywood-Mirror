import {startTransition, useEffect, useRef, useState} from 'react';

import {Layout} from '@/components/Layout';
import {ResultsTable} from '@/components/ResultsTable';
import {SearchForm} from '@/components/SearchForm';
import {fetchSearchCapabilities, fetchSimilarMovies, warmupBackend} from '@/services/api';
import {
  FALLBACK_CAPABILITIES,
  MovieResult,
  SearchCapabilitiesResponse,
  SimilarMoviesRequest,
} from '@/types';

export default function App() {
  const [capabilities, setCapabilities] = useState<SearchCapabilitiesResponse>(FALLBACK_CAPABILITIES);
  const [capabilitiesError, setCapabilitiesError] = useState<string | null>(null);
  const [isBackendWarming, setIsBackendWarming] = useState(true);
  const [results, setResults] = useState<MovieResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeRequestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const defaultModel = FALLBACK_CAPABILITIES.default_model;

    const loadCapabilities = async () => {
      try {
        const response = await fetchSearchCapabilities(controller.signal);
        if (controller.signal.aborted) {
          return;
        }
        startTransition(() => {
          setCapabilities(response);
          setCapabilitiesError(null);
        });
      } catch (requestError: unknown) {
        if (controller.signal.aborted) {
          return;
        }
        const message =
          requestError instanceof Error
            ? requestError.message
            : 'The app is using fallback frontend defaults.';
        setCapabilitiesError(message);
      }
    };

    const warmup = async () => {
      try {
        const response = await warmupBackend(defaultModel, controller.signal);
        if (controller.signal.aborted) {
          return;
        }
        startTransition(() => {
          setCapabilities((current) => ({
            ...current,
            default_model: response.model,
            models: current.models.map((model) =>
              model.id === response.model ? {...model, available: true} : model
            ),
          }));
          setCapabilitiesError(null);
        });
      } catch (requestError: unknown) {
        if (controller.signal.aborted) {
          return;
        }
        console.error('Warmup error:', requestError);
      } finally {
        if (!controller.signal.aborted) {
          setIsBackendWarming(false);
        }
      }
    };

    void loadCapabilities();
    void warmup();

    return () => controller.abort();
  }, []);

  const handleSearch = async (request: SimilarMoviesRequest) => {
    activeRequestRef.current?.abort();
    const controller = new AbortController();
    activeRequestRef.current = controller;
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetchSimilarMovies(request, controller.signal);
      if (activeRequestRef.current !== controller) {
        return;
      }
      startTransition(() => {
        setResults(response.results);
      });
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred while fetching results.');
      startTransition(() => {
        setResults(null);
      });
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null;
        setIsLoading(false);
      }
    }
  };

  useEffect(() => () => activeRequestRef.current?.abort(), []);

  return (
    <Layout>
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        <div className="lg:col-span-6 xl:col-span-5">
          <SearchForm
            capabilities={capabilities}
            capabilitiesError={capabilitiesError}
            isBackendWarming={isBackendWarming}
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
