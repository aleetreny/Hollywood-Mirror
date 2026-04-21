import type {ChangeEvent, FormEvent} from 'react';
import {useEffect, useState} from 'react';
import {LoaderCircle, Search} from 'lucide-react';

import {ModelCapability, SearchCapabilitiesResponse, SimilarMoviesRequest} from '@/types';

interface SearchFormProps {
  capabilities: SearchCapabilitiesResponse;
  capabilitiesError: string | null;
  isBackendWarming: boolean;
  isLoading: boolean;
  onSearch: (request: SimilarMoviesRequest) => void;
}

function getEnabledModels(models: ModelCapability[]): ModelCapability[] {
  return models.filter((model) => model.available);
}

export function SearchForm({
  capabilities,
  capabilitiesError,
  isBackendWarming,
  isLoading,
  onSearch,
}: SearchFormProps) {
  const [text, setText] = useState('');
  const [model, setModel] = useState(capabilities.default_model);
  const [k, setK] = useState<number>(5);
  const [error, setError] = useState<string | null>(null);
  const enabledModels = getEnabledModels(capabilities.models);
  const minK = 1;
  const maxK = capabilities.max_k;
  const activeModel = enabledModels.find((item) => item.id === model) ?? enabledModels[0];

  const normalizeK = (value: number): number => {
    if (!Number.isFinite(value)) return 5;
    return Math.max(minK, Math.min(maxK, Math.floor(value)));
  };

  useEffect(() => {
    setK((currentK) => normalizeK(currentK));
    if (!activeModel) {
      return;
    }
    setModel((currentModel) => (currentModel === activeModel.id ? currentModel : activeModel.id));
  }, [activeModel, maxK]);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);

    if (!text.trim()) {
      setError('Please enter a movie idea or script fragment.');
      return;
    }

    if (!activeModel) {
      setError('No embedding model is available on the backend right now.');
      return;
    }

    const clampedK = normalizeK(k);
    setK(clampedK);

    onSearch({text: text.trim(), model: activeModel.id, k: clampedK});
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex min-h-[560px] flex-col gap-6 rounded-3xl border border-white/10 bg-zinc-900/80 p-6 shadow-xl lg:sticky lg:top-24"
    >
      <div className="flex flex-1 flex-col gap-2">
        <label htmlFor="text" className="text-sm font-medium text-zinc-300">
          Your Idea or Script Fragment
        </label>
        <textarea
          id="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="A detective with insomnia tries to solve a murder in a city where the sun never sets..."
          className="min-h-[240px] flex-1 resize-none rounded-2xl border border-white/10 bg-zinc-950 p-4 text-zinc-100 transition-all placeholder:text-zinc-600 focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          disabled={isLoading}
        />
        <div className="flex items-center justify-between gap-4 text-xs text-zinc-500">
          <span>{text.trim().split(/\s+/).filter(Boolean).length} words</span>
          <span>Longer prompts usually produce better matches.</span>
        </div>
        {capabilitiesError ? (
          <p className="mt-1 text-sm text-red-400">{capabilitiesError}</p>
        ) : null}
        {error ? <p className="mt-1 text-sm text-red-400">{error}</p> : null}
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <label htmlFor="model" className="text-sm font-medium text-zinc-300">
            Embedding Model
          </label>
          <select
            id="model"
            value={model}
            onChange={(e: ChangeEvent<HTMLSelectElement>) =>
              setModel(e.target.value as SimilarMoviesRequest['model'])
            }
            className="w-full appearance-none rounded-2xl border border-white/10 bg-zinc-950 p-3 text-zinc-100 transition-all focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            disabled={isLoading || enabledModels.length === 0}
          >
            {capabilities.models.map((option) => (
              <option key={option.id} value={option.id} disabled={!option.available}>
                {option.id} ({option.dimension} dims){option.available ? '' : ' - unavailable'}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="k" className="text-sm font-medium text-zinc-300">
            Number of Results (Top K)
          </label>
          <div className="flex items-center gap-4">
            <input
              type="range"
              id="k-slider"
              min={minK}
              max={maxK}
              value={k}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setK(normalizeK(Number(e.target.value)))
              }
              className="flex-1 accent-emerald-500"
              disabled={isLoading}
            />
            <input
              type="number"
              id="k"
              min={minK}
              max={maxK}
              value={k}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setK(normalizeK(Number(e.target.value)))
              }
              className="w-20 rounded-2xl border border-white/10 bg-zinc-950 p-2 text-center text-zinc-100 transition-all focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
              disabled={isLoading}
            />
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading || isBackendWarming || !activeModel}
        className="mt-2 flex w-full items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-4 py-3 font-medium text-white transition-colors hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading || isBackendWarming ? (
          <LoaderCircle className="h-5 w-5 animate-spin" />
        ) : (
          <Search className="h-5 w-5" />
        )}
        {isLoading
          ? 'Searching...'
          : isBackendWarming
            ? 'Preparing search engine...'
            : 'Find similar movies'}
      </button>
    </form>
  );
}
