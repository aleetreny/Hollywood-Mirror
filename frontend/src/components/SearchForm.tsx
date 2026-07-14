import type {ChangeEvent, FormEvent} from 'react';
import {useState} from 'react';
import {Cpu, LoaderCircle, LockKeyhole, Search} from 'lucide-react';

import type {EngineStatus, SimilarMoviesRequest} from '@/types';

interface SearchFormProps {
  engineStatus: EngineStatus;
  isLoading: boolean;
  onSearch: (request: SimilarMoviesRequest) => void;
}

export function SearchForm({engineStatus, isLoading, onSearch}: SearchFormProps) {
  const [text, setText] = useState('');
  const [k, setK] = useState(5);
  const [error, setError] = useState<string | null>(null);
  const minK = 1;
  const maxK = 20;

  const normalizeK = (value: number): number => {
    if (!Number.isFinite(value)) return 5;
    return Math.max(minK, Math.min(maxK, Math.floor(value)));
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (!text.trim()) {
      setError('Please enter a movie idea or script fragment.');
      return;
    }

    if (engineStatus.state === 'error') {
      setError(engineStatus.message);
      return;
    }

    const clampedK = normalizeK(k);
    setK(clampedK);
    onSearch({text: text.trim(), k: clampedK});
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
          onChange={(event) => setText(event.target.value)}
          placeholder="A detective with insomnia tries to solve a murder in a city where the sun never sets..."
          className="min-h-[240px] flex-1 resize-none rounded-2xl border border-white/10 bg-zinc-950 p-4 text-zinc-100 transition-all placeholder:text-zinc-600 focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          disabled={isLoading}
        />
        <div className="flex items-center justify-between gap-4 text-xs text-zinc-500">
          <span>{text.trim().split(/\s+/).filter(Boolean).length} words</span>
          <span>Longer prompts usually produce better matches.</span>
        </div>
        {error ? <p className="mt-1 text-sm text-red-400">{error}</p> : null}
      </div>

      <div className="rounded-2xl border border-emerald-400/15 bg-emerald-500/5 p-4">
        <div className="flex items-start gap-3">
          {engineStatus.state === 'loading' ? (
            <LoaderCircle className="mt-0.5 h-5 w-5 shrink-0 animate-spin text-emerald-400" />
          ) : engineStatus.state === 'ready' ? (
            <Cpu className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
          ) : (
            <LockKeyhole className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
          )}
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-zinc-200">{engineStatus.message}</p>
            <p className="mt-1 text-xs leading-relaxed text-zinc-500">
              MiniLM runs on your device. Your text is never sent to our server, and the model
              stays cached after its first download.
            </p>
            {engineStatus.state === 'loading' && engineStatus.progress !== null ? (
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-emerald-500 transition-[width]"
                  style={{width: `${engineStatus.progress}%`}}
                />
              </div>
            ) : null}
          </div>
        </div>
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
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              setK(normalizeK(Number(event.target.value)))
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
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              setK(normalizeK(Number(event.target.value)))
            }
            className="w-20 rounded-2xl border border-white/10 bg-zinc-950 p-2 text-center text-zinc-100 transition-all focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            disabled={isLoading}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading || engineStatus.state === 'error'}
        className="mt-2 flex w-full items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-4 py-3 font-medium text-white transition-colors hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? (
          <LoaderCircle className="h-5 w-5 animate-spin" />
        ) : (
          <Search className="h-5 w-5" />
        )}
        {isLoading
          ? engineStatus.state === 'loading'
            ? 'Preparing local model...'
            : 'Searching...'
          : 'Find similar movies'}
      </button>
    </form>
  );
}
