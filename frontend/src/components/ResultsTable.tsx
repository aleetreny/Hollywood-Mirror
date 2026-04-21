import {AlertCircle, Clapperboard} from 'lucide-react';

import {formatMovieTitle} from '@/lib/format';
import {MovieResult} from '@/types';

interface ResultsTableProps {
  results: MovieResult[] | null;
  isLoading: boolean;
  error: string | null;
}

export function ResultsTable({results, isLoading, error}: ResultsTableProps) {
  if (isLoading) {
    return (
      <div className="flex min-h-[400px] flex-col items-center justify-center rounded-3xl border border-white/10 bg-zinc-900/80 p-8 text-zinc-400 shadow-xl">
        <div className="mb-4 h-10 w-10 animate-spin rounded-full border-4 border-emerald-500/30 border-t-emerald-500" />
        <p className="font-medium">Analyzing narrative embeddings...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[400px] flex-col items-center justify-center rounded-3xl border border-red-500/20 bg-red-500/10 p-8 text-center text-red-300 shadow-xl">
        <AlertCircle className="mb-4 h-12 w-12 opacity-80" />
        <h3 className="text-lg font-semibold mb-2">Search Failed</h3>
        <p className="max-w-md">{error}</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex min-h-[400px] flex-col items-center justify-center rounded-3xl border border-dashed border-white/10 bg-zinc-900/30 p-8 text-center text-zinc-500">
        <Clapperboard className="mb-4 h-12 w-12 opacity-50" />
        <p className="font-medium">Enter a script fragment to see similar movies</p>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="flex min-h-[400px] flex-col items-center justify-center rounded-3xl border border-white/10 bg-zinc-900/80 p-8 text-center text-zinc-400">
        <Clapperboard className="mb-4 h-12 w-12 opacity-50" />
        <p className="font-medium">No similar movies found.</p>
        <p className="text-sm mt-2 opacity-70">Try a different or longer description.</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-zinc-900/80 shadow-xl">
      <div className="border-b border-white/10 bg-zinc-900 px-6 py-4">
        <h2 className="text-lg font-semibold text-zinc-100">Top Matches</h2>
      </div>
      <div className="divide-y divide-white/5">
        {results.map((result, index) => {
          const affinity = Number.isFinite(result.affinity) ? result.affinity : 0;
          const affinityPercent = (affinity * 100).toFixed(1);
          const affinityBar = `${Math.max(0, Math.min(100, affinity * 100))}%`;
          const displayTitle = formatMovieTitle(result.title);
          
          return (
            <div
              key={`${result.title}-${index}`}
              className="flex flex-col gap-4 px-6 py-4 transition-colors hover:bg-white/5 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-full border border-white/5 bg-zinc-800 text-xs font-bold text-zinc-400">
                  {index + 1}
                </div>
                <span className="text-lg font-medium text-zinc-200">{displayTitle}</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="hidden h-2 w-24 overflow-hidden rounded-full bg-zinc-800 sm:block">
                  <div 
                    className="h-full rounded-full bg-emerald-500" 
                    style={{ width: affinityBar }}
                  />
                </div>
                <span className="w-16 text-right font-mono font-medium text-emerald-400">
                  {affinityPercent}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
