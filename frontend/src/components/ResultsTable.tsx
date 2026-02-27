import React from 'react';
import { MovieResult } from '../types';
import { Clapperboard, AlertCircle } from 'lucide-react';

interface ResultsTableProps {
  results: MovieResult[] | null;
  isLoading: boolean;
  error: string | null;
}

export function ResultsTable({ results, isLoading, error }: ResultsTableProps) {
  if (isLoading) {
    return (
      <div className="bg-zinc-900/80 border border-white/10 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px] text-zinc-400">
        <div className="w-10 h-10 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-4" />
        <p className="font-medium">Analyzing narrative embeddings...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px] text-red-400 text-center">
        <AlertCircle className="w-12 h-12 mb-4 opacity-80" />
        <h3 className="text-lg font-semibold mb-2">Search Failed</h3>
        <p className="max-w-md">{error}</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-zinc-900/30 border border-white/5 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px] text-zinc-500 text-center border-dashed">
        <Clapperboard className="w-12 h-12 mb-4 opacity-50" />
        <p className="font-medium">Enter a script fragment to see similar movies</p>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="bg-zinc-900/80 border border-white/10 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px] text-zinc-400 text-center">
        <Clapperboard className="w-12 h-12 mb-4 opacity-50" />
        <p className="font-medium">No similar movies found.</p>
        <p className="text-sm mt-2 opacity-70">Try a different or longer description.</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900/80 border border-white/10 rounded-2xl overflow-hidden shadow-xl">
      <div className="px-6 py-4 border-b border-white/10 bg-zinc-900">
        <h2 className="text-lg font-semibold text-zinc-100">Top Matches</h2>
      </div>
      <div className="divide-y divide-white/5">
        {results.map((result, index) => {
          const affinity = Number.isFinite(result.affinity) ? result.affinity : 0;
          const affinityPercent = (affinity * 100).toFixed(1);
          const affinityBar = `${Math.max(0, Math.min(100, affinity * 100))}%`;
          // Clean up the title if it has an ID appended like "Inception_1375666"
          const displayTitle = result.title.replace(/_\d+$/, '').replace(/_/g, ' ');
          
          return (
            <div key={`${result.title}-${index}`} className="px-6 py-4 flex items-center justify-between hover:bg-white/5 transition-colors">
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-400 border border-white/5">
                  {index + 1}
                </div>
                <span className="font-medium text-zinc-200 text-lg">{displayTitle}</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-24 h-2 bg-zinc-800 rounded-full overflow-hidden hidden sm:block">
                  <div 
                    className="h-full bg-emerald-500 rounded-full" 
                    style={{ width: affinityBar }}
                  />
                </div>
                <span className="font-mono text-emerald-400 font-medium w-16 text-right">
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
