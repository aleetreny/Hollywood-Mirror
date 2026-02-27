import React, { useState } from "react";
import { Search } from "lucide-react";
import { SimilarMoviesRequest } from "../types";

interface SearchFormProps {
  onSearch: (request: SimilarMoviesRequest) => void;
  isLoading: boolean;
}

export function SearchForm({ onSearch, isLoading }: SearchFormProps) {
  const [text, setText] = useState("");
  const [model, setModel] = useState<"mpnet" | "minilm">("minilm");
  const [k, setK] = useState<number>(5);
  const [error, setError] = useState<string | null>(null);
  const K_MIN = 1;
  const K_MAX = 20;

  const normalizeK = (value: number): number => {
    if (!Number.isFinite(value)) return 5;
    return Math.max(K_MIN, Math.min(K_MAX, Math.floor(value)));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!text.trim()) {
      setError("Please enter a movie idea or script fragment.");
      return;
    }

    const clampedK = normalizeK(k);
    setK(clampedK);

    onSearch({ text: text.trim(), model, k: clampedK });
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-zinc-900/80 border border-white/10 rounded-2xl p-6 shadow-xl flex flex-col gap-6 h-[calc(100vh-8rem)]"
    >
      <div className="flex flex-col gap-2 flex-1">
        <label htmlFor="text" className="text-sm font-medium text-zinc-300">
          Your Idea or Script Fragment
        </label>
        <textarea
          id="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="A detective with insomnia tries to solve a murder in a city where the sun never sets..."
          className="w-full flex-1 bg-zinc-950 border border-white/10 rounded-xl p-4 text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 resize-none transition-all"
          disabled={isLoading}
        />
        {error && <p className="text-sm text-red-400 mt-1">{error}</p>}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="flex flex-col gap-2">
          <label htmlFor="model" className="text-sm font-medium text-zinc-300">
            Embedding Model
          </label>
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value as "mpnet" | "minilm")}
            className="w-full bg-zinc-950 border border-white/10 rounded-xl p-3 text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all appearance-none"
            disabled={isLoading}
          >
            <option value="minilm">minilm (all-MiniLM-L6-v2)</option>
            <option value="mpnet">mpnet (all-mpnet-base-v2)</option>
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
              min={K_MIN}
              max={K_MAX}
              value={k}
              onChange={(e) => setK(normalizeK(Number(e.target.value)))}
              className="flex-1 accent-emerald-500"
              disabled={isLoading}
            />
            <input
              type="number"
              id="k"
              min={K_MIN}
              max={K_MAX}
              value={k}
              onChange={(e) => setK(normalizeK(Number(e.target.value)))}
              className="w-16 bg-zinc-950 border border-white/10 rounded-xl p-2 text-center text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all"
              disabled={isLoading}
            />
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="mt-2 w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : (
          <Search className="w-5 h-5" />
        )}
        {isLoading ? "Searching..." : "Find similar movies"}
      </button>
    </form>
  );
}
