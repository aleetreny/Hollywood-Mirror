import React from 'react';
import { Film } from 'lucide-react';

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500/30">
      <header className="border-b border-white/10 bg-zinc-900/50 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center gap-3">
          <Film className="w-6 h-6 text-emerald-500" />
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-white leading-tight">Hollywood Mirror</h1>
            <p className="text-xs text-zinc-400 font-medium">Find movies with a similar narrative style</p>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
}
