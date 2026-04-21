import type {ReactNode} from 'react';

import {Film} from 'lucide-react';

export function Layout({children}: {children: ReactNode}) {
  return (
    <div className="min-h-screen font-sans text-zinc-100 selection:bg-emerald-500/30">
      <header className="sticky top-0 z-20 border-b border-white/10 bg-zinc-950/75 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6 lg:px-8">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-emerald-400/25 bg-emerald-500/10 shadow-[0_0_32px_rgba(16,185,129,0.15)]">
            <Film className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-xl font-semibold leading-tight tracking-tight text-white">
              Hollywood Mirror
            </h1>
            <p className="text-xs font-medium text-zinc-400">
              Semantic movie search over screenplay embeddings
            </p>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        <section className="mb-8 rounded-3xl border border-white/10 bg-white/[0.03] px-6 py-6 shadow-2xl shadow-black/25 backdrop-blur-sm">
          <p className="max-w-3xl text-sm leading-6 text-zinc-300 sm:text-base">
            Paste a film premise, a scene fragment, or a narrative idea and compare it
            against the screenplay embedding archive. Choose a faster or richer model
            and inspect the closest matches by cosine affinity.
          </p>
        </section>
        {children}
      </main>
    </div>
  );
}
