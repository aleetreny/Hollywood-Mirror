import type {ReactNode} from 'react';

import {Film, Github} from 'lucide-react';

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
              Private semantic movie search in your browser
            </p>
          </div>
          <a
            className="ml-auto flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-zinc-900/70 text-zinc-300 transition hover:border-emerald-400/40 hover:text-white"
            href="https://github.com/aleetreny/Hollywood-Mirror"
            target="_blank"
            rel="noreferrer"
            aria-label="Open Hollywood Mirror repository on GitHub"
          >
            <Github className="h-5 w-5" />
          </a>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        {children}
      </main>
    </div>
  );
}
