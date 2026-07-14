/// <reference lib="webworker" />

import {env, pipeline} from '@huggingface/transformers';
import type {FeatureExtractionPipelineType} from '@huggingface/transformers';

import type {MovieResult} from '@/types';
import type {WorkerRequest, WorkerResponse} from './protocol';

const MODEL_ID = 'onnx-community/all-MiniLM-L6-v2-ONNX';
const DATA_VERSION = 'c8783c29d15fc7de388e5ae6d6a1167ea4ebeb99';
const DATA_BASE_URL = `https://cdn.jsdelivr.net/gh/aleetreny/Hollywood-Mirror@${DATA_VERSION}/data/processed`;
const EMBEDDINGS_URL = `${DATA_BASE_URL}/movie_embeddings_minilm.npy`;
const TITLES_URL = `${DATA_BASE_URL}/movie_embeddings_minilm.txt`;
const DIMENSION = 384;

type Extractor = FeatureExtractionPipelineType;

const createFeatureExtractor = pipeline as unknown as (
  task: 'feature-extraction',
  model: string,
  options: {dtype: 'q8'; progress_callback: (value: unknown) => void},
) => Promise<Extractor>;

interface MovieIndex {
  matrix: Float32Array;
  titles: string[];
  rows: number;
  columns: number;
}

interface NpyArray {
  data: Float32Array;
  shape: number[];
}

let extractorPromise: Promise<Extractor> | null = null;
let indexPromise: Promise<MovieIndex> | null = null;
let initializationPromise: Promise<void> | null = null;

env.allowLocalModels = false;

theGlobal().onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  if (message.type === 'initialize') {
    void initialize().catch((error: unknown) => {
      post({type: 'error', message: describeError(error)});
    });
    return;
  }

  void search(message.requestId, message.text, message.k).catch((error: unknown) => {
    post({type: 'error', requestId: message.requestId, message: describeError(error)});
  });
};

function theGlobal(): DedicatedWorkerGlobalScope {
  return self as unknown as DedicatedWorkerGlobalScope;
}

function post(message: WorkerResponse): void {
  theGlobal().postMessage(message);
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : 'The local search engine failed unexpectedly.';
}

function report(message: string, progress: number | null = null): void {
  post({type: 'status', message, progress});
}

function parseProgress(value: unknown): number | null {
  if (!value || typeof value !== 'object') return null;
  const payload = value as Record<string, unknown>;
  const progress = payload.progress;
  if (typeof progress === 'number' && Number.isFinite(progress)) {
    return Math.max(0, Math.min(100, progress));
  }
  const loaded = payload.loaded;
  const total = payload.total;
  if (typeof loaded === 'number' && typeof total === 'number' && total > 0) {
    return Math.max(0, Math.min(100, (loaded / total) * 100));
  }
  return null;
}

function parseProgressLabel(value: unknown): string {
  if (!value || typeof value !== 'object') return 'Downloading local AI model…';
  const payload = value as Record<string, unknown>;
  const file = typeof payload.file === 'string' ? payload.file.split('/').at(-1) : null;
  const status = typeof payload.status === 'string' ? payload.status : null;
  if (status === 'ready') return 'Local AI model ready';
  if (file) return `Downloading ${file}…`;
  return 'Downloading local AI model…';
}

async function loadExtractor(): Promise<Extractor> {
  if (!extractorPromise) {
    extractorPromise = createFeatureExtractor('feature-extraction', MODEL_ID, {
      dtype: 'q8',
      progress_callback: (value: unknown) => {
        report(parseProgressLabel(value), parseProgress(value));
      },
    });
  }
  return extractorPromise;
}

async function loadIndex(): Promise<MovieIndex> {
  if (!indexPromise) {
    indexPromise = (async () => {
      report('Downloading the movie index…', 0);
      const [matrixResponse, titlesResponse] = await Promise.all([
        fetch(EMBEDDINGS_URL),
        fetch(TITLES_URL),
      ]);

      if (!matrixResponse.ok) {
        throw new Error(`Could not download the movie embeddings (${matrixResponse.status}).`);
      }
      if (!titlesResponse.ok) {
        throw new Error(`Could not download the movie titles (${titlesResponse.status}).`);
      }

      const [matrixBuffer, titlesText] = await Promise.all([
        matrixResponse.arrayBuffer(),
        titlesResponse.text(),
      ]);
      const parsed = parseNpyFloat32(matrixBuffer);
      if (parsed.shape.length !== 2) {
        throw new Error(`Expected a 2D embedding matrix, received shape ${parsed.shape.join(' × ')}.`);
      }
      const rows = parsed.shape[0];
      const columns = parsed.shape[1];
      if (rows === undefined || columns === undefined || columns !== DIMENSION) {
        throw new Error(`Expected ${DIMENSION}-dimensional MiniLM embeddings.`);
      }

      const titles = titlesText.split(/\r?\n/).map((title) => title.trim()).filter(Boolean);
      if (titles.length !== rows) {
        throw new Error(`Movie index mismatch: ${rows} vectors but ${titles.length} titles.`);
      }

      normalizeRowsInPlace(parsed.data, rows, columns);
      report(`Movie index ready (${rows.toLocaleString()} titles)`, 100);
      return {matrix: parsed.data, titles, rows, columns};
    })();
  }
  return indexPromise;
}

async function initialize(): Promise<void> {
  if (!initializationPromise) {
    initializationPromise = (async () => {
      await Promise.all([loadIndex(), loadExtractor()]);
      post({type: 'ready'});
    })();
  }
  return initializationPromise;
}

async function search(requestId: number, text: string, requestedK: number): Promise<void> {
  const normalizedText = text.trim();
  if (!normalizedText) throw new Error('Please enter a movie idea or script fragment.');

  await initialize();
  report('Encoding your idea locally…', null);
  const [extractor, index] = await Promise.all([loadExtractor(), loadIndex()]);
  const output = await extractor(normalizedText, {pooling: 'mean', normalize: true});
  const query = output.data as Float32Array;
  if (query.length !== index.columns) {
    throw new Error(`The model returned ${query.length} dimensions instead of ${index.columns}.`);
  }

  report('Comparing against screenplay embeddings…', null);
  const k = Math.max(1, Math.min(50, Math.floor(requestedK)));
  const results = topK(index, query, k);
  post({type: 'results', requestId, results});
  post({type: 'ready'});
}

function topK(index: MovieIndex, query: Float32Array, k: number): MovieResult[] {
  const best: Array<{index: number; score: number}> = [];

  for (let row = 0; row < index.rows; row += 1) {
    const offset = row * index.columns;
    let score = 0;
    for (let column = 0; column < index.columns; column += 1) {
      score += index.matrix[offset + column]! * query[column]!;
    }

    const insertionIndex = best.findIndex((candidate) => score > candidate.score);
    if (insertionIndex === -1) {
      if (best.length < k) best.push({index: row, score});
    } else {
      best.splice(insertionIndex, 0, {index: row, score});
      if (best.length > k) best.pop();
    }
  }

  return best.map(({index: movieIndex, score}) => ({
    title: index.titles[movieIndex] ?? `Movie ${movieIndex + 1}`,
    affinity: score,
  }));
}

function normalizeRowsInPlace(matrix: Float32Array, rows: number, columns: number): void {
  for (let row = 0; row < rows; row += 1) {
    const offset = row * columns;
    let squaredNorm = 0;
    for (let column = 0; column < columns; column += 1) {
      const value = matrix[offset + column]!;
      squaredNorm += value * value;
    }
    const norm = Math.sqrt(squaredNorm);
    if (norm <= 1e-8) continue;
    for (let column = 0; column < columns; column += 1) {
      matrix[offset + column] = matrix[offset + column]! / norm;
    }
  }
}

function parseNpyFloat32(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(0, 6));
  if (magic !== '\u0093NUMPY') throw new Error('The embedding file is not a valid NumPy array.');

  const majorVersion = bytes[6];
  if (majorVersion === undefined) throw new Error('The NumPy header is incomplete.');
  const view = new DataView(buffer);
  const headerLength = majorVersion === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = majorVersion === 1 ? 10 : 12;
  const headerEnd = headerStart + headerLength;
  const header = new TextDecoder('latin1').decode(bytes.slice(headerStart, headerEnd));

  const dtype = /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(header)?.[1];
  if (dtype !== '<f4' && dtype !== '|f4' && dtype !== '=f4') {
    throw new Error(`Unsupported NumPy dtype: ${dtype ?? 'unknown'}.`);
  }
  if (/['"]fortran_order['"]\s*:\s*True/.test(header)) {
    throw new Error('Fortran-ordered NumPy arrays are not supported.');
  }

  const shapeText = /['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!shapeText) throw new Error('Could not read the NumPy array shape.');
  const shape = shapeText.split(',').map((part) => Number(part.trim())).filter(Number.isFinite);
  const length = shape.reduce((product, dimension) => product * dimension, 1);
  const dataEnd = headerEnd + length * Float32Array.BYTES_PER_ELEMENT;
  if (dataEnd > buffer.byteLength) throw new Error('The NumPy embedding file is truncated.');

  const dataBuffer = buffer.slice(headerEnd, dataEnd);
  return {data: new Float32Array(dataBuffer), shape};
}
