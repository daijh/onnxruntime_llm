/**
 * config.js — environment detection and per-environment settings.
 *
 * The same code runs in two environments:
 *   - Local dev:  served by `npm run serve` (http-server) on localhost. JS deps
 *     come from the installed node_modules; models come from the local /models/
 *     directory via directory-listing scan.
 *   - GitHub Pages (or any static host): JS deps come from the jsDelivr CDN and
 *     models are streamed from the Hugging Face Hub via a models.json manifest.
 *
 * Only the values below differ between the two; everything else is shared.
 *
 * NOTE: keep ORT_VERSION / the versions in index.html's bootstrap and
 * package.json in sync.
 */

const host = location.hostname;

/** True when served from a local development server. */
export const IS_LOCAL = host === 'localhost' || host === '127.0.0.1' || host === '';

/** onnxruntime-web version — the WASM/JSEP binaries must match the JS build. */
export const ORT_VERSION = '1.27.0';

/** Where onnxruntime-web loads its .wasm / .mjs backend files from. */
export const wasmPaths = IS_LOCAL
    ? '/web/node_modules/onnxruntime-web/dist/'
    : `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
