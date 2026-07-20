# Web — ONNX Runtime Web + WebGPU

Browser-based large language model demos running entirely on-device with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) and **WebGPU**.
No server-side inference — models run in your browser's GPU.

## Sub-projects

| Project | Description |
| --- | --- |
| [`llm_runner/`](./llm_runner/) | General-purpose LLM chat interface for any merged-decoder ONNX model. Hand-rolls the decode loop directly on ONNX Runtime Web; model capabilities are auto-detected from the ONNX session metadata, so no per-model configuration is required. |
| [`transformers_runner/`](./transformers_runner/) | Browser LLM chat interface powered by [Transformers.js](https://github.com/huggingface/transformers.js) (`AutoModelForCausalLM.generate()`) on WebGPU. The full generation pipeline runs in the library, so no per-model configuration is required. |

## Layout

```
web/
  package.json          # deps + serve script (run npm here)
  node_modules/         # installed deps (gitignored)
  llm_runner/           # LLM chat demo — hand-rolled ORT Web decode loop
  transformers_runner/  # LLM chat demo — Transformers.js generate()
```

The dev server runs from `web/` but serves the **repository root**, so both
`/web/…` (the app and its `node_modules`) and `/models/` (at the repo root) are
reachable.

## Requirements

- A WebGPU-capable browser (recent **Chrome** or **Edge**).
- [Node.js](https://nodejs.org/) (used only to serve static files and pull in dependencies).

## Getting started

```shell
cd web

# 1. Install JavaScript dependencies (onnxruntime-web, @huggingface/transformers)
npm install

# 2. Serve the repository root over HTTP (the script serves the parent dir)
npm run serve
```

Then open the app at <http://localhost:8080/web/llm_runner/> (or the top-level
landing page at <http://localhost:8080/>).

## How it adapts to its environment

The same code runs locally and on GitHub Pages with no build step, deciding two
things independently at runtime:

- **JS dependencies** — chosen by hostname (`llm_runner/config.js`): the local
  `/web/node_modules/` on `localhost`, the jsDelivr CDN everywhere else. This is
  why GitHub Pages needs no published `node_modules/`.
- **Models** — chosen by host: on `localhost` the app scans the local `/models/`
  directory first and falls back to the `llm_runner/models.json` manifest
  (Hugging Face Hub) only if no local models are found. Any other host (e.g.
  GitHub Pages) can't list directories, so it always uses the manifest.

## Using local models

On `localhost`, the app scans `/models/` first and uses whatever it finds there,
falling back to the manifest only when the folder is empty — so you don't need to
touch `models.json`. Model files are **not** included in this repository (they are
large and gitignored); place your ONNX models under the repository-root `models/`
directory (`../models` from here) so they are served at `/models/`:

```
models/
  Phi-4-mini-instruct-ONNX/
    config.json
    tokenizer.json
    ...
    onnx/
      model_q4f16.onnx
      model_q4f16.onnx_data   # external data, if present
```

The app auto-detects model folders under `/models/` (any directory that directly
contains a `config.json`, so org-nested layouts like `org/model/` work too) and
auto-selects the decoder `.onnx`.

Any merged-decoder ONNX export works. For example, download
[`onnx-community/Phi-4-mini-instruct-ONNX`](https://huggingface.co/onnx-community/Phi-4-mini-instruct-ONNX)
(or any other ONNX conversion from the
[Hugging Face Hub](https://huggingface.co/models?library=onnx)) into `../models/`.

## How it works

- Dependencies are loaded as ES modules via an [import map](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script/type/importmap)
  (local `/web/node_modules/` or the jsDelivr CDN), so the app runs without a bundler.
- `onnxruntime-web` runs the merged ONNX decoder on the WebGPU execution provider,
  keeping the KV cache in GPU buffers across decode steps.
- `@huggingface/transformers` provides tokenization and chat-template support.
- Model capabilities (position-id rank, `use_cache_branch`, `num_logits_to_keep`,
  KV cache shapes/dtypes) are auto-detected from the ONNX session metadata.

See [`llm_runner/README.md`](./llm_runner/README.md) for engine details.

## Deploying to GitHub Pages

No build step — Pages serves the repository's static files directly.

1. Edit [`llm_runner/models.json`](./llm_runner/models.json) to list the Hugging
   Face models you want to expose (verify each repo id and ONNX file path).
2. Push to GitHub.
3. In the repo, go to **Settings → Pages**, set **Source: Deploy from a branch**,
   branch **`main`**, folder **`/ (root)`**, and save.
4. Open `https://<user>.github.io/<repo>/web/llm_runner/`.

Notes:
- Dependencies load from the jsDelivr CDN, so `node_modules/` does not need to be
  published.
- Models stream from the Hugging Face Hub, so large weights are never committed.
- WebGPU requires HTTPS (Pages provides it) and a recent Chrome/Edge.
- The repository-root `.nojekyll` file disables Jekyll so all paths are served as-is.
