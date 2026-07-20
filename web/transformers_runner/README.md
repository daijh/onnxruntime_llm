# Transformers Runner

General-purpose LLM chat interface that runs entirely in the browser on
**[Transformers.js](https://github.com/huggingface/transformers.js)** with WebGPU.
The whole generation pipeline — decode loop, EOS handling, and KV cache — is
handled by Transformers.js' `AutoModelForCausalLM.generate()`, so no per-model
configuration is required.

## Features

- Uses Transformers.js' high-level `AutoModelForCausalLM` + `generate()` pipeline.
- Streaming token generation via `TextStreamer`, with performance metrics (TTFT, prefill/decode tokens/s).
- Chat template support via the `@huggingface/transformers` tokenizer (with `chat_template.jinja` fallback).
- `dtype` auto-derived from the ONNX file name (`model_<dtype>.onnx` → e.g. `q4f16`).
- Configurable prefill padding/truncation, decode length (max new tokens), max context, and loop count.
- No build step: dependencies load as ES modules via an import map; models load locally or from the Hugging Face Hub depending on where the app is served.

## Files

- `llm.js` — inference engine (`LLM` class wrapping `AutoModelForCausalLM` + `generate()`)
- `script.js` — UI controller (model discovery, prompt handling, token display)
- `config.js` — environment detection (local dev vs static host) and dependency source
- `index.html` — demo page (import map + UI)
- `models.json` — model manifest used on static hosts (e.g. GitHub Pages)

## Usage

From the `web/` directory (its serve script serves the repository root):

```shell
npm install   # first time only
npm run serve
```

Open `http://localhost:8080/web/transformers_runner/`.

## Two runtime modes

The code decides two things independently at runtime, with no build step.

**Dependencies** — chosen by hostname (`config.js`):

| | `localhost` | any other host (e.g. GitHub Pages) |
| --- | --- | --- |
| JS deps (import map) | local `/web/node_modules/` | jsDelivr CDN |
| `wasmPaths` | local `/web/node_modules/` | jsDelivr CDN |

**Models** — chosen by host, then by what is available (`detectModels()`):

| | `localhost` / `127.0.0.1` | any other host (e.g. GitHub Pages) |
| --- | --- | --- |
| Discovery | scan `/models/` first; fall back to `models.json` if empty | `models.json` manifest |
| Files | local `/models/…` (or Hugging Face on fallback) | streamed from the Hugging Face Hub |
| `transformers` env | `allowLocalModels` (or `allowRemoteModels` on fallback) | `allowRemoteModels` |

So on `localhost` the app uses your local `/models/` folder when it contains any
model, and otherwise falls back to the shipped `models.json` (Hugging Face). A
static host cannot list directories, so it always uses the manifest.

## models.json schema

Each entry describes one Hugging Face model:

```jsonc
{
  "models": [
    {
      "label": "Phi-4-mini-instruct (q4f16)",           // shown in the dropdown
      "hfId": "onnx-community/Phi-4-mini-instruct-ONNX", // HF repo id (tokenizer + config + weights)
      "modelUrl": "https://huggingface.co/onnx-community/Phi-4-mini-instruct-ONNX/resolve/main",
      "onnxFile": "onnx/model_q4f16.onnx"                // only its dtype (q4f16) is used
    }
  ]
}
```

- `hfId` — Hugging Face repo id; Transformers.js loads the tokenizer, config, and weights from here.
- `onnxFile` — used only to derive the `dtype` from its filename suffix (e.g.
  `model_q4f16.onnx` or a split `decoder_model_merged_q4f16.onnx` → `q4f16`). Transformers.js
  then selects the actual ONNX file set (including split `embed_tokens`/decoder exports) itself.
- `modelUrl` — base URL used only for the optional `chat_template.jinja` fallback.

## Using local models

On `localhost`, the app scans `/models/` first and uses whatever it finds there,
falling back to the manifest only when the folder is empty. Model files are not
included in this repository (they are large and gitignored); place your ONNX
models under the repository-root `models/` directory so they are served at
`/models/`:

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

Any directory that directly contains a `config.json` is treated as a model
folder (org-nested layouts like `org/model/` work too). Transformers.js loads
the folder directly; the local scan only samples an `.onnx` (preferring a
`q4f16` export) to pick the `dtype`.

## Supported models

Works with any **Transformers.js-supported causal-LM architecture** exported in
the `onnx-community` style — both single-file (`onnx/model_<dtype>.onnx`) and
split (`onnx/decoder_model_merged_<dtype>.onnx` + `onnx/embed_tokens_<dtype>.onnx`)
layouts. To load in the browser a model needs a **WebGPU-friendly quantization**
(e.g. `q4f16`, `fp16`, `q4`); large weights should use split external data so no
single file exceeds the browser's ~2 GB `ArrayBuffer` limit.

Vision-language repos (those that also ship a `vision_encoder`) load through
their **text decoder only** here — this runner sends text prompts, not images.
A model whose architecture Transformers.js does not map to a causal-LM class
will fail to load.

### Shipped in `models.json`

| Model | Hugging Face repo | dtype |
| --- | --- | --- |
| Phi-4-mini-instruct | [`onnx-community/Phi-4-mini-instruct-ONNX`](https://huggingface.co/onnx-community/Phi-4-mini-instruct-ONNX) | `q4f16` |
| Phi-4-mini-instruct (WebGPU) | [`jianhui42/Phi-4-mini-instruct-ONNX-webgpu`](https://huggingface.co/jianhui42/Phi-4-mini-instruct-ONNX-webgpu) | `q4f16` |
| Qwen3-4B | [`onnx-community/Qwen3-4B-ONNX`](https://huggingface.co/onnx-community/Qwen3-4B-ONNX) | `q4f16` |
| Qwen3-4B (WebGPU) | [`jianhui42/Qwen3-4B-ONNX-webgpu`](https://huggingface.co/jianhui42/Qwen3-4B-ONNX-webgpu) | `q4f16` |
| Qwen3.5-4B (onnx-community OPT) | [`onnx-community/Qwen3.5-4B-ONNX-OPT`](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT) | `q4f16` |
| Qwen3.5-4B (WebGPU) | [`jianhui42/Qwen3.5-4B-ONNX-webgpu`](https://huggingface.co/jianhui42/Qwen3.5-4B-ONNX-webgpu) | `q4f16` |

## Configuration

- **Model**: pick from the dropdown (populated from `/models/` or `models.json`).
- **Options**: chat template, thinking, verbose; and max context / prefill / decode / loop counts.
- **Dependencies**: `@huggingface/transformers` (which bundles `onnxruntime-web`),
  resolved via the import map in `index.html`.

## Requirements

- A WebGPU-capable browser (recent **Chrome** or **Edge**).
- [Node.js](https://nodejs.org/) — used only to serve static files and pull in dependencies.
