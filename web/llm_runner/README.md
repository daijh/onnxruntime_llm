# LLM Runner

General-purpose LLM chat interface using ONNX Runtime Web with WebGPU. Works with any merged-decoder ONNX model — capabilities are auto-detected from the ONNX session metadata, so no per-model configuration is required.

## Features

- Single-session merged decoder (no separate embed/vision sessions)
- Auto-detects model capabilities from session metadata:
  - `position_ids` rank (2D standard or 3D MRoPE)
  - `use_cache_branch` for merged prefill/decode models
  - `num_logits_to_keep` for selective logit output
- Static GPU-buffer KV cache pre-allocated to a configurable max context (default 8192)
  - Single-buffer (in-place) by default; double-buffer auto-enabled for models that need it (e.g. Qwen3.5)
- Auto-probes external data files (`.onnx.data` / `.onnx_data`)
- Chat template support via `@huggingface/transformers` tokenizer
- Configurable prefill padding/truncation, decode length, and max context
- Streaming token generation with performance metrics
- ORT session profiling support

## Files

- `llm.js` — inference engine (`LLM` class, auto-detect, decode loop)
- `script.js` — UI controller (model loading, prompt handling, token display)
- `config.js` — environment detection (local dev vs static host) and dependency/model source
- `index.html` — demo page
- `models.json` — model manifest used on static hosts (GitHub Pages)

## Usage

From the `web/` directory (its serve script serves the repository root):

```shell
npm install   # first time only
npm run serve
```

Open `http://localhost:8080/web/llm_runner/`.

## Two runtime modes

The code decides two things independently at runtime:

**Dependencies** — by hostname (`config.js`):

| | `localhost` | any other host (e.g. GitHub Pages) |
| --- | --- | --- |
| JS deps (import map) | local `/web/node_modules/` | jsDelivr CDN |
| `wasmPaths` | local `/web/node_modules/` | jsDelivr CDN |

**Models** — by host, then by what is available (`detectModels()`):

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
      "hfId": "onnx-community/Phi-4-mini-instruct-ONNX", // HF repo id (tokenizer + config)
      "modelUrl": "https://huggingface.co/onnx-community/Phi-4-mini-instruct-ONNX/resolve/main",
      "onnxFiles": [                                      // one or more decoder files
        { "label": "q4f16", "file": "onnx/model_q4f16.onnx" }
      ]
      // or a single: "onnxFile": "onnx/model_q4f16.onnx"
    }
  ]
}
```

- `hfId` — Hugging Face repo id; the tokenizer and config are fetched from here.
- `modelUrl` — base URL for the model's files (typically `…/resolve/main`).
- `file` — path to a decoder `.onnx`, relative to `modelUrl` or an absolute URL.
  External data files (`.onnx_data`) are auto-probed at load time.

Verify the repo ids and ONNX file names against the actual HF repos — they vary
per model.

## Supported models

Works with **any merged-decoder ONNX model** exported in the transformers.js /
`onnx-community` style — a single `model*.onnx` exposing `logits` and
`past_key_values` (optionally `use_cache_branch`, `position_ids`,
`num_logits_to_keep`). Capabilities are auto-detected, so no per-model config.

To load in the browser a model needs:

- a **WebGPU-friendly quantization** (e.g. `q4f16`, `fp16`, `q4`); and
- each `.onnx` / external-data file **under ~2 GB** (browser `ArrayBuffer` limit).
  Split external data (`*_data`, `*_data_1`, …) is supported and is how larger
  models fit.

### Shipped in `models.json`

| Model | Hugging Face repo | Decoder |
| --- | --- | --- |
| Phi-4-mini-instruct | [`onnx-community/Phi-4-mini-instruct-ONNX`](https://huggingface.co/onnx-community/Phi-4-mini-instruct-ONNX) | `onnx/model_q4f16.onnx` |
| Phi-4-mini-instruct (WebGPU) | [`jianhui42/Phi-4-mini-instruct-ONNX-webgpu`](https://huggingface.co/jianhui42/Phi-4-mini-instruct-ONNX-webgpu) | `onnx/model_q4f16.onnx` |
| Qwen3-4B | [`onnx-community/Qwen3-4B-ONNX`](https://huggingface.co/onnx-community/Qwen3-4B-ONNX) | `onnx/model_q4f16.onnx` |
| Qwen3-4B (WebGPU) | [`jianhui42/Qwen3-4B-ONNX-webgpu`](https://huggingface.co/jianhui42/Qwen3-4B-ONNX-webgpu) | `onnx/model_q4f16.onnx` |
| Qwen3.5-4B-Text (WebGPU) | [`jianhui42/Qwen3.5-4B-Text-ONNX-webgpu`](https://huggingface.co/jianhui42/Qwen3.5-4B-Text-ONNX-webgpu) | `onnx/model_q4f16.onnx` |

Other `onnx-community` conversions (Qwen3, Llama, Gemma, SmolLM, …) generally
work — add them to `models.json` (remote) or drop them under `/models/` (local).

### Not supported

- **Single-file decoders larger than ~2 GB** — a browser can't hold one in a
  single `ArrayBuffer`; use a split-data or smaller-quant variant.

## Configuration

- **Model**: pick from the dropdown (populated from `/models/` or `models.json`);
  the decoder `.onnx` is auto-selected (preferring `model_q4f16.onnx`).
- **Options**: chat template, thinking, verbose; and max context / prefill / decode / loop counts.
- **Dependencies**: `onnxruntime-web` and `@huggingface/transformers`, resolved
  via the import map in `index.html`.
