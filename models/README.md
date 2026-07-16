# Local models

Holds local ONNX models for **local development**. On `localhost` the app scans
this folder first and uses whatever it finds here, falling back to the Hugging
Face Hub manifest ([`web/llm_runner/models.json`](../web/llm_runner/models.json))
only when it is empty. Model files are large and are **not** committed (this
folder is gitignored apart from this README), so it can stay empty.

## Preferred layout: mirror the Hugging Face model id

Place each model under `models/<org>/<model>/`, matching its Hugging Face repo id
(`<org>/<model>`). The app then lists it in the dropdown by that exact id, so the
local entry looks identical to the remote one:

```
models/
  onnx-community/
    Phi-4-mini-instruct-ONNX/     # → shown as "onnx-community/Phi-4-mini-instruct-ONNX"
      config.json
      tokenizer.json
      ...
      onnx/
        model_q4f16.onnx
        model_q4f16.onnx_data      # external data, if present
```

The easiest way to populate it is to download the repo into the matching path:

```shell
# from the repository root
huggingface-cli download onnx-community/Phi-4-mini-instruct-ONNX \
  --local-dir models/onnx-community/Phi-4-mini-instruct-ONNX
# or: git clone https://huggingface.co/onnx-community/Phi-4-mini-instruct-ONNX \
#            models/onnx-community/Phi-4-mini-instruct-ONNX
```

## How detection works

A **model folder** is any directory that directly contains a `config.json`, so
the scan descends through `<org>/` wrappers and lists the real model directory by
its full path. The decoder `.onnx` is auto-selected (preferring
`model_q4f16.onnx`), and external data (`.onnx_data`) is picked up automatically.

See [`web/README.md`](../web/README.md) for the full local-vs-Hugging-Face
behavior.
