# ONNX Runtime LLM

Large language model demos running entirely on-device with
[ONNX Runtime](https://onnxruntime.ai/), organized by platform.

## Platforms

| Platform | Description |
| --- | --- |
| [`web/`](./web/README.md) | Browser — ONNX Runtime Web + WebGPU. No server-side inference; models run in the browser's GPU. |
| `py/` | Python — *planned.* |

## Layout

```
index.html   # top-level landing page linking to the platforms
web/         # browser platform (ONNX Runtime Web + WebGPU)
py/          # python platform (planned)
models/      # local ONNX models shared across platforms (gitignored)
```

`models/` holds local ONNX models used during local development. Model files are
large and are **not** committed (the folder is gitignored apart from its README).

See each platform's README for setup and usage — e.g. [`web/README.md`](./web/README.md).

## License

[MIT](./LICENSE)
