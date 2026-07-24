/**
 * llm.js — Generic single-session LLM inference using ONNX Runtime WebGPU.
 *
 * Works with any merged-decoder ONNX model, including hybrid architectures that
 * mix full-attention layers (past/present key & value) with linear-attention
 * layers (recurrent/conv states), e.g. Qwen3.5. Capabilities (position-id rank,
 * and KV/state cache shapes & dtypes) are auto-detected from the ONNX session
 * metadata at load time, so no per-model configuration is required.
 *
 * KV cache:
 *   - Static: pre-allocated once to `maxLen` (default 8192) as GPU buffers, so
 *     there is no per-step reallocation and the cache stays on-device. Recurrent
 *     linear-attention states are fixed-size and kept on-device the same way.
 *     The same buffer is bound as both past input and present output (in-place).
 */

import * as ort from 'onnxruntime-web';
import { wasmPaths } from './config.js';
ort.env.wasm.wasmPaths = wasmPaths;
// Single-threaded WASM: SharedArrayBuffer (multi-thread) needs cross-origin
// isolation (COOP/COEP), which static hosts like GitHub Pages can't set. The
// WebGPU EP runs compute on the GPU, so this doesn't affect inference speed.
ort.env.wasm.numThreads = 1;

// Static KV cache length. The KV cache is pre-allocated once to this sequence
// length so no per-step reallocation happens during decode. It is the hard cap
// on total context (prompt + generated); generation stops at maxLen. Overridable
// per-load via options.maxLen.
const MAX_LEN = 8192;

function log(i) { console.log(i); }

// ---------------------------------------------------------------------------
// Fetch with Browser Cache API
// ---------------------------------------------------------------------------

export async function fetchAndCache(url, onProgress) {
    const cache = await caches.open("onnx");
    const cachedResponse = await cache.match(url).catch(() => undefined);
    if (cachedResponse) {
        log(`${url} (cached)`);
        return cachedResponse.arrayBuffer();
    }

    // Network fetch — report each failure mode distinctly.
    let response;
    try {
        response = await fetch(url);
    } catch (err) {
        // TypeError: Failed to fetch — no HTTP status reached us. Usual causes:
        // server not running, connection reset, CORS, or the transfer being
        // aborted (e.g. an out-of-memory read of a multi-GB file).
        console.error(`[fetchAndCache] network error for ${url}: ${err.name}: ${err.message}`);
        throw new Error(`Network error fetching ${url} — ${err.message} (server down, CORS, or file too large?)`);
    }
    if (!response.ok) {
        console.error(`[fetchAndCache] HTTP ${response.status} ${response.statusText} for ${url}`);
        throw new Error(`HTTP ${response.status} ${response.statusText} fetching ${url}`);
    }

    const bytes = Number(response.headers.get('content-length')) || 0;
    const mb = (bytes / 1048576).toFixed(1);
    log(bytes ? `${url} (network, ${mb} MB)` : `${url} (network)`);

    // A single ArrayBuffer is capped at ~2 GiB in browsers, so Response.arrayBuffer()
    // can't read a file larger than that. Fail fast with a clear reason (checked up
    // front when the server sends Content-Length).
    const MAX_ARRAYBUFFER = 2 * 1024 * 1024 * 1024 - 1; // 2 GiB - 1 byte
    if (bytes > MAX_ARRAYBUFFER) {
        console.error(`[fetchAndCache] ${url} is ${mb} MB — exceeds the browser's ~2 GB max ArrayBuffer`);
        throw new Error(`${url} is ${mb} MB, exceeding the browser's ~2 GB single-buffer limit. ` +
            `Use a model whose weights are split into <2 GB files (e.g. the *_data / *_data_1 layout).`);
    }

    let buffer;
    if (onProgress && bytes > 0 && response.body?.getReader) {
        // Stream the body so download progress can be reported. A single buffer
        // pre-allocated to Content-Length keeps peak memory identical to
        // response.arrayBuffer() (no chunk-list doubling).
        try {
            const reader = response.body.getReader();
            const out = new Uint8Array(bytes);
            let loaded = 0;
            onProgress({ file: url, loaded: 0, total: bytes, progress: 0 });
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                out.set(value, loaded);   // throws RangeError if the server overshoots Content-Length
                loaded += value.byteLength;
                onProgress({ file: url, loaded, total: bytes, progress: (loaded / bytes) * 100 });
            }
            buffer = out.buffer;
        } catch (err) {
            const hint = err.name === 'RangeError'
                ? "exceeds the browser's ~2 GB max ArrayBuffer size, or the server's Content-Length was wrong"
                : 'likely out of memory';
            console.error(`[fetchAndCache] failed streaming ${url} (${mb} MB): ${err.name}: ${err.message}`);
            throw new Error(`Failed reading ${url} (${mb} MB) — ${hint}`);
        }
    } else {
        try {
            buffer = await response.arrayBuffer();
        } catch (err) {
            // Body read failed. RangeError = over the max ArrayBuffer size (when the
            // server didn't send Content-Length); otherwise usually out of memory.
            const hint = err.name === 'RangeError'
                ? "exceeds the browser's ~2 GB max ArrayBuffer size"
                : 'likely out of memory';
            console.error(`[fetchAndCache] failed reading body of ${url} (${mb} MB): ${err.name}: ${err.message}`);
            throw new Error(`Failed reading ${url} (${mb} MB) — ${hint}`);
        }
    }

    try {
        await cache.put(url, new Response(buffer));
    } catch (err) {
        // Non-fatal: the model still loads, it just won't be cached for next time.
        console.warn(`[fetchAndCache] cache put skipped for ${url} (quota exceeded?): ${err.message}`);
    }
    return buffer;
}

// ---------------------------------------------------------------------------
// Greedy argmax
// ---------------------------------------------------------------------------

function argmax(t) {
    const arr = t.data;
    const start = t.dims[2] * (t.dims[1] - 1);
    let max = arr[start], maxidx = 0;
    for (let i = 0; i < t.dims[2]; i++) {
        const val = arr[i + start];
        if (!isFinite(val)) throw new Error("non-finite value in logits");
        if (val > max) { max = val; maxidx = i; }
    }
    return maxidx;
}

// Search a byte array for an ASCII substring. Used to detect whether an ONNX
// graph references an external-data file (its filename is embedded as a
// "location" string) without issuing network probes that 404 in the console.
function bytesIncludesAscii(bytes, str) {
    const needle = new TextEncoder().encode(str);
    const n = needle.length;
    if (n === 0 || bytes.length < n) return false;
    const first = needle[0];
    for (let i = 0, max = bytes.length - n; i <= max; i++) {
        if (bytes[i] !== first) continue;
        let j = 1;
        while (j < n && bytes[i + j] === needle[j]) j++;
        if (j === n) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Static GPU-buffer KV cache helpers
// ---------------------------------------------------------------------------

// Build one descriptor per KV cache pair with a *static* shape: dynamic dims
// collapse to 1, except the sequence axis (dim 2 of key_values) which is fixed
// to maxLen so the buffer never needs to grow.
function buildStaticKvPairs(kvIndices, outputNames, inputMetadata, maxLen) {
    const metaByName = {};
    for (const meta of inputMetadata) {
        if (meta.isTensor) metaByName[meta.name] = meta;
    }
    const pairs = [];
    for (const [outIdx, pastName] of kvIndices) {
        const meta = metaByName[pastName];
        if (!meta) continue;
        const isSsm = pastName.includes('recurrent') || pastName.includes('conv');
        const isKv = pastName.includes('key_values');
        const dims = meta.shape.map((d, i) => {
            if (typeof d === 'number' && d > 0) return d;
            if (!isSsm && isKv && i === 2) return maxLen;
            return 1;
        });
        pairs.push({ outName: outputNames[outIdx], pastName, dims, dtype: meta.type });
    }
    return pairs;
}

// Allocate a zero-initialized GPU-buffer tensor. WebGPU guarantees new buffers
// are zero-filled, so past KV starts empty without an explicit clear.
function createGpuBufferTensor(device, dtype, dims) {
    const numel = dims.reduce((a, b) => a * b, 1);
    const bytesPerEl = (dtype === 'float16') ? 2 : 4;
    const byteLength = Math.ceil((numel * bytesPerEl) / 16) * 16; // 16-byte aligned
    const gpuBuffer = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return ort.Tensor.fromGpuBuffer(gpuBuffer, { dataType: dtype, dims });
}

// ---------------------------------------------------------------------------
// ORT environment configuration
// ---------------------------------------------------------------------------

function configureOrtEnv(sessionOpt, options) {
    ort.env.webgpu.profiling = {};
    if (options.verbose) {
        sessionOpt.logSeverityLevel = 0;
        sessionOpt.logVerbosityLevel = 0;
        ort.env.logLevel = "verbose";
        ort.env.debug = true;
        ort.env.webgpu.profiling.mode = 'default';
    }
    if (options.profiler) {
        ort.env.webgpu.profiling.mode = 'default';
        if (options.on_profiling_data) {
            ort.env.webgpu.profiling.ondata = options.on_profiling_data;
        }
        sessionOpt.enableProfiling = true;
    }
}

// ---------------------------------------------------------------------------
// LLM class — single-session pipeline (merged decoder)
// ---------------------------------------------------------------------------

export class LLM {
    // Session
    decoderSession = null;

    // Session metadata
    decoderOutNames = [];
    kvIndices       = [];   // [[outputIdx, pastInputName], ...]

    // Static GPU-buffer KV cache
    device         = null;
    maxLen         = MAX_LEN;
    kvPairs        = [];        // [{ outName, pastName, dims, dtype }]
    kvOutToPast    = {};        // present output name -> past input name
    kvSet          = {};        // pastName -> gpu Tensor (in-place past/present)

    // Runtime state
    feed          = {};
    output_tokens = [];
    stop          = false;

    // Config
    eos   = [];
    dtype = "float16";
    profiler          = false;
    on_profiling_data = null;

    // Auto-detected from model
    positionIdRank          = 2;     // 2 = [1,N], 3 = [3,1,N] MRoPE

    async load(model, options) {
        this.profiler = options.profiler;
        this.on_profiling_data = options.on_profiling_data;
        this.maxLen = options.maxLen || MAX_LEN;

        // Per-file download progress (large weight files only). Reports the
        // basename so the UI can show "model.onnx_data — 42%".
        const onProg = options.progress_callback
            ? (p) => options.progress_callback({
                file: p.file.split('/').pop(), loaded: p.loaded, total: p.total, progress: p.progress })
            : undefined;

        log(`loading... ${model}, webgpu`);
        // Load model config
        const configBytes = await fetchAndCache(model + "/config.json");
        const modelConfig = JSON.parse(new TextDecoder().decode(configBytes));

        // Multimodal configs (e.g. Qwen3.5 *ForConditionalGeneration) nest the
        // text-decoder settings under `text_config`; fall back to it for fields
        // that aren't present at the top level.
        const textConfig = modelConfig.text_config || modelConfig;

        // EOS token IDs
        let eosIds = modelConfig.eos_token_id ?? textConfig.eos_token_id;
        if (!Array.isArray(eosIds)) eosIds = [eosIds];
        this.eos = BigInt64Array.from(eosIds.filter(v => v != null), v => BigInt(v));

        const numLayers = modelConfig.num_hidden_layers ?? textConfig.num_hidden_layers;

        // Keep KV cache on GPU for fast decode. Pin every possible present.* output
        // to a gpu-buffer: standard attention layers emit key/value, while hybrid
        // linear-attention layers (e.g. Qwen3.5) emit conv_state/recurrent_state
        // instead. Names that don't match an actual output are ignored by ORT, so
        // listing all four patterns covers both architectures.
        const preferred = {};
        for (let i = 0; i < numLayers; ++i) {
            preferred[`present.${i}.key`] = 'gpu-buffer';
            preferred[`present.${i}.value`] = 'gpu-buffer';
            preferred[`present.${i}.conv_state`] = 'gpu-buffer';
            preferred[`present.${i}.recurrent_state`] = 'gpu-buffer';
        }
        const opt = {
            executionProviders: ["webgpu"],
            preferredOutputLocation: preferred,
            freeDimensionOverrides: { batch_size: 1 },
        };

        // Resolve ONNX file path (accepts an absolute URL or a root-relative path)
        if (!options.onnxFile) throw new Error("No ONNX file specified for the model.");
        const onnxPath = /^https?:\/\//.test(options.onnxFile)
            ? options.onnxFile
            : (options.onnxFile.startsWith('/') ? options.onnxFile : '/' + options.onnxFile);
        const lastSlash = onnxPath.lastIndexOf('/');
        const model_file_path = onnxPath.substring(0, lastSlash + 1);
        const model_file = onnxPath.substring(lastSlash + 1);

        const model_bytes = await fetchAndCache(model_file_path + model_file, onProg);
        let modelSize = model_bytes.byteLength;

        // External data: an ONNX graph that keeps weights in a separate file
        // embeds that file's name (as a "location" string) inside the .onnx. Scan
        // the already-downloaded graph for the expected name instead of issuing
        // HEAD probes — probes 404 (and clutter the console) for self-contained
        // models whose weights are embedded in the .onnx.
        //   transformers.js / optimum: "<model>.onnx_data"
        //   ONNX Runtime GenAI:        "<model>.onnx.data"
        const external_data_candidates = [
            model_file + '_data',
            model_file + '.data',
        ];
        const modelU8 = new Uint8Array(model_bytes);
        let actual_external_data_suffix = null;
        for (const candidate of external_data_candidates) {
            if (bytesIncludesAscii(modelU8, candidate)) { actual_external_data_suffix = candidate; break; }
        }

        if (actual_external_data_suffix) {
            const externalDataEntries = [];
            // Load the first external data file
            const firstData = await fetchAndCache(model_file_path + actual_external_data_suffix, onProg);
            modelSize += firstData.byteLength;
            externalDataEntries.push({ data: firstData, path: actual_external_data_suffix });
            // Split external data files (_data_1, _data_2, ...) are each referenced
            // by name inside the graph, so scan for them too rather than probing.
            for (let i = 1; ; i++) {
                const splitName = actual_external_data_suffix + '_' + i;
                if (!bytesIncludesAscii(modelU8, splitName)) break;
                const splitData = await fetchAndCache(model_file_path + splitName, onProg);
                modelSize += splitData.byteLength;
                externalDataEntries.push({ data: splitData, path: splitName });
            }
            log(`  external data: ${externalDataEntries.length} file(s)`);
            opt.externalData = externalDataEntries;
        }

        log(`model size ${Math.round(modelSize / 1024 / 1024)} MB`);

        configureOrtEnv(opt, {
            verbose: options.verbose,
            profiler: this.profiler,
            on_profiling_data: this.on_profiling_data,
        });

        this.decoderSession = await ort.InferenceSession.create(model_bytes, opt);
        this.decoderOutNames = this.decoderSession.outputNames;

        // ---------------------------------------------------------------
        // Detect model capabilities from session metadata
        // ---------------------------------------------------------------
        const inputNameSet = new Set(this.decoderSession.inputNames);

        // Build input metadata map (local, used for one-time config below)
        const inputMetaMap = {};
        for (const meta of this.decoderSession.inputMetadata) {
            if (meta.isTensor) inputMetaMap[meta.name] = meta;
        }

        // Detect position_ids rank from metadata (e.g. [3, 1, N] for 3D MRoPE)
        if (inputMetaMap["position_ids"]) {
            this.positionIdRank = inputMetaMap["position_ids"].shape.length;
        }

        // Log input/output counts for diagnostics
        log(`  inputs (${this.decoderSession.inputNames.length}), outputs (${this.decoderOutNames.length})`);

        // Build present→past cache mapping. Every cache output is named
        // "present.<layer>.<slot>" (slot = key/value for attention layers,
        // conv_state/recurrent_state for linear-attention layers) and pairs with
        // the "past_key_values.<layer>.<slot>" input of the same name.
        const presentToPast = {};
        for (const name of this.decoderOutNames) {
            if (!name.startsWith('present.')) continue;
            const past = 'past_key_values.' + name.slice('present.'.length);
            if (inputNameSet.has(past)) presentToPast[name] = past;
        }
        this.kvIndices = this.decoderOutNames
            .map((name, i) => [i, name])
            .filter(([, name]) => name in presentToPast)
            .map(([i, name]) => [i, presentToPast[name]]);

        // Static GPU-buffer KV cache.
        this.device = ort.env.webgpu.device;
        this.kvPairs = buildStaticKvPairs(
            this.kvIndices, this.decoderOutNames,
            this.decoderSession.inputMetadata, this.maxLen);
        this.kvOutToPast = {};
        for (const p of this.kvPairs) this.kvOutToPast[p.outName] = p.pastName;
        this.dtype = this.kvPairs.length ? this.kvPairs[0].dtype : "float16";

        this.resetFeed();
        log("Session loaded.");
    }

    // -----------------------------------------------------------------------
    // KV cache management
    // -----------------------------------------------------------------------

    initializeFeed() { this.resetFeed(); }
    resetFeed() {
        // Drop stale feed references (KV tensors are owned by kvSet).
        for (const key of Object.keys(this.feed)) delete this.feed[key];
        this._disposeKvSet();
        this.kvSet = this._allocKvSet();
        this.output_tokens = [];
    }

    /** Allocate a set of zeroed static GPU-buffer KV tensors. */
    _allocKvSet() {
        const set = {};
        for (const p of this.kvPairs) {
            set[p.pastName] = createGpuBufferTensor(this.device, p.dtype, p.dims);
        }
        return set;
    }

    /** Destroy the KV buffer set and its underlying GPU buffers. */
    _disposeKvSet() {
        for (const name in this.kvSet) {
            const buf = this.kvSet[name].gpuBuffer;
            if (buf && typeof buf.destroy === 'function') buf.destroy();
        }
        this.kvSet = {};
    }

    /** Signal the decode loop to stop. */
    abort() { this.stop = true; }

    _makePositionIds(startPos, length) {
        if (this.positionIdRank === 3) {
            const ids = BigInt64Array.from({ length }, (_, i) => BigInt(startPos + i));
            const buf = new BigInt64Array(3 * length);
            buf.set(ids, 0);
            buf.set(ids, length);
            buf.set(ids, 2 * length);
            return new ort.Tensor('int64', buf, [3, 1, length]);
        }
        return new ort.Tensor('int64',
            BigInt64Array.from({ length }, (_, i) => BigInt(startPos + i)), [1, length]);
    }

    _makeSinglePositionId(pos, buf3) {
        if (this.positionIdRank === 3) {
            const p = BigInt(pos);
            buf3[0] = p; buf3[1] = p; buf3[2] = p;
            return new ort.Tensor('int64', buf3, [3, 1, 1]);
        }
        buf3[0] = BigInt(pos);
        return new ort.Tensor('int64', buf3.subarray(0, 1), [1, 1]);
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    async generate(tokens, callback, options) {
        const feed = this.feed;
        const inputIdsBigInt = BigInt64Array.from(tokens.map(BigInt));
        const inputIdsTensor = new ort.Tensor('int64', inputIdsBigInt, [1, tokens.length]);
        this.stop = false;
        this.output_tokens.push(...inputIdsTensor.data);

        const seqLen     = tokens.length;
        // The static KV cache is sized to maxLen, so the prompt itself must fit and
        // total context (prompt + generated) can never exceed it.
        if (seqLen > this.maxLen) {
            throw new Error(`Prompt is ${seqLen} tokens but max context is ${this.maxLen}. Increase Max Context or shorten the prompt.`);
        }
        let   max_tokens = (options.max_tokens || 256) + seqLen;
        if (max_tokens > this.maxLen) max_tokens = this.maxLen;

        // Prepare decoder feed
        feed['input_ids'] = inputIdsTensor;
        feed['attention_mask'] = new ort.Tensor('int64',
            BigInt64Array.from({ length: seqLen }, () => 1n), [1, seqLen]);
        feed['position_ids'] = this._makePositionIds(0, seqLen);

        if (this.profiler) this.decoderSession.startProfiling();

        // Decode loop: greedy argmax, one token per step
        let lastToken      = -1n;
        let curLen          = seqLen;   // tracks total sequence length (prefill + decoded)
        let firstTokenTime  = 0;
        let tokenCount      = 0;

        // Pre-allocate buffers to avoid per-step allocation
        const maskBuffer       = new BigInt64Array(max_tokens + 1);
        maskBuffer.fill(1n, 0, seqLen);
        const decodeIdBuf      = new BigInt64Array(1);
        const posIdBuf         = new BigInt64Array(3);

        // Reused across decode steps: input_ids is always [1, 1] backed by
        // decodeIdBuf, so create the tensor once and just mutate decodeIdBuf[0].
        // (attention_mask can't be reused — its shape grows [1, curLen] each step.)
        const decodeIdTensor   = new ort.Tensor('int64', decodeIdBuf, [1, 1]);

        // Static KV buffers, bound in place: the same GPU buffer is fed as the
        // past input and requested as the present output each step. Pre-build the
        // fetches map once (cache outputs -> their buffer, everything else -> null
        // for ORT to allocate) so the decode loop reuses it instead of rebuilding
        // a map every step.
        const fetches = {};
        for (const name of this.decoderOutNames) {
            fetches[name] = (name in this.kvOutToPast) ? this.kvSet[this.kvOutToPast[name]] : null;
        }
        for (const p of this.kvPairs) feed[p.pastName] = this.kvSet[p.pastName];

        const startTime     = performance.now();
        while (!this.eos.includes(lastToken) && (seqLen + tokenCount) < max_tokens && !this.stop) {
            // Present KV outputs are written back into the same GPU buffers.
            const outputs = await this.decoderSession.run(feed, fetches);

            lastToken = BigInt(argmax(outputs.logits));
            this.output_tokens.push(lastToken);
            tokenCount++;

            if (callback) callback(this.output_tokens);
            if (firstTokenTime === 0) firstTokenTime = performance.now();

            if (this.eos.includes(lastToken)) break;

            // Prepare next decode step: input_ids, attention_mask, position_ids
            decodeIdBuf[0] = lastToken;
            feed['input_ids'] = decodeIdTensor;

            // Extend attention mask in pre-allocated buffer (no copy — uses subarray view)
            maskBuffer[curLen] = 1n;
            curLen++;
            feed['attention_mask'] = new ort.Tensor('int64',
                maskBuffer.subarray(0, curLen), [1, curLen]);

            // Position IDs for single decode token
            feed['position_ids'] = this._makeSinglePositionId(curLen - 1, posIdBuf);
        }

        const endTime = performance.now();
        if (this.profiler) this.decoderSession.endProfiling();

        return [
            this.output_tokens,
            (endTime - startTime) / 1000,           // total time
            (endTime - firstTokenTime) / 1000,       // decode time
            (firstTokenTime - startTime) / 1000,     // prefill time (TTFT)
        ];
    }
}
