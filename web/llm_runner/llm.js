/**
 * llm.js — Generic single-session LLM inference using ONNX Runtime WebGPU.
 *
 * Works with any merged-decoder ONNX model. Capabilities (position-id rank,
 * use_cache_branch, num_logits_to_keep, and KV cache shapes/dtypes) are
 * auto-detected from the ONNX session metadata at load time, so no per-model
 * configuration is required.
 *
 * KV cache:
 *   - Static: pre-allocated once to `maxLen` (default 8192) as GPU buffers, so
 *     there is no per-step reallocation and the cache stays on-device.
 *   - Single-buffer (default): the same buffer is bound as both past input and
 *     present output (in-place), using half the memory.
 *   - Double-buffer: two buffer sets alternate read/write roles each step.
 *     Required for models that fuse the KV read and scatter-write into one
 *     compute pass (WebGPU forbids read-only + read-write aliasing), e.g.
 *     Qwen3.5, which is auto-detected. Override via `options.doubleBuffer`.
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

export async function fetchAndCache(url) {
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
    hasCacheBranch  = false;

    // Static GPU-buffer KV cache
    doubleBuffer   = false;     // distinct read/write buffers (needed by e.g. Qwen3.5)
    device         = null;
    maxLen         = MAX_LEN;
    kvPairs        = [];        // [{ outName, pastName, dims, dtype }]
    kvOutToPast    = {};        // present output name -> past input name
    kvSets         = [];        // [{ pastName -> gpu Tensor }] (1 entry = single, 2 = double)

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
    need_position_ids       = false;
    need_num_logits_to_keep = false;
    numLogitsToKeepDims     = [1]; // resolved from metadata during load()
    positionIdRank          = 2;     // 2 = [1,N], 3 = [3,1,N] MRoPE

    async load(model, options) {
        const provider = options.provider || "webgpu";
        this.profiler = options.profiler;
        this.on_profiling_data = options.on_profiling_data;
        this.maxLen = options.maxLen || MAX_LEN;

        log(`loading... ${model}, ${provider}`);
        // Load model config
        const configBytes = await fetchAndCache(model + "/config.json");
        const modelConfig = JSON.parse(new TextDecoder().decode(configBytes));

        // EOS token IDs
        let eosIds = modelConfig.eos_token_id;
        if (!Array.isArray(eosIds)) eosIds = [eosIds];
        this.eos = BigInt64Array.from(eosIds.filter(v => v != null), v => BigInt(v));

        const numLayers = modelConfig.num_hidden_layers;

        // Keep KV cache on GPU for fast decode
        const preferred = {};
        if (provider === "webgpu") {
            for (let i = 0; i < numLayers; ++i) {
                preferred[`present.${i}.key`] = 'gpu-buffer';
                preferred[`present.${i}.value`] = 'gpu-buffer';
            }
        }
        const opt = {
            executionProviders: [provider],
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

        const model_bytes = await fetchAndCache(model_file_path + model_file);
        let modelSize = model_bytes.byteLength;

        // External data naming: transformers.js / optimum use "<model>.onnx_data";
        // ONNX Runtime GenAI uses "<model>.onnx.data". Probe the common one first so
        // typical models never request (and 404) the other. A miss here is expected
        // and not logged; split files (_data_1, _data_2, ...) are handled below.
        const external_data_candidates = [
            model_file + '_data',
            model_file + '.data',
        ];
        let actual_external_data_suffix = null;
        for (const candidate of external_data_candidates) {
            try {
                const resp = await fetch(model_file_path + candidate, { method: 'HEAD' });
                if (resp.ok) { actual_external_data_suffix = candidate; break; }
            } catch (_) { /* probe miss — try the next naming convention */ }
        }

        if (actual_external_data_suffix) {
            const externalDataEntries = [];
            // Load the first external data file
            const firstData = await fetchAndCache(model_file_path + actual_external_data_suffix);
            modelSize += firstData.byteLength;
            externalDataEntries.push({ data: firstData, path: actual_external_data_suffix });
            // Probe for split external data files: _data_1, _data_2, ...
            for (let i = 1; ; i++) {
                const splitName = actual_external_data_suffix + '_' + i;
                try {
                    const resp = await fetch(model_file_path + splitName, { method: 'HEAD' });
                    if (!resp.ok) break;
                    const splitData = await fetchAndCache(model_file_path + splitName);
                    modelSize += splitData.byteLength;
                    externalDataEntries.push({ data: splitData, path: splitName });
                } catch (_) { break; }
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
        this.need_position_ids = inputNameSet.has("position_ids");
        this.need_num_logits_to_keep = inputNameSet.has("num_logits_to_keep");
        this.hasCacheBranch = inputNameSet.has("use_cache_branch");

        // Build input metadata map (local, used for one-time config below)
        const inputMetaMap = {};
        for (const meta of this.decoderSession.inputMetadata) {
            if (meta.isTensor) inputMetaMap[meta.name] = meta;
        }

        // Detect position_ids rank from metadata (e.g. [3, 1, N] for 3D MRoPE)
        if (inputMetaMap["position_ids"]) {
            this.positionIdRank = inputMetaMap["position_ids"].shape.length;
        }

        // Resolve num_logits_to_keep dims (scalar [] vs 1D [1])
        if (this.need_num_logits_to_keep && inputMetaMap["num_logits_to_keep"]) {
            const meta = inputMetaMap["num_logits_to_keep"];
            this.numLogitsToKeepDims = meta.shape.length === 0 ? [] : [1];
        }

        // Log input/output counts for diagnostics
        log(`  inputs (${this.decoderSession.inputNames.length}), outputs (${this.decoderOutNames.length})`);

        // Build present→past KV cache mapping
        const presentToPast = {};
        for (const name of this.decoderOutNames) {
            if (!name.includes('present')) continue;
            const past = name.replace('present', 'past_key_values');
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

        // Single-buffer (same GPU buffer as past & present) is fastest and uses
        // half the memory, but fails on models that fuse the KV read and scatter-
        // write into one compute pass (WebGPU forbids read-only + read-write
        // aliasing). Qwen3.5 is such a model, so it needs double-buffering.
        // options.doubleBuffer forces the mode; otherwise auto-detect Qwen3.5.
        const nameStr = `${model} ${modelConfig._name_or_path || ''} ${(modelConfig.architectures || []).join(' ')}`.toLowerCase();
        const isQwen35 = /qwen-?3\.5|qwen3_5|qwen35/.test(nameStr);
        this.doubleBuffer = (options.doubleBuffer !== undefined) ? options.doubleBuffer : isQwen35;

        this.resetFeed();
        log("Session loaded.");
    }

    // -----------------------------------------------------------------------
    // KV cache management
    // -----------------------------------------------------------------------

    initializeFeed() { this.resetFeed(); }
    resetFeed() {
        // Drop stale feed references (KV tensors are owned by kvSets).
        for (const key of Object.keys(this.feed)) delete this.feed[key];
        this._disposeKvSets();
        const n = this.doubleBuffer ? 2 : 1;
        this.kvSets = Array.from({ length: n }, () => this._allocKvSet());
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

    /** Destroy all KV buffer sets and their underlying GPU buffers. */
    _disposeKvSets() {
        for (const set of this.kvSets) {
            for (const name in set) {
                const buf = set[name].gpuBuffer;
                if (buf && typeof buf.destroy === 'function') buf.destroy();
            }
        }
        this.kvSets = [];
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
        if (this.need_position_ids) {
            feed['position_ids'] = this._makePositionIds(0, seqLen);
        }
        if (this.need_num_logits_to_keep) {
            feed['num_logits_to_keep'] = new ort.Tensor('int64', BigInt64Array.from([1n]), this.numLogitsToKeepDims);
        }

        if (this.profiler) this.decoderSession.startProfiling();

        // Decode loop: greedy argmax, one token per step
        let lastToken      = -1n;
        let curLen          = seqLen;   // tracks total sequence length (prefill + decoded)
        let firstTokenTime  = 0;
        const startTime     = performance.now();
        let tokenCount      = 0;

        // Pre-allocate buffers to avoid per-step allocation
        const maskBuffer       = new BigInt64Array(max_tokens + 1);
        maskBuffer.fill(1n, 0, seqLen);
        const decodeIdBuf      = new BigInt64Array(1);
        const posIdBuf         = new BigInt64Array(3);

        // Pre-create reusable tensors for use_cache_branch
        const cacheBranchFalse = this.hasCacheBranch ? new ort.Tensor('bool', [false], [1]) : null;
        const cacheBranchTrue  = this.hasCacheBranch ? new ort.Tensor('bool', [true], [1]) : null;

        // Static KV buffers. Single-buffer: read == write (in-place). Double-
        // buffer: read from one set, write to the other, swap each step (distinct
        // read/write buffers are required on WebGPU for models like Qwen3.5).
        // Pre-build one fetches map per buffer set (KV outputs -> that set's
        // buffers, everything else -> null for ORT to allocate) so the decode
        // loop reuses them instead of rebuilding a map every step.
        const fetchesPerSet = this.kvSets.map((set) => {
            const f = {};
            for (const name of this.decoderOutNames) {
                f[name] = (name in this.kvOutToPast) ? set[this.kvOutToPast[name]] : null;
            }
            return f;
        });

        let readIdx = 0;
        let writeIdx = this.doubleBuffer ? 1 : 0;
        for (const p of this.kvPairs) feed[p.pastName] = this.kvSets[readIdx][p.pastName];

        while (!this.eos.includes(lastToken) && (seqLen + tokenCount) < max_tokens && !this.stop) {
            if (this.hasCacheBranch) {
                feed['use_cache_branch'] = tokenCount > 0 ? cacheBranchTrue : cacheBranchFalse;
            }

            // Present KV outputs are written into the write-set GPU buffers.
            const outputs = await this.decoderSession.run(feed, fetchesPerSet[writeIdx]);

            lastToken = BigInt(argmax(outputs.logits));
            this.output_tokens.push(lastToken);
            tokenCount++;

            if (callback) callback(this.output_tokens);
            if (firstTokenTime === 0) firstTokenTime = performance.now();

            // Feed the freshly written buffers as next-step past inputs, then swap
            // read/write roles (no-op for single-buffer, where read == write).
            for (const p of this.kvPairs) feed[p.pastName] = this.kvSets[writeIdx][p.pastName];
            if (this.doubleBuffer) [readIdx, writeIdx] = [writeIdx, readIdx];

            if (this.eos.includes(lastToken)) break;

            // Prepare next decode step: input_ids, attention_mask, position_ids
            decodeIdBuf[0] = lastToken;
            feed['input_ids'] = new ort.Tensor('int64', decodeIdBuf, [1, 1]);

            // Extend attention mask in pre-allocated buffer (no copy — uses subarray view)
            maskBuffer[curLen] = 1n;
            curLen++;
            feed['attention_mask'] = new ort.Tensor('int64',
                maskBuffer.subarray(0, curLen), [1, curLen]);

            // Position IDs for single decode token
            if (this.need_position_ids) {
                feed['position_ids'] = this._makeSinglePositionId(curLen - 1, posIdBuf);
            }
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
