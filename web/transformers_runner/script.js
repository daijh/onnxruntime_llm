/**
 * script.js — UI controller for Transformers Runner (Transformers.js + WebGPU).
 *
 * Handles model loading, prompt construction, token generation,
 * and performance metric display. Inference is delegated to the `LLM` class in
 * llm.js, which wraps Transformers.js' AutoModelForCausalLM + generate().
 */

import { env, Tensor } from '@huggingface/transformers';
import { LLM } from './llm.js';
import { IS_LOCAL, ORT_VERSION } from './config.js';

// Log the loaded dependency versions for diagnostics. The onnxruntime-web
// version is read from the actually-loaded module (resolved by index.html's
// import map, NOT bundled by Transformers.js), falling back to the config pin.
const _ortVersions = env.backends?.onnx?.versions;
console.log(`Transformers.js v${env.version} · onnxruntime-web v${_ortVersions?.web || _ortVersions?.common || ORT_VERSION}`);

// Model source, chosen at runtime by detectModels():
//   - On localhost, scan the local /models/ directory first, and fall back to the
//     models.json manifest (Hugging Face Hub) only if no local models are found.
//   - On any other host (e.g. GitHub Pages), use the manifest.
// The transformers.js env is configured per mode in the discovery functions, so it
// is independent of where the JS dependencies come from (see config.js).

// ---------------------------------------------------------------------------
// DOM elements
// ---------------------------------------------------------------------------
const modelSelect           = document.getElementById('modelSelect');
const loadModelButton       = document.getElementById('loadModelButton');
const promptInput           = document.getElementById('promptInput');
const generateResponseButton = document.getElementById('generateResponseButton');
const outputDiv             = document.getElementById('output');
const statusOutputDiv       = document.getElementById('statusOutput');
const webgpuInfoDiv         = document.getElementById('webgpuInfo');
const useChatTemplateCheckbox = document.getElementById('useChatTemplateCheckbox');
const enableThinkingCheckbox = document.getElementById('enableThinkingCheckbox');
const verboseCheckbox       = document.getElementById('verboseCheckbox');
const prefillNInput         = document.getElementById('prefillNInput');
const decodeNInput          = document.getElementById('decodeNInput');
const loopNInput            = document.getElementById('loopNInput');
const maxContextInput       = document.getElementById('maxContextInput');
const loadStatus            = document.getElementById('loadStatus');
const loadProgress          = document.getElementById('loadProgress');
const loadStatusText        = document.getElementById('loadStatusText');

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
let llm             = null;
let webGpuAvailable = false;

// Discovered models: option value -> { modelId, dtype, chatTemplateUrl, label }.
// Filled by detectModels() from either the manifest (remote) or a local /models/ scan.
const modelEntries = new Map();

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function updateOutput(message) {
    const scrollX = window.scrollX, scrollY = window.scrollY;
    outputDiv.textContent = message;
    window.scrollTo(scrollX, scrollY);
}
function updateOutputStatus(message) { statusOutputDiv.textContent = message; }

/** Show the loading progress bar under the Load button. pct null = indeterminate. */
function setLoadProgress(text, pct) {
    if (!loadStatus) return;
    loadStatus.hidden = false;
    loadProgress.hidden = false;
    if (pct == null || Number.isNaN(pct)) loadProgress.removeAttribute('value');
    else loadProgress.value = Math.max(0, Math.min(100, pct));
    loadStatusText.textContent = text || '';
}

/** Finish loading: hide the bar, optionally leaving a final status line. */
function endLoadProgress(text) {
    if (!loadStatus) return;
    loadProgress.hidden = true;
    if (text) { loadStatus.hidden = false; loadStatusText.textContent = text; }
    else loadStatus.hidden = true;
}

function setButtonStates(isLoading) {
    loadModelButton.disabled = isLoading || !webGpuAvailable;
    promptInput.disabled = isLoading || !webGpuAvailable;
    generateResponseButton.disabled = isLoading || !llm || !webGpuAvailable;
}

function detectWebGPU() {
    webGpuAvailable = !!navigator.gpu;
    if (!webGpuAvailable) {
        updateOutputStatus("WebGPU is NOT available. Please use a supported browser (Chrome/Edge).");
    }
    setButtonStates(false);
    renderWebGpuInfo();
}

/** Query the WebGPU adapter and render its details into the #webgpuInfo panel. */
async function renderWebGpuInfo() {
    if (!webgpuInfoDiv) return;

    const setUnavailable = (msg) => {
        webgpuInfoDiv.className = 'webgpu-info unavailable';
        webgpuInfoDiv.textContent = '⚠ ' + msg;
    };

    if (!navigator.gpu) { setUnavailable('WebGPU not available — use a recent Chrome or Edge.'); return; }

    webgpuInfoDiv.className = 'webgpu-info';
    webgpuInfoDiv.textContent = 'Querying WebGPU adapter…';

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { setUnavailable('No WebGPU adapter found.'); return; }

        // adapter.info is a property in recent Chrome; older builds use requestAdapterInfo().
        let info = adapter.info || {};
        if (!adapter.info && adapter.requestAdapterInfo) {
            try { info = await adapter.requestAdapterInfo(); } catch (_) { /* ignore */ }
        }
        console.log('GPU Adapter Info:', info);

        const items = [];
        for (const key of ['vendor', 'architecture', 'device', 'description']) {
            if (info[key]) items.push([key[0].toUpperCase() + key.slice(1), String(info[key])]);
        }
        const maxBuf = adapter.limits?.maxBufferSize;
        if (maxBuf) items.push(['Max buffer', (maxBuf / (1024 ** 3)).toFixed(2) + ' GB']);

        webgpuInfoDiv.className = 'webgpu-info';
        webgpuInfoDiv.replaceChildren();

        const badge = document.createElement('span');
        badge.className = 'wg-badge';
        badge.textContent = 'WebGPU';
        webgpuInfoDiv.appendChild(badge);

        if (items.length === 0) {
            webgpuInfoDiv.appendChild(document.createTextNode('adapter details hidden by the browser'));
            return;
        }
        for (const [k, v] of items) {
            const span = document.createElement('span');
            span.className = 'wg-item';
            const b = document.createElement('b');
            b.textContent = k + ': ';
            span.append(b, v);
            webgpuInfoDiv.appendChild(span);
        }
    } catch (e) {
        console.error('Error retrieving GPU info:', e);
        setUnavailable('Error querying WebGPU adapter (see console).');
    }
}

/**
 * Parse directory links from an http-server HTML listing.
 * Supports both ecstatic-style (#files a.icon-directory) and
 * table-style (td.name a) formats.
 */
function parseDirLinks(doc, { dirs = true, files = true } = {}) {
    // Strategy 1: ecstatic format — <ul id="files"><li><a class="icon icon-directory">
    let links = dirs
        ? [...doc.querySelectorAll('#files a.icon-directory')]
        : [...doc.querySelectorAll('#files a.icon:not(.icon-directory)')];
    if (links.length > 0) return links;

    // Strategy 2: table format — <table><td class="name"><a href="...">name/</a>
    links = [...doc.querySelectorAll('td.name a, td a')];
    if (links.length > 0) {
        return links.filter(a => {
            const href = a.getAttribute('href') || '';
            if (dirs && !files) return href.endsWith('/');
            if (files && !dirs) return !href.endsWith('/');
            return true;
        });
    }

    // Strategy 3: fallback — any <a> whose href looks like a path
    links = [...doc.querySelectorAll('a[href]')];
    return links.filter(a => {
        const href = a.getAttribute('href') || '';
        if (href === '/' || href === './' || href === '../' || href.startsWith('?')) return false;
        if (dirs && !files) return href.endsWith('/');
        if (files && !dirs) return !href.endsWith('/');
        return true;
    });
}

// Transformers.js dtype suffixes, checked longest-first so e.g. `q4f16` wins
// over `q4`. Only the dtype token matters — Transformers.js selects the actual
// ONNX file(s) itself, whether the export is a single `model_<dtype>.onnx` or a
// split `decoder_model_merged_<dtype>.onnx` (+ `embed_tokens_<dtype>.onnx`).
const KNOWN_DTYPES = ['q4f16', 'bnb4', 'q4', 'fp16', 'fp32', 'int8', 'uint8', 'q8', 'quantized'];

/** Derive the Transformers.js `dtype` from an ONNX file name (`*_<dtype>.onnx`). */
function dtypeFromOnnx(onnxPath) {
    if (!onnxPath) return 'q4f16';
    const base = (onnxPath.split('/').pop() || '').replace(/\.onnx$/i, '').toLowerCase();
    return KNOWN_DTYPES.find(d => base.endsWith('_' + d)) || 'fp32';
}

/**
 * Populate the model dropdown. On localhost, scan the local /models/ directory
 * first and only fall back to the models.json manifest if it is empty. On any
 * other host (which can't serve directory listings), use the manifest.
 */
async function detectModels() {
    if (IS_LOCAL) {
        const localCount = await detectLocalModels();
        if (localCount > 0) return;
        console.log('[detectModels] no local /models/ found — trying models.json');
    }
    // Static host, or localhost with no local models: use the manifest.
    try {
        const resp = await fetch('./models.json', { cache: 'no-cache' });
        if (resp.ok) {
            const manifest = await resp.json();
            const models = Array.isArray(manifest) ? manifest : (manifest.models || []);
            if (models.length > 0) { populateFromManifest(models); return; }
        }
    } catch (_) { /* no usable manifest */ }
    // Last resort on a static host with no manifest.
    if (!IS_LOCAL) await detectLocalModels();
}

/** Add a model to the dropdown and register its resolved sources. */
function addModelOption(key, label, descriptor) {
    modelEntries.set(key, descriptor);
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = label;
    modelSelect.appendChild(opt);
}

/** Populate the dropdown from a models.json manifest (remote Hugging Face models). */
function populateFromManifest(models) {
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
    modelEntries.clear();
    modelSelect.innerHTML = '';
    models.forEach((entry, i) => {
        const base = (entry.modelUrl || '').replace(/\/+$/, '');
        let onnx = entry.onnxFile;
        if (!onnx && Array.isArray(entry.onnxFiles) && entry.onnxFiles.length) {
            const f = entry.onnxFiles[0];
            onnx = typeof f === 'string' ? f : f.file;
        }
        const label = entry.label || entry.hfId || entry.modelUrl || ('model ' + i);
        // Transformers.js loads by repo id (tokenizer + config + weights); it
        // selects the ONNX file from `dtype`, so we only need the id + dtype.
        addModelOption('m' + i, label, {
            modelId:         entry.hfId || base,
            dtype:           dtypeFromOnnx(onnx),
            chatTemplateUrl: base ? base + '/chat_template.jinja' : '',
            label,
        });
    });
    if (models.length > 0) modelSelect.value = 'm0';
}

/**
 * Scan /models/ for local model folders and register each in the dropdown.
 * A "model folder" is any directory that directly contains a config.json, so
 * org-nested layouts (e.g. org/model/) are listed by their full path. A sample
 * .onnx is picked only to derive the dtype (preferring q4f16). Returns the count.
 */
async function detectLocalModels() {
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = '/';

    let modelRoots = [], onnxFiles = [];
    try {
        ({ modelRoots, onnxFiles } = await scanModelsTree('models'));
    } catch (e) {
        console.warn('Could not scan /models/:', e);
        return 0;
    }

    modelEntries.clear();
    modelSelect.innerHTML = '';
    modelRoots.sort();
    modelRoots.forEach((root, i) => {
        const abs   = '/' + root;
        const onnx  = pickOnnx(root, onnxFiles);
        const label = root.replace(/^models\//, '');
        addModelOption('local' + i, label, {
            modelId:         abs,
            dtype:           dtypeFromOnnx(onnx),
            chatTemplateUrl: abs + '/chat_template.jinja',
            label,
        });
    });
    console.log('[detectModels] found', modelRoots.length, 'local model(s)');
    if (modelRoots.length > 0) modelSelect.value = 'local0';
    return modelRoots.length;
}

/**
 * Pick a sample .onnx under a model root, used only to derive the dtype
 * (Transformers.js selects the real file set itself). Prefer a WebGPU-friendly
 * q4f16 export — covers both `model_q4f16.onnx` and split
 * `decoder_model_merged_q4f16.onnx` layouts.
 */
function pickOnnx(root, onnxFiles) {
    const candidates = onnxFiles.filter(f => f.startsWith(root + '/'));
    if (candidates.length === 0) return '';
    return candidates.find(f => /q4f16\.onnx$/i.test(f)) || candidates[0];
}

/**
 * Recursively walk directory listings under baseDir, collecting model roots
 * (directories that directly contain config.json) and every .onnx file path.
 */
async function scanModelsTree(baseDir) {
    const modelRoots = [];
    const onnxFiles  = [];
    const visited    = new Set();

    async function walk(dirPath) {
        const norm = dirPath.replace(/^\/+/, '').replace(/\/+$/, '');
        if (!norm || visited.has(norm)) return;
        visited.add(norm);

        let resp;
        try { resp = await fetch('/' + norm + '/'); } catch (_) { return; }
        if (!resp.ok) return;
        const doc = new DOMParser().parseFromString(await resp.text(), 'text/html');
        const links = parseDirLinks(doc, { dirs: true, files: true });

        let hasConfig = false;
        const subdirs = [];
        for (const a of links) {
            const href = a.getAttribute('href');
            if (!href || href === '../' || href === './../') continue;
            const full = new URL(href, resp.url).pathname.replace(/^\/+/, '').replace(/\/+$/, '');
            if (!full.startsWith(norm + '/')) continue;
            const name  = full.split('/').pop();
            const isDir = href.endsWith('/') || a.classList.contains('icon-directory');
            if (name === 'config.json') hasConfig = true;
            else if (name.endsWith('.onnx')) onnxFiles.push(full);
            else if (isDir && !name.startsWith('.')) subdirs.push(full);
        }
        if (hasConfig) modelRoots.push(norm);
        for (const d of subdirs) await walk(d);
    }

    await walk(baseDir);
    return { modelRoots, onnxFiles };
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/** Resolve the current dropdown selection into concrete model sources. */
function resolveSelection() {
    return modelEntries.get(modelSelect.value) || null;
}

async function loadAndInitializeModel() {
    if (!webGpuAvailable) { updateOutputStatus("WebGPU not available."); return; }
    const sel = resolveSelection();
    if (!sel)             { updateOutputStatus("Please select a model."); return; }

    setLoadProgress('Loading ' + sel.label + '…', null);
    setButtonStates(true);

    try {
        // Free the previously loaded model's GPU/WASM sessions before loading another.
        if (llm) { try { await llm.dispose(); } catch (_) { /* ignore */ } llm = null; }

        const verbose = verboseCheckbox.checked;
        llm = new LLM();
        await llm.load(sel.modelId, {
            device: "webgpu",
            dtype: sel.dtype,
            verbose,
            chatTemplateUrl: sel.chatTemplateUrl,
            progress_callback: (p) => {
                if (p.status === 'progress' && p.file) {
                    setLoadProgress(`${p.file} — ${Math.round(p.progress || 0)}%`, p.progress || 0);
                } else if ((p.status === 'initiate' || p.status === 'download') && p.file) {
                    setLoadProgress(`Fetching ${p.file}…`, null);
                }
            },
        });

        endLoadProgress(`Model loaded: ${sel.label} (dtype=${sel.dtype})`);
    } catch (error) {
        endLoadProgress(`Error: ${error.message}`);
        console.error("Error loading model:", error.stack || error);
        llm = null;
    } finally {
        setButtonStates(false);
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/**
 * Apply the "Prefill N" control to a tokenized inputs dict: pad (by repeating
 * the prompt) or truncate the token ids to exactly `prefillN`, then rebuild the
 * input_ids / attention_mask tensors. Returns the inputs unchanged when N <= 0.
 */
function applyPrefill(inputs, prefillN) {
    if (!prefillN || prefillN <= 0) return inputs;

    const ids = Array.from(inputs.input_ids.data, (v) => Number(v));
    let finalIds = ids;
    if (ids.length < prefillN) {
        const pad = [];
        while (pad.length < prefillN - ids.length) pad.push(...ids);
        finalIds = [...pad.slice(0, prefillN - ids.length), ...ids];
    } else if (ids.length > prefillN) {
        finalIds = ids.slice(0, prefillN);
    }

    const data = BigInt64Array.from(finalIds, BigInt);
    const dims = [1, finalIds.length];
    return {
        input_ids: new Tensor('int64', data, dims),
        attention_mask: new Tensor('int64', new BigInt64Array(finalIds.length).fill(1n), dims),
    };
}

async function generateResponse() {
    const userPrompt = promptInput.value.trim();
    if (!userPrompt)      { updateOutputStatus("Please enter a prompt."); return; }
    if (!llm)             { updateOutputStatus("Model not loaded."); return; }
    if (!webGpuAvailable) { updateOutputStatus("WebGPU not available."); return; }

    updateOutputStatus("Generating response...");
    setButtonStates(true);

    try {
        const baseInputs = llm.tokenize(userPrompt, {
            useChatTemplate: useChatTemplateCheckbox.checked,
            enableThinking: enableThinkingCheckbox.checked,
        });

        const prefillN = parseInt(prefillNInput.value) || 0;
        const inputs = applyPrefill(baseInputs, prefillN);

        // Max Context caps total tokens (prompt + generated); Decode N caps new tokens.
        const decodeN     = parseInt(decodeNInput.value) || 128;
        const maxContext  = parseInt(maxContextInput.value) || 8192;
        const promptLen   = Number(inputs.input_ids.dims.at(-1));
        const maxNewTokens = Math.max(1, Math.min(decodeN, maxContext - promptLen));
        const loopN       = parseInt(loopNInput.value) || 1;

        let allPerf = "";
        let generatedText = "";
        for (let run = 0; run < loopN; run++) {
            generatedText = "";
            const r = await llm.generate(inputs,
                (text) => { generatedText = text; updateOutput(text); },
                { max_tokens: maxNewTokens });

            generatedText = r.text;
            let perf = `=== Run ${run + 1}/${loopN} ===\n`;
            perf += `  Prefill tokens: ${r.promptTokens}\n`;
            perf += `  Decode tokens:  ${r.decodeTokens}\n`;
            perf += `  TTFT:           ${(r.ttft * 1000).toFixed(1)} ms\n`;
            if (r.ttft > 0)      perf += `  Prefill speed:  ${(r.promptTokens / r.ttft).toFixed(1)} tokens/s\n`;
            if (r.genTime > 0)   perf += `  Decode speed:   ${(r.decodeTokens / r.genTime).toFixed(1)} tokens/s\n`;
            perf += `  Total time:     ${(r.totalTime * 1000).toFixed(1)} ms\n`;

            console.log(perf);
            allPerf += perf;
            updateOutputStatus(allPerf);
        }

        updateOutput(generatedText);
    } catch (error) {
        updateOutput(`Error: ${error.message}`);
        console.error("Generation error:", error.stack || error);
    } finally {
        setButtonStates(false);
    }
}

// ---------------------------------------------------------------------------
// Event listeners & page init
// ---------------------------------------------------------------------------
loadModelButton.addEventListener('click', loadAndInitializeModel);
generateResponseButton.addEventListener('click', generateResponse);

window.addEventListener('load', () => {
    detectWebGPU();
    detectModels();
});
