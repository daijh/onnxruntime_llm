/**
 * script.js — UI controller for LLM Runner ONNX Runtime WebGPU demo.
 *
 * Handles model loading, prompt construction, token generation,
 * and performance metric display.
 */

import { AutoTokenizer, env } from '@huggingface/transformers';
import { LLM } from './llm.js';
import { IS_LOCAL, ORT_VERSION } from './config.js';

// Log the loaded dependency versions for diagnostics. The onnxruntime-web
// version is read from the actually-loaded module (resolved by index.html's
// import map), falling back to the config pin.
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
// Profiling has no UI control; the element is absent, so this stays null unless
// an #enableProfileCheckbox is re-added to index.html. Kept for future use.
const enableProfileCheckbox = document.getElementById('enableProfileCheckbox');
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
let tokenizer      = null;
let llm            = null;
let webGpuAvailable = false;

// Discovered models: option value -> { tokenizerId, configBaseUrl, onnxFileUrl, chatTemplateUrl, label }.
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
        const onnxUrl = !onnx ? ''
            : /^https?:\/\//.test(onnx) ? onnx
            : base + '/' + onnx.replace(/^\/+/, '');
        const label = entry.label || entry.hfId || entry.modelUrl || ('model ' + i);
        addModelOption('m' + i, label, {
            tokenizerId:     entry.hfId || base,
            configBaseUrl:   base,
            onnxFileUrl:     onnxUrl,
            chatTemplateUrl: base + '/chat_template.jinja',
            label,
        });
    });
    if (models.length > 0) modelSelect.value = 'm0';
}

/**
 * Scan /models/ for local model folders and register each in the dropdown.
 * A "model folder" is any directory that directly contains a config.json, so
 * org-nested layouts (e.g. org/model/) are listed by their full path. The
 * decoder .onnx is auto-selected (preferring model_q4f16.onnx). Returns the count.
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
            tokenizerId:     abs,
            configBaseUrl:   abs,
            onnxFileUrl:     onnx ? '/' + onnx : '',
            chatTemplateUrl: abs + '/chat_template.jinja',
            label,
        });
    });
    console.log('[detectModels] found', modelRoots.length, 'local model(s)');
    if (modelRoots.length > 0) modelSelect.value = 'local0';
    return modelRoots.length;
}

/** Choose the decoder .onnx under a model root, preferring model_q4f16.onnx. */
function pickOnnx(root, onnxFiles) {
    const candidates = onnxFiles.filter(f => f.startsWith(root + '/'));
    if (candidates.length === 0) return '';
    return candidates.find(f => f.endsWith('model_q4f16.onnx')) || candidates[0];
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
// Profiling
// ---------------------------------------------------------------------------
let profileData = '';
function onProfilingData(data) { profileData += JSON.stringify(data) + '\n'; }

function saveDataToFile(content, filename = 'profile-data.json') {
    const blob = new Blob([content], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
        tokenizer = await AutoTokenizer.from_pretrained(sel.tokenizerId);

        // If tokenizer has no chat_template, try loading chat_template.jinja from model dir
        if (!tokenizer.chat_template) {
            try {
                const resp = await fetch(sel.chatTemplateUrl);
                if (resp.ok) {
                    tokenizer.chat_template = await resp.text();
                    console.log('[loadModel] loaded chat_template.jinja');
                }
            } catch (_) { /* ignore */ }
        }

        const profiler = enableProfileCheckbox?.checked ?? false;
        const verbose  = verboseCheckbox.checked;
        const maxLen   = parseInt(maxContextInput?.value) || 8192;
        llm = new LLM();
        await llm.load(sel.configBaseUrl, {
            verbose,
            profiler,
            maxLen,
            onnxFile: sel.onnxFileUrl,
            on_profiling_data: onProfilingData,
            progress_callback: (p) => {
                if (p && p.total) setLoadProgress(`${p.file} — ${Math.round(p.progress)}%`, p.progress);
            },
        });

        endLoadProgress(`Model loaded: ${sel.label}`);
    } catch (error) {
        endLoadProgress(`Error: ${error.message}`);
        console.error("Error loading model:", error.stack || error);
        llm = null;
        tokenizer = null;
    } finally {
        setButtonStates(false);
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/** Decode token ids (from startIdx onward) back to text. */
function tokenToText(tok, tokens, startIdx) {
    const ids = tokens.slice(startIdx);
    return ids.length < 1 ? "" : tok.decode(ids, { skip_special_tokens: true });
}

async function generateResponse() {
    const userPrompt = promptInput.value.trim();
    if (!userPrompt)          { updateOutputStatus("Please enter a prompt."); return; }
    if (!llm || !tokenizer)   { updateOutputStatus("Model not loaded."); return; }
    if (!webGpuAvailable)     { updateOutputStatus("WebGPU not available."); return; }

    updateOutputStatus("Generating response...");
    setButtonStates(true);

    try {
        let prompt;
        if (useChatTemplateCheckbox.checked) {
            prompt = tokenizer.apply_chat_template(
                [{ role: 'user', content: userPrompt }],
                { add_generation_prompt: true, return_dict: false, tokenize: false,
                  enable_thinking: enableThinkingCheckbox.checked });
        } else {
            prompt = userPrompt;
        }

        const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });

        // Prefill padding / truncation
        const prefillN = parseInt(prefillNInput.value) || 0;
        let finalInputIds = input_ids;
        if (prefillN > 0) {
            if (input_ids.length < prefillN) {
                const pad = [];
                while (pad.length < prefillN - input_ids.length) pad.push(...input_ids);
                finalInputIds = [...pad.slice(0, prefillN - input_ids.length), ...input_ids];
            } else if (input_ids.length > prefillN) {
                finalInputIds = input_ids.slice(0, prefillN);
            }
        }

        const decodeN = parseInt(decodeNInput.value) || 128;
        const loopN = parseInt(loopNInput.value) || 1;

        let allPerf = "";
        let generatedText = "";
        for (let run = 0; run < loopN; run++) {
            llm.initializeFeed();

            generatedText = "";
            const [outputTokens, _took, genTime, promptTime] = await llm.generate(finalInputIds,
                (tokens) => {
                    generatedText = tokenToText(tokenizer, tokens, finalInputIds.length);
                    updateOutput(generatedText);
                },
                { max_tokens: decodeN });

            const promptTokens = finalInputIds.length;
            const decodeTokens = Math.max(outputTokens.length - promptTokens - 1, 0);
            let perf = `=== Run ${run + 1}/${loopN} ===\n`;
            perf += `  Prefill tokens: ${promptTokens}\n`;
            perf += `  Decode tokens:  ${decodeTokens}\n`;
            perf += `  TTFT:           ${(promptTime * 1000).toFixed(1)} ms\n`;
            if (promptTime > 0) perf += `  Prefill speed:  ${(promptTokens / promptTime).toFixed(1)} tokens/s\n`;
            if (genTime > 0)    perf += `  Decode speed:   ${(decodeTokens / genTime).toFixed(1)} tokens/s\n`;
            perf += `  Total time:     ${((promptTime + genTime) * 1000).toFixed(1)} ms\n`;

            console.log(perf);
            allPerf += perf;
            updateOutputStatus(allPerf);
        }

        updateOutput(generatedText);

        if (enableProfileCheckbox?.checked) saveDataToFile(profileData, 'ort-web-profile.log');
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
