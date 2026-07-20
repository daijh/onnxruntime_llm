/**
 * llm.js — Generic single-model LLM inference using Transformers.js + WebGPU.
 *
 * The full generation pipeline is handled by Transformers.js:
 *   - `AutoModelForCausalLM.from_pretrained` loads the merged decoder, selects
 *     the ONNX file by `dtype`, wires up the KV cache, and runs on WebGPU.
 *   - `model.generate({ ..., streamer })` runs the full generation loop
 *     (prefill + greedy/sampled decode, EOS handling, dynamic KV cache growth).
 *   - `TextStreamer` streams decoded text token-by-token for a live UI.
 *
 * Capabilities (position ids, use_cache_branch, KV cache shapes, EOS ids) are
 * all detected internally by Transformers.js from the model's config, so no
 * per-model configuration is required.
 */

import {
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    InterruptableStoppingCriteria,
    env,
} from '@huggingface/transformers';
import { wasmPaths } from './config.js';

// Point Transformers.js' bundled onnxruntime-web at the same WASM/JSEP binaries
// the import map resolves (local node_modules or the jsDelivr CDN). Single-
// threaded WASM: multi-thread needs cross-origin isolation (COOP/COEP), which
// static hosts like GitHub Pages can't set. WebGPU runs compute on the GPU, so
// this doesn't affect inference speed.
env.backends.onnx.wasm.wasmPaths = wasmPaths;
env.backends.onnx.wasm.numThreads = 1;

function log(i) { console.log(i); }

// ---------------------------------------------------------------------------
// LLM class — single-model pipeline (merged decoder via Transformers.js)
// ---------------------------------------------------------------------------

export class LLM {
    tokenizer = null;
    model     = null;

    // Per-run stopping criteria (recreated each generate() so abort() only
    // affects the in-flight run).
    stoppingCriteria = null;

    /**
     * Load tokenizer + model.
     *
     * @param {string} modelId  HF repo id (remote) or local path (e.g. /models/foo).
     * @param {object} options  { dtype, device, verbose, chatTemplateUrl, progress_callback }.
     */
    async load(modelId, options = {}) {
        const dtype  = options.dtype  || 'q4f16';
        const device = options.device || 'webgpu';

        if (options.verbose) env.backends.onnx.logLevel = 'verbose';

        log(`loading... ${modelId}, ${device}, dtype=${dtype}`);

        this.tokenizer = await AutoTokenizer.from_pretrained(modelId, {
            progress_callback: options.progress_callback,
        });

        // If the tokenizer has no chat_template, try loading chat_template.jinja
        // from the model dir (some repos ship the template as a separate file).
        if (!this.tokenizer.chat_template && options.chatTemplateUrl) {
            try {
                const resp = await fetch(options.chatTemplateUrl);
                if (resp.ok) {
                    this.tokenizer.chat_template = await resp.text();
                    log('[load] loaded chat_template.jinja');
                }
            } catch (_) { /* ignore */ }
        }

        this.model = await AutoModelForCausalLM.from_pretrained(modelId, {
            dtype,
            device,
            progress_callback: options.progress_callback,
        });

        log('Model loaded.');
    }

    /** Signal the in-flight generation to stop. */
    abort() { if (this.stoppingCriteria) this.stoppingCriteria.interrupt(); }

    /** Release the model's ONNX sessions (frees GPU/WASM memory). Safe to call twice. */
    async dispose() {
        if (this.model) { await this.model.dispose(); this.model = null; }
        this.tokenizer = null;
    }

    /**
     * Build model inputs from a prompt.
     *
     * @param {string} userPrompt
     * @param {object} opts  { useChatTemplate, enableThinking }.
     * @returns tokenized inputs dict ({ input_ids, attention_mask }) from the tokenizer.
     */
    tokenize(userPrompt, opts = {}) {
        if (opts.useChatTemplate) {
            return this.tokenizer.apply_chat_template(
                [{ role: 'user', content: userPrompt }],
                {
                    add_generation_prompt: true,
                    return_dict: true,
                    enable_thinking: opts.enableThinking,
                });
        }
        return this.tokenizer(userPrompt);
    }

    /**
     * Run generation over pre-tokenized inputs, streaming decoded text via
     * `onText`. Returns timing/token metrics.
     *
     * @param {object}   inputs   tokenized dict ({ input_ids, attention_mask }).
     * @param {function} onText   called with the full decoded text so far.
     * @param {object}   options  { max_tokens }.
     */
    async generate(inputs, onText, options = {}) {
        this.stoppingCriteria = new InterruptableStoppingCriteria();

        let text            = '';
        let firstTokenTime  = 0;

        const streamer = new TextStreamer(this.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (t) => {
                if (firstTokenTime === 0) firstTokenTime = performance.now();
                text += t;
                if (onText) onText(text);
            },
        });

        const promptTokens = Number(inputs.input_ids.dims.at(-1));

        const startTime = performance.now();
        const output = await this.model.generate({
            ...inputs,
            max_new_tokens: options.max_tokens || 128,
            do_sample: false,
            streamer,
            stopping_criteria: this.stoppingCriteria,
            // Force a plain token-id Tensor back (some models default this to
            // true in generation_config.json, which would return an object).
            return_dict_in_generate: false,
        });
        const endTime = performance.now();

        // `output` is a Tensor [1, promptTokens + generated] of full sequences.
        const totalTokens  = Number(output.dims.at(-1));
        const decodeTokens = Math.max(totalTokens - promptTokens, 0);

        // Final text from the full sequence (authoritative; the streamed text
        // should already match it).
        const finalText = this.tokenizer
            .batch_decode(output.slice(null, [promptTokens, totalTokens]), { skip_special_tokens: true })[0] || text;

        const ttft = (firstTokenTime ? firstTokenTime - startTime : 0) / 1000;    // prefill / time-to-first-token
        const genTime = (firstTokenTime ? endTime - firstTokenTime : endTime - startTime) / 1000;

        return {
            text: finalText,
            promptTokens,
            decodeTokens,
            ttft,
            genTime,
            totalTime: (endTime - startTime) / 1000,
        };
    }
}
