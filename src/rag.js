import { pipeline, cos_sim } from '@xenova/transformers';
import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = `${process.env.PUBLIC_URL}/static/js/`;

async function fetchAndCache(url) {
    try {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        return buffer;
    } catch (error) {
        console.error(`can't fetch ${url}`);
        throw error;
    }
}

export class RAG {
    extractor = undefined;
    sess = undefined;
    profiler = false;
    eos = 2;
    dtype = "float16";
    kv_dims = [];
    num_layers = 0;

    constructor() {}

    // Init method to load tokenizer and embedding model
    async load(model, options) {
        const provider = options.provider || "webgpu";
        const verbose = options.verbose;
        const hasFP16 = (provider === "wasm") ? false : (options.hasFP16 !== undefined ? options.hasFP16 : true);
        this.profiler = options.profiler;
        this.dtype = hasFP16 ? 'float16' : 'float32';

        const model_path = "modelRag";
        let model_file = "model_fp16.onnx";

        console.log(`loading... ${model},  ${provider}`);
        const json_bytes = await fetchAndCache(`${model_path}/config.json`);
        let textDecoder = new TextDecoder();
        const model_config = JSON.parse(textDecoder.decode(json_bytes));

        const model_bytes = await fetchAndCache(`${model_path}/${model_file}`);
        let modelSize = model_bytes.byteLength;
        console.log(`model size ${Math.round(modelSize / 1024 / 1024)} MB`);

        const opt = {
            executionProviders: [provider],
            preferredOutputLocation: {},
        };

        switch (provider) {
            case "webgpu":
                for (let i = 0; i < model_config.num_hidden_layers; ++i) {
                    opt.preferredOutputLocation[`present.${i}.key`] = 'cpu';
                    opt.preferredOutputLocation[`present.${i}.value`] = 'cpu';
                }
                break;
        }

        if (verbose) {
            opt.logSeverityLevel = 0;
            opt.logVerbosityLevel = 0;
            ort.env.logLevel = "verbose";
        }

        ort.env.webgpu.profiling = {};
        if (this.profiler) {
            opt.enableProfiling = true;
            ort.env.webgpu.profiling.mode = 'default';
        }

        this.sess = await ort.InferenceSession.create(model_bytes, opt);
        this.eos = model_config.eos_token_id;
        this.kv_dims = [1, model_config.num_key_value_heads, 0, model_config.hidden_size / model_config.num_attention_heads];
        this.num_layers = model_config.num_hidden_layers;

        // Load embedding model
        try {
            this.extractor = await pipeline('feature-extraction', 'Xenova/jina-embeddings-v2-base-en', { quantized: false });
        } catch (error) {
            console.error("Error loading the embedding model:", error);
            throw error;
        }
    }

    // Embedding Algorithm
    async getEmbeddings(query, kbContents) {
        const question = query;
        let sim_result = [];

        for (const content of kbContents) {
            try {
                const output = await this.extractor([question, content], { pooling: 'mean' });
                const sim = cos_sim(output[0].data, output[1].data);
                sim_result.push({ content, sim });
            } catch (error) {
                console.error("Error processing the embeddings:", error);
                throw error;
            }
        }

        sim_result.sort((a, b) => b.sim - a.sim);

        const answer = sim_result.length > 0 ? sim_result[0].content : 'No relevant content found';

        return answer;
    }
}
