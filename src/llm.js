

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

export class LLM {
  sess = undefined;
  profiler = false;
  feed = {};
  output_tokens = [];
  eos = 2;
  need_position_ids = true;
  stop = false;
  kv_dims = [];
  dtype = "float16";
  max_tokens = 9999;

  async load(model, options = {}) {
    // console.log("model:", model);
    // console.log("options:", options); // 确认 options 包含 hasFP16
    const provider = options.provider || "webgpu";
    const verbose = options.verbose;
    const hasFP16 = (provider === "wasm") ? false : (options.hasFP16 !== undefined ? options.hasFP16 : true);
    this.profiler = options.profiler;
    // console.log("hasFP16 with default value:", hasFP16); // 调试输出

    // 设置 dtype，根据 hasFP16 参数
    this.dtype = hasFP16 ? 'float16' : 'float32';
    console.log("this.dtype:", this.dtype); // 检查 dtype 设置

    const model_path = "models";
    let model_file = "model_q4f16.onnx";

    console.log(`loading... ${model},  ${provider}`);
    const json_bytes = await fetchAndCache(`${model_path}/config.json`);
    let textDecoder = new TextDecoder();
    const model_config = JSON.parse(textDecoder.decode(json_bytes));

    const model_bytes = await fetchAndCache(`${model_path}/${model_file}`);
    const externaldata = await fetchAndCache(`${model_path}/${model_file}_data`);
    let modelSize = model_bytes.byteLength;
    if (externaldata) {
      modelSize += externaldata.byteLength;
    }
    console.log(`model size ${Math.round(modelSize / 1024 / 1024)} MB`);

    const opt = {
      executionProviders: [provider],
      preferredOutputLocation: {},
    };

    switch (provider) {
      case "webgpu":
        for (let i = 0; i < model_config.num_hidden_layers; ++i) {
          opt.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer';
          opt.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer';
        }
        break;
    }

    if (externaldata) {
      opt.externalData = [
        {
          data: externaldata,
          path: `${model_file}_data`,
        },
      ];
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
    // console.log("model_config:", model_config);
    this.sess = await ort.InferenceSession.create(model_bytes, opt);
    this.eos = model_config.eos_token_id;
    this.kv_dims = [1, model_config.num_key_value_heads, 0, model_config.hidden_size / model_config.num_attention_heads];
    this.dtype = hasFP16 ? "float16" : "float32";
    // console.log("this.dtype after session create:", this.dtype);

    this.num_layers = model_config.num_hidden_layers;
    this.initilize_feed();
  }

  initilize_feed() {
    const feed = this.feed;

    // dispose of previous gpu buffers
    for (const name in feed) {
      const t = feed[name];
      if (t.location === 'gpu-buffer') {
        t.dispose();
      }
    }
    this.feed = {};
    // key value cache is zero copy, just pass gpu buffer as reference
    const empty = this.dtype === "float16" ? new Uint16Array() : [];
    
    for (let i = 0; i < this.num_layers; ++i) {
      this.feed[`past_key_values.${i}.key`] = new ort.Tensor(this.dtype, empty, this.kv_dims);
      this.feed[`past_key_values.${i}.value`] = new ort.Tensor(this.dtype, empty, this.kv_dims);
    }
    this.output_tokens = [];
  }

  argmax(t) {
    const arr = t.data;
    const start = t.dims[2] * (t.dims[1] - 1);
    let max = arr[start];
    let maxidx = 0;

    for (let i = 0; i < t.dims[2]; i++) {
      const val = arr[i + start];
      if (!isFinite(val)) {
        throw new Error("found infinitive in logits");
      }
      if (val > max) {
        max = arr[i + start];
        maxidx = i;
      }
    }
    return maxidx;
  }

  update_kv_cache(feed, outputs) {
    for (const name in outputs) {
      if (name.startsWith('present')) {
        let newName = name.replace('present', 'past_key_values');
        // dispose previous gpu buffers
        const t = feed[newName];
        if (t.location === 'gpu-buffer') {
          t.dispose();
        }
        feed[newName] = outputs[name];
      }
    }
  }

  abort() {
    this.stop = true;
  }

  async generate(tokens, callback, options) {
    const max_tokens = options.max_tokens || 256;
    const feed = this.feed;

    // console.log("Tokens before converting to BigInt:", tokens);
    // 将 tokens 转换为 BigInt 数组
    const input_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
    // console.log("Converted input_ids to BigInt Tensor:", input_ids);

    feed['input_ids'] = input_ids;
    this.stop = false;

    this.output_tokens.push(...input_ids.data);
    // console.log("Initial output_tokens:", this.output_tokens);

    let last_token = 0n;
    let seqlen = this.output_tokens.length;
    const input_len = input_ids.size;

    if (this.need_position_ids) {
      feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from({ length: input_len }, (_, i) => BigInt(seqlen - input_len + i)), [1, input_len]);
    //   console.log("Position IDs Tensor:", feed['position_ids']);
    }

    while (last_token !== this.eos && last_token !== 32007n && seqlen < max_tokens && !this.stop) {
      try {
        seqlen = this.output_tokens.length;
        feed['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen }, () => 1n), [1, seqlen]);
        // console.log("Attention Mask Tensor:", feed['attention_mask']);

        // console.log("Running session with feed:", feed);
        const outputs = await this.sess.run(feed);
        // console.log("Outputs from session run:", outputs);

        last_token = BigInt(this.argmax(outputs.logits));
        // console.log("Last token:", last_token);

        this.output_tokens.push(last_token);
        // console.log("Updated output_tokens:", this.output_tokens);

        if (callback && !this.profiler) {
          callback(this.output_tokens);
        }

        this.update_kv_cache(feed, outputs);

        feed['input_ids'] = new ort.Tensor('int64', BigInt64Array.from([last_token]), [1, 1]);
        // console.log("Next input_ids Tensor:", feed['input_ids']);

        if (this.need_position_ids) {
          feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from([BigInt(seqlen)]), [1, 1]);
        //   console.log("Updated Position IDs Tensor:", feed['position_ids']);
        }
      } catch (error) {
        console.error("Error during session run:", error);
        throw error;
      }
    }

    if (this.profiler) {
      this.sess.endProfiling();
    }

    // console.log("Final output_tokens:", this.output_tokens);
    return this.output_tokens;
  }
}

