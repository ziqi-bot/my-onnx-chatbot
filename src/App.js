





import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AutoTokenizer } from '@xenova/transformers';
import { LLM } from './llm';
import { marked } from 'marked';
import PdfParser from './PdfParser'; // Import PdfParser
import './App.css';

const preCannedQueries = {
  "1": "Tell me about the lighthouse of Alexandria.",
  "2": "Did the lighthouse of Alexandria existed at the same time the library of Alexandria existed?",
  "3": "How did the Pharos lighthouse impact ancient maritime trade?",
  "4": "Tell me about Constantinople.",
};

marked.use({ mangle: false, headerIds: false });

const App = () => {
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [llm, setLLM] = useState(null);
  const [tokenizer, setTokenizer] = useState(null);
  const [status, setStatus] = useState('');
  const [pdfContent, setPdfContent] = useState(''); // Change from pdfText to pdfContent
  const chatContainerRef = useRef(null);

  const config = {
    model: "phi3",
    provider: "webgpu",
    profiler: 0,
    verbose: 0,
    threads: 1,
    show_special: 0,
    csv: 0,
    max_tokens: 9999,  // Adjusted to prevent endless token generation
    local: 1,
  };

  const log = (message) => {
    setStatus((prevStatus) => `${prevStatus}\n${message}`);
    setTimeout(() => {
      setStatus('');
    }, 3000); // 3 seconds
  };

  const Init = useCallback(async (llmInstance, hasFP16) => {
    try {
      log("Loading model...");
      await llmInstance.load(config.model, {
        provider: config.provider,
        profiler: config.profiler,
        verbose: config.verbose,
        local: config.local,
        max_tokens: config.max_tokens,
        hasFP16: hasFP16 === 2, // Only set true if hasFP16 is 2
      });
      log("Ready.");
      if (hasFP16 === 2) {
        console.log("WebGPU supports fp16.");
      } else if (hasFP16 === 1) {
        console.log("WebGPU does not support fp16, using fp32 instead.");
      } else {
        console.log("WebGPU is not supported.");
      }
    } catch (error) {
      log(`Model loading failed: ${error.message}`);
    }
  }, []);

  useEffect(() => {
    async function initialize() {
      try {
        const llmInstance = new LLM();
        setLLM(llmInstance);

        log("Loading tokenizer...");
        const tokenizerInstance = await AutoTokenizer.from_pretrained('');  // Specify the model path here if needed
        setTokenizer(() => tokenizerInstance);
        log("Tokenizer loaded successfully.");

        const hasFP16 = await checkWebGPU();
        await Init(llmInstance, hasFP16);
      } catch (error) {
        console.error("Failed to initialize:", error);
        log(`Failed to initialize: ${error.message}`);
      }
    }
    initialize();
  }, [Init]);

  const checkWebGPU = async () => {
    if (!("gpu" in navigator)) {
      return 0;  // WebGPU is not supported
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter.features.has('shader-f16')) {
        return 2;  // Supports fp16
      }
      return 1;  // Does not support fp16, but supports WebGPU with fp32
    } catch (e) {
      return 0;  // WebGPU is not supported
    }
  };

  const token_to_text = (tokenizer, tokens, startidx) => {
    const txt = tokenizer.decode(tokens.slice(startidx), { skip_special_tokens: config.show_special !=1 });
    return txt;
  };

  const Query = async (continuation, query, cb) => {
    let prompt = (continuation) ? query : `<|system|>\nYou are a friendly assistant.<|end|>\n<|user|>\n${query}<|end|>\n<|assistant|>\n`;

    const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });

    llm.initilize_feed();

    const start_timer = performance.now();
    const output_index = llm.output_tokens.length + input_ids.length;

    try {
      const output_tokens = await llm.generate(input_ids, (output_tokens) => {
        if (output_tokens.length === input_ids.length + 1) {
          const took = (performance.now() - start_timer) / 1000;
          console.log(`time to first token in ${took.toFixed(1)}sec, ${input_ids.length} tokens`);
        }
      }, { max_tokens: 9999 });

      const took = (performance.now() - start_timer) / 1000;
      const responseText = token_to_text(tokenizer, output_tokens, output_index);
      cb(responseText);
      const seqlen = output_tokens.length - output_index;
      console.log(`${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec`);
    } catch (error) {
      console.error(`Error in LLM.generate: ${error.message}`);
      throw error;
    }
  };

  const submitRequest = async (e) => {
    e.preventDefault();

    if (isSending) {
      llm.abort();
      setIsSending(false);
      return;
    }

    if (!tokenizer) {
      log("Tokenizer not initialized.");
      return;
    }

    const userMessage = input;
    if (!userMessage.trim()) return;

    setChatHistory((prevHistory) => [...prevHistory, { type: 'user', text: userMessage }]);
    setInput('');
    setIsSending(true);

    try {
      const continuation = e.ctrlKey && e.key === 'Enter';
      const query = pdfContent ? `${pdfContent}\n\n${userMessage}` : userMessage;
      await Query(continuation, query, (word) => {
        setChatHistory((prevHistory) => [
          ...prevHistory,
          { type: 'bot', text: marked.parse(word) },
        ]);
      });
    } catch (error) {
      console.error(error);
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { type: 'error', text: `Error generating response: ${error.message}` },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleKeyDown = (e) => {
    if (e.ctrlKey) {
      if (e.key === 'Enter') {
        submitRequest(e);
      } else {
        const query = preCannedQueries[e.key];
        if (query) {
          setInput(query);
          submitRequest(e);
        }
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      submitRequest(e);
    }
  };

  const handlePdfParsed = (text) => {
    setPdfContent(text); // Store the parsed text
    setChatHistory((prevHistory) => [
      ...prevHistory,
      { type: 'bot', text: marked.parse(`PDF parsed successfully!`) },
    ]);
  };

  return (
    <div className="container">
      <div className="row pt-3">
        <div className="col-md-8 col-12">
          <h2>   My ChatBot ONNX</h2>
        </div>
        <div id="status" className="col-md-12 col-12">{status}</div>
      </div>
      <div id="chat-container" ref={chatContainerRef}>
        <div id="chat-history">
          {chatHistory.map((message, index) => (
            <div key={index} className={message.type === 'user' ? 'user-message mb-2' : 'response-message mb-2 text-start'}>
              <span dangerouslySetInnerHTML={{ __html: message.text }} />
            </div>
          ))}
        </div>
      </div>
      <div id="input-area">
        <textarea 
          id="user-input" 
          placeholder="Type your question here ..."
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          onKeyDown={handleKeyDown}
        ></textarea>
        <button id="send-button" onClick={submitRequest}>
          {isSending ? 'Stop' : 'Send'}
        </button>
        <PdfParser onPdfParsed={handlePdfParsed} /> {/* Use PdfParser component */}
      </div>
    </div>
  );
};

export default App;
























