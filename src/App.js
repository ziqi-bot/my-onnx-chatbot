

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AutoTokenizer } from '@xenova/transformers';
import { LLM } from './llm';
import { marked } from 'marked';
import PdfParser from './PdfParser';
import './App.css';

const preCannedQueries = {
  "1": "Tell me about the lighthouse of Alexandria.",
  "2": "Did the lighthouse of Alexandria existed at the same time the library of Alexandria existed?",
  "3": "How did the Pharos lighthouse impact ancient maritime trade?",
  "4": "Tell me about Constantinople.",
};

marked.use({ mangle: false, headerIds: false });

const BotB = ({ pdfContent, llm, tokenizer, setChatHistory }) => {
  const CHUNK_SIZE = 900000; // 定义每个片段的大小

  const splitIntoChunks = (text, size) => {
    const regex = new RegExp(`(.|[\r\n]){1,${size}}`, 'g');
    const chunks = text.match(regex) || [];
    console.log(`splitIntoChunks - Total Chunks: ${chunks.length}`);
    chunks.forEach((chunk, index) => {
       console.log(`Chunk ${index + 1}: ${chunk}`);
    });
    return chunks;
  };

  const normalizeText = (text) => {
    return text
      .toLowerCase()
      .replace(/[\W_]+/g, ' ')
      .trim();
  };

  const searchPDF = async (query) => {
    // console.log("searchPDF query:", query);
    // console.log("pdfContent:", pdfContent);

    // const normalizedQuery = normalizeText(query);
    const chunks = splitIntoChunks(pdfContent, CHUNK_SIZE);

    const intermediateResults = await Promise.all(
      chunks.map(async (chunk) => {
        const normalizedChunk = normalizeText(chunk);
       
          
          console.log("searchPDF query:",query);

          const prompt = `<|system|> \nYou are a friendly assistant. Based on the following content, answer the query.<|end|>\n <|user|>\n ${normalizedChunk},${query}<|end|>\n<|assistant|>\n`;
          const promptText='You are a friendly assistant. Based on the following content, answer the query.';

  
   
    
          const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });
          

          llm.initilize_feed();

          const output_tokens = await llm.generate(input_ids, null, { max_tokens: 9999 });
          const responseText = tokenizer.decode(output_tokens, { skip_special_tokens: true });
          console.log("middleResponse:",responseText);
          
          const promptTextLength = promptText.length;
          const normalizedChunkTextLength = normalizedChunk.length;
          const queryTextLength = query.length;
          const assistantresponse = responseText.substring(promptTextLength+normalizedChunkTextLength+queryTextLength);
          setChatHistory((prevHistory) => [
            ...prevHistory,
            { type: 'bot', text: marked.parse(assistantresponse.trim()), botType: 'botB' },
          ]);

          return assistantresponse.trim();
        
      })
    );

    const results = intermediateResults.filter(result => result !== null);
    console.log("searchPDF results:", results);
    return results;
  };

  return { searchPDF };
};

const BotA = ({ pdfContent,botB, llm, tokenizer }) => {
  const analyzeQuery = async (message) => {
    console.log("query:",message);
    
    const prompt = `<|system|>\nYou are a friendly assistant. Does the message ask about a PDF/pdf, a document/file or something in the file/pdf ? Only answer with yes or no.<|end|>\n<|user|>\n${message}<|end|>\n<|assistant|>\n`;
    const promptText = `You are a friendly assistant. Does the message ask about a PDF/pdf, a document/file or something in the file/pdf ? Only answer with yes or no.`;
    
    const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });
    console.log("input_ids",input_ids);
    llm.initilize_feed();


    const output_tokens = await llm.generate(input_ids, null, { max_tokens: 9999 }); //
    const responseText = tokenizer.decode(output_tokens, { skip_special_tokens: true });
    //  console.log('response:',responseText);
     // 解析解码结果，去掉提示内容
    const promptTextLength = promptText.length;
    const messageTextLength = message.length;
    const assistantresponse = responseText.substring(promptTextLength+messageTextLength).toLowerCase();
    console.log('assistantresponse:',assistantresponse);


    return assistantresponse.includes('yes');

  };

  const respond = async (message) => {
    const isPDFRelated = await analyzeQuery(message);
    console.log('isPDFRelated:',isPDFRelated);
    if (isPDFRelated) {
      // console.log("pdfcontent:",pdfContent);
      if (!pdfContent) {
        return { text: "Please upload a PDF document!", botType: 'botA' };
      }
      const intermediateResults = await botB.searchPDF(message);
      // console.log("intermediateResults:", intermediateResults);

      const finalResult = await generateResponse(intermediateResults.join('\n')+message);
      return { text: finalResult, botType: 'botA' };
    } else {
      const response = await generateResponse(message);
      return { text: response, botType: 'botA' };
    }
  };

  const generateResponse = async (message) => {
    let prompt = `<|system|>\nYou are a friendly assistant.<|end|>\n<|user|>\n${message}<|end|>\n<|assistant|>\n`;

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
      const responseText = tokenizer.decode(output_tokens.slice(output_index), { skip_special_tokens: true });
      const seqlen = output_tokens.length - output_index;
      console.log(`${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec`);
      return responseText;
    } catch (error) {
      console.error(`Error in LLM.generate: ${error.message}`);
      throw error;
    }
  };

  return { respond };
};

const App = () => {
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [llm, setLLM] = useState(null);
  const [tokenizer, setTokenizer] = useState(null);
  const [status, setStatus] = useState('');
  const [pdfContent, setPdfContent] = useState('');
  const chatContainerRef = useRef(null);

  const config = {
    model: "phi3",
    provider: "webgpu",
    profiler: 0,
    verbose: 0,
    threads: 1,
    show_special: 0,
    csv: 0,
    max_tokens: 9999,
    local: 1,
  };

  const log = (message) => {
    setStatus((prevStatus) => `${prevStatus}\n${message}`);
    setTimeout(() => {
      setStatus('');
    }, 3000);
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
        hasFP16: hasFP16 === 2,
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
        const tokenizerInstance = await AutoTokenizer.from_pretrained('');
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
      return 0;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter.features.has('shader-f16')) {
        return 2;
      }
      return 1;
    } catch (e) {
      return 0;
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
      const botB = BotB({ pdfContent, llm, tokenizer, setChatHistory });
      const botA = BotA({ pdfContent,botB, llm, tokenizer, setChatHistory });
      const response = await botA.respond(userMessage, setChatHistory);
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { type: 'bot', text: marked.parse(response.text), botType: response.botType },
      ]);
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
    console.log("Parsed PDF content:", text);
    setPdfContent(text);
    setChatHistory((prevHistory) => [
      ...prevHistory,
      { type: 'bot', text: marked.parse(`PDF parsed successfully!`), botType: 'botB' },
    ]);
  };

  return (
    <div className="container">
      <div className="row pt-3">
        <div className="col-md-8 col-12">
          <h2>My ChatBot ONNX</h2>
        </div>
        <div id="status" className="col-md-12 col-12">{status}</div>
      </div>
      <div id="chat-container" ref={chatContainerRef}>
        <div id="chat-history">
          {chatHistory.map((message, index) => (
            <div key={index} className={message.type === 'user' ? 'user-message mb-2' : `response-message mb-2 text-start ${message.botType}`}>
              <span dangerouslySetInnerHTML={{ __html: message.text }} />
              {message.botType && <div className="bot-label">{message.botType === 'botA' ? 'Bot A' : 'Bot B'}</div>}
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
        <PdfParser onPdfParsed={handlePdfParsed} />
      </div>
    </div>
  );
};

export default App;
