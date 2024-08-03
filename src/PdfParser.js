












import React, { useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist/webpack';
import Tesseract from 'tesseract.js';
import PropTypes from 'prop-types';
import { createCanvas } from 'canvas';

const PdfParser = ({ onPdfParsed }) => {
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState(0);

  const log = (message) => {
    setStatus(message);
    setTimeout(() => {
      setStatus('');
    }, 3000); // 3 seconds
  };

  const extractTextAndCheckForImages = async (arrayBuffer) => {
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    const totalPages = pdf.numPages;
    let textContent = '';
    let containsImages = false;
    let containsText = false;

    for (let i = 1; i <= totalPages; i++) {
      const page = await pdf.getPage(i);
      const textContentItems = await page.getTextContent();
      if (textContentItems.items.length > 0) {
        containsText = true;
      }
      const textItems = textContentItems.items.map((item) => item.str).join(' ');
      textContent += textItems + '\n';

      const ops = await page.getOperatorList();
      for (let j = 0; j < ops.fnArray.length; j++) {
        if (ops.fnArray[j] === pdfjsLib.OPS.paintJpegXObject || ops.fnArray[j] === pdfjsLib.OPS.paintImageXObject) {
          containsImages = true;
        }
      }

      setProgress(Math.round((i / totalPages) * 100));
    }

    return { textContent, containsImages, containsText, pdf };
  };

  const renderPageToImage = async (page) => {
    const viewport = page.getViewport({ scale: 2.0 });
    const canvas = createCanvas(viewport.width, viewport.height);
    const context = canvas.getContext('2d');

    const renderContext = {
      canvasContext: context,
      viewport: viewport
    };

    await page.render(renderContext).promise;
    return canvas.toDataURL();
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setStatus('');
      setProgress(0);

      log("Parsing PDF...");
      try {
        const arrayBuffer = await file.arrayBuffer();
        const { textContent, containsImages, containsText, pdf } = await extractTextAndCheckForImages(arrayBuffer);

        if (containsImages) {
          log("Performing OCR...");
          let ocrText = '';
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const image = await renderPageToImage(page);
            const ocrResult = await Tesseract.recognize(
              image,
              'eng',
              {
                logger: (m) => {
                  if (m.status === 'recognizing text') {
                    const pageProgress = (i - 1) / pdf.numPages;
                    const ocrProgress = m.progress / pdf.numPages;
                    const totalProgress = pageProgress + ocrProgress;
                    setProgress(Math.min(Math.round(totalProgress * 100), 100));
                  }
                },
              }
            );
            ocrText += ocrResult.data.text + '\n';
          }
          const combinedText = textContent + '\n' + ocrText;
          onPdfParsed(combinedText);
        } else if (containsText) {
          onPdfParsed(textContent);
        } else {
          log("No text or images found in PDF.");
        }

        log("PDF parsed successfully.");
        setProgress(100); // Set progress to 100% when done
      } catch (error) {
        console.error("Error processing file:", error);
        log(`Error processing file: ${error.message}`);
      }
    }
  };

  return (
    <div>
      <input type="file" accept="application/pdf" onChange={handleFileUpload} />
      <div>{status}</div>
      <div className="progress-bar">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress}%` }}
        >
          {progress}%
        </div>
      </div>
    </div>
  );
};

PdfParser.propTypes = {
  onPdfParsed: PropTypes.func.isRequired,
};

export default PdfParser;
