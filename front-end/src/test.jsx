import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { RotateCw, AlertCircle } from 'react-feather';

function TxtQAInterface() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [uploadStatus, setUploadStatus] = useState({ type: '', message: '' });
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState({ upload: false, question: false });
  const [uploadedDocuments, setUploadedDocuments] = useState([]);

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await axios.get('http://localhost:8000/documents');
        setUploadedDocuments(response.data.documents);
      } catch (error) {
        setUploadStatus({ type: 'error', message: 'Failed to load document list' });
      }
    };
    fetchDocuments();
  }, [uploadStatus]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'text/plain': ['.txt'], 'application/pdf': ['.pdf'] },
    maxSize: 10 * 1024 * 1024,
    onDrop: acceptedFiles => setFile(acceptedFiles[0]),
    disabled: isLoading.upload,
  });

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    try {
      setIsLoading(prev => ({ ...prev, upload: true }));
      setUploadStatus({ type: 'info', message: 'Processing document...' });

      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000
      });

      setUploadStatus({ type: 'success', message: response.data.message });
      setUploadedDocuments(prev => [...new Set([...prev, file.name])]);
      
    } catch (error) {
      console.log(error);
      
    } finally {
      setIsLoading(prev => ({ ...prev, upload: false }));
      setFile(null);
    }
  };

  const handleQuestion = async () => {
    if (!question.trim()) return;

    try {
      setIsLoading(prev => ({ ...prev, question: true }));
      const response = await axios.post('http://localhost:8000/ask', { question });
      
      setChatHistory(prev => [
        ...prev, 
        { type: 'question', content: question, timestamp: new Date() },
        { 
          type: 'answer', 
          content: response.data.answer,
          sources: response.data.sources,
          timestamp: new Date()
        }
      ]);
      
      setQuestion('');
    } catch (error) {
      console.log(error);
    } finally {
      setIsLoading(prev => ({ ...prev, question: false }));
    }
  };



  const formatAnswer = (content) => {
    const [answerPart, sourcesPart] = content.split(/\nالمصادر:/);
    const answerLines = answerPart.replace(/الإجابة:\s*/, '').split('\n').filter(l => l.trim());
    
    const formattedAnswer = answerLines.map((line, index) => {
      const isList = /^\d+\./.test(line);
      return {
        type: isList ? 'list' : 'paragraph',
        content: line.replace(/\[(\d+)\]/g, '<sup class="reference">[$1]</sup>')
      };
    });

    const sources = sourcesPart?.split('\n')
      .filter(line => line.trim())
      .map(line => {
        const match = line.match(/\[(.*?)\] \((.*?)\)/);
        return match && {
          number: match[1],
          file: match[2].split('،')[0].trim(),
          chunk: match[2].match(/المقطع (\d+)/)?.[1]
        };
      }).filter(Boolean);

    return { formattedAnswer, sources };
  };

  return (
    <div className="container">
<header className="header">
        <h1>IMAMU RAG CHATBOT</h1>
        <div className="documents-info">
          <div className="document-count">
            <span>Loaded Documents: {uploadedDocuments.length}</span>

          </div>
          {uploadedDocuments.length > 0 && (
            <div className="document-list">
              {uploadedDocuments.map((doc, index) => (
                <span key={index} className="document-tag">{doc}</span>
              ))}
            </div>
          )}
        </div>
      </header>

      <section className="upload-section">
        <div 
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''} ${isLoading.upload ? 'disabled' : ''}`}
        >
          <input {...getInputProps()} aria-label="File upload" />
          {file ? (
            <div className="file-preview">
              <p className="filename">{file.name}</p>
              <p className="filesize">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          ) : (
            <div className="dropzone-content">
              {isDragActive ? (
                <p>Drop TXT file here</p>
              ) : (
                <p>Drag/drop TXT file or click to browse</p>
              )}
              <small>Max file size: 10MB</small>
            </div>
          )}
        </div>

        <button 
          onClick={handleUpload}
          disabled={isLoading.upload || !file}
          className={`upload-btn ${isLoading.upload ? 'loading' : ''}`}
        >
          {isLoading.upload ? (
            <><RotateCw className="spin" size={18} /> Processing...</>
          ) : (
            'Analyze Document'
          )}
        </button>

        {uploadStatus.message && (
          <div className={`status-message ${uploadStatus.type}`}>
            {uploadStatus.type === 'error' && <AlertCircle size={16} />}
            {uploadStatus.message}
          </div>
        )}
      </section>


      <section className="chat-section">
        <div className="chat-history">
          {chatHistory.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.type}`}>
              {msg.type === 'question' && (
                <div className="question-bubble">
                  <div className="message-content">{msg.content}</div>
                  <div className="message-time">
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              )}
              
              {msg.type === 'answer' && (
                <div className="answer-bubble">
                  <div className="answer-content" dir="rtl" lang="ar">
                    {formatAnswer(msg.content).formattedAnswer.map((item, i) => (
                      item.type === 'list' ? (
                        <ol key={i} className="answer-list" start="١">
                          <li dangerouslySetInnerHTML={{ __html: item.content }} />
                        </ol>
                      ) : (
                        <p key={i} dangerouslySetInnerHTML={{ __html: item.content }} />
                      )
                    ))}
                  </div>

                  {formatAnswer(msg.content).sources?.length > 0 && (
                    <div className="source-grid">
                      {formatAnswer(msg.content).sources.map((source, i) => (
                        <div key={i} className="source-item">
                          <span className="source-icon">📄</span>
                          <div>
                            <div className="source-file">{source.file}</div>
                            <div className="source-meta">
                              <span>Reference {source.number}</span>
                              <span>• Chunk {source.chunk}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="message-time">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              )}

            </div>
          ))}
        </div>
     
        <div className="question-input">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about the document..."
            disabled={isLoading.question}
            onKeyPress={(e) => e.key === 'Enter' && handleQuestion()}
            dir="auto"
            aria-label="Question input"
          />
          <button 
            onClick={handleQuestion} 
            disabled={isLoading.question || !question.trim()}
            className={isLoading.question ? 'loading' : ''}
          >
            {isLoading.question ? (
              <><RotateCw className="spin" size={16} /> Analyzing...</>
            ) : (
              'Ask Question'
            )}
          </button>
        </div>
      </section>
    </div>
  );
}

export default TxtQAInterface;