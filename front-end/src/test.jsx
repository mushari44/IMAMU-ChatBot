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

  // Fetch uploaded documents on mount and after uploads
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await axios.get('http://localhost:8000/documents');
        setUploadedDocuments(response.data.documents);
      } catch (error) {
        console.error('Error fetching documents:', error);
        setUploadStatus({ type: 'error', message: 'Failed to load document list' });
      }
    };
    fetchDocuments();
  }, [uploadStatus]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: '.txt',
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
      setUploadedDocuments(prev => [...new Set([...prev, file.name])]); // Prevent duplicates
      
    } catch (error) {
      let message = 'Error processing document';
      if (error.response) {
        message = error.response.data.detail || message;
        if (error.response.status === 413) message = 'File size exceeds 10MB limit';
      } else if (error.request) {
        message = 'No response from server - check connection';
      }
      setUploadStatus({ type: 'error', message });
    } finally {
      setIsLoading(prev => ({ ...prev, upload: false }));
      setFile(null);
    }
  };

  const handleQuestion = async () => {
    if (!question.trim()) return;

    try {
      setIsLoading(prev => ({ ...prev, question: true }));
      const response = await axios.post('http://localhost:8000/ask', 
        { question },
        { timeout: 15000 }
      );
      console.log("response", response.data.answer);
      console.log("sources", response.data.sources);
      
      
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
      let message = 'Error getting answer';
      if (error.response) message = error.response.data.detail || message;
      if (error.code === 'ECONNABORTED') message = 'Request timed out';
      setChatHistory(prev => [...prev, { 
        type: 'error', 
        content: message,
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(prev => ({ ...prev, question: false }));
    }
  };

  const handleReset = async () => {
    try {
      await axios.delete('http://localhost:8000/reset');
      setUploadedDocuments([]);
      setChatHistory([]);
      setUploadStatus({ type: 'success', message: 'System reset successfully' });
    } catch (error) {
      setUploadStatus({ type: 'error', message: 'Error resetting system' });
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Arabic Document Analyzer</h1>
        <div className="documents-info">
          <div className="document-count">
            <span>Loaded Documents: {uploadedDocuments.length}</span>
            <button 
              onClick={handleReset}
              className="reset-btn"
              aria-label="Reset system"
            >
              <RotateCw size={16} /> Reset
            </button>
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
            <div 
              key={index}
              className={`chat-message ${msg.type} ${msg.type === 'answer' ? 'rtl' : ''}`}
              aria-live={msg.type === 'answer' ? 'polite' : 'off'}
            >
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
                  <div className="message-content">{msg.content}</div>
                  {msg.sources && (
                    <div className="sources">
                      <div className="sources-title">References:</div>
                      {msg.sources.map((source, i) => (
                        <div key={i} className="source-item">
                          <span className="source-file">{source.file}</span>
                          <span className="source-chunk">Chunk {source.chunk}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="message-time">
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              )}
              
              {msg.type === 'error' && (
                <div className="error-message">
                  <AlertCircle size={16} />
                  {msg.content}
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