import React, { useState, useEffect } from "react";
import axios from "axios";
import { RotateCw, ThumbsUp, ThumbsDown, Copy } from "react-feather";
function TxtQAInterface() {
  const [question, setQuestion] = useState("");
  const [uploadStatus, setUploadStatus] = useState({ type: "", message: "" });
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState({
    upload: false,
    question: false,
  });
  const [documentTitles, setDocumentTitles] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(
    localStorage.getItem("selectedDocument") || ""
  );
  const [selectedFeedbackIndex, setSelectedFeedbackIndex] = useState(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const handleFeedback = async (index, type) => {
    if (type !== "dislike") {
      const updatedChatHistory = [...chatHistory];
      updatedChatHistory[index].feedback = type;
      setChatHistory(updatedChatHistory);

      try {
        const answerIndex = index;
        const questionIndex = index - 1;

        await axios.post("http://localhost:8000/feedback", {
          messageId: index,
          feedback: type,
          question: chatHistory[questionIndex]?.content || "Unknown question",
          answer: chatHistory[answerIndex].content,
          document: selectedDocument,
          references: chatHistory[answerIndex].sources || [],
          context: chatHistory[answerIndex].context,
        });
      } catch (error) {
        console.error("Feedback submission failed:", error);
      }
    }
  };

  const handleCopy = (index) => {
    const answerContent = chatHistory[index].content;
    navigator.clipboard.writeText(answerContent.split(/المصادر:/)[0]);
  };
  const submitFeedback = async (index) => {
    try {
      const answerIndex = index;
      const questionIndex = index - 1;

      await axios.post("http://localhost:8000/feedback", {
        messageId: index,
        feedback: "dislike",
        comment: feedbackComment,
        question: chatHistory[questionIndex]?.content || "Unknown question",
        answer: chatHistory[answerIndex].content,
        document: selectedDocument,
        references: chatHistory[answerIndex].sources || [],
        context: chatHistory[answerIndex].context,
      });

      const updatedChatHistory = [...chatHistory];
      updatedChatHistory[index].feedback = "dislike";
      setChatHistory(updatedChatHistory);

      setSelectedFeedbackIndex(null);
      setFeedbackComment("");
    } catch (error) {
      console.error("Feedback submission failed:", error);
    }
  };
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await axios.get("http://localhost:8000/documents");
        setDocumentTitles(response.data.documents);
      } catch (error) {
        setUploadStatus({
          type: "error",
          message: "Failed to load document list",
        });
      }
    };
    fetchDocuments();
  }, []);

  const handleQuestion = async () => {
    if (!question.trim()) return;

    try {
      setIsLoading((prev) => ({ ...prev, question: true }));
      const response = await axios.post("http://localhost:8000/ask", {
        question,
        document: selectedDocument,
      });

      setChatHistory((prev) => [
        ...prev,
        { type: "question", content: question, timestamp: new Date() },
        {
          type: "answer",
          content: response.data.answer,
          context: response.data.context,
          sources: response.data.sources,
          timestamp: new Date(),
        },
      ]);

      setQuestion("");
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading((prev) => ({ ...prev, question: false }));
    }
  };

  const formatAnswer = (content) => {
    const [answerPart, sourcesPart] = content.split(/المصادر:/);
    const answerLines = answerPart
      .replace(/الإجابة:\s*/, "")
      .split("\n")
      .filter((l) => l.trim());

    const formattedAnswer = answerLines.map((line) => {
      const isList = /^[١-٩]\./.test(line);
      return {
        type: isList ? "list" : "paragraph",
        content: line.replace(
          /\[(\d+)\]/g,
          '<sup class="reference">[$1]</sup>'
        ),
      };
    });
    console.log("Sources Part:", sourcesPart);

    const sources = sourcesPart
      ?.split("\n")
      .filter((line) => line.trim())
      .map((line) => {
        const match = line.match(
          /\[([\d\u0660-\u0669]+)\] (.+?) \(المقطع ([\d\u0660-\u0669]+)\)/
        );
        return (
          match && {
            number: match[1],
            file: match[2],
            chunk: match[3],
          }
        );
      })
      .filter(Boolean);

    console.log("formated source : : ", sources);

    return { formattedAnswer, sources };
  };

  return (
    <div className="container">
      <header className="header">
        <h1>IMAMU RAG CHATBOT</h1>
      </header>

      <section className="chat-section">
        <div className="welcome-mssage" dir="rtl">
          <h2>مرحبًا بك في مساعد الجامعة</h2>
          <p>اختر الوثيقة التي تريد طرح الأسئلة منها:</p>

          <select
            className="question-dropdown"
            onChange={(e) => {
              setSelectedDocument(e.target.value);
              localStorage.setItem("selectedDocument", e.target.value);
            }}
            value={selectedDocument}
            dir="rtl"
          >
            <option value="" disabled>
              اختر وثيقة...
            </option>
            <option value="">جميع الوثائق</option>
            {documentTitles.map((title, index) => (
              <option key={index} value={title}>
                {title}
              </option>
            ))}
          </select>
        </div>

        <div className="chat-history directive" dir="rtl">
          {chatHistory.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.type}`}>
              {msg.type === "question" && (
                <div className="question-bubble">
                  <div className="message-content">{msg.content}</div>
                  <div className="message-time">
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              )}

              {msg.type === "answer" && (
                <div className="answer-bubble">
                  <div className="answer-content" dir="rtl" lang="ar">
                    {formatAnswer(msg.content).formattedAnswer.map((item, i) =>
                      item.type === "list" ? (
                        <ol key={i} className="answer-list" start="١">
                          <li
                            dangerouslySetInnerHTML={{ __html: item.content }}
                          />
                        </ol>
                      ) : (
                        <p
                          key={i}
                          dangerouslySetInnerHTML={{ __html: item.content }}
                        />
                      )
                    )}
                  </div>

                  {formatAnswer(msg.content).sources?.length > 0 && (
                    <div className="source-grid">
                      {formatAnswer(msg.content).sources.map((source, i) => (
                        <div key={i} className="source-item">
                          <div>
                            <div className="source-file">{source.file}</div>
                            <div className="source-meta">
                              <span>مرجع [{source.number}]</span>
                              <span>• المقطع {source.chunk}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="message-time">
                    {msg.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                  <div className="feedback-container text-red-950">
                    <div className="feedback-buttons">
                      <button
                        className={`feedback-btn ${
                          msg.feedback === "like" ? "active" : ""
                        }`}
                        onClick={() => handleFeedback(index, "like")}
                        disabled={!!msg.feedback}
                      >
                        <ThumbsUp size={16} />
                      </button>
                      <button
                        className={`feedback-btn ${
                          msg.feedback === "dislike" ? "active" : ""
                        }`}
                        onClick={() => {
                          if (!msg.feedback) {
                            setSelectedFeedbackIndex(index);
                          }
                        }}
                        disabled={!!msg.feedback}
                      >
                        <ThumbsDown size={16} />
                      </button>
                      <button
                        className="feedback-btn copy-btn"
                        onClick={() => handleCopy(index)}
                      >
                        <Copy size={16} />
                      </button>
                    </div>

                    {selectedFeedbackIndex === index && (
                      <div className="feedback-input-container">
                        <textarea
                          value={feedbackComment}
                          onChange={(e) => setFeedbackComment(e.target.value)}
                          placeholder="الرجاء توضيح سبب عدم الرضا..."
                          dir="rtl"
                          rows="2"
                        />
                        <div className="feedback-submit-btns">
                          <button
                            onClick={() => submitFeedback(index)}
                            disabled={!feedbackComment.trim()}
                          >
                            إرسال الملاحظات
                          </button>
                          <button
                            className="cancel-btn"
                            onClick={() => {
                              setSelectedFeedbackIndex(null);
                              setFeedbackComment("");
                            }}
                          >
                            إلغاء
                          </button>
                        </div>
                      </div>
                    )}
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
            placeholder="اسأل عن الوثيقة المختارة..."
            disabled={isLoading.question}
            onKeyDown={(e) => e.key === "Enter" && handleQuestion()}
            dir="auto"
          />
          <button
            onClick={handleQuestion}
            disabled={isLoading.question || !question.trim()}
            className={isLoading.question ? "loading" : ""}
          >
            {isLoading.question ? (
              <>
                <RotateCw className="spin" size={16} /> جارٍ التحليل...
              </>
            ) : (
              "إرسال"
            )}
          </button>
        </div>
      </section>
    </div>
  );
}

export default TxtQAInterface;
