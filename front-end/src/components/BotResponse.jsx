import React, { useContext, useState, useEffect } from "react";
import Loader from "./Loader";
import copyIcon from "../assets/copy.svg";
import checkIcon from "../assets/check.svg";
import likeIcon from "../assets/like.svg";
import disLikeIcon from "../assets/dislike.svg";
import readMore from "../assets/more.png";
import readLess from "../assets/less.png";
import { GlobalContext } from "../context";

const BotResponse = ({ response, original_text, refined_query, timestamp, index }) => {
  const { handleFeedBack } = useContext(GlobalContext);

  const [copied, setCopied] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const [typedContent, setTypedContent] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isPressed, setIsPressed] = useState("");

  const isBooksObject =
    response &&
    typeof response === "object" &&
    Array.isArray(response.books) &&
    response.books.length > 0;

 
  const buildTypingContent = () => {
    const content = [];
    return content;
  };

  
  useEffect(() => {
    let isCancelled = false; 

    const TYPING_SPEED = 10; 
    const CONTENT = buildTypingContent();
    let currentIndex = 0;
    let currentChar = 0;
    let timeout;

    const typeNext = () => {
      if (isCancelled) return;
      if (currentIndex < CONTENT.length) {
        const currentContent = CONTENT[currentIndex];
        if (!typedContent[currentIndex]) {
          typedContent[currentIndex] = { type: currentContent.type, text: "" };
        }

        if (currentChar < currentContent.text.length) {
          typedContent[currentIndex].text += currentContent.text[currentChar];
          setTypedContent([...typedContent]);
          currentChar++;
          timeout = setTimeout(typeNext, TYPING_SPEED);
        } else {
          currentIndex++;
          currentChar = 0;
          timeout = setTimeout(typeNext, 200); 
        }
      } else {
        setIsTyping(false);
      }
    };

    if (CONTENT.length > 0) {
      setIsTyping(true);
      setTypedContent([]); 
      currentIndex = 0;
      currentChar = 0;
      typeNext();
    }

    return () => {
      isCancelled = true;
      clearTimeout(timeout);
    };
  }, [response]);

  const handleCopy = () => {
    const textToCopy = isBooksObject ? JSON.stringify(response, null, 2) : response;
    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch((err) => {
        console.error("Failed to copy text: ", err);
      });
  };

  
  const toggleOriginalText = () => {
    setShowOriginal(!showOriginal);
  };

  return (
    <div className="Bot-Response">
      {!response ? (
        refined_query ? (
          <div
            className="bg-yellow-100 p-4 rounded-md mb-2"
            style={{ textAlign: "right", direction: "rtl" }}
          >
            <p className="text-yellow-800">تم تعديل الاستفسار إلى: {refined_query}</p>
          </div>
        ) : (
          <div className="loader">
            <div className="bg-blue-600 text-white rounded-md">
              <Loader />
            </div>
          </div>
        )
      ) : (
        <>
          <div className="Time-Icon">
            {/* <img src={} alt="Siraj icon" /> */}
            <div>
              <h2 className="text-lg">سِراج</h2>
              <p className="text-sm">{timestamp}</p>
            </div>
          </div>

          <div className="Response-Container">
            {typedContent.length > 0 ? (
              isBooksObject ? (
                <ol style={{ padding: "0 20px" }}>
                  {typedContent.map((content, i) => (
                    <li key={i} style={{ marginBottom: "15px" }}>
                      {content.type === "header" && (
                        <strong>
                          {content.text}
                          {isTyping && i === typedContent.length - 1 && <span className="typing-cursor">|</span>}
                        </strong>
                      )}
                      {content.type === "detail" && (
                        <ul style={{ listStyleType: "circle", paddingRight: "35px", marginTop: "5px" }}>
                          <li>
                            {content.text}
                            {isTyping && i === typedContent.length - 1 && <span className="typing-cursor">|</span>}
                          </li>
                        </ul>
                      )}
                    </li>
                  ))}
                </ol>
              ) : (
                <div>
                  <div>
  {typedContent.map((content, i) => {
    if (content.type === "paragraph") {
      return (
        <p key={i} style={{ marginBottom: "10px", padding: "15px" }}>
          {content.text}
          {isTyping && i === typedContent.length - 1 && <span className="typing-cursor">|</span>}
        </p>
      );
    }
    
    if (content.type === "header") {
      return (
        <h2 key={i} style={{ fontWeight: 600, fontSize: "larger", marginBottom: "20px" }}>
          {content.text}
          {isTyping && i === typedContent.length - 1 && <span className="typing-cursor">|</span>}
        </h2>
      );
    }
    
    if (content.type === "subtitle") {
      const subtitleNumber = typedContent
        .slice(0, i)
        .filter((c) => c.type === "subtitle").length + 1;
      
      return (
        <h3
          key={i}
          style={{
            fontSize: "medium",
            fontWeight: 500,
            marginBottom: "3px",
            paddingRight: "10px",
          }}
        >
          {subtitleNumber}. {content.text}
          {isTyping && i === typedContent.length - 1 && <span className="typing-cursor">|</span>}
        </h3>
      );
    }
    
    // Handle any other content types if necessary
    return null;
  })}
</div>

                </div>
              )
            ) : (
              <p>جاري كتابة الرد...</p>
            )}
          </div>

          {/* Show Original Text if toggled */}
          {showOriginal && !isBooksObject && (
            <div className="bg-gray-50 p-4 mt-4 border rounded-md Response-Container">
              <h3 className="text-lg font-semibold mb-2">النصوص الأصلية:</h3>
              {/* Display original_text with formatting */}
              {(() => {
                const lines = original_text.split("\n");
                return lines.map((line, i) => {
                  const trimmedLine = line.trim();
                  if (!trimmedLine) return null;

                  if (/^\*\*(.+?)\*\*$/.test(trimmedLine)) {
                    const headerText = trimmedLine.replace(/^\*\*(.+?)\*\*$/, "$1").trim();
                    return (
                      <h2 key={`original-h2-${i}`} style={{ fontWeight: 600, fontSize: "larger", marginBottom: "20px" }}>
                        {headerText}
                      </h2>
                    );
                  } else if (/^\*(.+?)\*$/.test(trimmedLine)) {
                    const subtitleText = trimmedLine.replace(/^\*(.+?)\*$/, "$1").trim();
                    return (
                      <h3
                        key={`original-h3-${i}`}
                        style={{
                          fontSize: "medium",
                          fontWeight: 500,
                          marginBottom: "3px",
                          paddingRight: "10px",
                        }}
                      >
                        {subtitleText}
                      </h3>
                    );
                  } else {
                    return (
                      <p key={`original-p-${i}`} style={{ marginBottom: "10px", padding: "15px" }}>
                        {trimmedLine}
                      </p>
                    );
                  }
                });
              })()}
            </div>
          )}

          <div
            className="flex items-center"
            style={{ justifyContent: "flex-end", direction: "rtl" }}
          >
            <button
              id="readmoreId"
              className="rounded-lg text-token-text-secondary hover:bg-token-main-surface-secondary relative group"
              aria-label="Read More"
              onClick={toggleOriginalText}
            >
              <span className="flex items-center justify-center h-[30px] w-[30px]">
                <img
                  className="feedback-icon read-more"
                  src={showOriginal && !isBooksObject ? readLess : readMore}
                  alt={showOriginal ? "Read Less" : "Read More"}
                />
              </span>
            </button>

            <button
              className="rounded-lg text-token-text-secondary hover:bg-token-main-surface-secondary"
              aria-label="Copy"
              onClick={handleCopy}
            >
              <span className="flex items-center justify-center h-[30px] w-[30px]">
                <img
                  className="feedback-icon"
                  src={copied ? checkIcon : copyIcon}
                  alt={copied ? "Check Icon" : "Copy Icon"}
                />
              </span>
            </button>

            <button
              id="feedback"
              className="rounded-lg text-token-text-secondary hover:bg-token-main-surface-secondary"
              aria-label="Like"
              onClick={() => {
                handleFeedBack(index, true);
                setIsPressed("like");
              }}
              hidden={isPressed === "dislike"}
              disabled={isPressed === "like"}
            >
              <span className="flex items-center justify-center h-[30px] w-[30px]">
                <img className="feedback-icon" src={likeIcon} alt="Like Icon" />
              </span>
            </button>

            <button
              id="feedback"
              className="rounded-lg text-token-text-secondary hover:bg-token-main-surface-secondary"
              aria-label="Dislike"
              onClick={() => {
                handleFeedBack(index, false);
                setIsPressed("dislike");
              }}
              hidden={isPressed === "like"}
              disabled={isPressed === "dislike"}
            >
              <span className="flex items-center justify-center h-[30px] w-[30px]">
                <img className="feedback-icon" src={disLikeIcon} alt="Dislike Icon" />
              </span>
            </button>
          </div>

          <style jsx>{`
            .typing-cursor {
              display: inline-block;
              width: 1px;
              background-color: black;
              animation: blink 1s step-start infinite;
              margin-left: 2px;
            }

            @keyframes blink {
              50% {
                opacity: 0;
              }
            }
          `}</style>
        </>
      )}
    </div>
  );
};

export default BotResponse;
