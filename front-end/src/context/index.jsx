import React, { createContext, useEffect, useState } from "react";
import axios from "axios";

export const GlobalContext = createContext(null);

export default function GlobalState({ children }) {
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [positive, setPositive] = useState(false);
  const handleFeedBack = async (index,feedBack) => {
    // console.log("Submitting feedback: ", feedBack);
    // console.log("QUERI : ",query);
    // console.log("CHAT : ",chatHistory);
    // console.log("index  : ",index);
    // console.log(chatHistory[index]);
      
      const currentQuery=chatHistory[index].query
      const currentRespose=chatHistory[index].response
      const currentBotTimeStamp=chatHistory[index].botTimestamp
      const currentUserTimeStamp=chatHistory[index].timestamp
      const currentDocIds=chatHistory[index].doc_ids
      const currentOriginalText=chatHistory[index].original_text
    // console.log("CUURENT IDDS : ",currentBotTimeStamp);
    // console.log("CUURENT IDsdasdDS : ",currentUserTimeStamp);

    try {
      const res = await axios.post("http://127.0.0.1:8000/feedback", {
        satisfied: feedBack,
        query:currentQuery,
        response:currentRespose,
        original_text:currentOriginalText,
        botTimestamp:currentBotTimeStamp,
        timestamp:currentUserTimeStamp,
      });
      // console.log("Feedback Response: ", res.data);
      // setIsPressed(feedBack?"like":"dislike")
    } catch (error) {
      console.error("Error sending feedback: ", error);
    }
  };


  const generateTimestamp = () => {
    return new Date()
      .toLocaleTimeString("ar-SA", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      })
      .replace("ص", "صباحًا")
      .replace("م", "مساءً");
  };

  const handleSend = async (e) => {
    if (e) e.preventDefault();

    if (!query.trim()) return;

    const timestamp = generateTimestamp();

    setChatHistory((prev) => [
      ...prev,
      { query, response: null,original_text: null, timestamp: timestamp,isPressed:null ,doc_ids:[]},
    ]);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/chatbot", { query });
      const botTimestamp = generateTimestamp();
      // console.log("CHAT BOT RES : ", res);
    
      setChatHistory((prev) =>
        prev.map((chat, index) =>
          index === prev.length - 1
            ? {
                ...chat,
                response: res.data.response,
                original_text: res.data.original_text, 
                botTimestamp,
                doc_ids: res.data.doc_ids,
              }
            : chat
        )
      );
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      const errorTimestamp = generateTimestamp();
      setChatHistory((prev) =>
        prev.map((chat, index) =>
          index === prev.length - 1
            ? {
                ...chat,
                response:
                  "حدث خطأ أثناء معالجة استفسارك. الرجاء المحاولة مرة أخرى.",
                botTimestamp: errorTimestamp,
              }
            : chat
        )
      );
        
    } finally {
      setLoading(false);
      // console.log("CHAT IN HANDLE SEND FINALLY : ",chatHistory);
      
    }
  };

  
  

  return (
    <GlobalContext.Provider
      value={{
        query,
        setQuery,
        loading,
        setLoading,
        chatHistory,
        setChatHistory,
        generateTimestamp,
        handleSend,
        positive,
        setPositive,
        handleFeedBack
      }}
    >
      {children}
    </GlobalContext.Provider>
  );
}
