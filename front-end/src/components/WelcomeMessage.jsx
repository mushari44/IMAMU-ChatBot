import React, { useContext } from "react";
import { GlobalContext } from "../context";

const WelcomeMessage = ({ timestamp }) => {
  const { setQuery } = useContext(GlobalContext);

  const welcomingText = [
    "👋 مرحباً بك في سراج!",
    "كيف يمكنني مساعدتك؟",
    "يمكنك الاستفسار حول أي جزء من محتوى المدونة",
  ];

  const suggestions = [
    "من هو مؤلف كتاب استراتيجيات القراءة ؟",
    "اذكر لي 3 كتب نشرت في السعودية",
  ];


  return (
    <div className="Message-Suggests">
      <div className="Time-Icon">
        {/* <img src={sirajIcon} alt="Siraj icon" /> */}
        <div>
          <h2>IMAMU BOT</h2>
          <p className="text-sm">{timestamp}</p>
        </div>
      </div>

      <div className="bg-blue-600 text-white rounded-md p-4 Welcome-Message">
        {welcomingText.map((line, index) => (
          <p className="" key={index}>
            {line}
          </p>
        ))}
      </div>

      <div className="SuggestQ">
        {suggestions.map((suggestion, index) => (
          <p
            key={index}
            className="bg-blue-600 text-white rounded-md p-4 suggest"
            onClick={() => setQuery(suggestion)}
          >
            {suggestion}
          </p>
        ))}
      </div>
    </div>
  );
};

export default WelcomeMessage;
