import React, { useState } from "react";
import IDSFrontend from "./IDSFrontend";

function App() {
  const [showChatbot, setShowChatbot] = useState(false);

  return (
    <div className="bg-black min-h-screen text-white p-4">
      {/* IDS Frontend Section */}
      <IDSFrontend />

      {/* Chatbot Toggle Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <button
          onClick={() => setShowChatbot(!showChatbot)}
          className="bg-gradient-to-r from-green-500 to-blue-500 px-4 py-2 rounded-lg shadow-lg hover:scale-105 transition-transform"
        >
          {showChatbot ? "Close chat" : "Talk to assistant"}
        </button>

        {/* Chatbot iframe */}
        {showChatbot && (
          <div className="mt-2 border border-gray-700 rounded-lg overflow-hidden shadow-lg">
            <iframe
              src="http://localhost:8501"
              title="Chatbot"
              className="w-[400px] h-[600px] rounded-lg shadow-2xl border border-gray-700"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
