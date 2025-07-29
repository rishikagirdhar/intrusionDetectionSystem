Intrusion Detection System (IDS)

This project is a web-based Intrusion Detection System with:
    Frontend: Built using React (for data input, file upload, and visualization).
    Backend: Powered by Flask (ML models for attack detection).
    Chatbot: Integrated via Streamlit (powered by Groq LLM).

Features
    Manual & Bulk Prediction: Detect attacks by entering values manually or uploading CSV/Excel files.
    Real-time Risk Classification: Predicts attack type, confidence level, and risk severity.
    Integrated Chatbot: A chatbot interface for interacting with the system.
    Single Command Startup: Start frontend, backend, and chatbot together using one command.

Project Structure

    intrusionDetectionSystem/
    │
    ├── ids-frontend/        # React frontend
    │   ├── public/
    │   ├── src/
    │   └── package.json
    │
    ├── ids-backend/         # Flask backend and chatbot
    │   ├── models/          # ML models (.pkl files)
    │   ├── backend.py       # Main API for IDS
    │   ├── app.py           # Streamlit chatbot app
    │   ├── requirements.txt # Python dependencies
    │
    └── README.md
    
Setup Instructions
  1. Clone the Repository
    git clone https://github.com/rishikagirdhar/intrusionDetectionSystem.git
    cd intrusionDetectionSystem
  
  3. Backend Setup (Flask + Streamlit)
  Install Python Dependencies
    cd ids-backend
    pip install -r requirements.txt
  Environment Variables- Create a .env file inside ids-backend/ and add:
    GROQ_API_KEY=your_api_key_here
  
  4. Frontend Setup (React)
    cd ../ids-frontend
    npm install
  
  5. Start the Entire Project
  From inside ids-frontend/, run:
    npm run dev
  
  This will:
    Start Flask backend (backend.py).
    Start Streamlit chatbot (app.py) in headless mode.
    Start React frontend.

Key Scripts
   
  Frontend
    npm start – Start React frontend only.
    npm run dev – Start frontend + backend + chatbot together.
  
  Backend
    cd ids-backend
    python backend.py
  
  Chatbot
    cd ids-backend
    streamlit run app.py --server.headless true

Technologies Used
  Frontend: React, Tailwind CSS.
  Backend: Flask, Scikit-learn, Pandas, NumPy.
  Chatbot: Streamlit, LangChain Groq API.
  Others: XLSX for Excel parsing, Concurrently for parallel startup.

License
    This project is licensed under the MIT License.

<img width="1351" height="866" alt="image" src="https://github.com/user-attachments/assets/813d8a2f-cd35-437a-a6f7-e11aafe239c4" />
<img width="1854" height="853" alt="image" src="https://github.com/user-attachments/assets/c4351007-83ad-4fea-a32f-8c393ecd1823" />
<img width="531" height="835" alt="image" src="https://github.com/user-attachments/assets/367231ff-ab60-4e54-829a-2b51bbe427eb" />
<img width="339" height="868" alt="image" src="https://github.com/user-attachments/assets/89751785-86c1-47a9-a60f-b70e73aaae5a" />


